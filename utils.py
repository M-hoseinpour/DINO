import torch
import os
from torch.nn import attention
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import random
import numpy as np
from huggingface_hub import hf_hub_download
from pathlib import Path

CIFAR10_Classes = ["frog", 'airplane', 'cat', 'dog', 'horse', 'automobile', 'truck', 'bird', 'deer', 'ship']

def seed_eveything(is_cuda):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if is_cuda:
        torch.cuda.manual_seed(0)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],     # ImageNet mean
    std=[0.229, 0.224, 0.225]       # ImageNet std
)

def clean_accuracy(test_loader, model, device):
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            preds  = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    print(f'Clean accuracy (DINOv2-B): {correct/total*100:.2f}%')

def load_prototypes(device, train_loader, normalize, dinov2):
    PROTOTYPE_PATH = './dinov2_b_prototypes.pt'

    if os.path.exists(PROTOTYPE_PATH):
        prototypes = torch.load(PROTOTYPE_PATH, map_location=device)
        if torch.isnan(prototypes).any():
            os.remove(PROTOTYPE_PATH)
            print('Corrupted — recomputing')
        else:
            print(f'Loaded prototypes: {prototypes.shape}')
    else:
        prototype_sums   = torch.zeros(10, 768, device=device)
        prototype_counts = torch.zeros(10, device=device)

        with torch.no_grad():
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                embeddings = dinov2(normalize(images))
                embeddings = F.normalize(embeddings, dim=-1)

                nan_mask = torch.isnan(embeddings).any(dim=1)
                if nan_mask.any():
                    embeddings = embeddings[~nan_mask]
                    labels     = labels[~nan_mask]

                for c in range(10):
                    mask = (labels == c)
                    if mask.any():
                        prototype_sums[c]   += embeddings[mask].sum(dim=0)
                        prototype_counts[c] += mask.sum()

        prototypes = prototype_sums / prototype_counts.unsqueeze(1)
        prototypes = F.normalize(prototypes, dim=-1)

        print(f'NaN: {torch.isnan(prototypes).any()}')
        print(f'Shape: {prototypes.shape}')
        torch.save(prototypes, PROTOTYPE_PATH)
    return prototypes

def topk_patches(x, dinov2, k=10):
    attn_list = []
    def hook(module, input, _output):
        batch, num_toknes, embedding_dim = input[0].shape
        num_heads = module.num_heads
        head_dim  = embedding_dim // num_heads
        qkv = module.qkv(input[0]).reshape(batch, num_toknes, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, kk, _ = qkv.unbind(0)

        attn = (q @ kk.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn_list.append(attn.detach())

    handle = dinov2.blocks[-1].attn.register_forward_hook(hook)
    with torch.no_grad():
        dinov2(normalize(x))
    handle.remove()

    cls_attn = attn_list[0][:, :, 0, 1:]  # (batch, num_heads, 256)
    cls_attn = cls_attn.mean(dim=1)       # (batch, 256) — average over heads

    topk_indices = cls_attn.topk(k, dim=-1).indices  # (batch, k)
    return topk_indices, cls_attn

def extract_crops(x, topk_indices, crop_size=84, target_size=224):
    """
    x:             (batch, 3, 224, 224)
    topk_indices:  (batch, k) — flat patch indices in 0..255
    returns:       (batch, k, 3, 224, 224)
    """
    batch, k    = topk_indices.shape
    half        = crop_size // 2 # 42
    patch_size  = 14
    grid_size   = 16  # 224 / 14 = 16 patches per side

    all_crops = []

    for b in range(batch):
        crops_b = []
        for i in range(k):
            idx = topk_indices[b, i].item()

            # convert flat index to grid position
            row = idx // grid_size      # 0..15
            col = idx  % grid_size      # 0..15

            # center of this patch in the 224×224 image
            cx = col * patch_size + patch_size // 2    # pixel x center
            cy = row * patch_size + patch_size // 2    # pixel y center

            # 84×84 window centered on (cx, cy)
            x0 = cx - half
            y0 = cy - half
            x1 = x0 + crop_size
            y1 = y0 + crop_size

            # clamp: shift window to stay inside 224×224
            if x0 < 0:
                x0, x1 = 0, crop_size
            if x1 > target_size:
                x1, x0 = target_size, target_size - crop_size
            if y0 < 0:
                y0, y1 = 0, crop_size
            if y1 > target_size:
                y1, y0 = target_size, target_size - crop_size

            crop = x[b:b+1, :, y0:y1, x0:x1]          # (1, 3, 84, 84)
            crop = F.interpolate(
                crop, size=(target_size, target_size),
                mode='bicubic', align_corners=False
            )                                            # (1, 3, 224, 224)
            crops_b.append(crop)

        all_crops.append(torch.cat(crops_b, dim=0))     # (k, 3, 224, 224)

    return torch.stack(all_crops, dim=0)                 # (batch, k, 3, 224, 224)


def ensure_weights(decoder_path: str, stats_path: str, dit_path: str) -> None:
    repo = 'nyu-visionx/RAE-collections'
    files = {
        decoder_path: 'decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
        stats_path:   'stats/dinov2/wReg_base/imagenet1k/stat.pt',
        dit_path:     'DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt',
    }
    for local, remote in files.items():
        if not Path(local).exists():
            print(f'Downloading {remote} ...')
            hf_hub_download(repo_id=repo, filename=remote, local_dir='RAE/models')