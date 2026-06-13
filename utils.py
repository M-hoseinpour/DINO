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

def load_prototypes(device, train_loader, normalize, dinov2, n_classes=10, cache_path=None):
    # load from cache if available
    if cache_path and os.path.exists(cache_path):
        print(f'[prototypes] loaded from {cache_path}')
        return torch.load(cache_path, map_location=device)

    print(f'[prototypes] computing for {n_classes} classes...')
    dinov2.eval()
    sums   = torch.zeros(n_classes, 768, device=device)
    counts = torch.zeros(n_classes,      device=device)

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc='prototypes'):
            images = images.to(device)
            labels = labels.to(device)
            feats  = dinov2(normalize(images))          # (B, 768)
            feats  = F.normalize(feats, dim=-1)
            for c in range(n_classes):
                mask = (labels == c)
                if mask.any():
                    sums[c]   += feats[mask].sum(0)
                    counts[c] += mask.sum()

    prototypes = F.normalize(sums / counts.unsqueeze(1), dim=-1)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(prototypes, cache_path)
        print(f'[prototypes] saved to {cache_path}')

    return prototypes

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