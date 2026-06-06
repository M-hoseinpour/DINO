import sys
import argparse
from pathlib import Path
from typing import cast
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from autoattack import AutoAttack
from classifier import DINO_classification, PurifiedClassifier, purify
from consts import DECODER_PATH, DIT_PATH, STATS_PATH
from data import cifar10_loader
from utils import ensure_weights, extract_crops, load_prototypes, normalize, seed_eveything, topk_patches

sys.path.insert(0, str(Path(__file__).parent / 'RAE' / 'src'))
from stage1.rae import RAE
from stage2.models.DDT import DiTwDDTHead

p = argparse.ArgumentParser()
p.add_argument('--decoder-path', default=DECODER_PATH)
p.add_argument('--stats-path',   default=STATS_PATH)
p.add_argument('--dit-path',     default=DIT_PATH)
p.add_argument('--t-noise',      type=float, default=0.3)
p.add_argument('--n-steps',      type=int,   default=20)
p.add_argument('--eps',          type=float, default=8/255)
p.add_argument('--batch-size',   type=int,   default=16)
p.add_argument('--n-samples',    type=int,   default=None)
p.add_argument('--n-batches',    type=int,   default=None)
p.add_argument('--crop-size', type=int, default=84)

if __name__ == "__main__":
    args = p.parse_args()

    is_cuda = torch.cuda.is_available()
    device  = torch.device('cuda' if is_cuda else 'cpu')
    seed_eveything(is_cuda)

    train_loader, test_loader = cifar10_loader(test_batch_size=args.batch_size)

    dinov2 = cast(nn.Module, torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'))
    dinov2 = dinov2.to(device).eval()

    prototypes = load_prototypes(device, train_loader, normalize, dinov2)
    classifier = DINO_classification(dinov2, prototypes, normalize).to(device).eval()
    ensure_weights(args.decoder_path, args.stats_path, args.dit_path)

    rae = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-with-registers-base',
        encoder_input_size=224,
        encoder_params={'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
        decoder_config_path='RAE/configs/decoder/ViTXL',
        pretrained_decoder_path=args.decoder_path,
        noise_tau=0.,
        reshape_to_2d=True,
        normalization_stat_path=args.stats_path,
    ).to(device).eval()

    dit = DiTwDDTHead(
        input_size=16, patch_size=1, in_channels=768,
        hidden_size=[1152, 2048], depth=[28, 2], num_heads=[16, 16],
        mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=1000,
        use_qknorm=False, use_swiglu=True, use_rope=True,
        use_rmsnorm=True, wo_shift=False, use_pos_embed=True,
    )
    dit.load_state_dict(torch.load(args.dit_path, map_location='cpu'))
    dit = dit.to(device).eval()

    purified_model = PurifiedClassifier(rae, dit, classifier, t_noise=args.t_noise, n_steps=args.n_steps).to(device).eval()

    adversary = AutoAttack(purified_model, norm='Linf', eps=args.eps, version='rand', device=str(device), verbose=False)

    total = 0
    for i, (images, labels) in enumerate(tqdm(test_loader, desc='eval')):
        if args.n_batches is not None and i >= args.n_batches:
            break
        if args.n_samples is not None and total >= args.n_samples:
            break

        images = images.to(device)
        labels = labels.to(device)
        batch = images.shape[0]

        indices, attn = topk_patches(images, dinov2, k=10)
        # crops = extract_crops(images, indices, crop_size=args.crop_size)  # (batch, 10, 3, 224, 224)

        local_token  = torch.zeros(batch, 768, 16, 16, device=device)
        global_token = torch.zeros(batch, 768, 16, 16, device=device)

        with torch.no_grad():
            results = []
            y_null  = torch.full((1,), dit.y_embedder.num_classes,
                                  dtype=torch.long, device=device)

            for b in range(batch):
                local_token  = torch.zeros(1, 768, 16, 16, device=device)
                global_token = torch.zeros(1, 768, 16, 16, device=device)
                z_full       = rae.encode(images[b:b+1])   # (1, 768, 16, 16)

                for step in range(indices.shape[1]):        # k=10 steps
                    row = indices[b, step].item() // 16
                    col = indices[b, step].item() % 16

                    z_i              = torch.zeros_like(z_full)
                    z_i[:, :, row, col] = z_full[:, :, row, col]

                    local_token = local_token + z_i
                    local_norm  = local_token / (step + 1)

                    noise = torch.randn_like(local_norm)
                    z_t   = (1 - args.t_noise) * local_norm + args.t_noise * noise

                    dt = args.t_noise / args.n_steps
                    for s in range(args.n_steps):
                        t_cur = args.t_noise - s * dt
                        t_vec = torch.full((1,), t_cur, device=device, dtype=torch.float32)
                        v     = dit(z_t, t_vec, y_null)
                        z_t   = z_t - v * dt

                    local_clean  = z_t
                    global_token = global_token + local_clean
                    local_token  = local_clean

                global_avg      = global_token / indices.shape[1]
                x_reconstructed = rae.decode(global_avg).clamp(0, 1)
                x_reconstructed = F.interpolate(x_reconstructed, size=(224, 224), mode='bicubic', align_corners=False)
                logits = classifier(x_reconstructed)
                results.append(logits.argmax(dim=1))
                torch.cuda.empty_cache()

            preds = torch.cat(results)
            torch.cuda.empty_cache()
            print(f'batch {i}: accuracy = {(preds == labels).sum().item()}/{batch}')
            total += batch

    # adversary.apgd.eot_iter    = 1
    # adversary.apgd.n_iter      = 10
    # adversary.apgd.n_restarts  = 1

    # correct = {'clean': 0, 'clean_purified': 0, 'adv': 0, 'adv_purified': 0}

    # for i, (images, labels) in enumerate(tqdm(test_loader, desc='eval')):
    #     if args.n_batches is not None and i >= args.n_batches:
    #         break
    #     if args.n_samples is not None and total >= args.n_samples:
    #         break
    #     images, labels = images.to(device), labels.to(device)

    #     with torch.no_grad():
    #         correct['clean']          += (classifier(images).argmax(1) == labels).sum().item()
    #         correct['clean_purified'] += (classifier(purify(rae, dit, images, args.t_noise, args.n_steps)).argmax(1) == labels).sum().item()

    #     x_adv = adversary.run_standard_evaluation(images, labels, bs=len(images))

    #     with torch.no_grad():
    #         correct['adv']          += (classifier(x_adv).argmax(1) == labels).sum().item()
    #         correct['adv_purified'] += (classifier(purify(rae, dit, x_adv, args.t_noise, args.n_steps)).argmax(1) == labels).sum().item()

    #     total += labels.size(0)

    # print()
    # for k, v in correct.items():
    #     print(f'{k:16s} accuracy: {v / total * 100:.2f}%')
