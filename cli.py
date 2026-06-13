import os
import sys
import argparse
from pathlib import Path
from typing import cast
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from autoattack import AutoAttack
from classifier import DINO_classification, PurifiedClassifier
from consts import DECODER_PATH, DIT_PATH, STATS_PATH
from data import cifar10_loader, imagenet_loader, imagenet_train_loader, imagenet_val_loader
from utils import ensure_weights, extract_crops, load_prototypes, normalize, seed_eveything

sys.path.insert(0, str(Path(__file__).parent / 'RAE' / 'src'))
from stage1.rae import RAE
from stage2.models.DDT import DiTwDDTHead

p = argparse.ArgumentParser()
p.add_argument('--decoder-path', default=DECODER_PATH)
p.add_argument('--stats-path',   default=STATS_PATH)
p.add_argument('--dit-path',     default=DIT_PATH)
p.add_argument('--t-noise',      type=float, default=0.9)
p.add_argument('--n-steps',      type=int,   default=5)
p.add_argument('--eps',          type=float, default=None)
p.add_argument('--batch-size',   type=int,   default=16)
p.add_argument('--n-samples',    type=int,   default=None)
p.add_argument('--topk',    type=int,   default=5)
p.add_argument('--weight-scheme', type=str, default='equal', choices=['equal', 'increasing', 'decreasing'])
p.add_argument('--start-idx', type=int, default=0,    help='Start sample index')
p.add_argument('--end-idx',   type=int, default=None, help='End sample index (exclusive)')
p.add_argument('--neighborhood-radius', type=int, default=3, help='Radius of neighborhood around attended position (1=3x3, 2=5x5)')
p.add_argument('--min-patch-dist', type=int, default=0, help='Minimum Chebyshev distance between selected patches. 0=standard top-k, set to neighborhood_radius to guarantee no overlap')
p.add_argument('--random-order', action='store_true', help='Shuffle patch indices randomly instead of attention ordering')
p.add_argument('--dataset',          type=str, default='cifar10', choices=['cifar10', 'imagenet'])
p.add_argument('--imagenet-train',   type=str, default=None, help='Path to ImageNet train directory')
p.add_argument('--imagenet-val',     type=str, default=None, help='Path to ImageNet val directory')
p.add_argument('--prototypes-path',  type=str, default=None, help='Path to cache/load prototypes (.pt file)')
p.add_argument('--crop-size', type=int, default=112, help='Pixel size of crop extracted at each attended patch (default 112 = 8x8 patches)')

if __name__ == "__main__":
    args = p.parse_args()

    is_cuda = torch.cuda.is_available()
    device  = torch.device('cuda' if is_cuda else 'cpu')
    seed_eveything(is_cuda)

    if args.dataset == 'imagenet':
        assert args.imagenet_val, 'provide --imagenet-val path'

        test_loader = imagenet_val_loader(args.imagenet_val, args.batch_size)

        if args.prototypes_path and os.path.exists(args.prototypes_path):
            train_loader = None   # skip — will load from cache
        else:
            assert args.imagenet_train, 'provide --imagenet-train or a cached --prototypes-path'
            train_loader = imagenet_train_loader(args.imagenet_train)

        n_classes  = 1000
        default_ep = 4/255
    else:
        train_loader, test_loader = cifar10_loader(test_batch_size=args.batch_size)
        n_classes  = 10
        default_ep = 8/255

    eps = args.eps if args.eps is not None else default_ep
    print(f'[attack] eps = {eps:.4f} ({eps*255:.1f}/255)')

    dinov2 = cast(nn.Module, torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'))
    dinov2 = dinov2.to(device).eval()

    prototypes = load_prototypes(
        device, train_loader, normalize, dinov2,
        n_classes=n_classes,
        cache_path=args.prototypes_path
    )
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

    purified_model = PurifiedClassifier(
        rae, dit, classifier,
        t_noise=args.t_noise, n_steps=args.n_steps, k=args.topk,
        weight_scheme=args.weight_scheme,
        min_patch_dist=args.min_patch_dist,
        crop_size=args.crop_size,
        random_order=args.random_order
    ).to(device).eval()

    n_eval = args.n_samples or 512
    start_idx = args.start_idx
    end_idx   = args.end_idx if args.end_idx is not None else n_eval

    adversary = AutoAttack(
        purified_model, norm='Linf', eps=eps,
        version='rand', device=str(device), verbose=True,
        log_path=f'attack_log_{start_idx}_{end_idx}.txt'
    )
    adversary.attacks_to_run       = ['apgd-ce', 'apgd-dlr', 'square']
    adversary.seed = 0

    adversary.apgd.eot_iter        = 20
    adversary.apgd.n_iter          = 100
    adversary.apgd.n_restarts      = 1

    adversary.square.n_queries     = 5000

    x_all, y_all = [], []
    for images, labels in test_loader:
        x_all.append(images)
        y_all.append(labels)
        if sum(x.shape[0] for x in x_all) >= n_eval:
            break

    x_test = torch.cat(x_all)[:n_eval].to(device)
    y_test = torch.cat(y_all)[:n_eval].to(device)
    print(f'Loaded {len(x_test)} test samples')

    print(f'Processing samples {start_idx} to {end_idx}')

    ckpt_dir = (f'ckpt_eps{int(eps*255)}_n{n_eval}'
            f'_top{args.topk}k_{args.n_steps}steps'
            f'_{args.weight_scheme}'
            f'_r{args.neighborhood_radius}_md{args.min_patch_dist}'
            f'_iter{adversary.apgd.n_iter}_eot{adversary.apgd.eot_iter}'
            f'_noise{args.t_noise}')

    os.makedirs(ckpt_dir, exist_ok=True)

    chunk_size = args.batch_size
    all_y_adv  = []
    all_labels = []

    for start in range(start_idx, end_idx, chunk_size):
        end        = min(start + chunk_size, end_idx)
        ckpt_path  = os.path.join(ckpt_dir, f'chunk_{start:04d}_{end:04d}.pt')

        if os.path.exists(ckpt_path):
            print(f'[resume] chunk {start}-{end}')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            all_y_adv.append(ckpt['y_adv'])
            all_labels.append(ckpt['labels'])
            continue

        x_chunk = x_test[start:end]
        y_chunk = y_test[start:end]

        x_adv_chunk, y_adv_chunk = adversary.run_standard_evaluation(x_chunk, y_chunk, bs=len(x_chunk), return_labels=True)

        torch.save({ 'y_adv':  y_adv_chunk.cpu(), 'labels': y_chunk.cpu() }, ckpt_path)

        all_y_adv.append(y_adv_chunk.cpu())
        all_labels.append(y_chunk.cpu())
        print(f'[saved] chunk {start}-{end}')

    y_adv_all  = torch.cat(all_y_adv)
    labels_all = torch.cat(all_labels)
    correct    = (y_adv_all == labels_all).sum().item()

    print(f'\n [GPU {start_idx}-{end_idx}] robust accuracy: {correct / len(labels_all) * 100:.2f}%' f'  ({correct}/{len(labels_all)})')
