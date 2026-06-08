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
from data import cifar10_loader
from utils import ensure_weights, extract_crops, load_prototypes, normalize, seed_eveything

sys.path.insert(0, str(Path(__file__).parent / 'RAE' / 'src'))
from stage1.rae import RAE
from stage2.models.DDT import DiTwDDTHead

p = argparse.ArgumentParser()
p.add_argument('--decoder-path', default=DECODER_PATH)
p.add_argument('--stats-path',   default=STATS_PATH)
p.add_argument('--dit-path',     default=DIT_PATH)
p.add_argument('--t-noise',      type=float, default=0.95)
p.add_argument('--n-steps',      type=int,   default=10)
p.add_argument('--eps',          type=float, default=8/255)
p.add_argument('--batch-size',   type=int,   default=16)
p.add_argument('--n-samples',    type=int,   default=None)
p.add_argument('--n-batches',    type=int,   default=None)
p.add_argument('--crop-size',    type=int,   default=84)

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

    purified_model = PurifiedClassifier(
        rae, dit, classifier,
        t_noise=args.t_noise, n_steps=args.n_steps, k=5
    ).to(device).eval()

    adversary = AutoAttack(
        purified_model, norm='Linf', eps=args.eps,
        version='rand', device=str(device), verbose=True,
        log_path='attack_log.txt'
    )
    adversary.attacks_to_run       = ['apgd-ce', 'apgd-dlr', 'square']

    adversary.apgd.eot_iter        = 20
    adversary.apgd.n_iter          = 100
    adversary.apgd.n_restarts      = 1

    adversary.apgd_targeted.eot_iter  = 20
    adversary.apgd_targeted.n_iter    = 100
    adversary.apgd_targeted.n_restarts = 1

    adversary.square.n_queries     = 5000

    n_eval = args.n_samples or 512
    x_all, y_all = [], []
    for images, labels in test_loader:
        x_all.append(images)
        y_all.append(labels)
        if sum(x.shape[0] for x in x_all) >= n_eval:
            break

    x_test = torch.cat(x_all)[:n_eval].to(device)
    y_test = torch.cat(y_all)[:n_eval].to(device)
    print(f'Loaded {len(x_test)} test samples')

    ckpt_dir = f'ckpt_eps{int(args.eps*255)}_n{n_eval}'
    os.makedirs(ckpt_dir, exist_ok=True)

    chunk_size = args.batch_size
    all_y_adv  = []
    all_labels = []

    for start in range(0, n_eval, chunk_size):
        end        = min(start + chunk_size, n_eval)
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

    print(f'\nrobust accuracy: {correct / len(labels_all) * 100:.2f}%' f'  ({correct}/{len(labels_all)})')
