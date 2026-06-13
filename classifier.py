import torch.nn as nn
import torch.nn.functional as F
import torch

class DINO_classification(nn.Module):
    def __init__(self, dinov2, prototypes, normalize):
        super().__init__()
        self.dinov2 = dinov2
        self.prototypes = prototypes
        self.normalize = normalize
    
    def forward(self, x):
        x = self.normalize(x)
        z = self.dinov2(x)
        z = F.normalize(z, dim=-1)
        logits = z @ self.prototypes.T * 100
        return logits

def get_neighborhood(row, col, grid_size=16, radius=1):
    """Return all positions in a radius*radius neighborhood of (row, col)."""
    positions = []
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            r, c = row + dr, col + dc
            if 0 <= r < grid_size and 0 <= c < grid_size:
                positions.append((r, c))
    return positions

def topk_patches(x, dinov2, k, min_dist=0):
    B    = x.shape[0]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    x_norm = (x - mean) / std

    qkv_holder: list[torch.Tensor] = []
    def hook(_module, _input, output):
        qkv_holder.append(output)

    last_attn = dinov2.blocks[-1].attn
    handle    = last_attn.qkv.register_forward_hook(hook)
    with torch.no_grad():
        dinov2(x_norm)
    handle.remove()

    qkv_out  = qkv_holder[0]
    N_plus1  = qkv_out.shape[1]
    H        = last_attn.num_heads
    head_dim = qkv_out.shape[2] // 3 // H

    qkv = qkv_out.reshape(B, N_plus1, 3, H, head_dim)
    q   = qkv[:, :, 0].transpose(1, 2)
    kk  = qkv[:, :, 1].transpose(1, 2)

    attn_weights = (q @ kk.transpose(-2, -1)) * (head_dim ** -0.5)
    attn_weights = attn_weights.softmax(dim=-1)

    attn_cls = attn_weights[:, :, 0, 1:]
    attn_avg = attn_cls.mean(dim=1)                      # (B, N)

    if min_dist == 0:
        sorted_indices = attn_avg.argsort(dim=-1, descending=True)
        if k == -1:
            return sorted_indices, attn_avg
        return sorted_indices[:, :k], attn_avg

    n_select = attn_avg.shape[1] if k == -1 else k
    indices_list = []

    for b in range(B):
        scores   = attn_avg[b].clone()    # (256,)
        selected = []

        for _ in range(n_select):
            available = (scores >= 0).sum().item()

            if available == 0:
                # pad with last selected if ran out of positions
                selected.append(selected[-1] if selected else 0)
                continue

            idx      = scores.argmax().item()
            selected.append(idx)

            # suppress all positions within Chebyshev distance min_dist
            row, col = idx // 16, idx % 16
            for r in range(max(0, row - min_dist), min(16, row + min_dist + 1)):
                for c in range(max(0, col - min_dist), min(16, col + min_dist + 1)):
                    scores[r * 16 + c] = -1.0

        indices_list.append(torch.tensor(selected, dtype=torch.long))

    return torch.stack(indices_list).to(x.device), attn_avg

class PurifiedClassifier(nn.Module):
    def __init__(self, rae, dit, classifier, t_noise, n_steps, k=5,
                 weight_scheme='equal', min_patch_dist=3,
                 crop_size=112, random_order=False):
        super().__init__()
        self.rae        = rae
        self.dit        = dit
        self.classifier = classifier
        self.t_noise    = t_noise
        self.n_steps    = n_steps
        self.k          = k
        self.min_dist   = min_patch_dist
        self.crop_size  = crop_size
        self.random_order = random_order

        if k != -1:
            if weight_scheme == 'increasing':
                w = torch.arange(1, k+1, dtype=torch.float32)
            elif weight_scheme == 'decreasing':
                w = torch.arange(k, 0, -1, dtype=torch.float32)
            else:
                w = torch.ones(k)
            self.register_buffer('weights', w / w.sum())
        else:
            self.weights = None

    def forward(self, x):
        B         = x.shape[0]
        device    = x.device
        y_null    = torch.full((B,), self.dit.y_embedder.num_classes,
                               dtype=torch.long, device=device)
        PATCH_PX  = 14                    # DINOv2-B14 patch size in pixels
        HALF      = self.crop_size // 2
        IMG_SIZE  = 224

        global_cls = torch.zeros(B, 768, device=device)

        with torch.no_grad():
            indices, _ = topk_patches(x, self.classifier.dinov2,
                                       k=self.k, min_dist=self.min_dist)
            n_steps    = indices.shape[1]

            if self.random_order:
                perm    = torch.randperm(n_steps, device=device)
                indices = indices[:, perm]

            weights = (torch.ones(n_steps, device=device) / n_steps
                       if self.weights is None else self.weights)

            for step in range(n_steps):
                idx_batch = indices[:, step]   # (B,)

                # ── extract crop at attended patch position ───────────
                crops = []
                for b in range(B):
                    idx      = idx_batch[b].item()
                    row, col = idx // 16, idx % 16

                    # patch centre in pixels
                    cy = row * PATCH_PX + PATCH_PX // 2
                    cx = col * PATCH_PX + PATCH_PX // 2

                    # crop boundaries — always crop_size × crop_size
                    y1 = max(0, min(cy - HALF, IMG_SIZE - self.crop_size))
                    x1 = max(0, min(cx - HALF, IMG_SIZE - self.crop_size))
                    y2 = y1 + self.crop_size
                    x2 = x1 + self.crop_size

                    crop = x[b:b+1, :, y1:y2, x1:x2]          # (1,3,H,W)
                    crop = F.interpolate(crop, (IMG_SIZE, IMG_SIZE),
                                         mode='bicubic',
                                         align_corners=False)   # (1,3,224,224)
                    crops.append(crop)

                x_crop = torch.cat(crops, dim=0)               # (B,3,224,224)

                # ── encode → noise → DiT denoise ─────────────────────
                z_crop = self.rae.encode(x_crop)
                noise  = torch.randn_like(z_crop)
                z_t    = (1 - self.t_noise) * z_crop + self.t_noise * noise

                dt = self.t_noise / self.n_steps
                for s in range(self.n_steps):
                    t_cur = self.t_noise - s * dt
                    t_vec = torch.full((B,), t_cur, device=device,
                                       dtype=torch.float32)
                    v     = self.dit(z_t, t_vec, y_null)
                    z_t   = z_t - v * dt

                # ── decode → DINOv2 → CLS vote ────────────────────────
                x_clean = self.rae.decode(z_t).clamp(0, 1)
                x_clean = F.interpolate(x_clean, (IMG_SIZE, IMG_SIZE),
                                         mode='bicubic', align_corners=False)
                cls_k   = self.classifier.dinov2(
                              self.classifier.normalize(x_clean))  # (B,768)
                cls_k   = F.normalize(cls_k, dim=-1)
                global_cls = global_cls + weights[step] * cls_k

        # ── consensus classification ──────────────────────────────────
        purified_logits = F.normalize(global_cls, dim=-1) @ self.classifier.prototypes.T * 100

        # ── BPDA: forward=purified, backward=undefended gradient ──────
        clean_logits = self.classifier(x)
        return clean_logits + (purified_logits - clean_logits).detach()