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

def topk_patches(x, dinov2, k):
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

    sorted_indices = attn_avg.argsort(dim=-1, descending=True)  # (B, 256)

    if k == -1:
        indices = sorted_indices                         # all 256 patches
    else:
        indices = sorted_indices[:, :k]                 # top-k only

    return indices, attn_avg

class PurifiedClassifier(nn.Module):
    def __init__(self, rae, dit, classifier, t_noise, n_steps, k=10, weight_scheme='equal', neighborhood_radius=3):
        super().__init__()
        self.rae = rae
        self.dit = dit
        self.classifier = classifier
        self.t_noise = t_noise
        self.n_steps = n_steps
        self.k = k
        self.neighborhood_radius = neighborhood_radius

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
        B      = x.shape[0]
        device = x.device
        y_null = torch.full((B,), self.dit.y_embedder.num_classes, dtype=torch.long, device=device)
        
        with torch.no_grad():
            indices, _ = topk_patches(x, self.classifier.dinov2, k=self.k)
        n_steps = indices.shape[1]

        if self.weights is None:
            weights = torch.ones(n_steps, device=device) / n_steps
        else:
            weights = self.weights

        global_token = torch.zeros(B, 768, 16, 16, device=device)

        local = None
        with torch.no_grad():
            z_full = self.rae.encode(x)
            for step in range(self.k):
                z_input = torch.randn_like(z_full) if local is None else local.clone()

                # inject adversarial content at attended position + neighborhood
                for b in range(B):
                    idx = indices[b, step].item()
                    row, col = idx // 16, idx % 16
                    for r, c in get_neighborhood(row, col, radius=self.neighborhood_radius):
                        z_input[b, :, r, c] = z_full[b, :, r, c]

                noise = torch.randn_like(z_input)
                z_t   = (1 - self.t_noise) * z_input + self.t_noise * noise

                # reverse ODE
                dt = self.t_noise / self.n_steps
                for s in range(self.n_steps):
                    t_cur = self.t_noise - s * dt
                    t_vec = torch.full((B,), t_cur, device=device,dtype=torch.float32)
                    v     = self.dit(z_t, t_vec, y_null)
                    z_t   = z_t - v * dt

                z_clean = z_t  # full dense (B, 768, 16, 16)
                global_token = global_token + weights[step] * z_clean
                local = z_clean

        x_rec  = self.rae.decode(global_token).clamp(0, 1)
        x_rec  = F.interpolate(x_rec, size=(224, 224), mode='bicubic', align_corners=False)

        # BPDA: forward uses purified, backward is straight-through
        x_bpda = x + (x_rec - x).detach()
        return self.classifier(x_bpda)