import torch.nn as nn
import torch.nn.functional as F
import torch

def purify(rae, dit, x, t_noise, n_steps):
    z = rae.encode(x)                                              # (B, 768, 16, 16)
    noise = torch.randn_like(z)
    z_t = (1 - t_noise) * z + t_noise * noise                     # forward to t_noise
    y_null = torch.full((x.shape[0],), 1000, dtype=torch.long, device=x.device)
    dt = t_noise / n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t_vec = torch.full((x.shape[0],), t_noise - i * dt, device=x.device)
            z_t = z_t - dit(z_t, t_vec, y_null) * dt              # Euler reverse step
    x_rec = rae.decode(z_t).clamp(0, 1)
    return F.interpolate(x_rec, size=x.shape[-2:], mode='bicubic', align_corners=False)

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

class PurifiedClassifier(nn.Module):
    def __init__(self, rae, dit, classifier, t_noise, n_steps):
        super().__init__()
        self.rae        = rae
        self.dit        = dit
        self.classifier = classifier
        self.t_noise    = t_noise
        self.n_steps    = n_steps

    def forward(self, x):
        x_purified = purify(self.rae, self.dit, x, self.t_noise, self.n_steps)
        # BPDA: forward uses purified image, backward is straight-through (identity)
        x_bpda = x + (x_purified - x).detach()
        return self.classifier(x_bpda)