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

class PurifiedClassifier(nn.Module):
    def __init__(self, rae, dit, classifier, t_noise, n_steps, k=10, weight_scheme='equal'):
        super().__init__()
        self.rae = rae
        self.dit = dit
        self.classifier = classifier
        self.t_noise = t_noise
        self.n_steps = n_steps
        self.k = k
        self.weight_scheme = weight_scheme

        w = torch.ones(k)
        if weight_scheme == 'increasing':
            w = torch.arange(1, k+1, dtype=torch.float32)
        elif weight_scheme == 'decreasing':
            w = torch.arange(k, 0, -1, dtype=torch.float32)

        self.weights = w / w.sum()

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        y_null = torch.full((1,), self.dit.y_embedder.num_classes, dtype=torch.long, device=device)
        logits_list = []

        for b in range(B):
            global_token = torch.zeros(1, 768, 16, 16, device=device)
            
            with torch.no_grad():
                z_full = self.rae.encode(x[b:b+1])
                local  = z_full.clone()
                for step in range(self.k):
                    noise = torch.randn_like(local)
                    z_t   = (1 - self.t_noise) * local + self.t_noise * noise

                    dt = self.t_noise / self.n_steps
                    for s in range(self.n_steps):
                        t_cur = self.t_noise - s * dt
                        t_vec = torch.full((1,), t_cur, device=device, dtype=torch.float32)
                        v     = self.dit(z_t, t_vec, y_null)
                        z_t   = z_t - v * dt

                    local_clean  = z_t
                    global_token = global_token + self.weights[step].to(device) * local_clean
                    local        = local_clean

            global_avg = global_token
            # global_avg = global_token / self.k
            x_rec      = self.rae.decode(global_avg).clamp(0, 1)
            x_rec      = F.interpolate(x_rec, size=(224, 224), mode='bicubic', align_corners=False)

            # BPDA: forward uses purified, backward is straight-through
            x_bpda = x[b:b+1] + (x_rec - x[b:b+1]).detach()
            logits_list.append(self.classifier(x_bpda))

        return torch.cat(logits_list)   # (B, 10) logits