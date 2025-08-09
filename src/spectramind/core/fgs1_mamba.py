import torch, torch.nn as nn

class FGS1MambaEncoder(nn.Module):
    """Placeholder FGS1 temporal encoder (Mamba-like stub)."""
    def __init__(self, in_dim=64, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))

    def forward(self, x):
        return self.net(x)
