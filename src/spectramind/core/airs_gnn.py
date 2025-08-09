import torch, torch.nn as nn

class AIRSSpectralGNN(nn.Module):
    """Tiny placeholder for AIRS encoder."""
    def __init__(self, in_dim=64, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))

    def forward(self, x):
        return self.net(x)
