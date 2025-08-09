import torch, torch.nn as nn

class MultiScaleDecoder(nn.Module):
    """Minimal multi-resolution decoder stub that encourages smooth output."""
    def __init__(self, latent_dim=64, out_bins=283):
        super().__init__()
        self.low = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU())
        self.mid = nn.Sequential(nn.Linear(128, 128), nn.GELU())
        self.high = nn.Linear(128, out_bins)

    def forward(self, z):
        h = self.low(z)
        h = self.mid(h)
        mu = self.high(h)
        return mu
