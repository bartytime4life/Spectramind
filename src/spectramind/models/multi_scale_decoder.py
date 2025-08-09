import torch, torch.nn as nn

class MultiScaleDecoder(nn.Module):
    """Simple multi-scale MLP decoder for μ."""
    def __init__(self, latent_dim: int = 64, out_bins: int = 283):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, out_bins)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
