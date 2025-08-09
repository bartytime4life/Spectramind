import torch, torch.nn as nn

class FGS1MambaEncoder(nn.Module):
    """Lightweight Mamba-style SSM stub for temporal features."""
    def __init__(self, in_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
