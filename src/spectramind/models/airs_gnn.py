import torch, torch.nn as nn

class AIRSSpectralGNN(nn.Module):
    """Tiny spectral encoder; replace with real torch_geometric graph model later."""
    def __init__(self, in_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
