import torch, torch.nn as nn

class FlowUncertaintyHead(nn.Module):
    """σ-head with Softplus to ensure positivity."""
    def __init__(self, latent_dim: int = 64, out_bins: int = 283):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, out_bins)
        )
        self.softplus = nn.Softplus()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.mlp(z)) + 1e-6
