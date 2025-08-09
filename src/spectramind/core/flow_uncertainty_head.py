import torch, torch.nn as nn

class FlowUncertaintyHead(nn.Module):
    """Softplus σ head (flow-ready placeholder)."""
    def __init__(self, latent_dim=64, out_bins=283):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, out_bins)
        )
        self.softplus = nn.Softplus()

    def forward(self, z):
        return self.softplus(self.mlp(z))
