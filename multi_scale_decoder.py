"""
SpectraMind V50 – Multi-Scale μ Decoder (Ultimate Version)
-------------------------------------------------------------
Decodes latent vector z into predicted μ across low, mid, and high frequency bands.
Includes dynamic fusion weights, multi-activation modeling, and diagnostics support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union


class MultiScaleDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden: int = 256,
        output_bins: int = 283,
        return_diagnostics: bool = False
    ):
        super().__init__()
        self.return_diagnostics = return_diagnostics

        self.low_band = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, output_bins)
        )

        self.mid_band = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, output_bins)
        )

        self.high_band = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, output_bins)
        )

        self.fusion_weights = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, z: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            μ: (B, output_bins) or diagnostics dict
        """
        mu_low = self.low_band(z)
        mu_mid = self.mid_band(z)
        mu_high = self.high_band(z)

        weights = F.softmax(self.fusion_weights, dim=0)
        mu = (
            weights[0] * mu_low +
            weights[1] * mu_mid +
            weights[2] * mu_high
        )

        if self.return_diagnostics:
            return {
                "mu": mu,
                "mu_low": mu_low.detach(),
                "mu_mid": mu_mid.detach(),
                "mu_high": mu_high.detach(),
                "fusion_weights": weights.detach()
            }
        return mu