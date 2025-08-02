"""
SpectraMind V50 – Flow-Based σ Decoder (Ultimate Version)
----------------------------------------------------------
Predicts per-bin uncertainty σ using shallow normalizing-flow-inspired residual MLP.
Can optionally fuse μ and z, return diagnostics, or use residual modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Union


class FlowUncertaintyHead(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        mu_dim: int = 283,
        hidden_dim: int = 256,
        use_mu: bool = True,
        residual_sigma: bool = False,
        output_type: Literal["sigma", "logvar"] = "sigma"
    ):
        """
        Args:
            latent_dim: input z dimension
            mu_dim: number of output bins (default 283)
            hidden_dim: internal MLP size
            use_mu: whether to fuse μ as decoder input
            residual_sigma: model Δσ rather than σ directly
            output_type: 'sigma' (std) or 'logvar' (log(σ²)) for stability
        """
        super().__init__()
        input_dim = latent_dim + mu_dim if use_mu else latent_dim

        self.residual_sigma = residual_sigma
        self.output_type = output_type
        self.use_mu = use_mu

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mu_dim)
        )
        self.softplus = nn.Softplus()

    def forward(
        self,
        z: torch.Tensor,            # (B, latent_dim)
        mu: Optional[torch.Tensor] = None,  # (B, 283) or None if use_mu=False
        base_sigma: Optional[torch.Tensor] = None,  # optional base σ if residual mode
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        Returns:
            σ: (B, 283) or log(σ²), or dict if return_diagnostics=True
        """
        if self.use_mu:
            if mu is None:
                raise ValueError("mu must be provided if use_mu=True")
            x = torch.cat([z, mu], dim=1)  # (B, D+283)
        else:
            x = z

        h = self.net(x)  # (B, 283)

        if self.output_type == "logvar":
            output = h
        else:
            output = self.softplus(h) + 1e-4

        if self.residual_sigma and base_sigma is not None:
            output = output + base_sigma

        if return_diagnostics:
            return {
                "sigma": output,
                "pre_activation": h.detach(),
                "z": z.detach(),
                "mu": mu.detach() if mu is not None else None
            }
        return output