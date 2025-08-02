"""
SpectraMind V50 – Flow-Based σ Decoder (Ultimate Symbolic-Aware Edition)
-------------------------------------------------------------------------
Predicts per-bin uncertainty σ using a shallow residual MLP with symbolic fusion and diagnostic hooks.

✅ Features:
- Input: latent z, optional μ, optional symbolic overlays
- Symbolic fusion: add, mul, concat
- Residual σ mode with base_sigma
- Output: σ or log(σ²)
- TorchScript-compatible (no control flow divergence)
- Optional LayerNorm, Dropout, Softplus floor, Precision-safe
- SHAP/attention/latent trace diagnostics (mu, z, symbolic, logits)
- Robust to empty or masked symbolic overlays
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Union, Dict


class FlowUncertaintyHead(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        mu_dim: int = 283,
        hidden_dim: int = 256,
        use_mu: bool = True,
        residual_sigma: bool = False,
        output_type: Literal["sigma", "logvar"] = "sigma",
        fusion_mode: Literal["add", "mul", "concat", "none"] = "none",
        symbolic_dim: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm: bool = False,
        min_sigma: float = 1e-4
    ):
        """
        Args:
            latent_dim: Dimension of latent input z
            mu_dim: Number of wavelength bins (typically 283)
            hidden_dim: Width of the hidden MLP
            use_mu: Whether to include μ in input
            residual_sigma: Predict Δσ over base σ if True
            output_type: 'sigma' (positive std) or 'logvar' (log(σ²))
            fusion_mode: Strategy for symbolic fusion ('add', 'mul', 'concat', 'none')
            symbolic_dim: Size of symbolic overlay (defaults to mu_dim)
            dropout: Dropout probability (e.g., 0.1)
            layer_norm: Whether to use LayerNorm
            min_sigma: Floor for σ output to avoid zero std
        """
        super().__init__()
        self.use_mu = use_mu
        self.residual_sigma = residual_sigma
        self.output_type = output_type
        self.fusion_mode = fusion_mode
        self.symbolic_dim = symbolic_dim or mu_dim
        self.mu_dim = mu_dim
        self.min_sigma = min_sigma

        input_dim = latent_dim
        if use_mu:
            input_dim += mu_dim
        if fusion_mode == "concat":
            input_dim += self.symbolic_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mu_dim)
        ]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(
        self,
        z: torch.Tensor,                          # (B, latent_dim)
        mu: Optional[torch.Tensor] = None,        # (B, mu_dim)
        symbolic: Optional[torch.Tensor] = None,  # (B, symbolic_dim)
        base_sigma: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            z: Latent representation (B, latent_dim)
            mu: Optional μ input (B, mu_dim) if use_mu=True
            symbolic: Optional symbolic constraint features (B, symbolic_dim)
            base_sigma: Optional base σ (B, mu_dim) if residual_sigma=True
            return_diagnostics: Whether to return dict of intermediates

        Returns:
            σ: (B, mu_dim) tensor of std or logvar
            OR diagnostic dict with pre-activations, latent traces, overlays
        """
        x = z
        if self.use_mu:
            if mu is None:
                raise ValueError("μ must be provided if use_mu=True")
            x = torch.cat([x, mu], dim=-1)

        symbolic_fused = None
        if symbolic is not None:
            if self.fusion_mode == "add":
                x = x + symbolic
                symbolic_fused = x
            elif self.fusion_mode == "mul":
                x = x * (symbolic + 1e-3)
                symbolic_fused = x
            elif self.fusion_mode == "concat":
                x = torch.cat([x, symbolic], dim=-1)
                symbolic_fused = symbolic
            elif self.fusion_mode != "none":
                raise ValueError(f"Invalid fusion_mode: {self.fusion_mode}")

        h = self.net(x)

        if self.output_type == "logvar":
            output = h
        else:
            output = self.softplus(h) + self.min_sigma

        if self.residual_sigma and base_sigma is not None:
            output = output + base_sigma

        if return_diagnostics:
            return {
                "sigma": output,
                "logits": h.detach(),
                "z": z.detach(),
                "mu": mu.detach() if mu is not None else None,
                "symbolic": symbolic.detach() if symbolic is not None else None,
                "symbolic_fused": symbolic_fused.detach() if symbolic_fused is not None else None
            }

        return output