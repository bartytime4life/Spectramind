"""
SpectraMind V50 – Symbolic Loss Functions (Ultimate Version)
-------------------------------------------------------------
Differentiable penalties for symbolic constraints on predicted μ spectra.
Includes: smoothness, monotonicity, nonnegativity, variance shaping, symmetry,
FFT penalty, and photonic band alignment using template YAML.
"""

import torch
import torch.nn.functional as F
from photonic_alignment import enforce_photonic_template

def smoothness_loss(mu: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(mu[:, 2:], 2 * mu[:, 1:-1] - mu[:, :-2])

def monotonicity_loss(mu: torch.Tensor, direction: str = "none") -> torch.Tensor:
    if direction == "none":
        return torch.tensor(0.0, device=mu.device)
    diffs = mu[:, 1:] - mu[:, :-1]
    if direction == "increasing":
        return F.relu(-diffs).mean()
    elif direction == "decreasing":
        return F.relu(diffs).mean()
    return torch.tensor(0.0, device=mu.device)

def nonnegative_loss(mu: torch.Tensor) -> torch.Tensor:
    return F.relu(-mu).mean()

def variance_shaping_loss(mu: torch.Tensor, target_std: float = 0.02) -> torch.Tensor:
    std = torch.std(mu, dim=1)
    return F.mse_loss(std, torch.full_like(std, target_std))

def asymmetry_penalty(mu: torch.Tensor) -> torch.Tensor:
    center = mu.shape[1] // 2
    left = mu[:, :center]
    right = torch.flip(mu[:, -center:], dims=[1])
    return F.mse_loss(left, right)

def fft_spectral_penalty(mu: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    fft_mag = torch.fft.rfft(mu, dim=-1).abs()
    high_freq = fft_mag[:, -20:]
    return F.relu(high_freq.mean(dim=1) - threshold).mean()

def compute_symbolic_losses(mu: torch.Tensor, config: dict, meta: dict = None) -> dict:
    """
    Compute symbolic penalties including optional photonic alignment.

    Args:
        mu: Tensor (B, 283)
        config: YAML-style symbolic config dict
        meta: Optional metadata dict (planet info)

    Returns:
        dict[str, Tensor] symbolic losses
    """
    losses = {}

    if config.get("smoothness", True):
        losses["smoothness"] = smoothness_loss(mu)

    if config.get("monotonicity", False):
        losses["monotonicity"] = monotonicity_loss(mu, config.get("monotonicity"))

    if config.get("nonnegativity", True):
        losses["nonnegative"] = nonnegative_loss(mu)

    if config.get("variance_shaping", False):
        losses["variance"] = variance_shaping_loss(mu, config.get("target_std", 0.02))

    if config.get("enable_asymmetry", False):
        losses["asymmetry"] = asymmetry_penalty(mu)

    if config.get("enable_fft", False):
        losses["fft_penalty"] = fft_spectral_penalty(mu, config.get("fft_threshold", 0.05))

    if config.get("enable_photonic", False):
        path = config.get("photonic_template", "photonic_basis.yaml")
        photonic_losses = enforce_photonic_template(mu, meta or {}, path, save_plots=config.get("save_plots", False))
        losses.update(photonic_losses)

    # Normalize all scalar losses
    B = mu.shape[0]
    for k in losses:
        if losses[k].ndim == 0:
            losses[k] = losses[k] / B

    return losses
