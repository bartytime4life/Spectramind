"""
SpectraMind V50 – Symbolic Loss Functions
-----------------------------------------
Differentiable penalty functions for enforcing symbolic constraints on predicted μ spectra.
Includes: smoothness, monotonicity, nonnegativity, variance shaping, symmetry, and FFT penalties.
Config-driven, batch-consistent, and fully differentiable.
"""

import torch
import torch.nn.functional as F

def smoothness_loss(mu: torch.Tensor) -> torch.Tensor:
    """Second-order difference penalty for local spectral continuity."""
    return F.mse_loss(mu[:, 2:], 2 * mu[:, 1:-1] - mu[:, :-2])

def monotonicity_loss(mu: torch.Tensor, direction: str = "none") -> torch.Tensor:
    """Enforces increasing/decreasing spectral trends."""
    if direction == "none":
        return torch.tensor(0.0, device=mu.device)
    diffs = mu[:, 1:] - mu[:, :-1]
    if direction == "increasing":
        return F.relu(-diffs).mean()
    elif direction == "decreasing":
        return F.relu(diffs).mean()
    return torch.tensor(0.0, device=mu.device)

def nonnegative_loss(mu: torch.Tensor) -> torch.Tensor:
    """Penalize μ values below zero."""
    return F.relu(-mu).mean()

def variance_shaping_loss(mu: torch.Tensor, target_std: float = 0.02) -> torch.Tensor:
    """Align μ batch stddev to reference value."""
    std = torch.std(mu, dim=1)
    return F.mse_loss(std, torch.full_like(std, target_std))

def asymmetry_penalty(mu: torch.Tensor) -> torch.Tensor:
    """Enforce bilateral symmetry around spectral center."""
    center = mu.shape[1] // 2
    left = mu[:, :center]
    right = torch.flip(mu[:, -center:], dims=[1])
    return F.mse_loss(left, right)

def fft_spectral_penalty(mu: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Suppress high-frequency oscillations beyond threshold."""
    fft_mag = torch.fft.rfft(mu, dim=-1).abs()
    high_freq = fft_mag[:, -20:]
    return F.relu(high_freq.mean(dim=1) - threshold).mean()

def compute_symbolic_losses(mu: torch.Tensor, config: dict) -> dict:
    """
    Compute all enabled symbolic constraints.

    Args:
        mu (Tensor): shape (B, 283)
        config (dict): constraint toggles + hyperparameters

    Returns:
        dict: symbolic loss components, batch-normalized
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

    # Normalize scalar losses by batch size
    B = mu.shape[0]
    for k in losses:
        if losses[k].ndim == 0:
            losses[k] = losses[k] / B

    return losses
