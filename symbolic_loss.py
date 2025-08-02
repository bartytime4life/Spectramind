"""
SpectraMind V50 – Symbolic Loss Functions (Ultimate Version)
-------------------------------------------------------------
Differentiable penalties for symbolic constraints on predicted μ spectra.
Includes: smoothness, monotonicity, nonnegativity, variance shaping, symmetry,
FFT penalty, and photonic band alignment using template YAML.
Logs raw loss values and supports diagnostics export.
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


def compute_symbolic_losses(mu: torch.Tensor, config: dict, meta: dict = None, log_raw: bool = False) -> dict:
    """
    Compute symbolic penalties including optional photonic alignment.

    Args:
        mu: Tensor (B, 283)
        config: YAML-style symbolic config dict
        meta: Optional metadata dict (planet info)
        log_raw: If True, returns a parallel dict of raw (unnormalized) values

    Returns:
        dict[str, Tensor] symbolic losses
    """
    losses = {}
    raw_losses = {} if log_raw else None

    if config.get("smoothness", True):
        val = smoothness_loss(mu)
        losses["smoothness"] = val / mu.shape[0]
        if log_raw: raw_losses["smoothness"] = val

    if config.get("monotonicity", False):
        val = monotonicity_loss(mu, config.get("monotonicity"))
        losses["monotonicity"] = val / mu.shape[0]
        if log_raw: raw_losses["monotonicity"] = val

    if config.get("nonnegativity", True):
        val = nonnegative_loss(mu)
        losses["nonnegative"] = val / mu.shape[0]
        if log_raw: raw_losses["nonnegative"] = val

    if config.get("variance_shaping", False):
        val = variance_shaping_loss(mu, config.get("target_std", 0.02))
        losses["variance"] = val / mu.shape[0]
        if log_raw: raw_losses["variance"] = val

    if config.get("enable_asymmetry", False):
        val = asymmetry_penalty(mu)
        losses["asymmetry"] = val / mu.shape[0]
        if log_raw: raw_losses["asymmetry"] = val

    if config.get("enable_fft", False):
        val = fft_spectral_penalty(mu, config.get("fft_threshold", 0.05))
        losses["fft_penalty"] = val / mu.shape[0]
        if log_raw: raw_losses["fft_penalty"] = val

    if config.get("enable_photonic", False):
        path = config.get("photonic_template", "photonic_basis.yaml")
        photonic_losses = enforce_photonic_template(mu, meta or {}, path, save_plots=config.get("save_plots", False))
        for k, v in photonic_losses.items():
            losses[k] = v / mu.shape[0]
            if log_raw: raw_losses[k] = v

    return (losses, raw_losses) if log_raw else losses