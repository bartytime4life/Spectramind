"""
SpectraMind V50 – Symbolic Loss Functions (Ultimate Version)
-------------------------------------------------------------
Differentiable penalties for symbolic constraints on predicted μ spectra.
Includes: smoothness, monotonicity, nonnegativity, variance shaping, symmetry,
entropy regularization, FFT penalty, photonic band alignment using YAML templates,
and rule-specific symbolic masking. Logs raw loss values for diagnostics.
"""

import torch
import torch.nn.functional as F
from photonic_alignment import enforce_photonic_template
from symbolic_rule_scorer import apply_symbolic_mask
from typing import Optional, Dict, Tuple


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


def entropy_regularization(mu: torch.Tensor) -> torch.Tensor:
    prob = mu / (mu.sum(dim=1, keepdim=True) + 1e-8)
    entropy = - (prob * (prob + 1e-8).log()).sum(dim=1)
    return -entropy.mean()


def asymmetry_penalty(mu: torch.Tensor) -> torch.Tensor:
    center = mu.shape[1] // 2
    left = mu[:, :center]
    right = torch.flip(mu[:, -center:], dims=[1])
    return F.mse_loss(left, right)


def fft_spectral_penalty(mu: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    fft_mag = torch.fft.rfft(mu, dim=-1).abs()
    high_freq = fft_mag[:, -20:]
    return F.relu(high_freq.mean(dim=1) - threshold).mean()


def compute_symbolic_losses(
    mu: torch.Tensor,
    config: Dict,
    meta: Optional[Dict] = None,
    log_raw: bool = False,
    symbolic_mask: Optional[torch.Tensor] = None
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """
    Compute symbolic penalties, optionally using symbolic masks and logging raw values.

    Args:
        mu: Tensor (B, 283)
        config: symbolic config dict
        meta: optional metadata dict for photonic rules
        log_raw: whether to return unnormalized raw loss values
        symbolic_mask: optional (B, 283) tensor weighting symbolic penalty regions

    Returns:
        (loss_dict, raw_loss_dict) if log_raw else loss_dict
    """
    losses = {}
    raw_losses = {} if log_raw else None

    def register(name: str, value: torch.Tensor):
        losses[name] = value / mu.shape[0]
        if log_raw:
            raw_losses[name] = value

    if config.get("smoothness", True):
        register("smoothness", smoothness_loss(mu))

    if config.get("monotonicity", False):
        register("monotonicity", monotonicity_loss(mu, config.get("monotonicity")))

    if config.get("nonnegativity", True):
        register("nonnegative", nonnegative_loss(mu))

    if config.get("variance_shaping", False):
        register("variance", variance_shaping_loss(mu, config.get("target_std", 0.02)))

    if config.get("enable_entropy", False):
        register("entropy", entropy_regularization(mu))

    if config.get("enable_asymmetry", False):
        register("asymmetry", asymmetry_penalty(mu))

    if config.get("enable_fft", False):
        register("fft_penalty", fft_spectral_penalty(mu, config.get("fft_threshold", 0.05)))

    if config.get("enable_photonic", False):
        path = config.get("photonic_template", "photonic_basis.yaml")
        photonic_losses = enforce_photonic_template(mu, meta or {}, path, save_plots=config.get("save_plots", False))
        for k, v in photonic_losses.items():
            register(k, v)

    if symbolic_mask is not None:
        # symbolic mask: higher weights in specific bin regions (shape: B x 283)
        symbolic_weight = symbolic_mask.to(mu.device)
        masked_loss = F.mse_loss(mu * symbolic_weight, mu.detach() * symbolic_weight)
        register("symbolic_mask", masked_loss)

    return (losses, raw_losses) if log_raw else losses