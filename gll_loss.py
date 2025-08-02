"""
SpectraMind V50 – GLL Loss Function (Beyond Ultimate)
-----------------------------------------------------
Implements numerically stable, symbolically weighted, and diagnostic-aware
Gaussian Log-Likelihood (GLL) loss for μ and σ predictions.
"""

import torch
from typing import Optional, Literal, Union, Dict


def gll_loss(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reduction: Literal["sum", "mean", "none", "per_sample", "per_bin"] = "sum",
    symbolic_weight: Optional[torch.Tensor] = None,
    clip_sigma: float = 1e2,
    return_diagnostics: bool = False
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Args:
        y: (B, 283) ground truth
        mu: (B, 283) predicted mean
        sigma: (B, 283) predicted std deviation
        reduction: type of output:
            - 'sum': scalar
            - 'mean': scalar
            - 'none': shape (B, 283)
            - 'per_sample': shape (B,)
            - 'per_bin': shape (283,)
        symbolic_weight: optional (B, 283) weighting for symbolic importance
        clip_sigma: max allowed σ value for stability
        return_diagnostics: if True, returns dict with residuals, entropy, etc.

    Returns:
        Scalar loss, tensor loss, or diagnostic dict
    """
    eps = 1e-6
    sigma = sigma.clamp(min=eps, max=clip_sigma)
    var = sigma**2

    # Full Gaussian log-likelihood
    log_term = torch.log(var + eps)
    diff_term = ((y - mu) ** 2) / (var + eps)
    gll = 0.5 * (log_term + diff_term + torch.log(torch.tensor(2.0 * torch.pi, device=y.device)))

    # Symbolic weighting (optional)
    if symbolic_weight is not None:
        gll = gll * symbolic_weight

    # Reduction
    if reduction == "sum":
        loss = gll.sum()
    elif reduction == "mean":
        loss = gll.mean()
    elif reduction == "none":
        loss = gll
    elif reduction == "per_sample":
        loss = gll.sum(dim=1)  # shape: (B,)
    elif reduction == "per_bin":
        loss = gll.mean(dim=0)  # shape: (283,)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    if not return_diagnostics:
        return loss

    # Diagnostics
    residual = y - mu
    diagnostics = {
        "loss": loss,
        "residual_mean": residual.mean(),
        "residual_std": residual.std(),
        "residual_max": residual.abs().max(),
        "sigma_mean": sigma.mean(),
        "sigma_min": sigma.min(),
        "sigma_max": sigma.max(),
        "entropy": (-sigma * torch.log(sigma + eps)).mean(),
        "mean_log_term": log_term.mean(),
    }

    return diagnostics