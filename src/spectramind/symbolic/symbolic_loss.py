import torch

def smoothness_penalty(mu: torch.Tensor) -> torch.Tensor:
    # second finite difference across spectral bins
    d2 = mu[..., :-2] - 2*mu[..., 1:-1] + mu[..., 2:]
    return (d2**2).mean()

def nonnegativity_penalty(mu: torch.Tensor) -> torch.Tensor:
    return torch.relu(-mu).mean()

def symbolic_loss(mu: torch.Tensor, smooth_lambda: float = 0.1, nonneg_lambda: float = 0.05) -> torch.Tensor:
    return smooth_lambda * smoothness_penalty(mu) + nonneg_lambda * nonnegativity_penalty(mu)
