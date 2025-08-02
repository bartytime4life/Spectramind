"""
SpectraMind V50 – Symbolic Influence Map (Beyond Ultimate)
-----------------------------------------------------------
Maps symbolic rule influence across spectral bins and computes binwise diagnostics:
- Per-rule normalized strength
- Composite fused score (max, entropy, weighted)
- Entropy of symbolic focus
- Top rule per bin
"""

import torch
import numpy as np
from typing import Dict, Union, Optional


def compute_influence_map(
    mu: Union[torch.Tensor, np.ndarray],
    rule_weights: Dict[str, Union[torch.Tensor, np.ndarray]],
    normalize: bool = True,
    attention: Optional[Union[torch.Tensor, np.ndarray]] = None,
    return_numpy: bool = False,
    return_diagnostics: bool = True
) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
    """
    Args:
        mu: (283,) predicted μ
        rule_weights: dict of rule_name → (283,) array or tensor
        normalize: whether to normalize per-rule weights
        attention: optional (283,) attention vector to reweight rule strengths
        return_numpy: if True, returns np.ndarray for all values
        return_diagnostics: if True, includes top_rule, entropy, total_influence

    Returns:
        Dict[str, Tensor or np.ndarray] including:
            - per-rule influence maps
            - "composite": fused influence (max, weighted, etc.)
            - "entropy": per-bin rule entropy
            - "top_rule": index of most influential rule per bin
    """
    # Convert to tensors if needed
    if isinstance(mu, np.ndarray):
        mu = torch.tensor(mu)

    influence = {}
    stack = []

    for rule, w in rule_weights.items():
        w = torch.tensor(w) if isinstance(w, np.ndarray) else w
        if normalize:
            w = w / (w.norm(p=2) + 1e-8)
        if attention is not None:
            attn = torch.tensor(attention) if isinstance(attention, np.ndarray) else attention
            w = w * attn
        influence[rule] = w
        stack.append(w.unsqueeze(0))  # shape: (1, 283)

    influence_tensor = torch.cat(stack, dim=0)  # shape: (R, 283)
    softmax_weights = torch.nn.functional.softmax(influence_tensor, dim=0)  # rule-wise focus

    # Composite metrics
    entropy = - (softmax_weights * (softmax_weights + 1e-8).log()).sum(dim=0)  # (283,)
    top_rule_idx = torch.argmax(influence_tensor, dim=0)  # (283,)
    composite_max = influence_tensor.max(dim=0).values  # (283,)
    composite_sum = influence_tensor.sum(dim=0)  # (283,)
    composite_weighted = (softmax_weights * influence_tensor).sum(dim=0)  # (283,)

    result = {k: v.numpy() if return_numpy else v for k, v in influence.items()}

    # Add composites
    result["composite_max"] = composite_max.numpy() if return_numpy else composite_max
    result["composite_sum"] = composite_sum.numpy() if return_numpy else composite_sum
    result["composite_weighted"] = composite_weighted.numpy() if return_numpy else composite_weighted
    result["entropy"] = entropy.numpy() if return_numpy else entropy
    result["top_rule"] = top_rule_idx.numpy() if return_numpy else top_rule_idx

    if return_diagnostics:
        rule_names = list(rule_weights.keys())
        result["top_rule_name"] = [rule_names[i] for i in top_rule_idx.tolist()]
        result["entropy_mean"] = float(entropy.mean())
        result["dominant_rules"] = {
            name: int((top_rule_idx == i).sum()) for i, name in enumerate(rule_names)
        }

    return result