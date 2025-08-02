"""
SpectraMind V50 – SHAP-Attention Fusion Overlay
-----------------------------------------------
Overlays SHAP values with attention weights for enhanced interpretability.
Supports multiple fusion strategies and diagnostics output.
"""

from typing import Union, Optional
import numpy as np
import torch


def overlay_shap_attention(
    shap_values: Union[np.ndarray, torch.Tensor],
    attention_weights: Union[np.ndarray, torch.Tensor],
    fusion_mode: str = "multiply",
    normalize: bool = True,
    attention_threshold: Optional[float] = None,
    return_metadata: bool = True,
    top_k: Optional[int] = None
) -> Union[torch.Tensor, dict]:
    """
    Combines SHAP values with attention map into a single importance overlay.

    Args:
        shap_values: (283,) SHAP contribution scores
        attention_weights: (283,) decoder attention per bin
        fusion_mode: 'multiply', 'average', 'max', 'geometric'
        normalize: whether to L1-normalize inputs before fusion
        attention_threshold: optional mask (e.g., 0.01) to ignore weak attention
        return_metadata: if True, returns dict with overlay + entropy, top-k bins, etc.
        top_k: return top-k bin indices (for overlay visualizer or latent matching)

    Returns:
        Tensor (283,) or dict with:
            - overlay: fused importance map
            - top_indices: top-k bin indices
            - sparsity: % of bins above 0
            - entropy: per-bin entropy of fused overlay
    """
    is_np = isinstance(shap_values, np.ndarray)
    to_tensor = lambda x: torch.tensor(x) if is_np else x
    shap = to_tensor(shap_values).float()
    attn = to_tensor(attention_weights).float()

    if normalize:
        shap = shap / (shap.abs().sum() + 1e-8)
        attn = attn / (attn.sum() + 1e-8)

    if attention_threshold is not None:
        attn_mask = (attn >= attention_threshold).float()
        attn = attn * attn_mask

    if fusion_mode == "multiply":
        overlay = shap * attn
    elif fusion_mode == "average":
        overlay = 0.5 * (shap + attn)
    elif fusion_mode == "max":
        overlay = torch.max(shap, attn)
    elif fusion_mode == "geometric":
        overlay = torch.sqrt(shap.abs() * attn)
    else:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    if normalize:
        overlay = overlay / (overlay.sum() + 1e-8)

    if not return_metadata:
        return overlay

    entropy = - (overlay + 1e-8) * torch.log2(overlay + 1e-8)
    entropy_val = float(entropy.sum())
    sparsity = float((overlay > 0).sum()) / overlay.shape[0]

    top_idx = torch.topk(overlay, top_k if top_k else 10).indices.tolist()

    return {
        "overlay": overlay,
        "top_indices": top_idx,
        "sparsity": sparsity,
        "entropy": entropy_val,
        "fusion_mode": fusion_mode,
        "normalized": normalize
    }