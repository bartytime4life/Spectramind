"""
SpectraMind V50 – SHAP Value Overlay
------------------------------------
Computes SHAP values for μ predictions using DeepExplainer.
Targets the μ head output (283,) per sample.
"""

import torch
import shap
from typing import Optional


def compute_shap_values(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    background_data: torch.Tensor,
    target_fn: Optional[callable] = None,
    use_cpu: bool = False,
    return_numpy: bool = False
) -> torch.Tensor:
    """
    Computes SHAP values for μ predictions.

    Args:
        model: trained SpectraMind model
        input_tensor: Tensor (B, ...) – input samples
        background_data: Tensor (N, ...) – background for SHAP
        target_fn: optional function to extract μ head: lambda out → out["mu"]
        use_cpu: move model and data to CPU for SHAP
        return_numpy: return np.ndarray instead of Tensor

    Returns:
        SHAP values: (B, 283)
    """
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    bg = background_data.to(device)
    inputs = input_tensor.to(device)

    def model_mu_output(x):
        with torch.no_grad():
            out = model(x)
            if isinstance(out, dict):
                return target_fn(out) if target_fn else out.get("mu", out)
            return out

    explainer = shap.DeepExplainer(model_mu_output, bg)
    shap_values = explainer.shap_values(inputs)

    # DeepExplainer returns a list (1 element for single-output head)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    if return_numpy:
        return shap_values
    return torch.tensor(shap_values)