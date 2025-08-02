"""
SpectraMind V50 – GLL Error Localizer (Ultimate Version)
--------------------------------------------------------
Computes and visualizes binwise GLL contributions for scientific error localization.
Integrates with:
- generate_diagnostic_summary.py (per-bin GLL export)
- generate_html_report.py (heatmap overlay)
- symbolic_violation_predictor.py (fusion with symbolic bins)
- spectral_event_miner.py (anomaly detection)
- shap_overlay.py (joint GLL+SHAP diagnostics)
- auto_ablate_v50.py (GLL-driven ablation targeting)
- submission_validator_v50.py (GLL score validation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def compute_binwise_gll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, return_matrix: bool = False) -> np.ndarray:
    """
    Args:
        y, mu, sigma: (N, 283)
        return_matrix: if True, returns (N, 283) matrix

    Returns:
        gll_bins: (283,) sum over batch OR full matrix
    """
    var = sigma**2 + 1e-6
    log_term = np.log(sigma + 1e-6)
    quad_term = ((y - mu)**2) / (2 * var)
    gll_matrix = log_term + quad_term

    if return_matrix:
        return gll_matrix
    return np.sum(gll_matrix, axis=0)


def plot_gll_heatmap(
    gll_bins: np.ndarray,
    outdir: str = "diagnostics/gll",
    save_name: str = "gll_bin_heatmap.png",
    annotate_max: bool = True,
    overlay_mask: Optional[np.ndarray] = None
):
    """
    Args:
        gll_bins: (283,) per-bin GLL values
        overlay_mask: optional binary array to overlay symbolic focus
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10, 3))
    x = np.arange(len(gll_bins))
    plt.plot(gll_bins, color="crimson", lw=2, label="GLL")

    if overlay_mask is not None:
        mask = (overlay_mask > 0.5)
        plt.fill_between(x, 0, gll_bins, where=mask, color="orange", alpha=0.3, label="Symbolic Overlay")

    if annotate_max:
        max_idx = np.argmax(gll_bins)
        plt.axvline(max_idx, color="black", linestyle="--", alpha=0.4)
        plt.text(max_idx + 2, np.max(gll_bins)*0.9, f"Peak: {max_idx}", color="black")

    plt.title("Binwise GLL Contribution")
    plt.xlabel("Spectral Bin")
    plt.ylabel("GLL Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, save_name)
    plt.savefig(path)
    print(f"✅ Saved GLL heatmap: {path}")


if __name__ == "__main__":
    sub = pd.read_csv("submission.csv")
    gt = pd.read_csv("ground_truth.csv")
    mu_cols = [f"mu_{i}" for i in range(283)]
    sigma_cols = [f"sigma_{i}" for i in range(283)]
    mu = sub[mu_cols].values
    sigma = sub[sigma_cols].values
    y = gt[mu_cols].values

    gll_bins = compute_binwise_gll(y, mu, sigma)
    plot_gll_heatmap(gll_bins)