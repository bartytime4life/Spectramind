"""
SpectraMind V50 – SHAP + Entropy Overlay (Ultimate Version)
-----------------------------------------------------------
Fuses SHAP attribution and spectral entropy to highlight high-variance, high-impact bins.
Integrates with:
- generate_diagnostic_summary.py (automatic overlay)
- diagnostic dashboard (HTML embed)
- symbolic_violation_predictor.py (fusion-based bin prioritization)
- auto_ablate_v50.py (ablation scoring)
- plot_umap_v50.py (SHAP+Entropy coloring)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_shap_entropy_overlay(
    shap: np.ndarray,
    entropy: np.ndarray,
    outdir: str = "diagnostics/shap_entropy",
    save_name: str = "shap_entropy_overlay.png",
    fusion_save: Optional[str] = "shap_entropy_fusion.npy"
):
    """
    Plots normalized SHAP and entropy and returns fusion signal.

    Args:
        shap: (283,) SHAP values
        entropy: (283,) entropy per bin
        outdir: save directory
        save_name: output filename
        fusion_save: if provided, saves fused signal as .npy

    Returns:
        fusion_signal: np.ndarray (283,) – normalized fusion score
    """
    os.makedirs(outdir, exist_ok=True)

    shap_norm = shap / (np.max(np.abs(shap)) + 1e-8)
    entropy_norm = entropy / (np.max(entropy) + 1e-8)
    fusion = shap_norm * entropy_norm

    # Plot
    plt.figure(figsize=(12, 3))
    plt.plot(shap_norm, label="SHAP", color="green")
    plt.plot(entropy_norm, label="Entropy", color="purple")
    plt.plot(fusion, label="SHAP × Entropy", color="black", lw=1.5)
    plt.fill_between(range(283), 0, fusion, alpha=0.2, color="gray")
    plt.title("SHAP vs Entropy Overlay and Fusion")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(outdir, save_name)
    plt.savefig(path)
    print(f"✅ Saved SHAP + Entropy overlay: {path}")

    if fusion_save:
        np.save(os.path.join(outdir, fusion_save), fusion)
        print(f"💾 Saved fusion signal: {fusion_save}")

    return fusion


if __name__ == "__main__":
    shap = (np.random.rand(283) - 0.5) * 2
    entropy = np.random.rand(283)
    plot_shap_entropy_overlay(shap, entropy)