"""
SpectraMind V50 – GLL Heatmap Per Bin
-------------------------------------
Visualizes the per-bin average Gaussian Log-Likelihood (GLL) across all samples.

✅ Diagnostics-compatible
✅ Highlights difficult wavelength regions
✅ Uses standard 567-bin layout (FGS + AIRS)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

TOTAL_BINS = 567
FGS_BINS = 1
AIRS_BINS = 566
FGS_WEIGHT = 0.4
AIRS_WEIGHT = 0.6 / AIRS_BINS
SPECTRAL_WEIGHTS = np.array([FGS_WEIGHT] + [AIRS_WEIGHT] * AIRS_BINS)

def plot_gll_heatmap_per_bin(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, outdir: str = "diagnostics"):
    os.makedirs(outdir, exist_ok=True)
    sigma = np.clip(sigma, 1e-8, 1e6)

    gll = ((y_true - mu) / sigma) ** 2 + 2 * np.log(sigma) + np.log(2 * np.pi)
    gll_weighted = gll * SPECTRAL_WEIGHTS
    mean_per_bin = gll_weighted.mean(axis=0)

    plt.figure(figsize=(14, 3))
    plt.plot(mean_per_bin, color="darkred", linewidth=1.5)
    plt.title("Average GLL Loss per Spectral Bin", fontsize=14)
    plt.xlabel("Spectral Bin Index (0 = FGS1, 1–566 = AIRS)", fontsize=12)
    plt.ylabel("Weighted GLL", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(outdir, "gll_heatmap_per_bin.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 GLL per-bin heatmap saved to {save_path}")