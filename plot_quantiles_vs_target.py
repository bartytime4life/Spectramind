"""
SpectraMind V50 – Quantile Plot vs Ground Truth (Ultimate)
-----------------------------------------------------------
Visualizes μ quantile bands, overlays y_true, and highlights bins outside expected range.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_quantiles(mu_bands: dict, y_true: np.ndarray, title: str = "Quantile Bands vs Ground Truth"):
    """
    Args:
        mu_bands: dict of quantiles, must include q10, q25, q50, q75, q90
        y_true: np.ndarray of shape (283,)
        title: optional plot title

    Returns:
        matplotlib.figure.Figure
    """
    assert all(k in mu_bands for k in ["q10", "q25", "q50", "q75", "q90"]), "Missing required quantiles"
    assert isinstance(y_true, np.ndarray) and y_true.shape[0] == mu_bands["q50"].shape[0]

    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 5))

    q10 = mu_bands["q10"]
    q25 = mu_bands["q25"]
    q50 = mu_bands["q50"]
    q75 = mu_bands["q75"]
    q90 = mu_bands["q90"]

    # Plot outer band
    ax.fill_between(x, q10, q90, color="lightblue", alpha=0.35, label="q10–q90")

    # Plot inner band
    ax.fill_between(x, q25, q75, color="cornflowerblue", alpha=0.3, label="q25–q75")

    # Plot median
    ax.plot(x, q50, color="navy", lw=2, label="Median (q50)")

    # Plot ground truth
    ax.plot(x, y_true, color="black", lw=1.5, linestyle="--", label="Ground Truth")

    # Highlight violations
    outside = (y_true < q10) | (y_true > q90)
    num_violations = outside.sum()
    total_bins = y_true.shape[0]
    percent_violated = 100 * num_violations / total_bins

    if num_violations > 0:
        ax.fill_between(x, q90.max() * 1.02, q90.max() * 1.03,
                        where=outside,
                        color="red", alpha=0.7, step='mid', label="Outside q10–q90")

    ax.set_ylim([min(q10.min(), y_true.min()) - 10, max(q90.max(), y_true.max()) + 10])
    ax.set_xlim([0, len(y_true) - 1])
    ax.set_xlabel("Spectral Bin")
    ax.set_ylabel("Transit Depth (ppm)")
    ax.set_title(f"{title}  |  Violations: {num_violations} / {total_bins} "
                 f"({percent_violated:.1f}%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    return fig