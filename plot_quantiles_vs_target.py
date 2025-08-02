"""
SpectraMind V50 – Quantile Plot vs Ground Truth
-----------------------------------------------
Visualizes predicted μ quantile bands and overlays with true transmission spectrum (y_true).
Used for uncertainty calibration diagnostics and symbolic QA overlays.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_quantiles(mu_bands: dict, y_true: np.ndarray, title: str = "Quantile Bands vs Ground Truth"):
    """
    Plot μ quantile envelope and overlay with y_true.

    Args:
        mu_bands: Dict[str, np.ndarray] with keys like q10, q25, q50, q75, q90
        y_true: np.ndarray of shape (283,) – ground-truth transmission spectrum
        title: plot title (optional)

    Returns:
        matplotlib.figure.Figure
    """
    assert isinstance(mu_bands, dict), "mu_bands must be a dict of quantile arrays"
    assert "q10" in mu_bands and "q90" in mu_bands, "Expected at least q10 and q90 keys"
    assert y_true.shape[0] == mu_bands["q50"].shape[0], "y_true shape mismatch"

    x = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 5))

    # Outer band (q10–q90)
    ax.fill_between(x, mu_bands["q10"], mu_bands["q90"],
                    color="lightblue", alpha=0.4, label="q10–q90 band")

    # Inner band (q25–q75)
    if "q25" in mu_bands and "q75" in mu_bands:
        ax.fill_between(x, mu_bands["q25"], mu_bands["q75"],
                        color="dodgerblue", alpha=0.3, label="q25–q75 band")

    # Median line
    ax.plot(x, mu_bands["q50"], color="navy", lw=2, label="Median (q50)")

    # Ground truth overlay
    ax.plot(x, y_true, color="black", lw=1.5, linestyle="--", label="Ground Truth")

    ax.set_xlabel("Spectral Bin")
    ax.set_ylabel("Transit Depth (ppm)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    return fig