"""
SpectraMind V50 – Transit Stack Generator (Ultimate Version)
------------------------------------------------------------
Simulates multiple noisy transit light curves and returns their average stack.
Integrates with:
- simulate_lightcurve_from_mu.py
- fgs1_mamba.py (pretrain signal injection)
- diagnostic visualizations (FFT, symbolic overlays)
- plot_umap_v50.py (latent visual trace validation)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional


def generate_single_transit(
    T: int = 1000,
    depth: float = 0.001,
    duration: float = 0.1,
    sigma: float = 0.0005,
    baseline: float = 1.0,
    random_phase: bool = False
) -> np.ndarray:
    time = np.linspace(0, 1, T)
    flux = np.ones(T) * baseline

    if random_phase:
        center = np.random.uniform(0.4, 0.6)
    else:
        center = 0.5

    in_transit = np.abs(time - center) < (duration / 2)
    flux[in_transit] -= depth
    return flux + np.random.normal(0, sigma, size=T)


def stack_transits(
    n: int = 5,
    T: int = 1000,
    depth: float = 0.001,
    duration: float = 0.1,
    sigma: float = 0.0005,
    random_phase: bool = True,
    return_all: bool = False
) -> np.ndarray:
    stack = np.array([
        generate_single_transit(T, depth, duration, sigma, random_phase=random_phase)
        for _ in range(n)
    ])
    return (stack if return_all else stack.mean(axis=0))


def plot_stacked_transit(
    stacked: np.ndarray,
    save_path: str = "diagnostics/stacked_transit.png",
    title: str = "Stacked Transit Light Curve"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 3))

    if stacked.ndim == 2:
        for i in range(stacked.shape[0]):
            plt.plot(stacked[i], color="gray", alpha=0.3)
        mean_curve = stacked.mean(axis=0)
        plt.plot(mean_curve, color="black", label="Mean")
    else:
        plt.plot(stacked, color="black")

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Flux")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved stacked transit plot: {save_path}")


if __name__ == "__main__":
    stacked = stack_transits(n=10, T=1000, depth=0.002, sigma=0.0007)
    plot_stacked_transit(stacked)