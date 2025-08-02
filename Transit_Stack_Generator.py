"""
SpectraMind V50 – Transit Stack Generator (Fully Loaded Version)
---------------------------------------------------------------
Simulates multiple noisy transit light curves and returns the average or full stack.
Integrates with:
- simulate_lightcurve_from_mu.py
- fgs1_mamba.py (pretrain injection)
- symbolic_loss.py (smoothness, monotonicity eval)
- plot_fft_power_cluster_compare.py
- generate_html_report.py (summary + PNG export)
- CLI: spectramind simulate-transit
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Tuple, Optional


def generate_single_transit(
    T: int = 1000,
    depth: float = 0.001,
    duration: float = 0.1,
    sigma: float = 0.0005,
    baseline: float = 1.0,
    random_phase: bool = True
) -> np.ndarray:
    time = np.linspace(0, 1, T)
    flux = np.ones(T) * baseline
    center = np.random.uniform(0.4, 0.6) if random_phase else 0.5
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
    return stack if return_all else stack.mean(axis=0)


def plot_stacked_transit(
    stacked: np.ndarray,
    save_path: str = "diagnostics/stacked_transit.png",
    title: str = "Stacked Transit Light Curve",
    save_npy: Optional[str] = None,
    save_json: Optional[str] = None,
    save_fft: Optional[str] = None
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 3))

    if stacked.ndim == 2:
        for i in range(stacked.shape[0]):
            plt.plot(stacked[i], color="gray", alpha=0.3)
        mean_curve = stacked.mean(axis=0)
        plt.plot(mean_curve, color="black", lw=2, label="Mean")
    else:
        plt.plot(stacked, color="black", lw=2)

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Flux")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved stacked transit plot: {save_path}")

    if save_npy:
        np.save(save_npy, stacked)
        print(f"💾 Saved stacked transit .npy: {save_npy}")

    if save_json and stacked.ndim == 1:
        with open(save_json, "w") as f:
            json.dump({"mean_flux": stacked.tolist()}, f)
        print(f"📝 Saved transit JSON summary: {save_json}")

    if save_fft and stacked.ndim == 1:
        fft = np.abs(np.fft.rfft(stacked))
        plt.figure(figsize=(8, 3))
        plt.plot(fft, color="purple")
        plt.title("FFT Power of Stacked Transit")
        plt.xlabel("Frequency Bin")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_fft)
        print(f"🔬 Saved FFT of stacked transit: {save_fft}")


if __name__ == "__main__":
    stacked = stack_transits(n=10, T=1200, depth=0.002, sigma=0.0007)
    plot_stacked_transit(
        stacked,
        save_path="diagnostics/stacked_transit.png",
        save_npy="diagnostics/stacked_transit.npy",
        save_json="diagnostics/stacked_transit.json",
        save_fft="diagnostics/stacked_transit_fft.png"
    )