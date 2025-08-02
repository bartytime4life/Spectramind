"""
SpectraMind V50 – Observatory Simulator (Ultimate Version)
-----------------------------------------------------------
Simulates end-to-end light curve observations from true μ + σ predictions.
Integrates with:
- simulate_lightcurve_from_mu.py (μ reconstruction)
- fgs1_mamba.py / airs_gnn.py (synthetic training)
- diagnostic overlay (e.g., violation_heatmap)
- transit_stack_generator.py (stack from μ)
- generate_html_report.py (simulation visual panel)
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import os

def simulate_detector_response(
    mu: np.ndarray,
    sigma: np.ndarray,
    n_airs_frames: int = 11250,
    n_fgs_frames: int = 135000,
    noise_scale: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates AIRS and FGS1 light curves from ground-truth μ and σ.
    
    Args:
        mu: (283,) true transit spectrum
        sigma: (283,) uncertainty spectrum
        n_airs_frames: number of AIRS time steps
        n_fgs_frames: number of FGS1 time steps
        noise_scale: multiplier for realistic noise amplitude
        seed: random seed (optional)

    Returns:
        Tuple of (AIRS lightcurve, FGS1 flux curve)
    """
    if seed is not None:
        np.random.seed(seed)

    mu_repeated = np.tile(mu, (n_airs_frames, 1))
    sigma_repeated = np.tile(sigma, (n_airs_frames, 1))
    noise_airs = np.random.normal(0, sigma_repeated * noise_scale)
    airs_signal = mu_repeated + noise_airs

    fgs_base = 1000.0 - np.mean(mu)
    fgs_noise = np.random.normal(0, np.std(sigma) * noise_scale, size=n_fgs_frames)
    fgs_flux = fgs_base + fgs_noise

    return airs_signal, fgs_flux


def plot_simulated_observations(
    airs: np.ndarray,
    fgs: np.ndarray,
    outdir: str = "diagnostics/simulation",
    tag: str = "simulated_observation"
):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 3))
    plt.imshow(airs.T, aspect="auto", cmap="inferno")
    plt.title("AIRS Simulated Light Curve (Time x Bin)")
    plt.colorbar(label="Flux")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{tag}_airs.png")
    plt.close()

    plt.figure(figsize=(10, 2.5))
    plt.plot(fgs, lw=0.5, color="black")
    plt.title("FGS1 Simulated Flux")
    plt.xlabel("Time Step")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{tag}_fgs.png")
    plt.close()

    print(f"✅ Saved simulated AIRS and FGS1 plots to {outdir}")


if __name__ == "__main__":
    mu = np.random.rand(283) * 100
    sigma = np.random.rand(283) * 5 + 1
    airs, fgs = simulate_detector_response(mu, sigma, seed=42)
    plot_simulated_observations(airs, fgs)