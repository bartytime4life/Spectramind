"""
SpectraMind V50 – Constraint Violation Overlay
----------------------------------------------
Overlays symbolic violation scores on predicted μ spectrum.
Supports batchwise averaging, dual-sided violation fill, and auto-saving to diagnostics dir.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_violation_overlay(mu: np.ndarray,
                           violations: np.ndarray,
                           outdir="outputs/diagnostics",
                           save_name="mu_violation_overlay.png",
                           normalize=False,
                           save_csv=True):
    """
    Args:
        mu: (283,) or (N, 283) predicted μ
        violations: (283,) or (N, 283) symbolic violation scores
        normalize: if True, normalize violation magnitude relative to μ
        save_csv: save overlay values as .csv
    """
    os.makedirs(outdir, exist_ok=True)

    if mu.ndim == 2:
        mu = mu.mean(axis=0)
    if violations.ndim == 2:
        violations = violations.mean(axis=0)

    if normalize:
        violations = violations / (np.abs(mu) + 1e-8)

    # Compute coverage
    coverage_rate = np.mean(violations > 0)
    print(f"🧪 Violation coverage: {coverage_rate*100:.1f}% of bins")

    x = np.arange(len(mu))
    plt.figure(figsize=(12, 4))
    plt.plot(x, mu, label="μ prediction", color="black", linewidth=1.5)
    plt.fill_between(x, mu, mu + violations, alpha=0.25, color="red", label="Violation ↑")
    plt.fill_between(x, mu, mu - violations, alpha=0.15, color="blue", label="Violation ↓")
    plt.title("Symbolic Constraint Violations Overlay")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Transit Depth (ppm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join(outdir, save_name)
    plt.savefig(outpath)
    plt.close()
    print(f"✅ Saved μ + violation overlay: {outpath}")

    # Save raw overlay data
    np.savez(os.path.join(outdir, "mu_violation_overlay.npz"), mu=mu, violations=violations)

    # Optional: save as CSV
    if save_csv:
        df = pd.DataFrame({"bin": x, "mu": mu, "violation": violations})
        df.to_csv(os.path.join(outdir, "mu_violation_overlay.csv"), index=False)
        print("📄 Saved overlay values to mu_violation_overlay.csv")


# Demo
if __name__ == "__main__":
    mu = np.random.rand(5, 283) * 100
    violations = (np.random.rand(5, 283) > 0.8) * 20
    plot_violation_overlay(mu, violations, normalize=False)