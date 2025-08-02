"""
SpectraMind V50 – Diagnostic Summary Generator (Ultimate)
----------------------------------------------------------
Performs full-spectrum diagnostics:
- GLL, MAE, RMSE, σ calibration
- Z-score and residual plots
- Binwise GLL heatmap
- Symbolic violation overlays
- SHAP + entropy + symbolic fusion miner
- Quantile band constraints
- JSON logs for CI dashboards and symbolic feedback
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from gll_error_localizer import compute_binwise_gll, plot_gll_heatmap
from constraint_violation_overlay import plot_violation_overlay
from spectral_event_miner import mine_anomalous_bins
from quantile_violation_chart import plot_quantile_vs_constraints
from generate_quantile_bands import compute_quantile_bands


def plot_zscore_and_fft(mu, sigma, y, outdir="diagnostics"):
    os.makedirs(outdir, exist_ok=True)
    residuals = mu - y
    zscores = residuals / (sigma + 1e-8)

    plt.figure()
    plt.hist(zscores.flatten(), bins=100, alpha=0.7, color="royalblue")
    plt.title("Z-score Distribution (Residual / σ)")
    plt.xlabel("Z")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "zscore_distribution.png"))
    plt.close()

    plt.figure()
    plt.plot(np.mean(sigma, axis=0), label="Mean σ", lw=2)
    plt.plot(np.std(residuals, axis=0), label="Empirical std", lw=2)
    plt.title("Calibration Check: σ vs Residual Spread")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "sigma_vs_residual_std.png"))
    plt.close()

    plt.figure()
    for i in range(min(3, residuals.shape[0])):
        fft = np.abs(np.fft.rfft(residuals[i]))
        plt.plot(fft, label=f"Planet {i}")
    plt.title("FFT Power of Residuals")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "fft_residuals.png"))
    plt.close()


def gll_score(y, mu, sigma):
    eps = 1e-6
    gll = 0.5 * np.log(2 * np.pi * (sigma ** 2 + eps)) + ((y - mu) ** 2) / (2 * (sigma ** 2 + eps))
    return gll.mean()


def generate_diagnostic_summary(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    shap: Optional[np.ndarray] = None,
    entropy: Optional[np.ndarray] = None,
    violations: Optional[np.ndarray] = None,
    outdir: str = "outputs/diagnostics",
    save_json: bool = True,
    symbolic_config: Optional[dict] = None,
):
    os.makedirs(outdir, exist_ok=True)
    summary = {}

    # --- Section 1: Core performance metrics ---
    residuals = mu - y
    summary.update({
        "gll_score": float(gll_score(y, mu, sigma)),
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt((residuals ** 2).mean())),
        "mean_sigma": float(sigma.mean()),
        "empirical_std": float(np.std(residuals)),
        "calibration_ratio": float(np.std(residuals) / (sigma.mean() + 1e-8)),
        "symbolic_config": symbolic_config or {},
    })

    plot_zscore_and_fft(mu, sigma, y, outdir)

    # --- Section 2: Binwise GLL ---
    print("🔍 Computing binwise GLL...")
    gll_bins = compute_binwise_gll(y, mu, sigma)
    plot_gll_heatmap(gll_bins, outdir)
    summary["gll_per_bin"] = gll_bins.tolist()

    # --- Section 3: Symbolic overlays ---
    if violations is not None:
        print("📏 Rendering symbolic violation overlay...")
        mu_avg = mu.mean(axis=0)
        viol_avg = violations.mean(axis=0)
        plot_violation_overlay(mu_avg, viol_avg, outdir)
        summary["violation_avg"] = viol_avg.tolist()

        if save_json:
            with open(os.path.join(outdir, "constraint_violation_log.json"), "w") as f:
                json.dump(violations.tolist(), f, indent=2)

    # --- Section 4: SHAP + entropy + symbolic fusion ---
    if shap is not None and entropy is not None and violations is not None:
        print("⚠️ Mining anomalous bins from SHAP+Entropy+Violations...")
        events = mine_anomalous_bins({
            "shap": shap.mean(axis=0),
            "entropy": entropy.mean(axis=0),
            "violations": violations.mean(axis=0),
            "gll": gll_bins
        })
        summary["anomalous_bins"] = events
        print(f"🚨 Anomalous bins identified: {events}")
        if save_json:
            with open(os.path.join(outdir, "anomalous_bins.json"), "w") as f:
                json.dump({"bins": events}, f, indent=2)

    # --- Section 5: Quantile band analysis ---
    print("📐 Computing quantile bands...")
    bands = compute_quantile_bands(torch.tensor(mu))
    summary["quantiles"] = {k: v.tolist() for k, v in bands.items()}

    if violations is not None:
        mask = (violations.mean(axis=0) > 0.1).astype(float)
        plot_quantile_vs_constraints(bands["q25"], bands["q75"], mask, outdir)

    # --- Section 6: Final summary export ---
    if save_json:
        with open(os.path.join(outdir, "diagnostic_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ diagnostic_summary.json written to {outdir}")