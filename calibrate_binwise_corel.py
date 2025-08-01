"""
SpectraMind V50 – Binwise Conformal Calibration Module
------------------------------------------------------
Applies conformal prediction to calibrate uncertainty estimates (σ)
per spectral bin using held-out calibration data. Supports visualization,
quantile control, log, and aggregated outputs.
"""

import pandas as pd
import numpy as np
import typer
import os
from pathlib import Path
import matplotlib.pyplot as plt

app = typer.Typer(help="SpectraMind V50 – COREL Binwise Conformal Calibration")

def compute_nonconformity(mu, y_true):
    return np.abs(mu - y_true)

def calibrate_sigma(nonconformity_scores, alpha=0.1):
    return np.quantile(nonconformity_scores, 1 - alpha, method='higher')

@app.command()
def calibrate_corel(
    mu_file: Path = typer.Option("calib_mu.csv", help="CSV with μ predictions (calibration set)"),
    y_file: Path = typer.Option("calib_targets.csv", help="CSV with ground-truth μ for calibration set"),
    out_file: Path = typer.Option("outputs/corel_binwise_sigma.csv", help="Output CSV with σ estimates"),
    alpha: float = typer.Option(0.1, help="Miscoverage level (e.g. 0.1 for 90% coverage)"),
    plot: bool = typer.Option(True, help="Whether to plot σ across bins"),
    log: Path = typer.Option("v50_debug_log.md", help="Log file to append summary")
):
    os.makedirs(out_file.parent, exist_ok=True)

    print("📂 Loading calibration predictions and targets...")
    mu = pd.read_csv(mu_file).values
    y_true = pd.read_csv(y_file).values
    assert mu.shape == y_true.shape, "Shape mismatch between μ and targets"

    n_bins = mu.shape[1]
    sigma_binwise = []

    print("📏 Computing binwise conformal σ estimates...")
    for i in range(n_bins):
        nonconf_scores = compute_nonconformity(mu[:, i], y_true[:, i])
        sigma_i = calibrate_sigma(nonconf_scores, alpha)
        sigma_binwise.append(sigma_i)

    sigma_binwise = np.array(sigma_binwise)
    df_sigma = pd.DataFrame(sigma_binwise.reshape(1, -1), columns=[f"sigma_{i}" for i in range(n_bins)])
    df_sigma.to_csv(out_file, index=False)
    print(f"✅ Binwise σ saved to: {out_file}")

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(sigma_binwise, color="black", linewidth=1.5)
        plt.title(f"Binwise COREL σ Estimates (α = {alpha})")
        plt.xlabel("Spectral Bin")
        plt.ylabel("Calibrated σ")
        plt.grid(True)
        plt.tight_layout()
        plot_path = out_file.with_suffix(".png")
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Plot saved to: {plot_path}")

    with open(log, "a") as f:
        f.write(f"\n### COREL Calibration Log\n")
        f.write(f"- Calibration α: {alpha}\n")
        f.write(f"- Output file: {out_file}\n")
        f.write(f"- Mean σ: {sigma_binwise.mean():.4f}, Std σ: {sigma_binwise.std():.4f}\n")

if __name__ == "__main__":
    app()
