"""
SpectraMind V50 – Predicted μ Quantiles vs Ground Truth Target
----------------------------------------------------------------
Aggregates predicted μ across all planets and overlays empirical target.
Supports optional COREL-calibrated σ envelope and logging.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import typer

app = typer.Typer(help="Plot predicted μ quantiles across planets vs target spectrum")

@app.command()
def compare_mu_quantiles(
    sub_csv: Path = typer.Option("submission.csv", help="Predicted μ+σ CSV"),
    gt_csv: Path = typer.Option("ground_truth.csv", help="Ground truth μ CSV"),
    corel_sigma: Path = typer.Option("outputs/corel_binwise_sigma.csv", help="Optional COREL σ file"),
    outdir: Path = typer.Option("outputs/diagnostics", help="Directory to save plot and logs"),
    log_file: Path = typer.Option("v50_debug_log.md", help="Log file for tracking summary")
):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(sub_csv)
    gt = pd.read_csv(gt_csv)

    mu_cols = [f"mu_{i}" for i in range(283)]
    mu = df[mu_cols].values  # shape: (N, 283)
    y = gt[mu_cols].mean(axis=0).values  # shape: (283,)

    q10 = np.percentile(mu, 10, axis=0)
    q25 = np.percentile(mu, 25, axis=0)
    q50 = np.percentile(mu, 50, axis=0)
    q75 = np.percentile(mu, 75, axis=0)
    q90 = np.percentile(mu, 90, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(y, label="Empirical Target μ", color="black", linewidth=2)
    plt.plot(q50, label="Predicted Median (q50)", color="blue")
    plt.fill_between(range(283), q10, q90, alpha=0.2, label="q10–q90", color="blue")
    plt.fill_between(range(283), q25, q75, alpha=0.4, label="q25–q75", color="cyan")

    if corel_sigma.exists():
        sigma_df = pd.read_csv(corel_sigma)
        sigma = sigma_df.iloc[0].values[:283].astype(float)
        mu_mean = q50
        band_lower = mu_mean - 1.2816 * sigma
        band_upper = mu_mean + 1.2816 * sigma
        plt.fill_between(range(283), band_lower, band_upper, alpha=0.25, label="COREL ±1.28σ", color="gray")

    plt.xlabel("Spectral Bin")
    plt.ylabel("Transit Depth (ppm)")
    plt.title("μ Quantile Envelope vs Ground Truth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = outdir / "quantiles_vs_target.png"
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")

    with open(log_file, "a") as f:
        f.write("\n### μ Quantile vs Target Summary\n")
        f.write(f"- N planets: {len(df)}\n")
        f.write(f"- Median predicted μ mean: {q50.mean():.3f}\n")
        f.write(f"- Target μ mean: {y.mean():.3f}\n")
        f.write(f"- Plot saved to: {path}\n")

if __name__ == "__main__":
    app()
