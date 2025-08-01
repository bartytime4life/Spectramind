"""
SpectraMind V50 – Dropout Impact Diagnostic
-------------------------------------------
Compares per-bin μ variance and σ mean across ensemble variants:
• with dropout/residual
• without dropout/residual

Input: two submission.csv files (dropout vs baseline)
Output: line plot of μ std dev and σ means across spectral bins
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer
from pathlib import Path

app = typer.Typer(help="Compare variance of μ predictions with vs without decoder dropout")

@app.command()
def compare_dropout_effect(
    submission_dropout: Path = typer.Option(..., help="submission.csv from model with dropout"),
    submission_baseline: Path = typer.Option(..., help="submission.csv from model without dropout"),
    out_file: Path = typer.Option("outputs/diagnostics/dropout_vs_nodropout_variance.png")
):
    os.makedirs(out_file.parent, exist_ok=True)
    
    def load_mu_sigma(file):
        df = pd.read_csv(file)
        mu = df[[c for c in df.columns if c.startswith("mu_")]].values
        sigma = df[[c for c in df.columns if c.startswith("sigma_")]].values
        return mu, sigma

    mu_drop, sigma_drop = load_mu_sigma(submission_dropout)
    mu_base, sigma_base = load_mu_sigma(submission_baseline)

    mu_std_drop = np.std(mu_drop, axis=0)
    mu_std_base = np.std(mu_base, axis=0)
    sigma_mean_drop = np.mean(sigma_drop, axis=0)
    sigma_mean_base = np.mean(sigma_base, axis=0)

    bins = np.arange(len(mu_std_drop))

    plt.figure(figsize=(12, 5))
    plt.plot(bins, mu_std_drop, label="μ std (dropout)", color="blue")
    plt.plot(bins, mu_std_base, label="μ std (baseline)", color="gray")
    plt.plot(bins, sigma_mean_drop, label="σ mean (dropout)", linestyle=":", color="blue")
    plt.plot(bins, sigma_mean_base, label="σ mean (baseline)", linestyle=":", color="gray")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Variance / Uncertainty (ppm)")
    plt.title("Effect of Dropout + Residual on μ Prediction Variance")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"✅ Saved: {out_file}")

if __name__ == "__main__":
    app()
