"""
SpectraMind V50 – Ensemble Prediction Aggregator
-------------------------------------------------
Combines predictions from multiple models using mean, median, or custom weighting.
Outputs final μ and σ curves to submission.csv. Includes diagnostics, logging, and fallback.
"""

import numpy as np
import pandas as pd
import typer
from pathlib import Path
import os
import matplotlib.pyplot as plt

app = typer.Typer(help="Run ensemble fusion from multiple model prediction CSVs")

@app.command()
def ensemble(
    prediction_dir: Path = typer.Option("outputs/ensemble_candidates", help="Folder containing candidate submission CSVs"),
    method: str = typer.Option("mean", help="Ensemble method: mean | median"),
    out_file: Path = typer.Option("submission.csv", help="Path to save final submission CSV"),
    log_file: Path = typer.Option("v50_debug_log.md", help="Optional log file to append"),
    plot_diagnostics: bool = typer.Option(True, help="Whether to plot std envelope across ensemble")
):
    files = list(prediction_dir.glob("*.csv"))
    assert len(files) >= 2, "Need at least 2 submission files for ensemble"
    print(f"📊 Loading {len(files)} predictions from: {prediction_dir}")

    all_mu = []
    all_sigma = []
    planet_ids = None

    for f in files:
        df = pd.read_csv(f)
        mu_cols = [c for c in df.columns if c.startswith("mu_")]
        sigma_cols = [c for c in df.columns if c.startswith("sigma_")]
        all_mu.append(df[mu_cols].values)
        all_sigma.append(df[sigma_cols].values)
        if planet_ids is None:
            planet_ids = df["planet_id"].tolist()

    mu_stack = np.stack(all_mu, axis=0)
    sigma_stack = np.stack(all_sigma, axis=0)

    if method == "mean":
        mu_ens = mu_stack.mean(axis=0)
        sigma_ens = sigma_stack.mean(axis=0)
    elif method == "median":
        mu_ens = np.median(mu_stack, axis=0)
        sigma_ens = np.median(sigma_stack, axis=0)
    else:
        raise ValueError("Unsupported method: choose 'mean' or 'median'")

    out_df = pd.DataFrame()
    out_df["planet_id"] = planet_ids
    for i in range(mu_ens.shape[1]):
        out_df[f"mu_{i}"] = mu_ens[:, i]
        out_df[f"sigma_{i}"] = sigma_ens[:, i]

    os.makedirs(out_file.parent, exist_ok=True)
    out_df.to_csv(out_file, index=False)
    print(f"✅ Ensemble submission saved to {out_file}")

    if plot_diagnostics:
        std_mu = mu_stack.std(axis=0).mean(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(std_mu, color="darkorange", label="μ Ensemble Std")
        plt.title("Mean Standard Deviation Across Ensemble (μ)")
        plt.xlabel("Spectral Bin")
        plt.ylabel("Std Dev")
        plt.grid(True)
        plt.tight_layout()
        fig_path = out_file.with_suffix("_ensemble_spread.png")
        plt.savefig(fig_path)
        print(f"📈 Saved diagnostics plot to {fig_path}")

    if log_file:
        with open(log_file, "a") as f:
            f.write("\n### Ensemble Summary\n")
            f.write(f"- Num models: {len(files)}\n")
            f.write(f"- Method: {method}\n")
            f.write(f"- Output: {out_file}\n")
            f.write(f"- μ mean ± std: {mu_ens.mean():.3f} ± {mu_ens.std():.3f}\n")
            f.write(f"- σ mean ± std: {sigma_ens.mean():.3f} ± {sigma_ens.std():.3f}\n")

if __name__ == "__main__":
    app()
