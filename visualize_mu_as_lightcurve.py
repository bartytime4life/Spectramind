"""
SpectraMind V50 – μ-to-Lightcurve Visualizer
--------------------------------------------
Generates transit lightcurve diagnostics from μ(λ) spectra using star/planet metadata.
Supports single or batch mode with CSV and image output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path

app = typer.Typer(help="Convert μ(λ) spectra to toy transit lightcurves")

def simulate_single_lightcurve(mu: np.ndarray, t: np.ndarray) -> np.ndarray:
    norm_mu = (mu - mu.min()) / (mu.max() - mu.min() + 1e-8)
    kernel = 1 - 0.002 * np.exp(-t**2 / 0.25)
    return np.convolve(kernel, norm_mu[::-1], mode='same')

@app.command()
@torch.no_grad()
def simulate_lightcurve(
    mu_csv: Path = typer.Option(..., help="CSV with μ predictions (e.g., submission.csv)"),
    meta_csv: Path = typer.Option(..., help="CSV with metadata (e.g., star_info.csv)"),
    output_dir: Path = typer.Option("outputs/lightcurves", help="Directory to save visualizations"),
    planet_index: int = typer.Option(0, help="Planet index if not using --all"),
    export_csv: bool = typer.Option(True, help="Save lightcurve values as CSV"),
    all: bool = typer.Option(False, help="Simulate lightcurves for all planets"),
    limit: int = typer.Option(0, help="If > 0, limit to first N planets")
):
    """
    Converts μ spectra to lightcurves using toy convolution kernel. Supports --all mode.
    """
    sub = pd.read_csv(mu_csv)
    meta = pd.read_csv(meta_csv)

    t = np.linspace(-2, 2, 300)
    output_dir.mkdir(parents=True, exist_ok=True)

    planet_rows = list(range(len(sub))) if all else [planet_index]
    if limit > 0:
        planet_rows = planet_rows[:limit]

    for i in planet_rows:
        row = sub.iloc[i]
        pid = row["planet_id"]
        mu = row[[f"mu_{j}" for j in range(283)]].values

        lightcurve = simulate_single_lightcurve(mu, t)
        baseline = np.ones_like(lightcurve)

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(t, baseline, linestyle="--", color="gray", label="Baseline")
        plt.plot(t, lightcurve, label=f"{pid}", color="black")
        plt.title(f"μ → Simulated Transit Light Curve for {pid}")
        plt.xlabel("Time (arb)")
        plt.ylabel("Flux")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        png_path = output_dir / f"simulated_lightcurve_{pid}.png"
        plt.savefig(png_path)
        plt.close()
        print(f"✅ Saved lightcurve plot to {png_path}")

        if export_csv:
            csv_path = output_dir / f"simulated_lightcurve_{pid}.csv"
            pd.DataFrame({"t": t, "flux": lightcurve}).to_csv(csv_path, index=False)
            print(f"📄 Saved lightcurve data to {csv_path}")

if __name__ == "__main__":
    app()
