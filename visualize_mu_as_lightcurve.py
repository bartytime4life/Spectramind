"""
SpectraMind V50 – μ-to-Lightcurve Visualizer
--------------------------------------------
Generates a diagnostic lightcurve from a μ(λ) spectrum using star/planet metadata.
Supports multi-planet files, transit kernel scaling, and CSV/PNG outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path

app = typer.Typer(help="Convert μ(λ) spectra to toy transit lightcurves")

@app.command()
@torch.no_grad()
def simulate_lightcurve(
    mu_csv: Path = typer.Option(..., help="CSV with μ predictions (e.g., submission.csv)"),
    meta_csv: Path = typer.Option(..., help="CSV with metadata (e.g., star_info.csv)"),
    output: Path = typer.Option("outputs/simulated_lightcurve.png", help="Output image path"),
    planet_index: int = typer.Option(0, help="Index of planet to plot (default: 0)"),
    export_csv: bool = typer.Option(True, help="Also save lightcurve values as CSV")
):
    """
    Visualizes a lightcurve shape synthesized from the μ transmission spectrum.
    """
    sub = pd.read_csv(mu_csv)
    meta = pd.read_csv(meta_csv)

    if planet_index >= len(sub):
        print(f"❌ Planet index {planet_index} out of bounds.")
        raise typer.Exit(1)

    mu = sub.iloc[planet_index, 1:284].values
    pid = sub.iloc[planet_index, 0]

    mp = meta.iloc[planet_index].get('Mp', 1.0)
    i = meta.iloc[planet_index].get('i', 90.0)
    p = meta.iloc[planet_index].get('P', 1.0)
    sma = meta.iloc[planet_index].get('sma', 1.0)

    # Normalize μ
    norm_mu = (mu - mu.min()) / (mu.max() - mu.min() + 1e-8)

    # Time grid and toy Gaussian transit kernel
    t = np.linspace(-2, 2, 300)
    transit_kernel = 1 - 0.002 * np.exp(-t**2 / 0.25)

    # μ to lightcurve via convolution
    lightcurve = np.convolve(transit_kernel, norm_mu[::-1], mode='same')
    baseline = np.ones_like(lightcurve)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, baseline, color="gray", linestyle="--", label="Baseline")
    plt.plot(t, lightcurve, label=f"Simulated Light Curve – {pid}", color="black")
    plt.title(f"μ → Simulated Transit Light Curve for {pid}")
    plt.xlabel("Time (arb units)")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(f"✅ Saved lightcurve to {output}")

    if export_csv:
        csv_out = output.with_suffix(".csv")
        pd.DataFrame({"t": t, "flux": lightcurve}).to_csv(csv_out, index=False)
        print(f"📄 Saved lightcurve CSV to {csv_out}")

if __name__ == "__main__":
    app()


