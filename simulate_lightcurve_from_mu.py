"""
SpectraMind V50 – μ-to-Lightcurve Visualizer (Challenge Version)
------------------------------------------------------------------
Generates simulated transit light curves from μ(λ) spectra and metadata.
Supports batch mode, metadata injection, CSV export, and diagnostic plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import typer
from pathlib import Path

app = typer.Typer(help="Simulate lightcurves from μ(λ) spectrum")

def generate_lightcurve(mu: np.ndarray, t: np.ndarray, depth: float = 0.002, width: float = 0.25) -> np.ndarray:
    norm_mu = (mu - mu.min()) / (mu.max() - mu.min() + 1e-8)
    kernel = 1 - depth * np.exp(-t**2 / width)
    return np.convolve(kernel, norm_mu[::-1], mode='same')

@app.command()
@torch.no_grad()
def simulate_lightcurve(
    mu_csv: Path = typer.Option(..., help="CSV with μ predictions (e.g., submission.csv)"),
    meta_csv: Path = typer.Option(..., help="CSV with metadata (e.g., star_info.csv)"),
    output: Path = typer.Option("outputs/simulated_lightcurve.png", help="Output image path"),
    planet_index: int = typer.Option(0, help="Which row (planet) to simulate"),
    depth: float = typer.Option(0.002, help="Transit depth factor"),
    width: float = typer.Option(0.25, help="Transit kernel width"),
    export_csv: bool = typer.Option(True, help="Save flux as CSV"),
    plot_components: bool = typer.Option(True, help="Overlay μ and kernel in inset plot")
):
    sub = pd.read_csv(mu_csv)
    meta = pd.read_csv(meta_csv)

    if planet_index >= len(sub):
        print("❌ Index out of range")
        raise typer.Exit(1)

    row = sub.iloc[planet_index]
    pid = row["planet_id"]
    mu = row[[f"mu_{i}" for i in range(283)]].values
    star = meta.iloc[planet_index] if planet_index < len(meta) else {}

    t = np.linspace(-2, 2, 300)
    lightcurve = generate_lightcurve(mu, t, depth=depth, width=width)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, np.ones_like(t), linestyle="--", color="gray", label="Baseline")
    plt.plot(t, lightcurve, color="black", label=f"Lightcurve – {pid}")
    plt.title(f"Simulated Lightcurve from μ – {pid}")
    plt.xlabel("Time (arb)")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.legend()

    if plot_components:
        axin = plt.gca().inset_axes([0.65, 0.5, 0.3, 0.4])
        axin.plot(mu, label="μ", alpha=0.6)
        axin.plot(np.ones_like(mu) * mu.mean(), linestyle=":", color="gray", linewidth=0.5)
        axin.set_title("μ Components", fontsize=8)
        axin.tick_params(labelsize=6)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    print(f"✅ Saved lightcurve to {output}")

    if export_csv:
        csv_path = output.with_suffix(".csv")
        pd.DataFrame({"t": t, "flux": lightcurve}).to_csv(csv_path, index=False)
        print(f"📄 Saved lightcurve CSV to {csv_path}")

if __name__ == "__main__":
    app()


