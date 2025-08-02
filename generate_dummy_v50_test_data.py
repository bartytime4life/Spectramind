"""
SpectraMind V50 – Dummy Test Data Generator
-------------------------------------------
Creates synthetic test data for μ, σ, y, COREL, edge_index, submission.csv,
and symbolic overlays compatible with the full SpectraMind V50 pipeline.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def generate(
    n: int = typer.Option(4, help="Number of dummy planets to generate"),
    outdir: str = typer.Option("outputs", help="Output directory for μ, σ, y")
):
    # --- Create directories ---
    dirs = [outdir, "models", "calibration_data", "diagnostics"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)

    B = 283
    planet_ids = [f"test_planet_{i}" for i in range(n)]

    # --- Dummy μ, σ, y ---
    mu = torch.randn(n, B).clamp(min=0) * 1000
    sigma = torch.abs(torch.randn(n, B)) + 10.0
    y = mu + torch.randn_like(mu) * 0.5

    torch.save(mu, f"{outdir}/mu.pt")
    torch.save(sigma, f"{outdir}/sigma.pt")
    torch.save(y, f"{outdir}/y.pt")

    # --- Dummy edge_index (linear) ---
    edge_index = torch.tensor(
        [[i, i+1] for i in range(B-1)] + [[i+1, i] for i in range(B-1)]
    ).T
    torch.save(edge_index, "calibration_data/edge_index.pt")

    # --- Dummy COREL model weights ---
    corel_state_dict = {
        'conv1.weight': torch.randn(64, 1),
        'conv1.bias': torch.randn(64),
        'conv2.weight': torch.randn(64, 64),
        'conv2.bias': torch.randn(64),
        'out_mean.weight': torch.randn(1, 64),
        'out_mean.bias': torch.randn(1),
        'out_radius.weight': torch.randn(1, 64),
        'out_radius.bias': torch.randn(1)
    }
    torch.save(corel_state_dict, "models/corel_gnn.pt")

    # --- Dummy submission.csv ---
    columns = ["planet_id"] + [f"mu_{i}" for i in range(B)] + [f"sigma_{i}" for i in range(B)]
    data = []
    for i in range(n):
        data.append([planet_ids[i]] + list(mu[i].numpy()) + list(sigma[i].numpy()))
    pd.DataFrame(data, columns=columns).to_csv("submission.csv", index=False)

    # --- Dummy symbolic overlays ---
    pd.DataFrame({
        "planet_id": planet_ids,
        "symbolic_class": ["water", "CO2", "clouds", "unknown"][:n]
    }).to_csv("diagnostics/symbolic_clusters.csv", index=False)

    typer.secho(f"\n✅ Dummy test data generated for {n} planets.", fg=typer.colors.GREEN)
    typer.echo("- μ, σ, y → outputs/")
    typer.echo("- COREL model → models/corel_gnn.pt")
    typer.echo("- Graph edges → calibration_data/edge_index.pt")
    typer.echo("- submission.csv created")
    typer.echo("- symbolic_clusters.csv → diagnostics/")

if __name__ == "__main__":
    app()