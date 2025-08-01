"""
SpectraMind V50 – Posterior Explorer Dashboard
----------------------------------------------
Generates an interactive HTML dashboard to explore μ, σ, SHAP, and symbolic overlays per planet.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import typer
from datetime import datetime
from pathlib import Path

app = typer.Typer(help="Generate interactive posterior dashboard for μ, σ, SHAP, and symbolic overlays.")

def build_dashboard(planet_id: str,
                    mu: np.ndarray,
                    sigma: np.ndarray,
                    shap: np.ndarray = None,
                    symbolic: np.ndarray = None,
                    outdir="outputs/html_report",
                    timestamp: bool = False):
    os.makedirs(outdir, exist_ok=True)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=mu, mode="lines", name="μ (spectrum)",
        line=dict(color="blue", width=2), hovertemplate="μ: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        y=mu + sigma, mode="lines", name="μ + σ",
        line=dict(color="lightblue", dash="dot"), hovertemplate="μ+σ: %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        y=mu - sigma, mode="lines", name="μ - σ",
        line=dict(color="lightblue", dash="dot"), hovertemplate="μ−σ: %{y:.2f}<extra></extra>",
        fill='tonexty', fillcolor="rgba(173, 216, 230, 0.2)"
    ))

    if shap is not None:
        fig.add_trace(go.Scatter(
            y=shap, mode="lines", name="SHAP",
            line=dict(color="green", dash="dash"), hovertemplate="SHAP: %{y:.2f}<extra></extra>"
        ))
    if symbolic is not None:
        fig.add_trace(go.Scatter(
            y=symbolic, mode="lines", name="Symbolic Influence",
            line=dict(color="red", dash="dot"), hovertemplate="Symbolic: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=f"Posterior Explorer – {planet_id}",
        xaxis_title="Spectral Bin",
        yaxis_title="Value (μ, σ, SHAP)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
        height=550,
        width=1000
    )

    filename = f"{planet_id}_dashboard.html"
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{planet_id}_dashboard_{ts}.html"

    output_path = os.path.join(outdir, filename)
    fig.write_html(output_path)
    print(f"✅ Saved interactive dashboard: {output_path}")


@app.command()
def standalone(
    planet_id: str = typer.Option(..., help="Planet ID to label dashboard"),
    mu_path: Path = typer.Option(..., help="Path to .npy or .csv file with μ values (283,)"),
    sigma_path: Path = typer.Option(..., help="Path to .npy or .csv file with σ values (283,)"),
    shap_path: Path = typer.Option(None, help="Optional path to SHAP values (283,)"),
    symbolic_path: Path = typer.Option(None, help="Optional path to symbolic scores (283,)"),
    outdir: Path = typer.Option("outputs/html_report", help="Output folder"),
    timestamp: bool = False
):
    """
    Generate a per-planet interactive dashboard from .npy or .csv arrays.
    """
    def load(path: Path):
        if path is None:
            return None
        return np.load(path) if path.suffix == ".npy" else pd.read_csv(path, header=None).values.flatten()

    mu = load(mu_path)
    sigma = load(sigma_path)
    shap = load(shap_path)
    symbolic = load(symbolic_path)

    build_dashboard(planet_id, mu, sigma, shap, symbolic, outdir=outdir, timestamp=timestamp)


if __name__ == "__main__":
    app()
