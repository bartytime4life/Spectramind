"""
SpectraMind V50 – Posterior Explorer Dashboard (Final Version)
--------------------------------------------------------------
Generates an interactive HTML dashboard for μ, σ, SHAP, entropy, symbolic overlays,
and quantile bands, per planet.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json


def build_dashboard(
    planet_id: str,
    mu: np.ndarray,
    sigma: np.ndarray,
    shap: np.ndarray = None,
    symbolic: np.ndarray = None,
    entropy: np.ndarray = None,
    quantiles: dict = None,  # expects q10, q25, q75, q90
    shap_attention: np.ndarray = None,
    outdir: str = "diagnostics/html_report"
):
    os.makedirs(outdir, exist_ok=True)
    x = np.arange(len(mu))
    fig = go.Figure()

    # μ ± σ envelope
    fig.add_trace(go.Scatter(
        x=x, y=mu, mode="lines", name="μ (mean)", line=dict(color="blue", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mu + sigma, mode="lines", name="μ + σ", line=dict(color="lightblue"), opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mu - sigma, mode="lines", name="μ - σ", line=dict(color="lightblue"), opacity=0.5,
        fill='tonexty'
    ))

    # Quantiles
    if quantiles:
        for qname in ["q10", "q25", "q75", "q90"]:
            if qname in quantiles:
                fig.add_trace(go.Scatter(
                    x=x, y=quantiles[qname], mode="lines", name=qname, line=dict(dash="dot", width=1)
                ))

    # SHAP
    if shap is not None:
        fig.add_trace(go.Scatter(
            x=x, y=shap, mode="lines", name="SHAP", line=dict(color="green")
        ))

    # SHAP + Attention fusion
    if shap_attention is not None:
        fig.add_trace(go.Scatter(
            x=x, y=shap_attention, mode="lines", name="SHAP-Attention Fusion", line=dict(color="darkgreen", dash="dash")
        ))

    # Symbolic
    if symbolic is not None:
        fig.add_trace(go.Scatter(
            x=x, y=symbolic, mode="lines", name="Symbolic Influence", line=dict(color="crimson")
        ))

    # Entropy
    if entropy is not None:
        fig.add_trace(go.Scatter(
            x=x, y=entropy, mode="lines", name="Entropy", line=dict(color="purple", dash="dot")
        ))

    fig.update_layout(
        title=f"Posterior Explorer – {planet_id}",
        xaxis_title="Spectral Bin",
        yaxis_title="Value",
        legend=dict(orientation="h"),
        template="plotly_white",
        height=500,
        width=1000
    )

    html_path = os.path.join(outdir, f"{planet_id}_dashboard.html")
    fig.write_html(html_path)

    # Optional export summary JSON
    summary_json = {
        "planet_id": planet_id,
        "mu_mean": float(mu.mean()),
        "sigma_mean": float(sigma.mean()),
        "shap_max": float(np.max(shap)) if shap is not None else None,
        "entropy_mean": float(np.mean(entropy)) if entropy is not None else None
    }
    json_path = os.path.join(outdir, f"{planet_id}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"✅ Dashboard HTML saved: {html_path}")
    print(f"📝 Summary JSON saved: {json_path}")