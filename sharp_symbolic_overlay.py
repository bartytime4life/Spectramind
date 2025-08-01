"""
SpectraMind V50 – SHAP × Symbolic Constraint Fusion Overlay
-----------------------------------------------------------
Overlays SHAP values with symbolic constraint violations per spectral bin
for unified visual debugging and symbolic interpretability enhancement.
Supports multiple fusion strategies, stats export, and per-bin logging.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import typer

app = typer.Typer(help="Generate SHAP × Symbolic Fusion Overlay Diagnostic")

@app.command()
def explain_symbolic_fusion(
    submission: Path = typer.Option("submission.csv", help="Path to submission CSV (μ, σ)"),
    shap_file: Path = typer.Option("shap_overlay.json", help="SHAP values per bin (planet_id → list[283])"),
    constraint_file: Path = typer.Option("constraint_violation_log.json", help="Constraint violations (planet_id → list[bin_idx])"),
    outdir: Path = typer.Option("outputs/diagnostics", help="Directory to save visualizations and metrics"),
    normalize_shap: bool = typer.Option(True, help="Normalize SHAP importance to sum=1"),
    save_json: bool = typer.Option(True, help="Save bin-wise overlay summary as JSON")
):
    """
    Generates a diagnostic overlay plot of SHAP importance vs symbolic violation rate.
    Computes per-bin scores and optionally saves metrics to JSON.
    """
    os.makedirs(outdir, exist_ok=True)

    print("🔍 Loading SHAP values...")
    with open(shap_file) as f:
        shap_data = json.load(f)

    print("📥 Loading symbolic constraint violations...")
    with open(constraint_file) as f:
        violations = json.load(f)

    print("📊 Loading submission data...")
    df = pd.read_csv(submission)
    planet_ids = df['planet_id'].tolist()
    mu_vals = df[[col for col in df.columns if col.startswith("mu_")]].values

    n_bins = mu_vals.shape[1]
    shap_importance = np.zeros(n_bins)
    violation_freq = np.zeros(n_bins)

    for pid in planet_ids:
        if pid not in shap_data:
            continue
        shap_importance += np.array(shap_data[pid])
        if pid in violations:
            for bin_idx in violations[pid]:
                if 0 <= int(bin_idx) < n_bins:
                    violation_freq[int(bin_idx)] += 1

    shap_importance /= len(planet_ids)
    violation_freq /= len(planet_ids)

    if normalize_shap:
        shap_importance = shap_importance / (shap_importance.sum() + 1e-6)

    fused = shap_importance * violation_freq

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    ax1.plot(shap_importance, label="SHAP Importance", color="blue")
    ax2.plot(violation_freq, label="Violation Rate", color="red", alpha=0.5)

    ax1.set_xlabel("Spectral Bin")
    ax1.set_ylabel("SHAP Importance", color="blue")
    ax2.set_ylabel("Violation Rate", color="red")
    ax1.set_title("SHAP × Symbolic Violation Overlay")
    ax1.grid(True)
    fig.tight_layout()

    out_img = outdir / "shap_symbolic_overlay.png"
    plt.savefig(out_img)
    print(f"✅ Overlay saved to: {out_img}")

    if save_json:
        overlay_stats = {
            f"bin_{i}": {
                "shap": float(shap_importance[i]),
                "violation": float(violation_freq[i]),
                "fused": float(fused[i])
            } for i in range(n_bins)
        }
        with open(outdir / "shap_symbolic_overlay.json", "w") as f:
            json.dump(overlay_stats, f, indent=2)
        print("📊 Saved bin-wise overlay metrics as JSON")

if __name__ == "__main__":
    app()
