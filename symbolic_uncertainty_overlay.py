"""
SpectraMind V50 – Symbolic-Uncertainty Overlay
----------------------------------------------
Highlights bins with high constraint violations AND high residuals vs σ.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def overlay_symbolic_uncertainty(
    submission_csv: str,
    ground_truth_csv: str,
    symbolic_json: str,
    outdir="outputs/diagnostics"
):
    os.makedirs(outdir, exist_ok=True)

    sub = pd.read_csv(submission_csv).set_index("planet_id")
    gt = pd.read_csv(ground_truth_csv).set_index("planet_id")

    with open(symbolic_json) as f:
        sym = pd.read_json(f).T  # expected format: {planet_id: [violation flags]}

    mu = sub[[f"mu_{i}" for i in range(283)]].values
    sigma = np.clip(sub[[f"sigma_{i}" for i in range(283)]].values, 1e-8, None)
    y = gt[[f"mu_{i}" for i in range(283)]].values
    residual = np.abs(y - mu)

    symbolic = sym.loc[sub.index].values.astype(float)
    outlier_mask = residual > sigma
    combined = symbolic * outlier_mask

    # Average across planets
    mean_symbolic = symbolic.mean(axis=0)
    mean_outlier = outlier_mask.mean(axis=0)
    mean_combined = combined.mean(axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(mean_symbolic, label="Symbolic Violations", color="red")
    plt.plot(mean_outlier, label="Uncertainty Failures", color="blue")
    plt.plot(mean_combined, label="Both (Overlap)", color="black", linewidth=2)
    plt.legend()
    plt.title("Symbolic vs Uncertainty Overlap per Spectral Bin")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Fraction of Planets")
    plt.grid(True)
    plt.tight_layout()
    fpath = os.path.join(outdir, "symbolic_uncertainty_overlay.png")
    plt.savefig(fpath)
    print(f"📊 Saved overlay plot: {fpath}")