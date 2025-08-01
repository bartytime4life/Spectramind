"""
SpectraMind V50 – Uncertainty Report Generator
---------------------------------------------
Generates combined diagnostics on σ calibration, residual spread, and GLL.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gll_loss(y, mu, sigma):
    eps = 1e-8
    sigma = np.clip(sigma, eps, None)
    return np.mean(((y - mu) / sigma) ** 2 + 2 * np.log(sigma))


def generate_uncertainty_report(sub_csv: str, gt_csv: str, outdir="outputs/diagnostics"):
    os.makedirs(outdir, exist_ok=True)

    sub = pd.read_csv(sub_csv).set_index("planet_id")
    gt = pd.read_csv(gt_csv).set_index("planet_id")

    mu_cols = [f"mu_{i}" for i in range(283)]
    sigma_cols = [f"sigma_{i}" for i in range(283)]

    # Align index
    common = sub.index.intersection(gt.index)
    sub = sub.loc[common]
    gt = gt.loc[common]

    mu = sub[mu_cols].values
    sigma = sub[sigma_cols].values
    y = gt[mu_cols].values

    residual = y - mu
    abs_resid = np.abs(residual)
    coverage = (abs_resid < sigma)
    coverage_rate = np.mean(coverage)
    mean_sigma = np.mean(sigma)
    mae = np.mean(abs_resid)
    gll = gll_loss(y, mu, sigma)

    # Scatter plot
    plt.figure(figsize=(10, 4))
    plt.scatter(sigma.flatten(), abs_resid.flatten(), alpha=0.3, s=2, c='navy')
    plt.xlabel("Predicted σ")
    plt.ylabel("|y − μ|")
    plt.title(f"Residual vs σ  (Coverage = {coverage_rate:.3f})")
    plt.grid(True)
    plt.tight_layout()

    scatter_path = os.path.join(outdir, "residual_vs_sigma_scatter.png")
    plt.savefig(scatter_path)
    plt.close()

    print(f"✅ Coverage rate: {coverage_rate:.3f}")
    print(f"📊 Scatter plot saved to: {scatter_path}")

    # Save metrics
    metrics = {
        "coverage_rate": float(coverage_rate),
        "mean_sigma": float(mean_sigma),
        "mae": float(mae),
        "gll": float(gll),
        "n_planets": int(mu.shape[0]),
        "n_bins": int(mu.shape[1])
    }

    metrics_path = os.path.join(outdir, "uncertainty_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 Uncertainty metrics saved to: {metrics_path}")

    # Optional CSV of per-bin coverage
    coverage_df = pd.DataFrame(coverage.reshape(-1, 283), index=common)
    coverage_df.to_csv(os.path.join(outdir, "binwise_coverage_mask.csv"))
    print(f"📄 Per-bin coverage mask saved to: binwise_coverage_mask.csv")


if __name__ == "__main__":
    generate_uncertainty_report("submission.csv", "ground_truth.csv")