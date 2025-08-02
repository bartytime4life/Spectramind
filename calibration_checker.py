"""
SpectraMind V50 – Calibration Checker (Ultimate Version)
---------------------------------------------------------
Evaluates how well predicted σ uncertainty matches the residual error from μ vs y.
Integrates with:
- generate_diagnostic_summary.py (summary export)
- generate_html_report.py (calibration plot panel)
- symbolic_violation_predictor.py (to score calibration failures)
- plot_zscore_and_fft.py (z-score support)
- uncertainty_calibrator.py (can tune based on this result)
- auto_ablate_v50.py (to remove poorly calibrated bins)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Optional

def check_calibration(
    sub_csv: str,
    gt_csv: str,
    outdir: str = "diagnostics/calibration",
    save_name: str = "calibration_hist.png",
    export_json: Optional[str] = "calibration_summary.json"
):
    os.makedirs(outdir, exist_ok=True)
    sub = pd.read_csv(sub_csv)
    gt = pd.read_csv(gt_csv)

    mu_cols = [f"mu_{i}" for i in range(283)]
    sigma_cols = [f"sigma_{i}" for i in range(283)]

    mu = sub[mu_cols].values
    sigma = sub[sigma_cols].values
    y = gt[mu_cols].values

    residual = np.abs(y - mu)
    coverage_mask = residual < sigma
    coverage_rate = np.mean(coverage_mask)
    residual_mean = float(residual.mean())
    sigma_mean = float(sigma.mean())
    calibration_gap = residual_mean - sigma_mean

    print(f"✅ Coverage rate (|y - μ| < σ): {coverage_rate:.3f}")
    print(f"    Mean residual: {residual_mean:.3f}, Mean σ: {sigma_mean:.3f}, Gap: {calibration_gap:.3f}")

    plt.figure(figsize=(9, 4))
    plt.hist(residual.flatten(), bins=100, alpha=0.5, label="|y - μ| (residual)", color="darkred")
    plt.hist(sigma.flatten(), bins=100, alpha=0.5, label="σ predicted", color="teal")
    plt.axvline(residual_mean, color="darkred", linestyle="--", label="mean |y - μ|")
    plt.axvline(sigma_mean, color="teal", linestyle=":", label="mean σ")
    plt.title("Distribution of Residuals vs Predicted σ")
    plt.xlabel("Value (ppm)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(outdir, save_name)
    plt.savefig(plot_path)
    print(f"📊 Saved calibration plot: {plot_path}")

    if export_json:
        summary = {
            "coverage_rate": float(coverage_rate),
            "residual_mean": residual_mean,
            "sigma_mean": sigma_mean,
            "calibration_gap": calibration_gap
        }
        json_path = os.path.join(outdir, export_json)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📝 Exported calibration summary: {json_path}")


if __name__ == "__main__":
    check_calibration("submission.csv", "ground_truth.csv")