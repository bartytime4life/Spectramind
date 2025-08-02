"""
SpectraMind V50 – Integration Test Runner (Ultimate Version)
------------------------------------------------------------
Runs and verifies every core pipeline stage to ensure full system operation.
Integrates with:
- cli_core_v50.py (train, predict, validate, calibrate, finalize, package)
- submission_validator_v50.py (L score)
- generate_html_report.py (diagnostic dashboard)
- generate_quantile_bands.py (quantile plotting)
- calibration_checker.py (uncertainty evaluation)
- generate_uncertainty_report.py (μ-σ scatter)
- plot_quantile_overlay.py (interactive quantile logic)
- generate_diagnostic_summary.py (QA pipeline log)
"""

import os
import pandas as pd
import torch

from src.spectramind.cli.commands import (
    run_train,
    run_predict,
    run_validate,
    run_finalize,
    run_package,
    run_diagnostics
)
from validate import score_submission
from calibration_checker import check_calibration
from generate_uncertainty_report import generate_uncertainty_report
from generate_html_report import generate_html
from generate_quantile_bands import compute_quantile_bands
from plot_quantile_overlay import plot_quantiles
from plot_quantiles_vs_target import plot_quantiles_vs_target


def run_integration():
    print("\n🚀 SpectraMind V50 – Full Integration Pipeline Test")

    print("\n🔁 [1] Training...")
    run_train()

    print("\n🔁 [2] Prediction...")
    run_predict()

    print("\n🔍 [3] Validating submission.csv...")
    run_validate()

    print("\n📦 [4] Finalizing run hash...")
    run_finalize()

    print("\n📁 [5] Packaging submission ZIP...")
    run_package()

    print("\n📈 [6] Running core diagnostics...")
    run_diagnostics()

    print("\n📊 [7] Scoring submission.csv vs ground_truth.csv...")
    L, score = score_submission("submission.csv", "ground_truth.csv")

    print("\n🔬 [8] Checking σ calibration metrics...")
    check_calibration("submission.csv", "ground_truth.csv")

    print("\n🧪 [9] Generating uncertainty scatter diagnostics...")
    generate_uncertainty_report("submission.csv", "ground_truth.csv")

    print("\n📄 [10] Writing interactive HTML dashboard...")
    generate_html(score=score, L=L)

    print("\n📊 [11] Plotting μ quantile bands...")
    df = pd.read_csv("submission.csv")
    mu_cols = [f"mu_{i}" for i in range(283)]
    mu_tensor = torch.tensor(df[mu_cols].values, dtype=torch.float32)
    bands = compute_quantile_bands(mu_tensor)

    gt = pd.read_csv("ground_truth.csv")
    y_true = gt[mu_cols].mean(axis=0).values

    plot_quantiles(bands, y_true)

    print("\n📊 [12] Plotting quantiles vs ground truth...")
    plot_quantiles_vs_target("submission.csv", "ground_truth.csv")

    print("\n✅ [✓] All integration tests completed successfully.")


if __name__ == "__main__":
    run_integration()