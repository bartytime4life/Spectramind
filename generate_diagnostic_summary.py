"""
SpectraMind V50 – Diagnostic Summary Generator
---------------------------------------------
Aggregates key QA diagnostics:
- Binwise GLL
- Symbolic violation overlays
- SHAP + entropy + violation fusion
- Quantile band constraints
- JSON summary logs for CI + dashboards
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

from gll_error_localizer import compute_binwise_gll, plot_gll_heatmap
from constraint_violation_overlay import plot_violation_overlay
from spectral_event_miner import mine_anomalous_bins
from quantile_violation_chart import plot_quantile_vs_constraints
from generate_quantile_bands import compute_quantile_bands

def generate_diagnostic_summary(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    shap: Optional[np.ndarray] = None,
    entropy: Optional[np.ndarray] = None,
    violations: Optional[np.ndarray] = None,
    outdir: str = "diagnostics",
    save_json: bool = True
):
    """
    Runs full-spectrum QA diagnostics and generates overlay visualizations + logs.

    Args:
        mu (np.ndarray): Predicted mean spectra (N, 283)
        sigma (np.ndarray): Predicted uncertainty spectra (N, 283)
        y (np.ndarray): Ground truth spectra (N, 283)
        shap (np.ndarray, optional): SHAP values (N, 283)
        entropy (np.ndarray, optional): Entropy (N, 283)
        violations (np.ndarray, optional): Violation matrix (N, 283)
        outdir (str): Output directory for diagnostics
        save_json (bool): Whether to write logs to JSON
    """
    os.makedirs(outdir, exist_ok=True)
    summary_log = {}

    # 1. Binwise GLL diagnostics
    print("🔍 Computing binwise GLL...")
    gll_bins = compute_binwise_gll(y, mu, sigma)
    plot_gll_heatmap(gll_bins, outdir)
    summary_log["gll_per_bin"] = gll_bins.tolist()

    # 2. Symbolic violation overlay
    if violations is not None:
        print("📏 Rendering violation overlay...")
        mu_avg = mu.mean(axis=0)
        viol_avg = violations.mean(axis=0)
        plot_violation_overlay(mu_avg, viol_avg, outdir)
        summary_log["violation_avg"] = viol_avg.tolist()

        if save_json:
            with open(os.path.join(outdir, "constraint_violation_log.json"), "w") as f:
                json.dump(violations.tolist(), f, indent=2)

    # 3. SHAP + entropy + symbolic fusion event miner
    if shap is not None and entropy is not None and violations is not None:
        print("⚠️ Mining anomalous bins...")
        events = mine_anomalous_bins({
            "shap": shap.mean(axis=0),
            "entropy": entropy.mean(axis=0),
            "violations": violations.mean(axis=0),
            "gll": gll_bins
        })
        summary_log["anomalous_bins"] = events
        print(f"🚨 Anomalous bins identified: {events}")
        if save_json:
            with open(os.path.join(outdir, "anomalous_bins.json"), "w") as f:
                json.dump({"bins": events}, f, indent=2)

    # 4. Quantile band constraint check
    print("📐 Computing quantile bands...")
    bands = compute_quantile_bands(torch.tensor(mu))
    summary_log["quantiles"] = {k: v.tolist() for k, v in bands.items()}

    if violations is not None:
        mask = (violations.mean(axis=0) > 0.1).astype(float)
        plot_quantile_vs_constraints(bands["q25"], bands["q75"], mask, outdir)

    # 5. Summary log
    if save_json:
        with open(os.path.join(outdir, "diagnostic_summary.json"), "w") as f:
            json.dump(summary_log, f, indent=2)
        print(f"✅ diagnostic_summary.json written to {outdir}")