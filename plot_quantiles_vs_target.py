"""
SpectraMind V50 – Quantile Overlay Plotter (Ultimate Version)
--------------------------------------------------------------
Visualizes predicted μ quantile bands against ground truth μ spectrum.
Highlights out-of-band violations and logs coverage stats.
Integrates with:
- generate_diagnostic_summary.py (optional overlay plot)
- generate_html_report.py (interactive summary panel)
- symbolic_violation_predictor.py (visual coverage against logic zones)
- auto_ablate_v50.py (helps target uncertain or underfit bins)
- shap_overlay.py (combined with influence maps)
- diagnostic overlay suite (via CLI or notebook)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional
import json

def plot_quantiles(
    mu_bands: Dict[str, np.ndarray],
    y_true: np.ndarray,
    outdir: str = "diagnostics/quantile_overlay",
    save_name: str = "quantile_overlay.png",
    return_fig: bool = False,
    export_json: Optional[str] = None
):
    """
    Args:
        mu_bands: dict with keys q10, q25, q50, q75, q90
        y_true: (283,) ground truth μ
        outdir: directory to save figure
        save_name: PNG filename
        return_fig: if True, return matplotlib figure
        export_json: if set, writes violation stats and bands to .json
    """
    assert all(k in mu_bands for k in ["q10", "q25", "q50", "q75", "q90"]), "Missing quantiles"
    assert y_true.shape[0] == mu_bands["q50"].shape[0], "Shape mismatch"

    os.makedirs(outdir, exist_ok=True)
    x = np.arange(len(y_true))
    q10, q25, q50, q75, q90 = mu_bands["q10"], mu_bands["q25"], mu_bands["q50"], mu_bands["q75"], mu_bands["q90"]

    outside = (y_true < q10) | (y_true > q90)
    num_violations = outside.sum()
    percent_violated = 100.0 * num_violations / len(y_true)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, q10, q90, color="lightblue", alpha=0.35, label="q10–q90")
    ax.fill_between(x, q25, q75, color="cornflowerblue", alpha=0.3, label="q25–q75")
    ax.plot(x, q50, color="navy", lw=2, label="Predicted Median μ (q50)")
    ax.plot(x, y_true, color="black", lw=1.5, linestyle="--", label="Ground Truth")

    if num_violations > 0:
        ax.fill_between(x, q90.max() * 1.01, q90.max() * 1.03,
                        where=outside, step="mid", color="red", alpha=0.6, label="Outside q10–q90")

    ax.set_xlim([0, len(y_true) - 1])
    ax.set_ylim([min(q10.min(), y_true.min()) - 10, max(q90.max(), y_true.max()) + 10])
    ax.set_xlabel("Spectral Bin")
    ax.set_ylabel("Transit Depth (μ, ppm)")
    ax.set_title(f"μ Quantile Bands vs Ground Truth  |  Violated: {num_violations}/{len(y_true)} ({percent_violated:.1f}%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize="small")

    save_path = os.path.join(outdir, save_name)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved quantile overlay plot: {save_path}")

    if export_json:
        summary = {
            "violated_bins": list(np.where(outside)[0]),
            "percent_violated": percent_violated,
            "bands": {k: v.tolist() for k, v in mu_bands.items()},
            "y_true": y_true.tolist()
        }
        with open(export_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📝 Saved quantile coverage report to {export_json}")

    return fig if return_fig else None


if __name__ == "__main__":
    dummy_bands = {
        "q10": np.random.rand(283) * 100,
        "q25": np.random.rand(283) * 100,
        "q50": np.random.rand(283) * 100,
        "q75": np.random.rand(283) * 100,
        "q90": np.random.rand(283) * 100
    }
    y_true = np.random.rand(283) * 100
    plot_quantiles(dummy_bands, y_true)