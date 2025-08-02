"""
SpectraMind V50 – Quantile Violation Chart (Ultimate Version)
-------------------------------------------------------------
Compares μ quantile bands (Q25, Q75) against symbolic constraint masks.
Integrates with:
- generate_diagnostic_summary.py (auto chart export)
- symbolic_loss.py (mask bin derivation)
- generate_quantile_bands.py (inputs)
- generate_html_report.py (dashboard embedding)
- shap_program_extractor.py (to score band rule overlaps)
- symbolic_violation_predictor.py (to visualize mask zones)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional


def plot_quantile_vs_constraints(
    q25: np.ndarray,
    q75: np.ndarray,
    symbolic_mask: np.ndarray,
    outdir: str = "diagnostics/quantile",
    save_name: str = "quantile_violation_chart.png",
    summary_path: Optional[str] = None
):
    """
    Args:
        q25, q75: (283,) arrays for 25th and 75th percentile of μ
        symbolic_mask: (283,) binary or float array where constraints must hold
        outdir: output directory
        save_name: name of output plot
        summary_path: optional path to save list of violated bins
    """
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(10, 3))
    x = np.arange(len(q25))
    plt.plot(q25, label="μ 25th percentile", color="blue")
    plt.plot(q75, label="μ 75th percentile", color="cyan")
    plt.fill_between(x, q25, q75, alpha=0.3, color="lightblue")

    mask_bool = symbolic_mask > 0.5
    plt.fill_between(x, 0, 1, where=mask_bool, color="red", alpha=0.2,
                     transform=plt.gca().get_xaxis_transform(), label="Symbolic mask")

    plt.title("μ Quantile Range vs Symbolic Constraint Mask")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Predicted μ Range")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(outdir, save_name)
    plt.savefig(out_path)
    print(f"✅ Saved quantile violation chart: {out_path}")

    if summary_path:
        violated_bins = list(np.where(mask_bool)[0])
        with open(summary_path, "w") as f:
            f.write(json.dumps({"violated_bins": violated_bins}, indent=2))
        print(f"📝 Saved violation bin list: {summary_path}")


if __name__ == "__main__":
    q25 = np.random.rand(283) * 100
    q75 = q25 + np.random.rand(283) * 20
    mask = (np.random.rand(283) > 0.85).astype(float)
    plot_quantile_vs_constraints(q25, q75, mask)