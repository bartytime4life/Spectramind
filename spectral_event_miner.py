"""
SpectraMind V50 – Spectral Event Miner (Ultimate Version)
----------------------------------------------------------
Detects anomalous spectral bins by fusing SHAP, entropy, symbolic violations, and GLL.
Integrates with:
- generate_diagnostic_summary.py (auto event scan)
- generate_html_report.py (event overlay table)
- shap_overlay.py (SHAP source)
- shap_entropy_overlay.py (entropy fusion)
- symbolic_violation_predictor.py (violation input)
- gll_error_localizer.py (binwise GLL)
- anomaly_feedback_trainer.py (retraining bin generator)
"""

import numpy as np
from typing import Dict, List
import json
import os

def mine_anomalous_bins(
    inputs: Dict[str, np.ndarray],
    shap_thresh: float = 0.2,
    ent_thresh: float = 1.0,
    viol_thresh: float = 0.1,
    gll_thresh: float = 5.0,
    fusion_save_path: str = "diagnostics/events/anomalous_bin_mask.npy",
    summary_path: str = "diagnostics/events/anomalous_bins.json"
) -> List[int]:
    """
    Detects bin indices flagged by all anomaly metrics.

    Args:
        inputs: dict of 'shap', 'entropy', 'violations', 'gll'
        *_thresh: thresholds for anomaly detection
        fusion_save_path: optional .npy file to save binary anomaly mask
        summary_path: optional .json file to list anomaly bins

    Returns:
        List of anomaly bin indices
    """
    os.makedirs(os.path.dirname(fusion_save_path), exist_ok=True)

    shap_flags = np.abs(inputs["shap"]) > shap_thresh
    entropy_flags = inputs["entropy"] > ent_thresh
    violation_flags = inputs["violations"] > viol_thresh
    gll_flags = inputs["gll"] > gll_thresh

    combined = shap_flags & entropy_flags & violation_flags & gll_flags
    anomaly_bins = np.where(combined)[0]

    np.save(fusion_save_path, combined.astype(np.uint8))
    with open(summary_path, "w") as f:
        json.dump({"anomalous_bins": anomaly_bins.tolist()}, f, indent=2)

    print(f"✅ Saved anomaly mask: {fusion_save_path}")
    print(f"📝 Saved anomalous bin list: {summary_path}")

    return list(anomaly_bins)


if __name__ == "__main__":
    dummy_inputs = {
        "shap": (np.random.rand(283) - 0.5) * 0.5,
        "entropy": np.random.rand(283) * 2,
        "violations": np.random.rand(283),
        "gll": np.random.rand(283) * 10
    }
    events = mine_anomalous_bins(dummy_inputs)
    print(f"🔍 Found {len(events)} anomalous bins: {events[:10]} ...")