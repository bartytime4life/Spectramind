"""
SpectraMind V50 – SHAP Program Extractor (Ultimate Version)
-----------------------------------------------------------
Translates SHAP patterns into symbolic rule programs for downstream logic.
Integrates with:
- symbolic_logic_engine.py (runtime symbolic execution)
- symbolic_rule_scorer.py (rule quality validation)
- generate_diagnostic_summary.py (auto rule discovery)
- anomaly_feedback_trainer.py (retraining triggers)
- plot_violation_overlay.py (visual rule alignment)
- auto_symbolic_rule_miner.py (bootstraps symbolic set)
- shap_overlay.py (generates SHAP used here)
- shap_entropy_overlay.py (fusion-based rule bin filtering)
"""

import numpy as np
from typing import List, Dict
import os
import json


def extract_symbolic_programs(
    shap_array: List[np.ndarray],
    threshold: float = 0.3,
    min_len: int = 3,
    rule_prefix: str = "shap_band"
) -> List[Dict]:
    """
    Args:
        shap_array: list of (283,) SHAP arrays
        threshold: abs(SHAP) minimum to consider high impact
        min_len: minimum bin length to form a symbolic rule
        rule_prefix: base name for rules

    Returns:
        List of symbolic programs: [{"rule_name": str, "bins": [...], "type": "positive"}, ...]
    """
    stack = np.stack(shap_array)
    avg = np.mean(np.abs(stack), axis=0)
    high = avg > threshold

    programs = []
    current = []
    for i, active in enumerate(high):
        if active:
            current.append(i)
        elif current:
            if len(current) >= min_len:
                programs.append({
                    "rule_name": f"{rule_prefix}_{current[0]}_{current[-1]}",
                    "bins": current.copy(),
                    "type": "positive"
                })
            current = []
    if current and len(current) >= min_len:
        programs.append({
            "rule_name": f"{rule_prefix}_{current[0]}_{current[-1]}",
            "bins": current.copy(),
            "type": "positive"
        })

    return programs


def save_programs(programs: List[Dict], out_path: str = "outputs/shap_programs.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(programs, f, indent=2)
    print(f"✅ Saved symbolic programs: {out_path}")


if __name__ == "__main__":
    fake_shaps = [(np.random.rand(283) - 0.5) * 2 for _ in range(5)]
    rules = extract_symbolic_programs(fake_shaps)
    save_programs(rules)