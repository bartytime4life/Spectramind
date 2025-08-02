"""
SpectraMind V50 – SHAP-Conditioned Rule Miner (Ultimate Version)
----------------------------------------------------------------
Learns symbolic rule triggers by identifying consistent SHAP bin activations.
Integrates with:
- symbolic_logic_engine.py (rule injection)
- symbolic_program_hypotheses.py (rule testing)
- shap_overlay.py (input SHAP arrays)
- generate_diagnostic_summary.py (auto rule mining)
- anomaly_feedback_trainer.py (retrains based on rule span violations)
- auto_symbolic_rule_miner.py (adds to rule mining pool)
- generate_html_report.py (shows mined rules in diagnostics)
"""

import numpy as np
from typing import List, Dict
import os
import json

def mine_rule_candidates(
    shap_values: List[np.ndarray],
    threshold: float = 0.25,
    min_support: float = 0.6,
    min_len: int = 3,
    rule_prefix: str = "shap_rule"
) -> List[Dict]:
    """
    Args:
        shap_values: list of (283,) SHAP arrays
        threshold: SHAP threshold to consider a bin "activated"
        min_support: fraction of samples required to activate a bin
        min_len: minimum span length to qualify as rule
        rule_prefix: name prefix

    Returns:
        List of rule candidates: [{"bins": [...], "direction": "positive", "rule_name": ...}, ...]
    """
    shap_stack = np.stack(shap_values)
    bin_counts = (np.abs(shap_stack) > threshold).sum(axis=0)
    support_thresh = int(len(shap_values) * min_support)
    active_mask = bin_counts >= support_thresh

    rules = []
    current = []
    for i, active in enumerate(active_mask):
        if active:
            current.append(i)
        elif current:
            if len(current) >= min_len:
                rules.append({
                    "bins": current.copy(),
                    "direction": "positive",
                    "rule_name": f"{rule_prefix}_{current[0]}_{current[-1]}"
                })
            current.clear()
    if current and len(current) >= min_len:
        rules.append({
            "bins": current.copy(),
            "direction": "positive",
            "rule_name": f"{rule_prefix}_{current[0]}_{current[-1]}"
        })

    return rules


def save_rule_candidates(rules: List[Dict], out_path: str = "outputs/mined_rules/shap_conditioned_rules.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"✅ Saved SHAP-conditioned rules: {out_path}")


if __name__ == "__main__":
    fake_shaps = [(np.random.rand(283) - 0.5) * 2 for _ in range(10)]
    rules = mine_rule_candidates(fake_shaps)
    save_rule_candidates(rules)