"""
symbolic_rule_scorer.py – SHAP × Symbolic Rule Attribution
----------------------------------------------------------
Scores symbolic rules based on alignment between SHAP attribution and violation patterns.
Useful for understanding which symbolic rules affect model predictions most strongly.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def score_symbolic_rules(
    shap_path="outputs/shap_values.npy",
    violations_path="outputs/constraint_violation_log.json",
    output_csv="outputs/symbolic_rule_scores.csv",
    summary_csv="outputs/symbolic_rule_rankings.csv"
):
    # Load SHAP values (B, 283)
    shap_vals = np.load(shap_path)  # shape: (B, 283)
    B, D = shap_vals.shape

    # Load symbolic violation log
    with open(violations_path) as f:
        violations = json.load(f)  # dict: rule::planet_id → [bins]

    # Determine unique rules
    rule_ids = sorted(set(k.split("::")[0] for k in violations.keys()))
    rule_scores = {}

    # Initialize score matrix (B, rule_id)
    for rid in rule_ids:
        rule_mask = np.zeros_like(shap_vals)

        for k, bins in violations.items():
            if not k.startswith(rid):
                continue
            try:
                idx = int(k.split("::")[1])
                for b in bins:
                    rule_mask[idx, int(b)] = 1
            except:
                continue

        # Multiply SHAP × violation mask
        influence = np.abs(shap_vals) * rule_mask
        rule_scores[rid] = influence.mean(axis=1)  # (B,) mean SHAP value on violated bins

    # Create DataFrame
    df = pd.DataFrame(rule_scores)
    df.loc["mean"] = df.mean()

    # Save detailed sample-level scores
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv)
    print(f"✅ Saved sample-level symbolic rule scores: {output_csv}")

    # Save global mean ranking
    ranking = df.loc["mean"].sort_values(ascending=False).reset_index()
    ranking.columns = ["rule_id", "mean_alignment_score"]
    ranking.to_csv(summary_csv, index=False)
    print(f"📊 Saved symbolic rule ranking: {summary_csv}")


if __name__ == "__main__":
    score_symbolic_rules()