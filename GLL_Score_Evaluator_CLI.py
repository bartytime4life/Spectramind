"""
SpectraMind V50 – GLL Score Evaluator CLI
-----------------------------------------
Evaluates predicted μ and σ against ground truth using weighted Gaussian log-likelihood (GLL).
Includes symbolic-compliant weighting, JSON log export, and reproducibility features.

✅ FGS/AIRS bin weighting
✅ NumPy stable σ clipping
✅ Optional JSON log export for CI/dashboard use
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- Spectral Constants ---
TOTAL_BINS = 567
FGS_BINS = 1
AIRS_BINS = TOTAL_BINS - FGS_BINS
FGS_WEIGHT = 0.4
AIRS_WEIGHT = 0.6 / AIRS_BINS
SPECTRAL_WEIGHTS = np.array([FGS_WEIGHT] + [AIRS_WEIGHT] * AIRS_BINS, dtype=np.float32)

# --- GLL Computation ---
def compute_gll_score(y_true: np.ndarray, mu_pred: np.ndarray, sigma_pred: np.ndarray) -> float:
    sigma_pred = np.clip(sigma_pred, 1e-8, 1e6)
    squared_term = ((y_true - mu_pred) / sigma_pred) ** 2
    log_term = 2 * np.log(sigma_pred) + np.log(2 * np.pi)
    gll = squared_term + log_term
    weighted_gll = gll * SPECTRAL_WEIGHTS
    return float(np.mean(np.sum(weighted_gll, axis=1)))

def extract_arrays(labels_df: pd.DataFrame, preds_df: pd.DataFrame):
    labels_df = labels_df.set_index("planet_id")
    preds_df = preds_df.set_index("planet_id")

    if not labels_df.index.equals(preds_df.index):
        missing = set(labels_df.index) - set(preds_df.index)
        raise ValueError(f"Mismatch in planet IDs. Missing predictions for: {list(missing)[:5]}")

    mu_cols = [f"mu_{i}" for i in range(TOTAL_BINS)]
    sigma_cols = [f"sigma_{i}" for i in range(TOTAL_BINS)]

    y_true = labels_df[mu_cols].values.astype(np.float32)
    mu_pred = preds_df[mu_cols].values.astype(np.float32)
    sigma_pred = preds_df[sigma_cols].values.astype(np.float32)

    return y_true, mu_pred, sigma_pred

def evaluate_and_log_gll(
    labels_path: str,
    preds_path: str,
    json_log_path: str = None,
    tag: str = None
) -> float:
    """
    Evaluates GLL score and optionally writes to a JSON log file.

    Args:
        labels_path: Path to CSV with ground truth
        preds_path: Path to CSV with model predictions
        json_log_path: Optional path to output JSON
        tag: Optional run ID or model name for tracking

    Returns:
        float GLL score
    """
    labels_df = pd.read_csv(labels_path)
    preds_df = pd.read_csv(preds_path)

    y_true, mu_pred, sigma_pred = extract_arrays(labels_df, preds_df)
    score = compute_gll_score(y_true, mu_pred, sigma_pred)

    print(f"✅ V50 GLL Score: {score:.6f}")

    if json_log_path:
        Path(json_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_log_path, "w") as f:
            json.dump({
                "score_name": "GLL",
                "score_value": score,
                "num_planets": len(labels_df),
                "bins": TOTAL_BINS,
                "tag": tag or "submission"
            }, f, indent=2)
        print(f"📄 GLL score log written to: {json_log_path}")

    return score

# --- CLI Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🧠 SpectraMind V50 – GLL Score Evaluator")
    parser.add_argument("--labels", type=str, required=True, help="Path to ground truth labels CSV")
    parser.add_argument("--preds", type=str, required=True, help="Path to predictions CSV")
    parser.add_argument("--json", type=str, help="Optional path to write JSON log")
    parser.add_argument("--tag", type=str, help="Optional tag for run hash, experiment name, etc.")

    args = parser.parse_args()
    evaluate_and_log_gll(args.labels, args.preds, json_log_path=args.json, tag=args.tag)