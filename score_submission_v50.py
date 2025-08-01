"""
SpectraMind V50 – Scientific Submission Scorer
----------------------------------------------
Computes Gaussian log-likelihood (GLL), symbolic submission score,
and logs results against L_ref and L_ideal to v50_debug_log.md.
Optionally exports JSON with score breakdown.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime

def gaussian_log_likelihood(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    var = sigma**2 + 1e-6
    log_prob = np.log(sigma + 1e-6) + (y - mu)**2 / (2 * var)
    return float(np.sum(log_prob))

def score_submission(sub_csv: str, gt_csv: str, L_ref: float = 19055.98, L_ideal: float = 6842.11, out_json: str = None):
    sub = pd.read_csv(sub_csv)
    gt = pd.read_csv(gt_csv)

    if not all(sub["planet_id"] == gt["planet_id"]):
        raise ValueError("❌ Planet ID mismatch between submission and ground truth")

    mu_cols = [f"mu_{i}" for i in range(283)]
    sigma_cols = [f"sigma_{i}" for i in range(283)]

    mu = sub[mu_cols].values
    sigma = sub[sigma_cols].values
    y = gt[mu_cols].values

    L = gaussian_log_likelihood(y, mu, sigma)
    score = (L_ref - L) / (L_ref - L_ideal)
    score = max(score, 0.0)

    summary = {
        "GLL_total": round(L, 3),
        "score": round(score, 5),
        "L_ref": L_ref,
        "L_ideal": L_ideal,
        "submission_rows": int(len(sub)),
        "timestamp": datetime.utcnow().isoformat()
    }

    print(f"✅ GLL: {L:.2f}")
    print(f"✅ Score: {score:.5f} (ideal={L_ideal}, ref={L_ref})")

    if out_json:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📃 Score written to {out_json}")

    with open("v50_debug_log.md", "a") as log:
        log.write(f"\n### Score Summary\n")
        for k, v in summary.items():
            log.write(f"- {k}: {v}\n")

    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", default="submission.csv")
    parser.add_argument("--gt", default="ground_truth.csv")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument("--lref", type=float, default=19055.98)
    parser.add_argument("--lideal", type=float, default=6842.11)
    args = parser.parse_args()

    score_submission(args.sub, args.gt, L_ref=args.lref, L_ideal=args.lideal, out_json=args.out)
