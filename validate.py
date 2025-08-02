"""
SpectraMind V50 – Submission Validator (Ultimate)
-------------------------------------------------
Checks that submission.csv conforms to challenge requirements:
- 567 columns: 1 planet_id + 283 mu + 283 sigma
- Column headers match required format
- No NaNs, Infs, or negative sigmas
- planet_id is unique and valid type
- Logs validation to v50_debug_log.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime


def validate_submission(path: str = "submission.csv", return_summary: bool = False):
    log_path = Path("v50_debug_log.md")
    timestamp = datetime.utcnow().isoformat()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load submission: {e}")

    with open(log_path, "a") as log:
        log.write(f"\n### ✅ Submission Validation Started [{timestamp}]\n")
        log.write(f"- File: {path}\n")

    if df.shape[1] != 567:
        raise ValueError(f"❌ Expected 567 columns, found {df.shape[1]}")

    expected_cols = ["planet_id"] + [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)]
    actual_cols = list(df.columns)

    if actual_cols != expected_cols:
        mismatches = [i for i, (a, b) in enumerate(zip(actual_cols, expected_cols)) if a != b]
        mismatch_sample = [f"Col {i}: expected '{expected_cols[i]}', found '{actual_cols[i]}'" for i in mismatches[:5]]
        msg = "❌ Column headers do not match required format.\n" + "\n".join(mismatch_sample)
        raise ValueError(msg)

    if df["planet_id"].duplicated().any():
        raise ValueError("❌ Duplicate planet_id entries detected")

    if not np.issubdtype(df["planet_id"].dtype, np.integer) and not np.issubdtype(df["planet_id"].dtype, np.object_):
        raise ValueError("❌ planet_id must be string or integer type")

    mu = df[[f"mu_{i}" for i in range(283)]].values
    sigma = df[[f"sigma_{i}" for i in range(283)]].values

    if not np.all(np.isfinite(mu)):
        raise ValueError("❌ Non-finite values found in μ (NaN or Inf)")

    if not np.all(np.isfinite(sigma)):
        raise ValueError("❌ Non-finite values found in σ (NaN or Inf)")

    if np.any(sigma <= 0):
        raise ValueError("❌ σ values must be strictly positive (found ≤ 0)")

    mu_mean = float(mu.mean())
    mu_std = float(mu.std())
    sigma_mean = float(sigma.mean())
    sigma_std = float(sigma.std())

    with open(log_path, "a") as log:
        log.write(f"- Shape: {df.shape}\n")
        log.write(f"- μ mean ± std: {mu_mean:.3f} ± {mu_std:.3f}\n")
        log.write(f"- σ mean ± std: {sigma_mean:.3f} ± {sigma_std:.3f}\n")
        log.write("✅ submission.csv passed all validation checks\n")

    print("✅ submission.csv format is valid")

    if return_summary:
        return {
            "file": path,
            "shape": df.shape,
            "mu_mean": mu_mean,
            "mu_std": mu_std,
            "sigma_mean": sigma_mean,
            "sigma_std": sigma_std,
            "num_rows": len(df),
            "timestamp": timestamp
        }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_submission(sys.argv[1])
    else:
        validate_submission()