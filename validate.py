"""
SpectraMind V50 – Submission Validator
--------------------------------------
Checks that submission.csv conforms to challenge requirements:
- 567 columns: 1 planet_id + 283 mu + 283 sigma
- Column headers match format
- No NaNs, Infs, or negative sigmas
- planet_id is unique
- Logs to v50_debug_log.md
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def validate_submission(path: str = "submission.csv"):
    log_path = Path("v50_debug_log.md")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load submission: {e}")

    with open(log_path, "a") as log:
        log.write(f"\n### Validating Submission: {path}\n")

    if df.shape[1] != 567:
        raise ValueError(f"❌ Expected 567 columns, found {df.shape[1]}")

    expected_cols = ["planet_id"] + [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)]
    if list(df.columns) != expected_cols:
        raise ValueError("❌ Column headers do not match required format")

    if df["planet_id"].duplicated().any():
        raise ValueError("❌ Duplicate planet_id entries detected")

    mu = df[[f"mu_{i}" for i in range(283)]].values
    sigma = df[[f"sigma_{i}" for i in range(283)]].values

    if not np.all(np.isfinite(mu)):
        raise ValueError("❌ Non-finite values found in μ")

    if not np.all(np.isfinite(sigma)):
        raise ValueError("❌ Non-finite values found in σ")

    if np.any(sigma <= 0):
        raise ValueError("❌ σ must be strictly positive")

    with open(log_path, "a") as log:
        log.write(f"- Shape: {df.shape}\n")
        log.write(f"- μ mean ± std: {mu.mean():.3f} ± {mu.std():.3f}\n")
        log.write(f"- σ mean ± std: {sigma.mean():.3f} ± {sigma.std():.3f}\n")
        log.write("✅ Submission format valid\n")

    print("✅ submission.csv format is valid")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_submission(sys.argv[1])
    else:
        validate_submission()
