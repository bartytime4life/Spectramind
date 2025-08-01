"""
SpectraMind V50 – Submission Validator
--------------------------------------
Checks that submission.csv conforms to competition format:
- 567 columns: 1 (planet_id) + 283 (mu) + 283 (sigma)
- Column headers match format exactly
- No NaNs, Infs, or negative sigmas
- Unique planet_ids only
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def validate_submission(path: str = "submission.csv", log_output: bool = True):
    log_path = Path("v50_debug_log.md")
    df_path = Path(path)

    if not df_path.exists():
        raise FileNotFoundError(f"❌ File not found: {path}")

    try:
        df = pd.read_csv(df_path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load submission: {e}")

    if log_output:
        with open(log_path, "a") as log:
            log.write(f"\n## 🔍 Validating Submission File: `{path}`\n")

    # Shape
    if df.shape[1] != 567:
        raise ValueError(f"❌ Expected 567 columns, found {df.shape[1]}")

    # Header
    expected_cols = ["planet_id"] + [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)]
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        mismatch = [i for i, (a, b) in enumerate(zip(actual_cols, expected_cols)) if a != b]
        msg = f"❌ Column headers do not match required format at indices: {mismatch[:5]}"
        raise ValueError(msg)

    # planet_id
    if df["planet_id"].duplicated().any():
        raise ValueError("❌ Duplicate planet_id entries detected")

    if not pd.api.types.is_string_dtype(df["planet_id"]):
        raise TypeError("❌ planet_id column must be string type")

    # μ and σ values
    mu = df[[f"mu_{i}" for i in range(283)]].values
    sigma = df[[f"sigma_{i}" for i in range(283)]].values

    if not np.all(np.isfinite(mu)):
        raise ValueError("❌ Non-finite values (NaN/Inf) found in μ")

    if not np.all(np.isfinite(sigma)):
        raise ValueError("❌ Non-finite values (NaN/Inf) found in σ")

    if np.any(sigma <= 0):
        raise ValueError("❌ σ must be strictly positive")

    # Summary log
    if log_output:
        with open(log_path, "a") as log:
            log.write(f"- ✅ Columns: OK (567)\n")
            log.write(f"- ✅ Header: OK\n")
            log.write(f"- ✅ planet_id: OK\n")
            log.write(f"- ✅ μ: finite, mean = {mu.mean():.3f} ± {mu.std():.3f}\n")
            log.write(f"- ✅ σ: finite & > 0, mean = {sigma.mean():.3f} ± {sigma.std():.3f}\n")
            log.write("✅ Submission format passed\n")

    print("✅ Submission format is valid")


if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    validate_submission(file_arg)
