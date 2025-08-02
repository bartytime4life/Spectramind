"""
SpectraMind V50 – Submission Validator (Final)
----------------------------------------------
Checks that submission.csv conforms to competition specs:
- 567 columns: 1 (planet_id) + 283 (mu) + 283 (sigma)
- Column headers must match expected order
- No NaNs, Infs, or zero/negative σ
- Unique planet_ids only
- ✅ Auto-evaluates GLL score vs train.csv
- ✅ Logs to v50_debug_log.md
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from evaluate_gll_v50 import evaluate_and_log_gll  # must be present in project

def validate_submission(path: str = "submission.csv", log_output: bool = True, gll_eval: bool = True):
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

    # Header check
    expected_cols = ["planet_id"] + [f"mu_{i}" for i in range(283)] + [f"sigma_{i}" for i in range(283)]
    if list(df.columns) != expected_cols:
        bad_cols = [i for i, (a, b) in enumerate(zip(df.columns, expected_cols)) if a != b]
        raise ValueError(f"❌ Column headers do not match at indices: {bad_cols[:5]}")

    # planet_id checks
    if df["planet_id"].duplicated().any():
        raise ValueError("❌ Duplicate planet_id entries detected")

    if not pd.api.types.is_string_dtype(df["planet_id"]):
        raise TypeError("❌ planet_id column must be string type")

    # μ and σ
    mu = df[[f"mu_{i}" for i in range(283)]].values
    sigma = df[[f"sigma_{i}" for i in range(283)]].values

    if not np.all(np.isfinite(mu)):
        raise ValueError("❌ Non-finite values found in μ")

    if not np.all(np.isfinite(sigma)):
        raise ValueError("❌ Non-finite values found in σ")

    if np.any(sigma <= 0):
        raise ValueError("❌ σ must be strictly positive (>0)")

    # Format summary log
    if log_output:
        with open(log_path, "a") as log:
            log.write(f"- ✅ Columns: OK ({df.shape[1]})\n")
            log.write(f"- ✅ Header: OK\n")
            log.write(f"- ✅ planet_id: OK (unique, string)\n")
            log.write(f"- ✅ μ: finite | mean = {mu.mean():.3f} ± {mu.std():.3f}\n")
            log.write(f"- ✅ σ: > 0 | mean = {sigma.mean():.3f} ± {sigma.std():.3f}\n")

    # Auto GLL eval
    if gll_eval:
        try:
            print("📏 Evaluating GLL score against training set...")
            gll_score = evaluate_and_log_gll(
                labels_path="data/train.csv",
                preds_path=path,
                json_log_path="diagnostics/gll_score_submission.json",
                tag="submission"
            )
            if log_output:
                with open(log_path, "a") as log:
                    log.write(f"- ✅ GLL score (vs train): {gll_score:.6f}\n")
        except Exception as e:
            print(f"⚠️ GLL evaluation skipped due to error: {e}")
            if log_output:
                with open(log_path, "a") as log:
                    log.write(f"- ⚠️ GLL evaluation failed: {str(e)}\n")

    print("✅ Submission format is valid")

    if log_output:
        with open(log_path, "a") as log:
            log.write("✅ Submission validation passed\n")

if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    validate_submission(file_arg)