"""
SpectraMind V50 – Flexible Submission Format Checker
----------------------------------------------------
Supports both 283-bin and 567-bin formats.
Checks:
- Column count (1 + N + N)
- Column names
- NaNs or negative σ
- Unique planet_id
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def check_submission_format(file_path: str) -> bool:
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Could not load submission: {e}")
        return False

    log = Path("v50_debug_log.md")
    with open(log, "a") as f:
        f.write(f"\n### Submission Format Check: {file_path}\n")

    if df.shape[1] < 10:
        print("❌ Too few columns. Are you using the correct submission format?")
        return False

    if df.columns[0] != "planet_id":
        print(f"❌ First column must be 'planet_id', found: {df.columns[0]}")
        return False

    n_bins = (df.shape[1] - 1) // 2
    mu_cols = [f"mu_{i}" for i in range(n_bins)]
    sigma_cols = [f"sigma_{i}" for i in range(n_bins)]

    expected_cols = ["planet_id"] + mu_cols + sigma_cols
    if list(df.columns) != expected_cols:
        print("❌ Columns are not in expected format.")
        missing = [c for c in expected_cols if c not in df.columns]
        print(f"Missing example columns: {missing[:5]}")
        return False

    if df["planet_id"].duplicated().any():
        print("❌ Duplicate planet_id entries found")
        return False

    if df.isnull().any().any():
        print("❌ Submission contains NaN values")
        return False

    if (df[sigma_cols] < 0).any().any():
        print("❌ Submission contains negative σ values")
        return False

    print(f"✅ Format valid for {n_bins} spectral bins")
    with open(log, "a") as f:
        f.write(f"- Detected {n_bins} bins\n")
        f.write("- Format valid ✅\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Ariel submission CSV format")
    parser.add_argument("--file", required=True, help="Path to submission CSV file")
    args = parser.parse_args()

    valid = check_submission_format(args.file)
    sys.exit(0 if valid else 1)
