"""
SpectraMind V50 – Submission Diff Viewer
----------------------------------------
Compares two submission.csv files and highlights bin-wise μ and σ differences
for one or more overlapping planets. Saves diagnostic plots and logs summary statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from typing import Optional, List

def compare_submissions(sub1: str, sub2: str, outdir: str = "diagnostics/diff", planet_idx: Optional[int] = None):
    sub1, sub2 = Path(sub1), Path(sub2)
    os.makedirs(outdir, exist_ok=True)

    df1 = pd.read_csv(sub1)
    df2 = pd.read_csv(sub2)

    if "planet_id" not in df1.columns or "planet_id" not in df2.columns:
        raise ValueError("❌ Missing 'planet_id' column in one or both submissions.")

    if df1.shape[1] != 567 or df2.shape[1] != 567:
        raise ValueError("❌ Submission must contain exactly 567 columns.")

    planet_ids = sorted(set(df1["planet_id"]).intersection(df2["planet_id"]))

    if not planet_ids:
        raise ValueError("❌ No common planet_ids found between submissions.")

    # If index is given, restrict to single planet
    if planet_idx is not None:
        if planet_idx >= len(planet_ids):
            raise IndexError(f"❌ Index {planet_idx} out of bounds (only {len(planet_ids)} shared planet_ids).")
        planet_ids = [planet_ids[planet_idx]]

    mu_cols = [f"mu_{i}" for i in range(283)]
    sig_cols = [f"sigma_{i}" for i in range(283)]

    log_path = Path("v50_debug_log.md")
    with open(log_path, "a") as log:
        log.write("\n## 📊 Submission Comparison Summary\n")

    for pid in planet_ids:
        row1 = df1[df1["planet_id"] == pid]
        row2 = df2[df2["planet_id"] == pid]

        if row1.empty or row2.empty:
            print(f"⚠️ Skipping planet_id {pid} (not found in both)")
            continue

        mu1 = row1[mu_cols].values[0]
        mu2 = row2[mu_cols].values[0]
        sig1 = row1[sig_cols].values[0]
        sig2 = row2[sig_cols].values[0]

        dmu = mu1 - mu2
        dsig = sig1 - sig2

        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(dmu, label="Δμ (mu diff)", color="blue", linewidth=1.5)
        plt.plot(dsig, label="Δσ (sigma diff)", color="orange", linewidth=1.5)
        plt.title(f"Submission Difference: {pid}")
        plt.xlabel("Spectral Bin")
        plt.ylabel("Δ Value (ppm)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        safe_pid = str(pid).replace("/", "_").replace(":", "_")
        outpath = os.path.join(outdir, f"sub_diff_{safe_pid}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()

        print(f"✅ Saved diff plot for {pid} to {outpath}")

        # Log
        with open(log_path, "a") as log:
            log.write(f"\n### {pid}\n")
            log.write(f"- Δμ mean ± std: {dmu.mean():.4f} ± {dmu.std():.4f}\n")
            log.write(f"- Δσ mean ± std: {dsig.mean():.4f} ± {dsig.std():.4f}\n")
            log.write(f"- Plot: `{outpath}`\n")

if __name__ == "__main__":
    # CLI mode: python submission_diff_viewer.py file1.csv file2.csv [planet_idx]
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: python submission_diff_viewer.py sub1.csv sub2.csv [planet_idx]")
    else:
        sub1 = args[0]
        sub2 = args[1]
        planet_idx = int(args[2]) if len(args) > 2 else None
        compare_submissions(sub1, sub2, planet_idx=planet_idx)
