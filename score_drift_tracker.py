"""
SpectraMind V50 – Score Drift Tracker (Ultimate Version)
--------------------------------------------------------
Tracks GLL loss and final score across submission versions.
Provides:
- JSON history log
- PNG drift plot
- CSV + Markdown table for CI + dashboard
- Anomaly detection for score regression
- Integration with submit CLI + diagnostics report
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def log_score(
    version: str,
    L: float,
    score: float,
    log_path: str = "outputs/logs/score_history.json",
    write_debug_log: bool = True
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {"version": version, "L": L, "score": score}

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"📊 Logged score: {version} → score={score:.4f}, L={L:.2f}")

    if write_debug_log:
        with open("v50_debug_log.md", "a") as f:
            f.write(f"\n### Submission Version: {version}\n")
            f.write(f"- Score: {score:.4f}\n")
            f.write(f"- GLL Loss: {L:.2f}\n")


def plot_score_drift(
    log_path: str = "outputs/logs/score_history.json",
    outdir: str = "diagnostics",
    save_csv: bool = True,
    save_md: bool = True,
    detect_regression: bool = True
):
    with open(log_path, 'r') as f:
        history = json.load(f)

    versions = [h["version"] for h in history]
    scores = [h["score"] for h in history]
    L_vals = [h["L"] for h in history]

    df = pd.DataFrame({"version": versions, "score": scores, "L": L_vals})
    os.makedirs(outdir, exist_ok=True)

    # Save CSV and Markdown
    if save_csv:
        df.to_csv(os.path.join(outdir, "score_history.csv"), index=False)
    if save_md:
        md_path = os.path.join(outdir, "score_history.md")
        with open(md_path, "w") as f:
            f.write("| Version | Score | L_total |\n")
            f.write("|---------|--------|---------|\n")
            for row in history:
                f.write(f"| {row['version']} | {row['score']:.4f} | {row['L']:.2f} |\n")

    # Detect anomaly: drop in score or spike in loss
    regress = []
    if detect_regression and len(scores) > 1:
        for i in range(1, len(scores)):
            if scores[i] < scores[i-1] or L_vals[i] > L_vals[i-1]:
                regress.append((versions[i], scores[i]))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(versions, scores, marker="o", color="green", label="Score")
    ax.plot(versions, L_vals, marker="x", color="red", label="GLL Loss")

    if regress:
        for r in regress:
            ax.axvline(x=r[0], color="orange", linestyle="--", alpha=0.3)
            ax.text(r[0], min(scores), "↓", ha="center", va="bottom", fontsize=12, color="orange")

    ax.set_title("Score & GLL Drift across Submissions")
    ax.set_xlabel("Version")
    ax.set_ylabel("Metric")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(outdir, "score_drift.png")
    fig.savefig(plot_path)

    print(f"✅ Drift plot saved: {plot_path}")
    print(f"📄 CSV: {outdir}/score_history.csv")
    print(f"📝 Markdown: {outdir}/score_history.md")

    if regress:
        print("⚠️  Regression detected in:", ", ".join([r[0] for r in regress]))


if __name__ == "__main__":
    # Example logging + plot generation
    log_score("v50r1", 11425.02, 0.7198)
    plot_score_drift()