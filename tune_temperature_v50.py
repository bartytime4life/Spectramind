"""
SpectraMind V50 – Temperature Scaling for Sigma Calibration
-----------------------------------------------------------
Calibrates predictive uncertainty σ using a scalar temperature T to optimize GLL.
"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

from validate_dataset_v50 import validate_dataset
from gll_score_v50 import calculate_gll_score


def gll_loss(y, mu, sigma):
    eps = 1e-8
    sigma = np.clip(sigma, eps, None)
    return np.mean(((y - mu) / sigma)**2 + 2 * np.log(sigma))


def run_temperature_tuning(config_path="configs/config_v50.yaml", labels_path=None, dry_run=False):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    label_file = labels_path or paths.get("metadata_file", "data/train.csv")
    submission_path = paths.get("submission_csv", "submission.csv")
    submission_dir = os.path.dirname(submission_path)

    if dry_run:
        print(f"[DRY RUN] Would calibrate σ using config: {config_path}")
        return

    print(f"📋 Validating dataset using: {label_file}")
    validate_dataset(paths["train_data_dir"], label_file, skip_label=False)

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission not found: {submission_path}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Labels not found: {label_file}")

    label_df = pd.read_csv(label_file).set_index("planet_id")
    pred_df = pd.read_csv(submission_path).set_index("planet_id")

    pred_df = pred_df.loc[label_df.index]  # Align rows

    mu_cols = [f"mu_{i}" for i in range(283)]
    sigma_cols = [f"sigma_{i}" for i in range(283)]

    mu = pred_df[mu_cols].values
    sigma = pred_df[sigma_cols].values
    y_true = label_df[mu_cols].values

    def loss_fn(log_temp):
        temp = np.exp(log_temp)
        return gll_loss(y_true, mu, sigma * temp)

    result = minimize(loss_fn, x0=0.0, method="L-BFGS-B")
    best_temp = float(np.exp(result.x[0]))
    best_loss = float(result.fun)

    print(f"\n✅ Optimal temperature: {best_temp:.4f}")
    print(f"📉 Calibrated GLL loss: {best_loss:.6f}")

    calibrated_sigma = sigma * best_temp
    out_df = pd.concat([
        pred_df.reset_index()[["planet_id"]],
        pd.DataFrame(mu, columns=mu_cols),
        pd.DataFrame(calibrated_sigma, columns=sigma_cols)
    ], axis=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuned_path = os.path.join(submission_dir, f"submission_v50_tuned_{timestamp}.csv")
    out_df.to_csv(tuned_path, index=False)
    print(f"💾 Tuned submission saved: {tuned_path}")

    gll_score = calculate_gll_score(label_df.reset_index(), out_df)
    print(f"📊 GLL Score on calibrated submission: {gll_score:.6f}")

    with open(os.path.join(submission_dir, "temp_scalar.txt"), "w") as f:
        f.write(f"{best_temp:.6f}\n")

    with open(os.path.join(submission_dir, "tuning_log.txt"), "a") as log:
        log.write(f"{timestamp} | config={config_path} | T={best_temp:.6f} | GLL={gll_score:.6f}\n")

    # Diagnostic plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(sigma.flatten(), bins=100, alpha=0.6, label="Original", color="orange")
    plt.hist(calibrated_sigma.flatten(), bins=100, alpha=0.6, label="Tuned", color="blue")
    plt.title("Sigma Distribution (All Bins)")
    plt.xlabel("σ")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 2, 2)
    delta = calibrated_sigma.flatten() - sigma.flatten()
    plt.hist(delta, bins=100, alpha=0.75, color="green")
    plt.title("Δσ (Tuned - Original)")
    plt.xlabel("Δσ")
    plt.tight_layout()

    plot_path = os.path.join(submission_dir, f"sigma_calibration_plot_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"🖼️ Saved diagnostic plot: {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Temperature scaling for σ calibration (SpectraMind V50)")
    parser.add_argument("--config", type=str, default="configs/config_v50.yaml", help="YAML config file")
    parser.add_argument("--labels", type=str, default=None, help="Optional path to labels CSV")
    parser.add_argument("--dry_run", action="store_true", help="Show actions without running")
    args = parser.parse_args()

    run_temperature_tuning(args.config, args.labels, dry_run=args.dry_run)
