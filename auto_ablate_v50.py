"""
SpectraMind V50 – Full Scientific Ablation Suite
--------------------------------------------------
Executes a configurable set of model + symbolic config ablations.
Tracks metrics, hashes, debug logs, and plots comparative graphs
for GLL, symbolic_weight, and other loss components.
"""

import os
import json
import shutil
import hashlib
import matplotlib.pyplot as plt
import argparse
from train_v50 import train_from_config

ABLATION_CONFIGS = [
    {
        "name": "moe_full",
        "model": {"decoder_type": "moe"},
        "symbolic": {"smoothness": True, "nonnegativity": True, "variance_shaping": True},
        "training": {"lr": 1e-4, "max_epochs": 10}
    },
    {
        "name": "diffusion_no_symbolic",
        "model": {"decoder_type": "diffusion"},
        "symbolic": {},
        "training": {"lr": 1e-4, "max_epochs": 10}
    },
    {
        "name": "quantile_partial",
        "model": {"decoder_type": "quantile"},
        "symbolic": {"smoothness": True},
        "training": {"lr": 1e-4, "max_epochs": 10}
    }
]

DATA_REF = "configs/base_data_config.json"
OUTDIR = "outputs/ablations"


def hash_config(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def plot_metric_comparison(summary, metric="loss", outpath="outputs/ablations/ablation_plot.png"):
    plt.figure(figsize=(10, 5))
    names = list(summary.keys())
    values = [summary[name].get(metric, 0.0) for name in names]

    plt.bar(names, values, color="dodgerblue")
    plt.title(f"Ablation Comparison – {metric}")
    plt.ylabel(metric.replace("_", " "))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"📊 Saved ablation bar plot to {outpath}")

def plot_all_metrics(summary, metrics):
    for m in metrics:
        plot_metric_comparison(summary, metric=m, outpath=f"outputs/ablations/{m}_comparison.png")

def run_ablation_suite(include=None, exclude=None):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(DATA_REF) as f:
        base_data = json.load(f)

    summary = {}
    selected = [cfg for cfg in ABLATION_CONFIGS if (not include or cfg["name"] in include) and (not exclude or cfg["name"] not in exclude)]

    for cfg in selected:
        name = cfg["name"]
        cfg_full = {"data": base_data, **cfg}
        config_hash = hash_config(cfg_full)

        print(f"\n🚀 Running ablation: {name}")
        train_from_config(cfg_full)

        metrics_path = "outputs/training/training_metrics.json"
        debug_log_path = "v50_debug_log.md"

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            final = metrics[-1]
            final["hash"] = config_hash
            summary[name] = final

            shutil.copy(metrics_path, os.path.join(OUTDIR, f"{name}_metrics.json"))
            if os.path.exists(debug_log_path):
                shutil.copy(debug_log_path, os.path.join(OUTDIR, f"{name}_debug_log.md"))

    summary_path = os.path.join(OUTDIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n🌍 All ablations complete. Summary written to", summary_path)
    plot_all_metrics(summary, metrics=["loss", "symbolic_weight", "smoothness", "variance"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="*", help="Limit to specific configs")
    parser.add_argument("--exclude", nargs="*", help="Exclude certain configs")
    args = parser.parse_args()

    run_ablation_suite(include=args.include, exclude=args.exclude)