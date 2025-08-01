"""
SpectraMind V50 – Full Scientific Ablation Suite
------------------------------------------------
Executes a configurable set of model + symbolic config ablations.
Tracks metrics, hashes, debug logs, and plots comparative graphs
for GLL, symbolic_weight, smoothness, variance, etc.
"""

import os
import json
import shutil
import hashlib
import argparse
import matplotlib.pyplot as plt
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
DEFAULT_OUTDIR = "outputs/ablations"

def hash_config(cfg: dict) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def plot_metric_comparison(summary, metric="loss", outdir="outputs/ablations"):
    plt.figure(figsize=(10, 5))
    names = list(summary.keys())
    values = [summary[name].get(metric, None) for name in names]
    valid = [v for v in values if v is not None]

    if not valid:
        print(f"⚠️ No valid values found for metric: {metric}")
        return

    plt.bar(names, values, color="skyblue")
    plt.title(f"Ablation Comparison – {metric}")
    plt.ylabel(metric.replace("_", " "))
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_path = os.path.join(outdir, f"{metric}_comparison.png")
    plt.savefig(save_path)
    print(f"📊 Saved plot: {save_path}")

def plot_all_metrics(summary, metrics, outdir):
    for m in metrics:
        plot_metric_comparison(summary, metric=m, outdir=outdir)

def run_ablation_suite(include=None, exclude=None, outdir=DEFAULT_OUTDIR, metrics=None):
    os.makedirs(outdir, exist_ok=True)
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

        # Gather logs
        metrics_path = "outputs/training/training_metrics.json"
        debug_log = "v50_debug_log.md"
        violation_log = "constraint_violation_log.json"

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                mlog = json.load(f)
            final = mlog[-1] if isinstance(mlog, list) else mlog
            final["config_hash"] = config_hash
            summary[name] = final

            shutil.copy(metrics_path, os.path.join(outdir, f"{name}_metrics.json"))
            if os.path.exists(debug_log):
                shutil.copy(debug_log, os.path.join(outdir, f"{name}_debug_log.md"))
            if os.path.exists(violation_log):
                shutil.copy(violation_log, os.path.join(outdir, f"{name}_violation_log.json"))
        else:
            print(f"⚠️ Metrics missing for {name} — skipping.")

    summary_path = os.path.join(outdir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ All ablations complete. Summary saved to: {summary_path}")

    # Determine metrics to plot
    all_keys = set(k for s in summary.values() for k in s.keys())
    metrics_to_plot = metrics or sorted(k for k in all_keys if isinstance(summary[next(iter(summary))].get(k, None), (float, int)))
    plot_all_metrics(summary, metrics_to_plot, outdir=outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="*", help="Only run selected config names")
    parser.add_argument("--exclude", nargs="*", help="Exclude specific config names")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--metrics", nargs="*", help="Metrics to plot (optional override)")
    args = parser.parse_args()

    run_ablation_suite(include=args.include, exclude=args.exclude, outdir=args.outdir, metrics=args.metrics)