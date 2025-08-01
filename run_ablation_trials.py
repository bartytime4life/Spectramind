"""
SpectraMind V50 – Run Ablation Trials (Hybrid)
----------------------------------------------
Combines YAML-based structured sweeps (Typer CLI) with dynamic JSON-based config sampling.
Supports full config sweeps, param randomization, and CLI-triggered execution.
"""

import os
import json
import random
import shutil
import subprocess
from pathlib import Path
import typer
import yaml

app = typer.Typer(help="Run ablation experiments: sweep YAML configs or randomize JSON trials")

# --- STRUCTURED YAML SWEEP MODE ---

ABLATION_CONFIGS = {
    "decoder": [
        "configs/ablation/decoder_moe.yaml",
        "configs/ablation/decoder_flow.yaml",
        "configs/ablation/decoder_quantile.yaml"
    ],
    "symbolic": [
        "configs/ablation/symbolic_none.yaml",
        "configs/ablation/symbolic_smooth.yaml",
        "configs/ablation/symbolic_full.yaml"
    ]
}

@app.command()
def sweep(category: str = typer.Argument(..., help="decoder | symbolic")):
    configs = ABLATION_CONFIGS.get(category)
    if not configs:
        typer.echo(f"❌ Unknown ablation category: {category}")
        raise typer.Exit()

    for config in configs:
        typer.echo(f"\n🔁 Running ablation: {config}")
        subprocess.run(["python", "cli_v50.py", "train", "--config", config])

@app.command()
def single(config_path: Path):
    typer.echo(f"🔬 Running single ablation config: {config_path}")
    subprocess.run(["python", "cli_v50.py", "train", "--config", str(config_path)])

# --- JSON-RANDOMIZED TRIAL MODE ---

def randomize_config(base_config, seed):
    cfg = json.loads(json.dumps(base_config))  # deep copy
    random.seed(seed)
    cfg["training"]["loss_weights"]["smooth"] = round(random.uniform(0.0, 0.3), 3)
    cfg["training"]["loss_weights"]["nonneg"] = round(random.uniform(0.0, 0.2), 3)
    cfg["training"]["loss_weights"]["entropy"] = round(random.uniform(0.0, 0.2), 3)
    cfg["training"]["loss_weights"]["pinball"] = round(random.uniform(0.0, 0.3), 3)
    return cfg

def run_trial(config_base_path, trial_id, out_dir):
    with open(config_base_path, "r") as f:
        base_config = json.load(f)

    trial_cfg = randomize_config(base_config, seed=trial_id)
    trial_cfg["paths"]["model_save_dir"] = os.path.join(out_dir, f"trial_{trial_id}")
    os.makedirs(trial_cfg["paths"]["model_save_dir"], exist_ok=True)

    trial_config_path = os.path.join(trial_cfg["paths"]["model_save_dir"], f"config_trial_{trial_id}.json")
    with open(trial_config_path, "w") as f:
        json.dump(trial_cfg, f, indent=2)

    print(f"🚀 Running ablation trial #{trial_id}...")
    try:
        subprocess.run(["python", "train_v50.py", "--config", trial_config_path], check=True)
        result_path = os.path.join(trial_cfg["paths"]["model_save_dir"], "fold_metrics.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                metrics = json.load(f)
            gll = metrics.get("fold_0", {}).get("best_val_loss", -1)
        else:
            gll = -1
        return trial_config_path, gll
    except Exception as e:
        print(f"❌ Trial {trial_id} failed: {e}")
        return trial_config_path, -1

@app.command()
def random_trials(
    config_base_path: str = "config_v33.json",
    out_dir: str = "ablation",
    n_trials: int = 5
):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "ablation_log.txt")
    all_results = []

    for trial_id in range(n_trials):
        config_path, gll = run_trial(config_base_path, trial_id, out_dir)
        all_results.append((config_path, gll))
        with open(log_path, "a") as f:
            f.write(f"{config_path}\nGLL Score: {gll:.6f}\n\n")

    print("✅ All ablation trials completed. Summary written to ablation_log.txt")

if __name__ == "__main__":
    app()
