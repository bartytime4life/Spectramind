"""
SpectraMind V50 – Submission CLI (Final)
----------------------------------------
Orchestrates full training → inference → validation → diagnostics → packaging.
Includes:
- Self-test enforcement
- CLI call logging
- Config hash tracking
"""

import typer
import yaml
import os
import json
import subprocess
import torch
import sys
from pathlib import Path
from datetime import datetime

from train_v50 import train_from_config
from predict_v50 import run as run_inference
from submission_validator_v50 import validate_submission
from generate_diagnostic_summary import generate_diagnostic_summary
from generate_html_report import generate_html_report
from generate_submission_package import generate_zip
from corel_inference import load_corel_model, apply_corel

app = typer.Typer(help="SpectraMind V50 – Full Pipeline CLI")

# --- Logging setup ---
__VERSION__ = "v50.1.0"
__HASH_FILE__ = Path("run_hash_summary_v50.json")
__LOG_FILE__ = Path("v50_debug_log.md")

def get_latest_config_hash():
    if __HASH_FILE__.exists():
        with open(__HASH_FILE__) as f:
            data = json.load(f)
            if data:
                last_tag = list(data.keys())[-1]
                return data[last_tag].get("hash", "unknown")
    return "unknown"

def log_cli_call():
    cmd = " ".join(sys.argv)
    hash_val = get_latest_config_hash()
    now = datetime.utcnow().isoformat()
    entry = f"\n### CLI Call @ {now}\n- Command: `{cmd}`\n- Version: {__VERSION__}\n- Config Hash: {hash_val}\n"
    if __LOG_FILE__.exists():
        __LOG_FILE__.write_text(__LOG_FILE__.read_text() + entry)
    else:
        __LOG_FILE__.write_text(entry)

@app.callback()
def log():
    log_cli_call()

# --- Pipeline logic ---

def log_hash(cfg, tag="default"):
    import hashlib
    hash_str = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    summary_path = Path("run_hash_summary_v50.json")
    record = {
        "run_tag": tag,
        "hash": hash_str,
        "timestamp": datetime.utcnow().isoformat()
    }
    if summary_path.exists():
        data = json.load(open(summary_path))
    else:
        data = {}
    data[tag] = record
    json.dump(data, open(summary_path, "w"), indent=2)
    return hash_str


def run_selftest():
    typer.echo("🛡️ Running system selftest...")
    result = subprocess.run("python selftest.py run --mode deep", shell=True)
    if result.returncode != 0:
        typer.secho("❌ Selftest failed. Aborting submission pipeline.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho("✅ Selftest passed.\n", fg=typer.colors.GREEN)


@app.command()
def make_submission(
    config: Path = typer.Option("configs/config_v50.yaml"),
    conformalize: bool = typer.Option(True),
    bundle: bool = typer.Option(True),
    validate: bool = typer.Option(True),
    diagnostics: bool = typer.Option(True),
    tag: str = typer.Option("spectramind_v50")
):
    """Run full pipeline: selftest → train → infer → COREL → validate → diagnose → zip"""
    run_selftest()

    cfg = yaml.safe_load(open(config))
    typer.echo(f"\n🚀 Training model...")
    train_from_config(cfg, tag=tag)

    typer.echo(f"\n🧠 Running inference...")
    run_inference()

    if conformalize:
        typer.echo("\n🔐 Applying COREL conformalization...")
        mu = torch.load("outputs/mu.pt")
        sigma = torch.load("outputs/sigma.pt")
        edge_index = torch.load("calibration_data/edge_index.pt")
        model = load_corel_model("models/corel_gnn.pt")
        mu_corr, sigma_corr = apply_corel(model, mu, sigma, edge_index)
        torch.save(mu_corr, "outputs/mu.pt")
        torch.save(sigma_corr, "outputs/sigma.pt")
        typer.echo("✅ COREL applied.")

    if validate:
        typer.echo(f"\n📏 Validating submission...")
        validate_submission("outputs/submission.csv")

    if diagnostics:
        typer.echo(f"\n🩺 Generating diagnostics...")
        mu = torch.load("outputs/mu.pt").numpy()
        sigma = torch.load("outputs/sigma.pt").numpy()
        y = torch.load("outputs/y.pt").numpy()
        generate_diagnostic_summary(mu, sigma, y)
        generate_html_report()

    if bundle:
        typer.echo(f"\n📦 Creating submission bundle...")
        generate_zip()

    typer.echo(f"\n🔁 Logging config hash...")
    log_hash(cfg, tag=tag)

    typer.secho(f"\n✅ SpectraMind V50 submission pipeline complete.", fg=typer.colors.GREEN)


@app.command()
def full_run(config: Path = typer.Option("configs/config_v50.yaml")):
    """Alias: run full pipeline"""
    make_submission(config=config)


if __name__ == "__main__":
    app()