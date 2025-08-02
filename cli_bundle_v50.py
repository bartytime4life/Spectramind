"""
SpectraMind V50 – Orchestration CLI (Full)
-------------------------------------------
Automates end-to-end challenge pipeline:
• Train → Infer → COREL → Validate → Bundle
• Logs config hash, generates dashboard HTML
• Prepares full leaderboard-ready submission ZIP
"""

import typer
import yaml
import os
import json
import torch
from pathlib import Path
from datetime import datetime

from train_v50 import train_from_config
from predict_v50 import run as run_inference
from submission_validator_v50 import validate_submission
from generate_diagnostic_summary import generate_diagnostic_summary
from generate_html_report import generate_html_report
from generate_submission_package import generate_zip
from corel_inference import load_corel_model, apply_corel


app = typer.Typer(help="SpectraMind V50 – Full Challenge Pipeline CLI")


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


@app.command()
def make_submission(
    config: Path = typer.Option("configs/config_v50.yaml"),
    conformalize: bool = typer.Option(True),
    bundle: bool = typer.Option(True),
    validate: bool = typer.Option(True),
    diagnostics: bool = typer.Option(True),
    tag: str = typer.Option("spectramind_v50")
):
    """Run full pipeline: train, infer, calibrate, diagnose, bundle"""

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
        typer.echo(f"\n🩺 Generating diagnostic overlays...")
        mu = torch.load("outputs/mu.pt").numpy()
        sigma = torch.load("outputs/sigma.pt").numpy()
        y = torch.load("outputs/y.pt").numpy()
        generate_diagnostic_summary(mu, sigma, y)
        generate_html_report()

    if bundle:
        typer.echo(f"\n📦 Creating submission bundle...")
        generate_zip()

    typer.echo(f"\n🔁 Logging hash...")
    log_hash(cfg, tag=tag)

    typer.echo(f"\n✅ Submission ready! Use ZIP in leaderboard upload.")


@app.command()
def full_run(config: Path = typer.Option("configs/config_v50.yaml")):
    """Alias for make-submission with defaults"""
    make_submission(config=config)


if __name__ == "__main__":
    app()