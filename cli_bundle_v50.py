"""
SpectraMind V50 – Submission Orchestration & Bundling CLI
----------------------------------------------------------
Typer-based CLI to automate:
• Training → inference → validation → submission.zip
• Symbolic + diagnostic artifact bundling
• Hash logging, manifest zipping, and HTML overlay capture

Includes: `make-submission`, `bundle-submission`, `full-run`
"""

import typer
import yaml
import os
import json
from datetime import datetime
from pathlib import Path
from generate_submission_package import generate_zip
from submission_validator_v50 import validate_submission
from train_v50 import train_from_config
from predict_v50 import run as run_inference
from corel_inference import load_corel_model, apply_corel
import torch

app = typer.Typer(help="SpectraMind V50 – CLI for bundling, submission, and automation")

@app.command()
def bundle_submission(
    config: Path = typer.Option("configs/config_v50.yaml"),
    symbolic_only: bool = typer.Option(False, help="Only include symbolic & reproducibility files"),
    finalize_only: bool = typer.Option(False, help="Only finalize pipeline outputs, skip diagnostics"),
    diagnostics_only: bool = typer.Option(False, help="Run diagnostics but skip finalizer")
):
    """Bundle submission.zip from outputs, manifest, logs, diagnostics"""
    generate_zip(
        symbolic_only=symbolic_only,
        finalize_only=finalize_only,
        diagnostics_only=diagnostics_only
    )

@app.command()
def make_submission(
    config: Path = typer.Option("configs/config_v50.yaml"),
    conformalize: bool = typer.Option(True, help="Apply COREL conformal bounds to μ/σ"),
    bundle: bool = typer.Option(True, help="Bundle final ZIP after run"),
    validate: bool = typer.Option(True, help="Validate submission.csv contents")
):
    """Train → inference → conformalize → validate → package"""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    typer.echo("\n🚀 Training model...")
    train_from_config(cfg)

    typer.echo("\n🧠 Running inference...")
    run_inference()

    if conformalize:
        typer.echo("\n🔐 Applying COREL conformalization...")
        mu = torch.load("outputs/mu.pt")
        sigma = torch.load("outputs/sigma.pt")
        edge_index = torch.load("calibration_data/edge_index.pt")
        model = load_corel_model("models/corel_gnn.pt")
        mu_corr, sigma_corr = apply_corel(model, mu, sigma, edge_index)
        torch.save(mu_corr, "outputs/mu_corel.pt")
        torch.save(sigma_corr, "outputs/sigma_corel.pt")
        typer.echo("✅ COREL complete")

    if validate:
        typer.echo("\n📏 Validating submission.csv...")
        validate_submission("submission.csv")

    if bundle:
        typer.echo("\n📦 Bundling outputs...")
        generate_zip()

    typer.echo("\n✅ Done!")

@app.command()
def full_run(
    config: Path = typer.Option("configs/config_v50.yaml"),
):
    """Alias: run entire pipeline and generate ZIP"""
    make_submission(config=config, conformalize=True, bundle=True, validate=True)

if __name__ == "__main__":
    app()
