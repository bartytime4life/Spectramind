"""
SpectraMind V50 – Diagnostic CLI
--------------------------------
Run scientific QA tools: GLL maps, symbolic overlays, UMAP latents, HTML dashboards.
"""

import typer
import subprocess
from pathlib import Path

app = typer.Typer(help="SpectraMind V50 – Diagnostics Toolkit")


@app.command("summary")
def run_diagnostic_summary(
    mu_path: str = "outputs/mu.pt",
    sigma_path: str = "outputs/sigma.pt",
    y_path: str = "outputs/y.pt",
    shap_path: str = None,
    entropy_path: str = None,
    violations_path: str = None
):
    import torch
    import numpy as np
    from generate_diagnostic_summary import generate_diagnostic_summary

    mu = torch.load(mu_path).numpy()
    sigma = torch.load(sigma_path).numpy()
    y = torch.load(y_path).numpy()
    shap = torch.load(shap_path).numpy() if shap_path else None
    entropy = torch.load(entropy_path).numpy() if entropy_path else None
    violations = torch.load(violations_path).numpy() if violations_path else None

    generate_diagnostic_summary(mu, sigma, y, shap, entropy, violations)


@app.command("umap-latents")
def umap_latent_plot(
    config: str = typer.Option("configs/config_v50.yaml"),
    checkpoint: str = typer.Option("outputs/model.pt"),
    tag: str = typer.Option("v50"),
    out_png: str = typer.Option("diagnostics/umap_latents.png"),
    out_html: str = typer.Option("diagnostics/umap_latents.html"),
    batch_size: int = typer.Option(64),
    n_neighbors: int = typer.Option(30),
    min_dist: float = typer.Option(0.1)
):
    """
    UMAP latent visualization from V50 encoder.
    """
    cmd = [
        "python", "plot_umap_v50.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--tag", tag,
        "--out_png", out_png,
        "--out_html", out_html,
        "--batch_size", str(batch_size),
        "--n_neighbors", str(n_neighbors),
        "--min_dist", str(min_dist)
    ]
    print("🚀 Running latent UMAP...")
    subprocess.run(cmd, check=True)


@app.command("validate-submission")
def validate_sub(
    submission: str = "submission.csv",
    gll_eval: bool = True
):
    from submission_validator_v50 import validate_submission
    validate_submission(submission, gll_eval=gll_eval)


@app.command("score-gll")
def score_gll(
    labels: str = "data/train.csv",
    preds: str = "submission.csv",
    json: str = "diagnostics/gll_score_submission.json",
    tag: str = "submission"
):
    from evaluate_gll_v50 import evaluate_and_log_gll
    evaluate_and_log_gll(labels, preds, json_log_path=json, tag=tag)


@app.command("dashboard")
def full_dashboard():
    """
    Builds a full diagnostics report (UMAP + GLL + violations + summary).
    """
    print("📊 Running diagnostic summary...")
    run_diagnostic_summary()

    print("📐 Plotting latent UMAP...")
    umap_latent_plot()

    print("🌐 Generating HTML dashboard...")
    from generate_html_report import generate_html_report
    generate_html_report()


if __name__ == "__main__":
    app()