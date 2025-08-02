"""
SpectraMind V50 – Diagnostic CLI
--------------------------------
Run all QA tools: UMAP latents, symbolic overlays, GLL scoring, submission validation,
and render a full HTML diagnostics dashboard.
"""

import typer
import subprocess
from pathlib import Path
import re
import webbrowser
from generate_html_report import generate_html_report
from submission_validator_v50 import validate_submission
from evaluate_gll_v50 import evaluate_and_log_gll
from generate_diagnostic_summary import generate_diagnostic_summary

app = typer.Typer(help="SpectraMind V50 – Diagnostics CLI")


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
    overlay_csv: str = typer.Option(None, help="Optional .csv with planet_id and label"),
    overlay_column: str = typer.Option("symbolic_class", help="Label column to color by"),
    batch_size: int = 64,
    n_neighbors: int = 30,
    min_dist: float = 0.1
):
    """
    Generate UMAP plot from V50 latent space. Overlay symbolic/cluster labels if provided.
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
    if overlay_csv:
        cmd += ["--overlay_csv", overlay_csv, "--overlay_column", overlay_column]

    print("📐 Running UMAP latent visualizer...")
    subprocess.run(cmd, check=True)


@app.command("validate-submission")
def validate_sub(
    submission: str = "submission.csv",
    gll_eval: bool = True
):
    """
    Validate submission format and optionally compute GLL score.
    """
    validate_submission(submission, gll_eval=gll_eval)


@app.command("score-gll")
def score_gll(
    labels: str = "data/train.csv",
    preds: str = "submission.csv",
    json: str = "diagnostics/gll_score_submission.json",
    tag: str = "submission"
):
    """
    Compute GLL score between predictions and ground truth.
    """
    evaluate_and_log_gll(labels, preds, json_log_path=json, tag=tag)


@app.command("dashboard")
def full_dashboard(
    open_browser: bool = typer.Option(True, help="Open HTML in browser after generation"),
    versioned: bool = typer.Option(True, help="Use versioned filenames like diagnostic_report_v2.html")
):
    """
    Build and render full diagnostics dashboard: UMAP + GLL + violations + HTML report.
    """
    print("📊 Running symbolic/GLL diagnostics...")
    run_diagnostic_summary()

    print("📐 Generating UMAP latent plots...")
    umap_latent_plot()

    # Versioned HTML file
    diagnostics_dir = Path("diagnostics")
    if versioned:
        existing = list(diagnostics_dir.glob("diagnostic_report_v*.html"))
        nums = [int(re.search(r"v(\d+)", f.name).group(1)) for f in existing if re.search(r"v(\d+)", f.name)]
        next_v = max(nums) + 1 if nums else 1
        out_path = diagnostics_dir / f"diagnostic_report_v{next_v}.html"
    else:
        out_path = diagnostics_dir / "diagnostic_report.html"

    print(f"🌐 Generating HTML dashboard: {out_path}")
    generate_html_report(out_path=out_path)

    if open_browser:
        print("🧭 Opening dashboard in browser...")
        webbrowser.open(out_path.resolve().as_uri())

    print(f"✅ Dashboard saved: {out_path}")


if __name__ == "__main__":
    app()