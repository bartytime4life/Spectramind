"""
SpectraMind V50 – Diagnostic CLI
--------------------------------
Run symbolic overlays, UMAP/t-SNE projections, scoring checks,
and full diagnostics dashboard generation with CLI logging.
"""

import typer
import subprocess
from pathlib import Path
import re
import webbrowser
import json
import sys
from datetime import datetime

from submission_validator_v50 import validate_submission
from evaluate_gll_v50 import evaluate_and_log_gll
from generate_diagnostic_summary import generate_diagnostic_summary
from generate_html_report import generate_html_report

app = typer.Typer(help="SpectraMind V50 – Diagnostics CLI")

# --- CLI Logging Setup ---
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

# ------------------------------- Commands -------------------------------

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
    config: str = "configs/config_v50.yaml",
    checkpoint: str = "outputs/model.pt",
    tag: str = "v50",
    out_png: str = "diagnostics/umap_latents.png",
    out_html: str = "diagnostics/umap_latents.html",
    overlay_csv: str = None,
    overlay_column: str = "symbolic_class",
    batch_size: int = 64,
    n_neighbors: int = 30,
    min_dist: float = 0.1
):
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
    subprocess.run(cmd, check=True)


@app.command("tsne-latents")
def tsne_latent_plot(
    config: str = "configs/config_v50.yaml",
    checkpoint: str = "outputs/model.pt",
    html_out: str = "diagnostics/tsne_latents.html",
    overlay_csv: str = None,
    overlay_column: str = "symbolic_class"
):
    cmd = [
        "python", "plot_tsne_interactive.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--html_out", html_out,
        "--overlay_column", overlay_column
    ]
    if overlay_csv:
        cmd += ["--overlay_csv", overlay_csv]
    subprocess.run(cmd, check=True)


@app.command("validate-submission")
def validate_sub(
    submission: str = "submission.csv",
    gll_eval: bool = True
):
    validate_submission(submission, gll_eval=gll_eval)


@app.command("score-gll")
def score_gll(
    labels: str = "data/train.csv",
    preds: str = "submission.csv",
    json: str = "diagnostics/gll_score_submission.json",
    tag: str = "submission"
):
    evaluate_and_log_gll(labels, preds, json_log_path=json, tag=tag)


@app.command("dashboard")
def full_dashboard(
    open_browser: bool = typer.Option(True),
    versioned: bool = typer.Option(True),
    overlay_csv: str = typer.Option(None),
    overlay_column: str = typer.Option("symbolic_class"),
    no_umap: bool = typer.Option(False, help="Skip UMAP projection"),
    no_tsne: bool = typer.Option(False, help="Skip t-SNE projection")
):
    print("📊 Running summary diagnostics...")
    run_diagnostic_summary()

    if not no_umap:
        print("📐 Plotting UMAP...")
        umap_latent_plot(
            overlay_csv=overlay_csv,
            overlay_column=overlay_column
        )

    if not no_tsne:
        print("📐 Plotting t-SNE...")
        tsne_latent_plot(
            overlay_csv=overlay_csv,
            overlay_column=overlay_column
        )

    diagnostics_dir = Path("diagnostics")
    if versioned:
        existing = list(diagnostics_dir.glob("diagnostic_report_v*.html"))
        nums = [int(re.search(r"v(\d+)", f.name).group(1)) for f in existing if re.search(r"v(\d+)", f.name)]
        next_v = max(nums) + 1 if nums else 1
        out_path = diagnostics_dir / f"diagnostic_report_v{next_v}.html"
    else:
        out_path = diagnostics_dir / "diagnostic_report.html"

    print(f"🌐 Generating HTML dashboard to {out_path}")
    generate_html_report(out_path=out_path)

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "umap_included": not no_umap,
        "tsne_included": not no_tsne,
        "html_file": str(out_path),
        "overlay_column": overlay_column,
        "overlay_csv": overlay_csv or "None"
    }
    with open(diagnostics_dir / "diagnostics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Dashboard ready: {out_path}")
    print("📝 Summary written to: diagnostics/diagnostics_report.json")


if __name__ == "__main__":
    app()