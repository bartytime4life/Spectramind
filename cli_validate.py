"""
SpectraMind V50 – Validation CLI
-------------------------------
Validates dataset structure, submission format, FFT integrity, and symbolic alignment.
"""

import typer
import yaml
import os
from pathlib import Path

from validate_dataset_v50 import validate_dataset
from submission_format_checker import validate_submission
from fft_variance_heatmap import run_fft_heatmap
from photonic_alignment import run_photonic_overlay
from cli_guardrails import guardrails_wrapper, _log_cli_call

app = typer.Typer(help="Validation tools for dataset, submission, and symbolic QA")

@app.command("data")
def validate_data(
    data_dir: Path = typer.Option(Path("./data/train"), help="Root of preprocessed training data"),
    csv: Path = typer.Option(Path("./data/train.csv"), help="Metadata or label CSV"),
    skip_label: bool = typer.Option(False, help="Skip label check for test mode"),
    dry_run: bool = False,
    confirm: bool = True
):
    """Check dataset structure and metadata consistency"""
    args = dict(data_dir=str(data_dir), csv=str(csv), skip_label=skip_label)
    guardrails_wrapper("validate_data", args, dry_run, confirm)
    validate_dataset(str(data_dir), str(csv), skip_label=skip_label)

@app.command("submission")
def validate_sub(
    path: Path = typer.Option("submission.csv", help="Path to submission.csv"),
    dry_run: bool = False,
    confirm: bool = True
):
    """Check submission format and contents (NaNs, shape, headers)"""
    args = dict(path=str(path))
    guardrails_wrapper("validate_submission", args, dry_run, confirm)
    validate_submission(str(path))

@app.command("full")
def validate_all(
    config: Path = typer.Option(Path("configs/config_v50.yaml"), help="Path to config YAML"),
    include_fft: bool = True,
    include_symbolic: bool = True,
    dry_run: bool = False,
    confirm: bool = True
):
    """Run full validation pipeline: dataset, submission, FFT, symbolic alignment"""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("train_data_dir", "./data/train"))
    label_csv = Path(paths.get("metadata_file", "./data/train.csv"))
    submission_csv = Path(paths.get("submission_csv", "submission.csv"))

    args = {
        "config": str(config),
        "include_fft": include_fft,
        "include_symbolic": include_symbolic
    }
    guardrails_wrapper("validate_all", args, dry_run, confirm)

    typer.echo("📁 Validating dataset...")
    validate_dataset(str(data_dir), str(label_csv), skip_label=False)

    typer.echo("📦 Validating submission file...")
    validate_submission(str(submission_csv))

    if include_symbolic:
        typer.echo("🔬 Running symbolic overlay check...")
        run_photonic_overlay(str(submission_csv), outdir="outputs/diagnostics")

    if include_fft:
        typer.echo("📊 Running FFT variance heatmap...")
        run_fft_heatmap(str(submission_csv), outdir="outputs/diagnostics")

    typer.echo("✅ Validation complete.")

if __name__ == "__main__":
    app()
