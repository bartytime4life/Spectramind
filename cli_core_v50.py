"""
SpectraMind V50 – Unified Command Line Interface
--------------------------------------------------
Central Typer-based CLI entry point for all operations: training, calibration, inference,
diagnostics, validation, packaging, tuning, clustering, symbolic QA, monitoring, and explanation.
"""

import typer
import subprocess
from pathlib import Path
from train_v50 import train as run_training
from predict_v50 import run as run_prediction
from evaluate_gll_v50 import evaluate_and_log_gll
from submission_validator_v50 import validate_submission
from generate_submission_package import generate_zip

app = typer.Typer(help="SpectraMind V50 – Exoplanetary Spectrum Inference CLI")


@app.command()
def train(config: Path = typer.Option("configs/config_v50.yaml", help="YAML config")):
    """Train model using symbolic-aware pipeline"""
    run_training(config_path=config)


@app.command()
def predict():
    """Run μ + σ prediction and generate submission.csv"""
    run_prediction()


@app.command()
def diagnose(
    overlay_csv: str = typer.Option("diagnostics/symbolic_clusters.csv"),
    overlay_column: str = typer.Option("symbolic_class")
):
    """Run full diagnostics dashboard"""
    cmd = [
        "python", "cli_diagnose.py", "dashboard",
        "--overlay_csv", overlay_csv,
        "--overlay_column", overlay_column
    ]
    subprocess.run(cmd)


@app.command("score-gll")
def score_gll(
    labels: str = typer.Option("data/train.csv"),
    preds: str = typer.Option("outputs/submission.csv"),
    out: str = typer.Option("diagnostics/gll_score_submission.json"),
    tag: str = typer.Option("submission")
):
    """Evaluate GLL score against ground truth"""
    evaluate_and_log_gll(labels, preds, out, tag=tag)


@app.command()
def validate(
    submission: str = "outputs/submission.csv",
    gll_eval: bool = True
):
    """Validate submission.csv format + optional GLL check"""
    validate_submission(submission, gll_eval=gll_eval)


@app.command()
def package():
    """Bundle submission.zip with logs, diagnostics, and model"""
    generate_zip()


@app.command("tune-temp")
def tune_temp():
    """(Placeholder) Tune temperature or conformal bounds"""
    print("🔧 Tuning placeholder. Use COREL or conformal calibrator module.")


if __name__ == "__main__":
    app()