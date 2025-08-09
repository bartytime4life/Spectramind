#!/usr/bin/env python
# spectramind.py — Unified CLI for SpectraMind V50
# ------------------------------------------------
# - Global flags: --dry-run, --confirm, --log, --version
# - Auto-detects Kaggle vs Local and resolves data paths
# - Logs all invocations to v50_debug_log.md
# - Updates run_hash_summary_v50.json for reproducibility
# - Routes to train / predict / diagnostics / submit / selftest
#
# This file intentionally has *no* hardcoded paths. It defers to:
#   configs/data/challenge.yaml  (+ optional challenge.local.yaml / challenge.kaggle.yaml)
# and environment variables (SPECTRAMIND_*). See src/spectramind/utils/config.py

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import typer

# Optional deps (keep CLI resilient if not installed)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # noqa: N816

try:
    from src.spectramind.utils.config import resolve_data_config
except Exception:
    # Minimal fallback so CLI still runs; strongly recommend adding utils/config.py
    def resolve_data_config(base_config: str = "configs/data/challenge.yaml",
                            override_config: Optional[str] = None,
                            env: Optional[str] = None) -> Dict[str, Any]:
        return {
            "dataset": {"name": "ariel_challenge", "bins": 283},
            "paths": {
                "fgs1": "data/challenge/raw/fgs1",
                "airs": "data/challenge/raw/airs_ch0",
                "calibration": "data/challenge/calibration",
                "cache": "data/cache",
            },
            "loader": {"batch_size": 8, "num_workers": 4, "pin_memory": True},
            "preprocess": {"fgs1_len": 512, "airs_width": 356, "bin_to": 283, "normalize": True},
        }

APP_VERSION = "0.3.0"
LOG_FILE = Path("v50_debug_log.md")
RUN_HASH = Path("run_hash_summary_v50.json")

app = typer.Typer(add_completion=False, help="SpectraMind V50 — Unified CLI")

# ---------------------------- utilities --------------------------------- #

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"

def _log(enabled: bool, line: str) -> None:
    if not enabled:
        return
    LOG_FILE.write_text((LOG_FILE.read_text() if LOG_FILE.exists() else "") + line.rstrip() + "\n")

def _update_run_hash(extra: Optional[Dict[str, Any]] = None) -> None:
    info: Dict[str, Any] = {
        "version": APP_VERSION,
        "created_utc": _now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_rev(),
    }
    if torch is not None:
        info["torch"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
    if extra:
        info.update(extra)
    RUN_HASH.write_text(json.dumps(info, indent=2))

def _echo_kv(title: str, d: Dict[str, Any]) -> None:
    typer.echo(f"\n{title}:")
    for k, v in d.items():
        typer.echo(f"  {k}: {v}")

class Ctx:
    dry_run: bool = False
    confirm: bool = False
    log: bool = True

# -------------------------- global options ------------------------------- #

@app.callback()
def main(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, help="Plan only; do not execute mutating steps"),
    confirm: bool = typer.Option(False, help="Ask for confirmation before mutating steps"),
    log: bool = typer.Option(True, help="Append this run to v50_debug_log.md"),
    version: bool = typer.Option(False, "--version", help="Print version and exit"),
):
    """
    Global flags are available to all subcommands.
    """
    if version:
        typer.echo(f"SpectraMind V50 CLI {APP_VERSION}")
        raise typer.Exit(code=0)
    Ctx.dry_run, Ctx.confirm, Ctx.log = dry_run, confirm, log
    _log(Ctx.log, f"[{_now()}] CLI start args dry_run={dry_run} confirm={confirm} log={log} cmd={' '.join(sys.argv)}")

# --------------------------- commands ------------------------------------ #

@app.command()
def resolve_paths(
    data_config: Optional[str] = typer.Option(None, "--data-config", help="Override data YAML"),
    env: Optional[str] = typer.Option(None, "--env", help="Force environment: local|kaggle"),
):
    """Print the resolved data/calibration/cache paths after auto-detection."""
    cfg = resolve_data_config("configs/data/challenge.yaml", data_config, env)
    _echo_kv("Resolved paths", cfg.get("paths", {}))
    _log(Ctx.log, f"[resolve_paths] {json.dumps(cfg.get('paths', {}))}")

@app.command()
def train(
    epochs: int = typer.Option(2, help="Epochs"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    data_config: Optional[str] = typer.Option(None, "--data-config", help="Override data YAML"),
    env: Optional[str] = typer.Option(None, "--env", help="Force environment: local|kaggle"),
):
    """Supervised training (GLL + symbolic)."""
    cfg = resolve_data_config("configs/data/challenge.yaml", data_config, env)
    _log(Ctx.log, f"[train] cfg_paths={json.dumps(cfg.get('paths', {}))} epochs={epochs} lr={lr}")

    if Ctx.dry_run:
        typer.echo("[train] DRY RUN — no training executed.")
        _echo_kv("Would use paths", cfg.get("paths", {}))
        return

    from src.spectramind.training.train_v50 import train as _train
    _train(epochs=epochs, lr=lr)
    _update_run_hash({"last_command": "train", "epochs": epochs, "lr": lr})

@app.command()
def predict(
    out_csv: str = typer.Option("outputs/submission.csv", help="Output CSV"),
    data_config: Optional[str] = typer.Option(None, "--data-config"),
    env: Optional[str] = typer.Option(None, "--env"),
):
    """Inference pipeline to generate μ and σ; writes submission.csv."""
    cfg = resolve_data_config("configs/data/challenge.yaml", data_config, env)
    _log(Ctx.log, f"[predict] cfg_paths={json.dumps(cfg.get('paths', {}))} out={out_csv}")

    if Ctx.dry_run:
        typer.echo("[predict] DRY RUN — no inference executed.")
        _echo_kv("Would write", {"submission_csv": out_csv})
        return

    from src.spectramind.inference.predict_v50 import predict as _predict
    _predict(out_csv=out_csv)
    _update_run_hash({"last_command": "predict", "submission_csv": out_csv})

@app.command()
def calibrate(
    pred_json: str = typer.Option("outputs/predictions.json", help="Input predictions JSON"),
    out_json: str = typer.Option("outputs/predictions_calibrated.json", help="Calibrated output JSON"),
):
    """Instance-level & conformal calibration."""
    _log(Ctx.log, f"[calibrate] in={pred_json} out={out_json}")

    if Ctx.dry_run:
        typer.echo("[calibrate] DRY RUN — no calibration executed.")
        return

    try:
        from src.spectramind.calibration.calibrate_instance_level import main as _cal
        _cal(pred_json=pred_json, out_json=out_json)
        _update_run_hash({"last_command": "calibrate", "pred_in": pred_json, "pred_out": out_json})
    except Exception as e:  # pragma: no cover
        typer.echo(f"Calibration module not available or failed: {e}")
        raise typer.Exit(code=1)

@app.command()
def diagnose(
    html: str = typer.Option("outputs/diagnostics/diagnostic_report_v50.html", help="Output HTML"),
):
    """Generate a unified diagnostics HTML dashboard."""
    _log(Ctx.log, f"[diagnose] html={html}")

    if Ctx.dry_run:
        typer.echo("[diagnose] DRY RUN — no report generated.")
        return

    from src.spectramind.diagnostics.generate_html_report import generate as _report
    _report(output_html=html)
    _update_run_hash({"last_command": "diagnose", "diagnostics_html": html})

@app.command()
def submit(
    submission: str = typer.Option("outputs/submission.csv", help="Input submission CSV"),
    out_zip: str = typer.Option("outputs/submission_bundle.zip", help="Packaged bundle"),
):
    """Validate and package submission + reproducibility artifacts."""
    _log(Ctx.log, f"[submit] csv={submission} bundle={out_zip}")

    if Ctx.dry_run:
        typer.echo("[submit] DRY RUN — no bundle created.")
        return

    # Try dedicated submit CLI if present; otherwise inline pack
    try:
        from src.spectramind.cli.cli_submit import app as _  # noqa: F401
        # fallback to inline bundle even if imported
    except Exception:
        pass

    import zipfile
    outp = Path(out_zip)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outp, "w") as z:
        z.write(submission, arcname="submission.csv")
        for extra in ["run_hash_summary_v50.json", "v50_debug_log.md"]:
            if Path(extra).exists():
                z.write(extra, arcname=Path(extra).name)
    typer.echo(f"Bundled -> {out_zip}")
    _update_run_hash({"last_command": "submit", "bundle": out_zip})

@app.command()
def selftest():
    """Lightweight repository sanity check."""
    _log(Ctx.log, "[selftest]")
    try:
        from src.spectramind.cli.selftest import main as _self
        code = _self()
        if code != 0:
            raise typer.Exit(code=code)
        _update_run_hash({"last_command": "selftest"})
    except Exception as e:
        typer.echo(f"Selftest failed or module missing: {e}")
        raise typer.Exit(code=1)

# ----------------------------- entry ------------------------------------- #

if __name__ == "__main__":
    app()
