#!/usr/bin/env python
# spectramind.py — Unified CLI for SpectraMind V50
# ------------------------------------------------
# Global flags: --dry-run, --confirm, --log, --version
# Kaggle/local auto-detection via src/spectramind/utils/config.py
# Logs to v50_debug_log.md, updates run_hash_summary_v50.json
# Commands: resolve-paths / check / train / predict / calibrate / calibrate-data
#           explain / diagnose / dashboard / submit / selftest

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import typer

# Hard-require the project resolver (no built-in fallback):
from src.spectramind.utils.config import resolve_data_config  # type: ignore

# Optional dependency: torch (keep CLI usable without it)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # noqa: N816

APP_VERSION = "0.5.0"
LOG_FILE = Path("v50_debug_log.md")
RUN_HASH = Path("run_hash_summary_v50.json")
CONFIG_BASE = "configs/data/challenge.yaml"

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
        "cmdline": " ".join(sys.argv),
    }
    if torch is not None:
        try:
            info["torch"] = str(torch.__version__)
            info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    if extra:
        info.update(extra)
    RUN_HASH.write_text(json.dumps(info, indent=2))

def _echo_kv(title: str, d: Dict[str, Any]) -> None:
    typer.echo(f"\n{title}:")
    for k, v in d.items():
        typer.echo(f"  {k}: {v}")

def _maybe_confirm(what: str) -> None:
    if not Ctx.confirm:
        return
    if not typer.confirm(f"Proceed to {what}?"):
        raise typer.Exit(code=1)

def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()

def _snapshot_configs(cfg_paths: Iterable[Path]) -> Optional[Path]:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs/config_snapshots") / ts
    to_copy = [p for p in cfg_paths if p.exists()]
    if not to_copy:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in to_copy:
        shutil.copy2(p, out_dir / p.name)
    return out_dir

# ----------------------- ripple‑update awareness ------------------------- #

@dataclass
class RippleRule:
    touch: Tuple[str, ...]      # if any of these change…
    then_run: Tuple[str, ...]   # …remind to run these

RIPPLE_MAP: Tuple[RippleRule, ...] = (
    # Core model/heads
    RippleRule(
        touch=(
            "src/spectramind/model_v50_ar.py",
            "src/spectramind/core/multi_scale_decoder.py",
            "src/spectramind/core/flow_uncertainty_head.py",
            "src/spectramind/heads/quantile_head.py",
        ),
        then_run=(
            "src/spectramind/training/train_v50.py",
            "src/spectramind/inference/predict_v50.py",
            "src/spectramind/diagnostics/generate_html_report.py",
            "configs/config_v50.yaml",
            "spectramind.py",
        ),
    ),
    # Encoders / auxiliary models
    RippleRule(
        touch=(
            "src/spectramind/models/airs_gnn.py",
            "src/spectramind/models/fgs1_mamba.py",
        ),
        then_run=(
            "src/spectramind/diagnostics/fft_variance_heatmap.py",
            "src/spectramind/diagnostics/coherence_curve_plot.py",
            "src/spectramind/diagnostics/generate_html_report.py",
        ),
    ),
    # Symbolic rules & logic
    RippleRule(
        touch=(
            "src/spectramind/symbolic/symbolic_loss.py",
            "src/spectramind/symbolic/symbolic_logic_engine.py",
            "src/spectramind/symbolic/symbolic_program_executor.py",
            "src/spectramind/symbolic/symbolic_program_hypotheses.py",
            "src/spectramind/symbolic/symbolic_profile_switcher.py",
        ),
        then_run=(
            "src/spectramind/diagnostics/symbolic_influence_map.py",
            "src/spectramind/diagnostics/violation_heatmap.py",
            "src/spectramind/diagnostics/generate_html_report.py",
        ),
    ),
    # Diagnostics surfaces
    RippleRule(
        touch=(
            "src/spectramind/diagnostics/spectral_smoothness_map.py",
            "src/spectramind/diagnostics/fft_variance_heatmap.py",
            "src/spectramind/diagnostics/plot_quantiles_vs_target.py",
            "src/spectramind/diagnostics/plot_gll_heatmap_per_bin.py",
        ),
        then_run=("src/spectramind/diagnostics/generate_html_report.py",),
    ),
    # Data loading & calibration
    RippleRule(
        touch=(
            "src/spectramind/data/dataloader.py",
            "src/spectramind/calibration/calibrate_instance_level.py",
            "src/spectramind/calibration/spectral_conformal.py",
        ),
        then_run=(
            "src/spectramind/training/train_v50.py",
            "src/spectramind/inference/predict_v50.py",
            "src/spectramind/diagnostics/generate_html_report.py",
        ),
    ),
)

def _git_modified(paths: Iterable[str]) -> Iterable[str]:
    modified = []
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
        changed = {line.split()[-1] for line in status.strip().splitlines() if line.strip()}
        for p in paths:
            if p in changed:
                modified.append(p)
    except Exception:
        for p in paths:
            fp = Path(p)
            if fp.exists():
                age = (datetime.utcnow() - datetime.utcfromtimestamp(fp.stat().st_mtime)).total_seconds()
                if age < 86400:
                    modified.append(p)
    return modified

def _ripple_warnings() -> list[str]:
    notes: list[str] = []
    for rule in RIPPLE_MAP:
        touched = list(_git_modified(rule.touch))
        if touched:
            notes.append(f"Detected edits in {touched} → consider re-running {list(rule.then_run)}")
    return notes

# ----------------------------- context ----------------------------------- #

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
    if version:
        typer.echo(f"SpectraMind V50 CLI {APP_VERSION}")
        raise typer.Exit(code=0)
    Ctx.dry_run, Ctx.confirm, Ctx.log = dry_run, confirm, log
    _log(Ctx.log, f"[{_now()}] CLI start args dry_run={dry_run} confirm={confirm} log={log} cmd={' '.join(sys.argv)}")
    for note in _ripple_warnings():
        _log(True, "[ripple] " + note)
        typer.echo(f"⚠️  {note}")

# --------------------------- commands ------------------------------------ #

@app.command("resolve-paths")
def resolve_paths(
    data_config: Optional[str] = typer.Option(None, "--data-config", help="Override data YAML"),
    env: Optional[str] = typer.Option(None, "--env", help="Force environment: local|kaggle"),
    dump: Optional[str] = typer.Option(None, "--dump", help="Write resolved dict to JSON"),
):
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)
    _echo_kv("Resolved paths", cfg.get("paths", {}))
    if dump:
        Path(dump).write_text(json.dumps(cfg, indent=2))
        typer.echo(f"Wrote resolved config -> {dump}")
    _log(Ctx.log, f"[resolve_paths] {json.dumps(cfg.get('paths', {}))}")

@app.command()
def check(
    data_config: Optional[str] = typer.Option(None, "--data-config"),
    env: Optional[str] = typer.Option(None, "--env"),
):
    _ = resolve_data_config(CONFIG_BASE, data_config, env)
    rc_total = 0
    try:
        from src.spectramind.cli.selftest import main as _self
        rc = _self()
        rc_total |= (rc != 0)
        typer.echo("selftest: " + ("✅ ok" if rc == 0 else "❌ failed"))
    except Exception as e:
        typer.echo(f"selftest: ❌ error: {e}")
        rc_total |= 1
    try:
        mod = "src/spectramind/cli/pipeline_consistency_checker.py"
        if Path(mod).exists():
            r = subprocess.run([sys.executable, mod], capture_output=True, text=True)
            typer.echo("consistency: " + ("✅ ok" if r.returncode == 0 else "❌ failed"))
            if r.stdout:
                _log(Ctx.log, "[consistency]\n" + r.stdout)
            rc_total |= (r.returncode != 0)
        else:
            typer.echo("consistency: (skipped; module not present)")
    except Exception as e:
        typer.echo(f"consistency: ❌ error: {e}")
        rc_total |= 1
    _update_run_hash({"last_command": "check"})
    if rc_total:
        raise typer.Exit(code=1)

@app.command()
def train(
    epochs: int = typer.Option(2, help="Epochs"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    data_config: Optional[str] = typer.Option(None, "--data-config"),
    env: Optional[str] = typer.Option(None, "--env"),
):
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)
    _log(Ctx.log, f"[train] cfg_paths={json.dumps(cfg.get('paths', {}))} epochs={epochs} lr={lr}")
    if Ctx.dry_run:
        typer.echo("[train] DRY RUN — no training executed.")
        _echo_kv("Would use paths", cfg.get("paths", {}))
        return
    _maybe_confirm("start training")
    snap = _snapshot_configs([Path(CONFIG_BASE), Path(f"configs/data/challenge.{ 'kaggle' if _is_kaggle() else 'local' }.yaml")])
    if snap: _log(True, f"[snapshot] configs -> {snap}")
    from src.spectramind.training.train_v50 import train as _train
    _train(epochs=epochs, lr=lr, cfg=cfg)
    _update_run_hash({"last_command": "train", "epochs": epochs, "lr": lr})

@app.command()
def predict(
    out_csv: str = typer.Option("outputs/submission.csv", help="Output CSV"),
    data_config: Optional[str] = typer.Option(None, "--data-config"),
    env: Optional[str] = typer.Option(None, "--env"),
):
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)
    _log(Ctx.log, f"[predict] cfg_paths={json.dumps(cfg.get('paths', {}))} out={out_csv}")
    if Ctx.dry_run:
        typer.echo("[predict] DRY RUN — no inference executed.")
        _echo_kv("Would write", {"submission_csv": out_csv})
        return
    _maybe_confirm(f"run inference → {out_csv}")
    snap = _snapshot_configs([Path(CONFIG_BASE)])
    if snap: _log(True, f"[snapshot] configs -> {snap}")
    from src.spectramind.inference.predict_v50 import predict as _predict
    _predict(out_csv=out_csv, cfg=cfg)
    _update_run_hash({"last_command": "predict", "submission_csv": out_csv})

@app.command()
def calibrate(
    pred_json: str = typer.Option("outputs/predictions.json", help="Input predictions JSON"),
    out_json: str = typer.Option("outputs/predictions_calibrated.json", help="Calibrated output JSON"),
):
    _log(Ctx.log, f"[calibrate] in={pred_json} out={out_json}")
    if Ctx.dry_run:
        typer.echo("[calibrate] DRY RUN — no calibration executed."); return
    _maybe_confirm("apply calibration")
    from src.spectramind.calibration.calibrate_instance_level import main as _cal
    _cal(pred_json=pred_json, out_json=out_json)
    _update_run_hash({"last_command": "calibrate", "pred_in": pred_json, "pred_out": out_json})

@app.command("calibrate-data")
def calibrate_data(
    data_config: Optional[str] = typer.Option(None, "--data-config"),
    env: Optional[str] = typer.Option(None, "--env"),
):
    """Run raw data calibration/preprocessing (if module available)."""
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)
    _log(Ctx.log, "[calibrate-data] start")
    if Ctx.dry_run:
        typer.echo("[calibrate-data] DRY RUN — no action."); return
    try:
        from src.spectramind.data.calibrate_raw_data import main as _cd  # optional module
        _cd(cfg=cfg)
    except Exception as e:
        typer.echo(f"[calibrate-data] Skipped (module missing or error): {e}")
    _update_run_hash({"last_command": "calibrate-data"})

@app.command()
def explain(
    html: str = typer.Option("outputs/diagnostics/explain_report.html", help="Output HTML"),
):
    """Generate SHAP + symbolic overlays (optional; runs if modules exist)."""
    _log(Ctx.log, f"[explain] html={html}")
    if Ctx.dry_run:
        typer.echo("[explain] DRY RUN — no report."); return
    try:
        from src.spectramind.diagnostics.shap_overlay import generate as _shap
        from src.spectramind.diagnostics.symbolic_influence_map import generate as _sym
        _shap(output_html=html.replace(".html", "_shap.html"))
        _sym(output_html=html.replace(".html", "_symbolic.html"))
        typer.echo(f"Wrote overlays near {html}")
    except Exception as e:
        typer.echo(f"[explain] Skipped (modules missing or error): {e}")
    _update_run_hash({"last_command": "explain", "html": html})

@app.command()
def diagnose(
    html: str = typer.Option("outputs/diagnostics/diagnostic_report_v50.html", help="Output HTML"),
):
    _log(Ctx.log, f"[diagnose] html={html}")
    if Ctx.dry_run:
        typer.echo("[diagnose] DRY RUN — no report generated."); return
    _maybe_confirm("generate diagnostics dashboard")
    from src.spectramind.diagnostics.generate_html_report import generate as _report
    _report(output_html=html)
    _update_run_hash({"last_command": "diagnose", "diagnostics_html": html})

@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(7860, help="Port"),
):
    """Start optional FastAPI/Gradio dashboard if bundled."""
    _log(Ctx.log, f"[dashboard] {host}:{port}")
    if Ctx.dry_run:
        typer.echo("[dashboard] DRY RUN — not starting server."); return
    try:
        from src.spectramind.dashboard.app import run as _run  # optional module
        _run(host=host, port=port)
    except Exception as e:
        typer.echo(f"[dashboard] Skipped (module missing or error): {e}")
        raise typer.Exit(code=1)

@app.command()
def submit(
    submission: str = typer.Option("outputs/submission.csv", help="Input submission CSV"),
    out_zip: str = typer.Option("outputs/submission_bundle.zip", help="Packaged bundle"),
):
    _log(Ctx.log, f"[submit] csv={submission} bundle={out_zip}")
    if Ctx.dry_run:
        typer.echo("[submit] DRY RUN — no bundle created."); return
    _maybe_confirm(f"package submission → {out_zip}")
    import zipfile
    outp = Path(out_zip); outp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outp, "w") as z:
        z.write(submission, arcname="submission.csv")
        for extra in [
            "run_hash_summary_v50.json",
            "v50_debug_log.md",
            "configs/data/challenge.yaml",
            "configs/data/challenge.local.yaml",
            "configs/data/challenge.kaggle.yaml",
        ]:
            p = Path(extra)
            if p.exists():
                z.write(p, arcname=str(p))
        snaps = sorted(Path("outputs/config_snapshots").glob("*"), reverse=True)
        if snaps:
            for p in snaps[0].glob("*"):
                z.write(p, arcname=f"config_snapshot/{p.name}")
    typer.echo(f"Bundled -> {out_zip}")
    _update_run_hash({"last_command": "submit", "bundle": out_zip})

@app.command()
def selftest():
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
