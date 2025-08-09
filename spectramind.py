#!/usr/bin/env python
# spectramind.py — Unified CLI for SpectraMind V50
# ------------------------------------------------
# - Global flags: --dry-run, --confirm, --log, --version
# - Auto-detect Kaggle vs Local and resolve data paths
# - Logs to v50_debug_log.md and updates run_hash_summary_v50.json
# - Routes to: resolve-paths / train / predict / calibrate / diagnose / submit / selftest
#
# NOTE: If src/spectramind/utils/config.py exists, we import its resolver.
#       Otherwise we fall back to the built-in lightweight resolver below.

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

# Optional dependency: torch (we avoid hard-failing on import to keep CLI usable)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # noqa: N816

# Try to import the project-wide resolver; fallback provided if missing
try:
    from src.spectramind.utils.config import resolve_data_config as _external_resolver  # type: ignore
except Exception:
    _external_resolver = None

APP_VERSION = "0.3.0"
LOG_FILE = Path("v50_debug_log.md")
RUN_HASH = Path("run_hash_summary_v50.json")

app = typer.Typer(add_completion=False, help="SpectraMind V50 — Unified CLI")

# ---------------------------- utilities --------------------------------- #

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
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
        try:
            info["torch"] = str(torch.__version__)
            info["cuda_available"] = bool(torch.cuda.is_available())
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

class Ctx:
    dry_run: bool = False
    confirm: bool = False
    log: bool = True

# ----------------------- lightweight built-in resolver ------------------- #

def _is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()

def _probe_first(candidates) -> Optional[str]:
    for p in candidates:
        if p and Path(p).exists():
            return str(Path(p).resolve())
    return None

def _read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    pp = Path(path)
    if not pp.exists():
        return {}
    try:
        import yaml  # lazy import to keep CLI light if user didn't install
    except Exception:
        return {}
    return (yaml.safe_load(pp.read_text()) or {})

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def _builtin_resolve_data_config(
    base_config: str = "configs/data/challenge.yaml",
    override_config: Optional[str] = None,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _read_yaml(base_config)

    effective_env = env or ("kaggle" if _is_kaggle() else "local")
    auto_yaml = f"configs/data/challenge.{effective_env}.yaml"
    cfg = _merge(cfg, _read_yaml(auto_yaml))
    cfg = _merge(cfg, _read_yaml(override_config))

    env_root = os.environ.get("SPECTRAMIND_DATA_ROOT", "")
    env_fgs1 = os.environ.get("SPECTRAMIND_FGS1", "")
    env_airs = os.environ.get("SPECTRAMIND_AIRS", "")
    env_cal  = os.environ.get("SPECTRAMIND_CALIB", "")

    kaggle_candidates = dict(
        fgs1=[env_fgs1, f"{env_root}/raw/fgs1", "/kaggle/input/ariel-fgs1/raw/fgs1", "/kaggle/input/fgs1/raw/fgs1"],
        airs=[env_airs, f"{env_root}/raw/airs_ch0", "/kaggle/input/ariel-airs-ch0/raw/airs_ch0", "/kaggle/input/airs/raw/airs_ch0"],
        calibration=[env_cal, f"{env_root}/calibration", "/kaggle/input/ariel-calibration/calibration"],
    )
    local_candidates = dict(
        fgs1=[env_fgs1, f"{env_root}/raw/fgs1", "data/challenge/raw/fgs1", "/data/ariel/raw/fgs1", str(Path.home() / "datasets/ariel/raw/fgs1")],
        airs=[env_airs, f"{env_root}/raw/airs_ch0", "data/challenge/raw/airs_ch0", "/data/ariel/raw/airs_ch0", str(Path.home() / "datasets/ariel/raw/airs_ch0")],
        calibration=[env_cal, f"{env_root}/calibration", "data/challenge/calibration", "/data/ariel/calibration", str(Path.home() / "datasets/ariel/calibration")],
    )

    probe = kaggle_candidates if _is_kaggle() else local_candidates
    paths = cfg.setdefault("paths", {})
    resolved = dict(
        fgs1 = paths.get("fgs1") if paths.get("fgs1") and Path(paths["fgs1"]).exists() else _probe_first(probe["fgs1"]),
        airs = paths.get("airs") if paths.get("airs") and Path(paths["airs"]).exists() else _probe_first(probe["airs"]),
        calibration = paths.get("calibration") if paths.get("calibration") and Path(paths["calibration"]).exists() else _probe_first(probe["calibration"]),
        metadata = paths.get("metadata", ""),
        splits = paths.get("splits", ""),
        cache = paths.get("cache", "data/cache" if not _is_kaggle() else "/kaggle/working/.cache"),
    )
    cfg.setdefault("paths", {}).update({k: v for k, v in resolved.items() if v})

    # Absolutize for robustness
    for k, v in list(cfg["paths"].items()):
        if v:
            cfg["paths"][k] = str(Path(v).resolve())

    _log(True, f"[config] env={effective_env} kaggle={_is_kaggle()} resolved_paths=" + json.dumps(cfg["paths"]))
    # Defaults for loader/preprocess if not present
    cfg.setdefault("dataset", {"name": "ariel_challenge", "bins": 283})
    cfg.setdefault("loader", {"batch_size": 8, "num_workers": 4, "pin_memory": True})
    cfg.setdefault("preprocess", {"fgs1_len": 512, "airs_width": 356, "bin_to": 283, "normalize": True})
    return cfg

def resolve_data_config(
    base_config: str = "configs/data/challenge.yaml",
    override_config: Optional[str] = None,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    if _external_resolver is not None:
        return _external_resolver(base_config, override_config, env)
    return _builtin_resolve_data_config(base_config, override_config, env)

# -------------------------- global options ------------------------------- #

@app.callback()
def main(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, help="Plan only; do not execute mutating steps"),
    confirm: bool = typer.Option(False, help="Ask for confirmation before mutating steps"),
    log: bool = typer.Option(True, help="Append this run to v50_debug_log.md"),
    version: bool = typer.Option(False, "--version", help="Print version and exit"),
):
    """Global flags are available to all subcommands."""
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

    _maybe_confirm("start training")
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

    _maybe_confirm(f"run inference → {out_csv}")
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

    _maybe_confirm("apply calibration")
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

    _maybe_confirm("generate diagnostics dashboard")
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

    _maybe_confirm(f"package submission → {out_zip}")
    # If a dedicated submit CLI exists, we still create a simple bundle here.
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
