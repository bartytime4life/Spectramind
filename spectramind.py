#!/usr/bin/env python
# spectramind.py — Unified CLI for SpectraMind V50
# ------------------------------------------------
# Global flags: --dry-run, --confirm, --log, --version
# Auto Kaggle/local detection + smart data path resolution
# Logs every run to v50_debug_log.md, updates run_hash_summary_v50.json
# Commands: resolve-paths / check / train / predict / calibrate / diagnose / submit / selftest
#
# If src/spectramind/utils/config.py exists, we import its resolver.
# Otherwise, we fall back to a built-in lightweight resolver below.

from __future__ import annotations

import hashlib
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

# Optional dependency: torch (keep CLI usable without it)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # noqa: N816

# Try to import project resolver; fallback if missing
try:
    from src.spectramind.utils.config import resolve_data_config as _external_resolver  # type: ignore
except Exception:
    _external_resolver = None

APP_VERSION = "0.3.0"
LOG_FILE = Path("v50_debug_log.md")
RUN_HASH = Path("run_hash_summary_v50.json")
CONFIG_BASE = "configs/data/challenge.yaml"

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

def _probe_first(candidates: Iterable[str]) -> Optional[str]:
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
        import yaml  # lazy import
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
    base_config: str = CONFIG_BASE,
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

    # Absolutize
    for k, v in list(cfg["paths"].items()):
        if v:
            cfg["paths"][k] = str(Path(v).resolve())

    _log(True, f"[config] env={effective_env} kaggle={_is_kaggle()} resolved_paths=" + json.dumps(cfg["paths"]))
    cfg.setdefault("dataset", {"name": "ariel_challenge", "bins": 283})
    cfg.setdefault("loader", {"batch_size": 8, "num_workers": 4, "pin_memory": True})
    cfg.setdefault("preprocess", {"fgs1_len": 512, "airs_width": 356, "bin_to": 283, "normalize": True})
    return cfg

def resolve_data_config(
    base_config: str = CONFIG_BASE,
    override_config: Optional[str] = None,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    if _external_resolver is not None:
        return _external_resolver(base_config, override_config, env)
    return _builtin_resolve_data_config(base_config, override_config, env)

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

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

# Minimal, extensible map (aligns with your protocol)
RIPPLE_MAP: Tuple[RippleRule, ...] = (
    RippleRule(
        touch=("src/spectramind/model_v50_ar.py", "src/spectramind/core/multi_scale_decoder.py", "src/spectramind/core/flow_uncertainty_head.py"),
        then_run=("src/spectramind/training/train_v50.py", "src/spectramind/inference/predict_v50.py", "configs/config_v50.yaml", "spectramind.py"),
    ),
    RippleRule(
        touch=("src/spectramind/models/airs_gnn.py", "src/spectramind/models/fgs1_mamba.py"),
        then_run=("src/spectramind/diagnostics/generate_html_report.py", "src/spectramind/diagnostics/fft_variance_heatmap.py"),
    ),
    RippleRule(
        touch=("src/spectramind/symbolic/symbolic_loss.py", "src/spectramind/symbolic/symbolic_logic_engine.py"),
        then_run=("src/spectramind/diagnostics/symbolic_violation_overlay.py", "v50_debug_log.md"),
    ),
)

def _git_modified(paths: Iterable[str]) -> Iterable[str]:
    """Return subset of paths that are modified vs HEAD (or simply exist if git not available)."""
    modified = []
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
        changed = {line.split()[-1] for line in status.strip().splitlines() if line.strip()}
        for p in paths:
            if p in changed:
                modified.append(p)
    except Exception:
        # fall back: if file exists and mtime < 1 day, consider as 'touched'
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
            notes.append(
                f"Detected edits in {touched} → consider re-running {list(rule.then_run)}"
            )
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
    """Global flags are available to all subcommands."""
    if version:
        typer.echo(f"SpectraMind V50 CLI {APP_VERSION}")
        raise typer.Exit(code=0)
    Ctx.dry_run, Ctx.confirm, Ctx.log = dry_run, confirm, log
    _log(Ctx.log, f"[{_now()}] CLI start args dry_run={dry_run} confirm={confirm} log={log} cmd={' '.join(sys.argv)}")
    # Ripple warnings at entry
    for note in _ripple_warnings():
        _log(True, "[ripple] " + note)
        typer.echo(f"⚠️  {note}")

# --------------------------- commands ------------------------------------ #

@app.command("resolve-paths")
def resolve_paths(
    data_config: Optional[str] = typer.Option(None, "--data-config", help="Override data YAML"),
    env: Optional[str] = typer.Option(None, "--env", help="Force environment: local|kaggle"),
):
    """Print the resolved data/calibration/cache paths after auto-detection."""
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)
    _echo_kv("Resolved paths", cfg.get("paths", {}))
    _log(Ctx.log, f"[resolve_paths] {json.dumps(cfg.get('paths', {}))}")

@app.command()
def check(
    data_config: Optional[str] = typer.Option(None, "--data-config"),
    env: Optional[str] = typer.Option(None, "--env"),
):
    """Run repo self-checks (selftest + optional pipeline_consistency_checker)."""
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)  # ensures resolver runs/logs
    rc_total = 0

    # selftest
    try:
        from src.spectramind.cli.selftest import main as _self
        rc = _self()
        rc_total |= (rc != 0)
        typer.echo("selftest: " + ("✅ ok" if rc == 0 else "❌ failed"))
    except Exception as e:
        typer.echo(f"selftest: ❌ error: {e}")
        rc_total |= 1

    # optional pipeline_consistency_checker
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
    data_config: Optional[str] = typer.Option(None, "--data-config", help="Override data YAML"),
    env: Optional[str] = typer.Option(None, "--env", help="Force environment: local|kaggle"),
):
    """Supervised training (GLL + symbolic)."""
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)
    _log(Ctx.log, f"[train] cfg_paths={json.dumps(cfg.get('paths', {}))} epochs={epochs} lr={lr}")

    if Ctx.dry_run:
        typer.echo("[train] DRY RUN — no training executed.")
        _echo_kv("Would use paths", cfg.get("paths", {}))
        return

    _maybe_confirm("start training")
    # Snapshot configs used this run (best-effort)
    snap = _snapshot_configs([
        Path(CONFIG_BASE),
        Path(f"configs/data/challenge.{ 'kaggle' if _is_kaggle() else 'local' }.yaml")
    ])
    if snap:
        _log(True, f"[snapshot] configs saved -> {snap}")

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
    cfg = resolve_data_config(CONFIG_BASE, data_config, env)
    _log(Ctx.log, f"[predict] cfg_paths={json.dumps(cfg.get('paths', {}))} out={out_csv}")

    if Ctx.dry_run:
        typer.echo("[predict] DRY RUN — no inference executed.")
        _echo_kv("Would write", {"submission_csv": out_csv})
        return

    _maybe_confirm(f"run inference → {out_csv}")
    # Snapshot configs
    snap = _snapshot_configs([Path(CONFIG_BASE)])
    if snap:
        _log(True, f"[snapshot] configs saved -> {snap}")

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
    # Bundle minimal reproducibility artifacts
    import zipfile
    outp = Path(out_zip)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outp, "w") as z:
        z.write(submission, arcname="submission.csv")
        for extra in ["run_hash_summary_v50.json", "v50_debug_log.md"]:
            if Path(extra).exists():
                z.write(extra, arcname=Path(extra).name)
        # include latest config snapshot if present
        snaps = sorted(Path("outputs/config_snapshots").glob("*"), reverse=True)
        if snaps:
            for p in snaps[0].glob("*"):
                z.write(p, arcname=f"config_snapshot/{p.name}")
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
