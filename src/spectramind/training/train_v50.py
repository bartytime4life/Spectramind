#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 – Minimal Train Entrypoint (Hydra-ready, package layout)
------------------------------------------------------------------------
Location: src/spectramind/training/train_v50.py

This stub:
  • Loads Hydra config from configs/config_v50.yaml (project root)
  • Logs a config hash and environment info to v50_debug_log.md
  • Verifies input data paths (warns if missing)
  • Creates output/diagnostics/checkpoint folders
  • Simulates a tiny training loop and writes a placeholder checkpoint

Run:
  python -m spectramind.training.train_v50

Replace the "SIMULATED TRAINING" section with your real training code.
"""

from __future__ import annotations

import os
import json
import time
import math
import hashlib
import socket
import getpass
import platform
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


# ---------------------------
# Utilities
# ---------------------------

def _utcnow() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def config_hash(cfg: DictConfig) -> str:
    """Stable-ish hash of the resolved config."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    payload = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def log_cli_call(cfg: DictConfig, run_dir: Path, log_file: Path) -> None:
    """Append a single-line, grep-friendly log entry to v50_debug_log.md."""
    ensure_dir(log_file.parent)
    entry = {
        "ts": _utcnow(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "cmd": "spectramind.training.train_v50",
        "config_name": "config_v50.yaml",
        "run_dir": str(run_dir),
        "config_hash": config_hash(cfg),
        "project": cfg.get("project", {}),
        "toggles": cfg.get("toggles", {}),
    }
    line = (
        f"{entry['ts']} | TRAIN | hash={entry['config_hash']} | "
        f"user={entry['user']}@{entry['host']} | run_dir={entry['run_dir']} | cmd={entry['cmd']}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def verify_input_paths(cfg: DictConfig) -> None:
    """Check common dataset path keys; warn (do not fail) if missing."""
    data = cfg.get("data", {})
    for key in ("fgs1_path", "airs_path", "calibration_dir"):
        raw = data.get(key, "")
        p = Path(raw) if raw else Path()
        if not raw:
            print(f"⚠️  [data.{key}] not set")
        elif not p.exists():
            print(f"⚠️  Path missing: data.{key} -> {p}")
        else:
            print(f"✅  Found: data.{key} -> {p}")


def write_placeholder_checkpoint(path: Path) -> None:
    """
    Create a tiny placeholder "checkpoint" so downstream scripts have something to load.
    If PyTorch is installed, we save an empty state dict; otherwise, we drop a JSON stub.
    """
    ensure_dir(path.parent)
    try:
        import torch  # type: ignore
        torch.save({"state_dict": {}, "meta": {"placeholder": True, "ts": _utcnow()}}, str(path))
        print(f"💾  Wrote torch checkpoint: {path}")
    except Exception as e:
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps({"placeholder": True, "ts": _utcnow()}, indent=2), encoding="utf-8")
        print(f"💾  Torch not available ({e}). Wrote JSON placeholder: {json_path}")


# ---------------------------
# Main
# ---------------------------

@hydra.main(config_path="../../../configs", config_name="config_v50", version_base=None)
def main(cfg: DictConfig) -> None:
    # When Hydra is configured with job.chdir=true, CWD becomes the job run dir
    HydraConfig.get()
    run_dir = Path(os.getcwd())

    # Directories from config
    paths = cfg.get("paths", {})
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    checkpoints_dir = Path(paths.get("checkpoints_dir", artifacts_dir / "checkpoints"))
    diagnostics_dir = Path(cfg.get("diagnostics", {}).get("artifacts", {}).get("dir", "diagnostics"))
    log_file = Path(cfg.get("logging", {}).get("log_file", "v50_debug_log.md"))

    # Create dirs
    ensure_dir(artifacts_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(Path(diagnostics_dir))

    # Log CLI call
    log_cli_call(cfg, run_dir, log_file)

    # Print and verify key paths
    print("========== SpectraMind V50 – TRAIN ==========")
    print("Resolved Config (excerpt):")
    print(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True, indent=2)[:2000])
    print("=============================================")
    verify_input_paths(cfg)

    # Simulated training loop (replace with real training)
    max_epochs = int(cfg.get("training", {}).get("scheduler", {}).get("max_epochs", 5))
    log_interval = int(cfg.get("training", {}).get("logging", {}).get("log_interval_steps", 50))
    print(f"Starting SIMULATED TRAINING for {max_epochs} epochs ...")

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss = 1.0 / math.sqrt(epoch)
        val_loss = train_loss * 1.05

        if epoch % max(1, (log_interval // 10)) == 0 or epoch == 1 or epoch == max_epochs:
            print(f"[Epoch {epoch:03d}] Train GLL: {train_loss:.6f} | Val GLL: {val_loss:.6f}")

        # Rolling diagnostic summary (lightweight JSON)
        diag_path = Path(diagnostics_dir) / "diagnostic_summary.json"
        ensure_dir(diag_path.parent)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ts": _utcnow(),
                    "epoch": epoch,
                    "metrics": {"train_gll": train_loss, "val_gll": val_loss},
                    "config_hash": config_hash(cfg),
                },
                f,
                indent=2,
            )

        # Write a "best" checkpoint at final epoch
        if epoch == max_epochs:
            write_placeholder_checkpoint(Path(paths.get("checkpoints_dir", "artifacts/checkpoints")) / "best.ckpt")

    dt = time.time() - t0
    print(f"✅ Training complete in {dt:.2f}s. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()