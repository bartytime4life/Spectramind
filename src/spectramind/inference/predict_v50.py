#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 – Minimal Predict Entrypoint (Hydra-ready, package layout)
---------------------------------------------------------------------------
Location: src/spectramind/inference/predict_v50.py

This stub:
  • Loads Hydra config from configs/config_v50.yaml (project root)
  • Logs a config hash and environment info to v50_debug_log.md
  • Loads (or tolerates) a placeholder checkpoint
  • Generates a placeholder submission.csv with the right shape

Run:
  python -m spectramind.inference.predict_v50

Replace the "SIMULATED INFERENCE" section with your real inference code.
"""

from __future__ import annotations

import os
import csv
import json
import socket
import getpass
import platform
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict

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
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    payload = json.dumps(cfg_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def log_cli_call(cfg: DictConfig, run_dir: Path, log_file: Path) -> None:
    ensure_dir(log_file.parent)
    entry = {
        "ts": _utcnow(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "cmd": "spectramind.inference.predict_v50",
        "config_name": "config_v50.yaml",
        "run_dir": str(run_dir),
        "config_hash": config_hash(cfg),
        "project": cfg.get("project", {}),
        "toggles": cfg.get("toggles", {}),
    }
    line = (
        f"{entry['ts']} | PREDICT | hash={entry['config_hash']} | "
        f"user={entry['user']}@{entry['host']} | run_dir={entry['run_dir']} | cmd={entry['cmd']}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def try_load_checkpoint(path: Path) -> None:
    if not path.exists():
        print(f"⚠️  Checkpoint not found at {path}. Proceeding with placeholder inference.")
        return
    try:
        import torch  # type: ignore
        _ = torch.load(str(path), map_location="cpu")
        print(f"✅  Loaded torch checkpoint: {path}")
    except Exception as e:
        # Might be our JSON placeholder or an unsupported format here
        try:
            _ = json.loads(Path(path.with_suffix(".json")).read_text(encoding="utf-8"))
            print(f"✅  Loaded JSON placeholder for checkpoint: {path.with_suffix('.json')}")
        except Exception:
            print(f"⚠️  Failed to parse checkpoint ({e}). Proceeding anyway.")


# ---------------------------
# Main
# ---------------------------

@hydra.main(config_path="../../../configs", config_name="config_v50", version_base=None)
def main(cfg: DictConfig) -> None:
    HydraConfig.get()
    run_dir = Path(os.getcwd())

    # Paths/IO from config
    logging_cfg = cfg.get("logging", {})
    log_file = Path(logging_cfg.get("log_file", "v50_debug_log.md"))

    predict_cfg = cfg.get("predict", {}) or {}
    export_cfg = predict_cfg.get("export", {})
    submission_csv = Path(export_cfg.get("submission_csv", "outputs/submission.csv"))
    weights_path = Path(predict_cfg.get("weights_path", "artifacts/checkpoints/best.ckpt"))

    # Log CLI call
    log_cli_call(cfg, run_dir, log_file)

    # Load (or tolerate) checkpoint
    try_load_checkpoint(weights_path)

    # Pull dataset shape (bins)
    data_cfg = cfg.get("data", {})
    bins = int(data_cfg.get("bins", 283))

    # SIMULATED INFERENCE:
    # Create a small, valid submission file with 5 dummy planets and μ/σ columns.
    n_planets = 5
    mu_cols = [f"mu_{i}" for i in range(bins)]
    sig_cols = [f"sigma_{i}" for i in range(bins)]
    header = ["planet_id"] + mu_cols + sig_cols

    ensure_dir(submission_csv.parent)
    with open(submission_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(n_planets):
            planet_id = f"planet_{k+1:04d}"
            mu = [0.0] * bins
            sigma = [0.1] * bins
            writer.writerow([planet_id] + mu + sigma)

    print(f"✅ Wrote placeholder submission to: {submission_csv}")
    print("⚠️  Replace the SIMULATED INFERENCE block with real model inference when ready.")


if __name__ == "__main__":
    main()