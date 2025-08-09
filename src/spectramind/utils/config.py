# src/spectramind/utils/config.py
from __future__ import annotations

"""
Unified Config Resolver for SpectraMind V50
-------------------------------------------
- Auto-detects Kaggle vs Local
- Merges base YAML -> env YAML -> explicit override YAML
- Applies ENV overrides (SPECTRAMIND_DATA_ROOT / FGS1 / AIRS / CALIB)
- Probes common mount locations to fill missing paths
- Expands ~, makes paths absolute, validates existence (warns if missing)
- Populates sane defaults for dataset/loader/preprocess if absent
- Logs resolution to v50_debug_log.md

Usage:
    from src.spectramind.utils.config import resolve_data_config
    cfg = resolve_data_config("configs/data/challenge.yaml",
                              override_config=None,
                              env=None)
"""

import os
import json
import socket
from pathlib import Path
from typing import Dict, Optional, Any

import yaml

LOG_FILE = Path("v50_debug_log.md")


# ----------------------------- helpers ---------------------------------- #

def _log(line: str) -> None:
    LOG_FILE.write_text((LOG_FILE.read_text() if LOG_FILE.exists() else "") + line.rstrip() + "\n")


def _is_kaggle() -> bool:
    # Kaggle sets an env var and mounts /kaggle
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()


def _read_yaml(p: Optional[str | Path]) -> Dict[str, Any]:
    if not p:
        return {}
    pp = Path(p)
    if not pp.exists():
        return {}
    try:
        return yaml.safe_load(pp.read_text()) or {}
    except Exception:
        return {}


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _probe_first(candidates) -> Optional[str]:
    for p in candidates:
        if p and Path(p).expanduser().exists():
            return str(Path(p).expanduser().resolve())
    return None


def _expand_abs_paths(d: Dict[str, Any], keys=("paths",)) -> Dict[str, Any]:
    for sect in keys:
        val = d.get(sect)
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str):
                    d[sect][k] = str(Path(v).expanduser().resolve())
    return d


def _warn_missing_paths(paths: Dict[str, str]) -> None:
    for k, v in (paths or {}).items():
        if v and not Path(v).exists():
            print(f"[config] ⚠ missing path for '{k}': {v}")


# ----------------------------- resolver --------------------------------- #

def resolve_data_config(
    base_config: str = "configs/data/challenge.yaml",
    override_config: Optional[str] = None,
    env: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolution order:
      1) base YAML (portable defaults)
      2) auto env YAML (challenge.kaggle.yaml or challenge.local.yaml) if present
      3) explicit override_config (CLI flag)
      4) ENV overrides + auto-probed paths

    Returns:
      dict with normalized absolute paths under cfg["paths"] and metadata in cfg["meta"].
    """
    effective_env = (env or ("kaggle" if _is_kaggle() else "local")).lower()

    # Load layered YAMLs
    cfg_base = _read_yaml(base_config)
    cfg_env  = _read_yaml(Path(base_config).with_name(Path(base_config).stem + f".{effective_env}.yaml"))
    cfg_ovr  = _read_yaml(override_config)

    cfg = {}
    for layer in (cfg_base, cfg_env, cfg_ovr):
        cfg = _merge(cfg, layer)

    # Defaults (if missing)
    cfg.setdefault("dataset", {"name": "ariel_challenge", "bins": 283})
    cfg.setdefault("loader", {"batch_size": 8, "num_workers": 4, "pin_memory": True, "persistent_workers": False})
    cfg.setdefault("preprocess", {"fgs1_len": 512, "airs_width": 356, "bin_to": 283, "normalize": True, "detrend": "median"})
    cfg.setdefault("paths", {})

    paths = cfg["paths"]

    # ENV overrides
    env_root = os.environ.get("SPECTRAMIND_DATA_ROOT", "")
    env_fgs1 = os.environ.get("SPECTRAMIND_FGS1", "")
    env_airs = os.environ.get("SPECTRAMIND_AIRS", "")
    env_cal  = os.environ.get("SPECTRAMIND_CALIB", "")

    # Candidate lists to probe
    kaggle_candidates = dict(
        fgs1=[paths.get("fgs1"), env_fgs1, f"{env_root}/raw/fgs1", "/kaggle/input/ariel-fgs1/raw/fgs1", "/kaggle/input/fgs1/raw/fgs1"],
        airs=[paths.get("airs"), env_airs, f"{env_root}/raw/airs_ch0", "/kaggle/input/ariel-airs-ch0/raw/airs_ch0", "/kaggle/input/airs/raw/airs_ch0"],
        calibration=[paths.get("calibration"), env_cal, f"{env_root}/calibration", "/kaggle/input/ariel-calibration/calibration"],
        cache=[paths.get("cache"), "/kaggle/working/.cache"],
    )
    local_candidates = dict(
        fgs1=[paths.get("fgs1"), env_fgs1, f"{env_root}/raw/fgs1", "data/challenge/raw/fgs1", "/data/ariel/raw/fgs1", str(Path.home() / "datasets/ariel/raw/fgs1")],
        airs=[paths.get("airs"), env_airs, f"{env_root}/raw/airs_ch0", "data/challenge/raw/airs_ch0", "/data/ariel/raw/airs_ch0", str(Path.home() / "datasets/ariel/raw/airs_ch0")],
        calibration=[paths.get("calibration"), env_cal, f"{env_root}/calibration", "data/challenge/calibration", "/data/ariel/calibration", str(Path.home() / "datasets/ariel/calibration")],
        cache=[paths.get("cache"), "data/cache"],
    )

    probe = kaggle_candidates if (effective_env == "kaggle") else local_candidates

    resolved = {
        "fgs1": _probe_first(probe["fgs1"]),
        "airs": _probe_first(probe["airs"]),
        "calibration": _probe_first(probe["calibration"]),
        "metadata": paths.get("metadata", ""),
        "splits": paths.get("splits", ""),
        "cache": _probe_first(probe["cache"]),
    }
    # Keep any user-specified values that exist
    for k, v in list(paths.items()):
        if v and Path(v).expanduser().exists():
            resolved[k] = str(Path(v).expanduser().resolve())

    cfg["paths"].update({k: v for k, v in resolved.items() if v})

    # Normalize
    cfg = _expand_abs_paths(cfg, keys=("paths",))

    # Attach runtime metadata
    cfg.setdefault("meta", {})
    cfg["meta"].update({
        "runtime_env": effective_env,
        "hostname": socket.gethostname(),
    })

    # Warn about missing paths (non-fatal)
    _warn_missing_paths(cfg["paths"])

    # Log resolution
    _log("[config] env="
         + effective_env
         + " kaggle=" + str(_is_kaggle())
         + " resolved_paths=" + json.dumps(cfg["paths"])
    )

    return cfg


# ------------------------------ CLI shim -------------------------------- #

if __name__ == "__main__":
    # Handy to debug in isolation: `python src/spectramind/utils/config.py`
    import json
    out = resolve_data_config()
    print(json.dumps(out, indent=2))
