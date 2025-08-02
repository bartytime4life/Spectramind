"""
SpectraMind V50 – Pipeline Finalizer (Ultimate)
-----------------------------------------------
Validates submission.csv, computes SHA256, records Git hash, and writes manifest + TOML
for full reproducibility, integrity audit, and scientific compliance.
"""

import hashlib
import json
import os
import datetime
import subprocess
import toml
import platform
from submission_validator_v50 import validate_submission


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def finalize_spectramind_toml(toml_path: str, git_hash: str, timestamp: str):
    if not os.path.exists(toml_path):
        print(f"⚠️ TOML file {toml_path} not found. Skipping TOML update.")
        return

    with open(toml_path, "r") as f:
        config = toml.load(f)

    config.setdefault("reproducibility", {})
    config["reproducibility"]["git_hash"] = git_hash
    config["reproducibility"]["timestamp"] = timestamp

    with open(toml_path, "w") as f:
        toml.dump(config, f)

    print(f"📝 Updated {toml_path} with Git hash and timestamp.")


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unversioned"


def finalize_submission():
    submission_file = "submission.csv"
    config_file = "config_v50.yaml"
    toml_file = "spectramind.toml"
    version_tag = "v50.0.0"

    if not os.path.exists(submission_file):
        raise FileNotFoundError("❌ submission.csv not found.")

    print("🔍 Validating submission...")
    validate_submission(submission_file)

    print("🔐 Computing SHA256 hash...")
    sha256 = compute_sha256(submission_file)

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    git_hash = get_git_hash()

    manifest = {
        "version": version_tag,
        "timestamp": timestamp,
        "submission_file": submission_file,
        "sha256": sha256,
        "author": "SpectraMind Team",
        "git_hash": git_hash,
        "generated_by": "v50_pipeline_finalizer.py",
        "python_version": platform.python_version(),
        "config_file": config_file if os.path.exists(config_file) else "missing",
    }

    print("🗂 Writing manifest_v50.json...")
    with open("manifest_v50.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("🗂 Writing run_hash_summary_v50.json...")
    with open("run_hash_summary_v50.json", "w") as f:
        json.dump({
            "submission_hash": sha256,
            "git_hash": git_hash,
            "timestamp": timestamp,
            "config_file": config_file,
            "generated_by": "pipeline_finalizer"
        }, f, indent=2)

    finalize_spectramind_toml(toml_file, git_hash, timestamp)

    print("✅ Finalization complete.\n"
          f"   - SHA256: {sha256}\n"
          f"   - Git: {git_hash}\n"
          f"   - Timestamp: {timestamp}\n"
          "   - Logs: manifest_v50.json, run_hash_summary_v50.json")


if __name__ == "__main__":
    finalize_submission()