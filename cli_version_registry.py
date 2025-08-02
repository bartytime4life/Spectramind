"""
SpectraMind V50 – CLI Version Registry (Ultimate Version)
---------------------------------------------------------
Tracks CLI command executions, versions, config hashes, timestamps.
Provides:
- Run hash logging
- Config fingerprint (MD5)
- Auto-logging to v50_debug_log.md
- Export to JSON for diagnostics
- Called by all major CLI commands (train, predict, calibrate, submit)
"""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path


def compute_config_fingerprint(config_path: str) -> str:
    with open(config_path, 'rb') as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()


def log_cli_run(
    command: str,
    config_file: str,
    version: str = "v50",
    log_path: str = "outputs/logs/cli_version_log.json",
    debug_log_path: str = "v50_debug_log.md"
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fingerprint = compute_config_fingerprint(config_file)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "command": command,
        "version": version,
        "config": config_file,
        "fingerprint": fingerprint
    }

    # Append to JSON
    if Path(log_path).exists():
        with open(log_path, 'r') as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Append to debug log
    with open(debug_log_path, 'a') as f:
        f.write(f"\n### CLI Run: {command}\n")
        f.write(f"- Timestamp: {entry['timestamp']}\n")
        f.write(f"- Config: {config_file}\n")
        f.write(f"- Fingerprint: {fingerprint}\n")
        f.write(f"- Version: {version}\n")

    print(f"🧾 CLI run logged: {command} @ {entry['timestamp']}")


if __name__ == "__main__":
    # Example manual call
    log_cli_run("predict", "configs/predict/submission.yaml")