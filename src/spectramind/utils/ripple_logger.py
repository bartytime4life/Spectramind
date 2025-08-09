import json, hashlib, os
from pathlib import Path
from datetime import datetime

def hash_file(path: str) -> str:
    p = Path(path)
    if not p.exists(): return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def update_run_hash_summary(command: str, config_path: str = "configs/config_v50.yaml"):
    p = Path("run_hash_summary_v50.json")
    obj = json.loads(p.read_text()) if p.exists() else {}
    cfg_hash = hash_file(config_path)
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "command": command,
        "config_hash": cfg_hash,
        "cwd": os.getcwd(),
    }
    obj.setdefault("runs", []).append(entry)
    p.write_text(json.dumps(obj, indent=2))
