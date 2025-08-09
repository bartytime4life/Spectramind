from __future__ import annotations
from pathlib import Path
import sys

REQUIRED = [
    "configs/config_v50.yaml",
    "src/spectramind/model_v50_ar.py",
    "src/spectramind/training/train_v50.py",
    "src/spectramind/inference/predict_v50.py",
    "src/spectramind/diagnostics/generate_html_report.py",
]

def main() -> int:
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if missing:
        print("❌ Missing files:"); [print(" -", m) for m in missing]
        return 1
    print("✅ Selftest passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
