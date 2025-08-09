# SpectraMind V50 – NeurIPS 2025 Ariel Data Challenge

[![CI Pipeline](https://github.com/bartytime4life/SpectraMind/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/bartytime4life/SpectraMind/actions/workflows/ci_pipeline.yml)

Mission‑grade, symbolically constrained, scientifically explainable, fully auditable.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train → Predict → Package → Diagnostics
python -m spectramind.cli.main train --confirm
python -m spectramind.cli.main predict --confirm
python -m spectramind.cli.main submit --confirm
python -m spectramind.cli.main diagnose --confirm
```

Artifacts: `outputs/submission.csv`, `outputs/model.pt`, `constraint_violation_log.json`, `diagnostics/diagnostic_report_v50.html`, `v50_debug_log.md`.
