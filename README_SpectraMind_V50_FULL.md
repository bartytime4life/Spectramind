# 🪐 SpectraMind V50 – NeurIPS 2025 Ariel Data Challenge

Welcome to **SpectraMind V50**, a fully modular, CLI-driven scientific AI pipeline developed for the ESA + NeurIPS 2025 Ariel Data Challenge.

> **Mission:** Recover the transmission spectra (μ) and predictive uncertainty (σ) of exoplanet atmospheres using time-series detector data from Ariel’s AIRS-CH0 and FGS1 instruments.

This repository represents a top-down, engineering-grade architecture for scientific machine learning, symbolic reasoning, and astrophysical inference — designed for reproducibility, traceability, and scientific integrity.

---

## 🚀 Pipeline Capabilities

| Component         | Description |
|------------------|-------------|
| 🔁 Dual Encoder   | FGS1: Mamba SSM • AIRS: Spectral GNN with edge construction |
| 🔬 Multi-scale Decoder | Low / mid / high band μ prediction |
| 📉 Uncertainty Modeling | Softplus-constrained Flow-based σ estimator |
| 🧠 Symbolic Modules | Spectral basis checks, photonic alignment, FFT smoothness |
| ⚙️ CLI Orchestration | Full pipeline via Typer + Hydra + Poetry |
| 🧪 Calibration + QA | GLL scoring, σ temperature scaling, violation overlays |
| 📦 Reproducibility | Manifest + TOML + SHA256 + Submission ZIP builder |

---

## 🧱 Project Architecture

```
spectramind-v50/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── poetry.lock
├── spectramind.toml
├── manifest_v50.csv
├── run_hash_summary_v50.json
├── constraint_violation_log.json
├── configs/
│   ├── config.yaml
│   ├── science_constraints_v50.yaml
│   ├── photonic_basis.yaml
│   ├── fft_templates.yaml
├── src/spectramind/
│   ├── cli/
│   │   ├── cli_v50.py
│   │   ├── commands.py
│   │   └── selftest.py
│   ├── core/
│   │   ├── model_v50_ar.py
│   │   ├── multi_scale_decoder.py
│   │   └── flow_uncertainty_head.py
│   ├── models/
│   │   ├── fgs1_mamba.py
│   │   └── airs_gnn.py
│   ├── utils/
│   │   ├── gll_loss.py
│   │   ├── calibrate.py
│   │   └── dataloader.py
│   ├── symbolic/
│   │   ├── symbolic_loss.py
│   │   └── photonic_alignment.py
│   ├── diagnostics/
│   │   ├── fft_variance_heatmap.py
│   │   └── violation_heatmap.py
│   ├── training/
│   │   └── train_v50.py
│   ├── inference/
│   │   └── predict_v50.py
│   ├── evaluation/
│   │   └── validate.py
├── scripts/
│   ├── submission.py
│   ├── submission_validator_v50.py
│   ├── generate_submission_package.py
│   ├── v50_pipeline_finalizer.py
│   └── auto_ablate_v50.py
├── outputs/
│   ├── v50_debug_log.md
│   └── submission.csv
└── data/
    ├── train/
    │   ├── fgs1_tensor.npy
    │   ├── airs_tensor.npy
    │   ├── gt_mu.npy
    │   └── gt_sigma.npy
    └── test/
        ├── fgs1_tensor.npy
        └── airs_tensor.npy
```

---

## 🛠️ Installation

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

---

## 🧪 Usage

### Train
```bash
poetry run python src/spectramind/training/train_v50.py
```

### Predict
```bash
poetry run python src/spectramind/inference/predict_v50.py
```

### Submit
```bash
poetry run python scripts/submission.py
```

### Package
```bash
poetry run python scripts/generate_submission_package.py
```

### Validate + Calibrate
```bash
poetry run python src/spectramind/evaluation/validate.py
poetry run python src/spectramind/utils/calibrate.py
```

---

## 🔬 Scientific & Symbolic Tools

- `symbolic_loss.py`
- `photonic_alignment.py`
- `fft_variance_heatmap.py`
- `violation_heatmap.py`

---

## 📜 License

MIT License © 2025 Andy Barta

---

## 🧠 Contributions Welcome

Forks, extensions, symbolic logic proposals, GNN improvements, or scientific validation are all encouraged.

---

Let the science begin.
