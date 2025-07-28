# 🪐 SpectraMind V50 – NeurIPS 2025 Ariel Data Challenge

Welcome to **SpectraMind V50**, the complete scientific AI pipeline for the ESA + NeurIPS 2025 Ariel Data Challenge.  
Built with modular architecture, symbolic intelligence, diagnostic tooling, and full reproducibility — this repository delivers state-of-the-art exoplanet atmosphere recovery using multi-instrument detector data.

---

## 🎯 Mission

> Predict exoplanet transmission spectra (μ) and associated uncertainty (σ) from detector-level time series acquired by Ariel’s AIRS-CH0 and FGS1 instruments.

Models are evaluated using the **Gaussian Log-Likelihood (GLL)** metric over 283 spectral bins and compared against baseline and ideal models.

---

## 🚀 Pipeline Capabilities

| Feature                | Description |
|------------------------|-------------|
| 🔁 Dual Encoder         | Mamba SSM (FGS1) + Spectral GNN (AIRS) |
| 🔬 Multi-scale Decoder  | μ split into low, mid, high bands |
| 📉 Uncertainty Modeling | Flow-based σ head with Softplus activation |
| 🧠 Symbolic Constraints | Smoothness, non-negativity, molecular match |
| 🔍 Diagnostics          | FFT overlays, rule violations, QA dashboards |
| ⚙️ CLI Control          | Fully orchestrated via Typer + Hydra |
| 🔐 Reproducibility      | Manifest, TOML, config, SHA256 |
| 🛰 Challenge Compliance | Submission format, runtime, hash trail |

---

## 🗂 Project Structure

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
│   │   ├── photonic_alignment.py
│   │   └── symbolic_logic_engine.py
│   ├── diagnostics/
│   │   ├── fft_variance_heatmap.py
│   │   ├── violation_heatmap.py
│   │   ├── coherence_curve_plot.py
│   │   └── generate_diagnostic_summary.py
│   ├── explain/
│   │   └── shap_overlay.py
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
│   ├── submission.csv
│   ├── model.pt
│   └── v50_debug_log.md
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

## 🛠 Installation

```bash
# Poetry installation
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

> Python 3.10+, CUDA 12.1+, PyTorch ≥ 2.1 required

---

## 🧪 Usage

### 🧠 Train
```bash
poetry run python src/spectramind/training/train_v50.py
```

### 🔮 Predict
```bash
poetry run python src/spectramind/inference/predict_v50.py
```

### 📑 Generate Submission
```bash
poetry run python scripts/submission.py
```

### ✅ Validate GLL
```bash
poetry run python src/spectramind/evaluation/validate.py
```

### 🔧 Calibrate σ
```bash
poetry run python src/spectramind/utils/calibrate.py
```

### 📦 Package for Submission
```bash
poetry run python scripts/generate_submission_package.py
```

### 🔁 CLI Health Check
```bash
poetry run python src/spectramind/cli/selftest.py
```

---

## 📑 Submission Format

- 1 column: `planet_id`
- 283 columns: `mu_1` → `mu_283`
- 283 columns: `sigma_1` → `sigma_283`
- ✅ Total: 567 columns
- ✅ CSV output: `outputs/submission.csv`

---

## 🧬 Symbolic + Diagnostic Modules

- `symbolic_loss.py`: symbolic rule loss routing
- `photonic_alignment.py`: checks μ dips vs CH₄, H₂O, CO₂ templates
- `symbolic_logic_engine.py`: programmable rule logic execution
- `violation_heatmap.py`: visual overlay of broken constraints
- `fft_variance_heatmap.py`: FFT variance scoring per bin
- `coherence_curve_plot.py`: smoothness proxy of μ
- `generate_diagnostic_summary.py`: auto QA summary
- `shap_overlay.py`: SHAP × attention × symbolic fusion
- `v50_debug_log.md`: captures QA + runtime diagnostics

---

## 🔐 Reproducibility Infrastructure

| File | Purpose |
|------|---------|
| `spectramind.toml`         | Tracks version, modules, config paths |
| `manifest_v50.csv`         | SHA256 hash list of all tracked files |
| `run_hash_summary_v50.json`| Artifact tracking & hash recording |
| `constraint_violation_log.json` | Symbolic & physics violation logs |
| `outputs/v50_debug_log.md` | CLI + inference notes |
| `poetry.lock`              | Frozen Python environment |

---

## 🛰 NeurIPS 2025 Compliance

- ✅ 567-column `submission.csv` format
- ✅ GLL scoring with symbolic constraints
- ✅ GPU runtime < 9h (Kaggle A100-compatible)
- ✅ Manifest, TOML, diagnostic, config trace
- ✅ Self-validating submission tool

---

## 📜 License

MIT License © 2025 Andy Barta

---

## 🙌 Contributions

SpectraMind is designed for extensibility in astrophysics, symbolic AI, diagnostics, and model transparency.

We welcome:
- PRs with new symbolic modules
- Scientific constraint designs
- CLI tools and runtime testing
- Contributions to diagnostics or reproducibility tooling

---

Let the science begin.
