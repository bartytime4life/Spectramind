# 🪐 SpectraMind V50 – NeurIPS 2025 Ariel Data Challenge

Welcome to the official repository for **SpectraMind V50**, an AI system designed to solve the ESA + NeurIPS 2025 Ariel Data Challenge. This version leverages hybrid symbolic + neural reasoning, dual-instrument fusion (FGS1 + AIRS), and rigorous uncertainty modeling.

---

## 🚀 Mission

Predict the **exoplanet atmospheric transmission spectra** (μ) and corresponding uncertainties (σ) from time-series imagery captured by ESA's Ariel instruments (FGS1 and AIRS-CH0).  

Submissions are evaluated using a **Gaussian Log-Likelihood (GLL)**-based score. Our pipeline is optimized for scientific fidelity, runtime constraints, and full reproducibility.

---

## 🧠 Core Features

| Component         | Description |
|------------------|-------------|
| 🔁 Dual Encoder   | `FGS1` via Mamba SSM and `AIRS` via Graph Attention GNN |
| 🔬 Decoder        | Multi-scale μ head with 3 spectral bands |
| 📉 Uncertainty    | Flow-based σ estimation with Softplus constraint |
| ⚙️ CLI            | Unified `typer` CLI with Hydra config overrides |
| 📦 Packaging      | Manifest + hash-based ZIP builder for submission |
| 🔍 Diagnostics    | GLL validation, temperature calibration |
| 🧪 Reproducibility| Poetry + DVC + config hashing + version pinning |

---

## 📁 Project Structure

```
spectramind-v50/
├── README.md
├── pyproject.toml
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
│   ├── core/
│   ├── diagnostics/
│   ├── models/
│   ├── utils/
│   ├── training/
├── scripts/
│   ├── train_v50.py
│   ├── predict_v50.py
│   ├── submission.py
│   ├── generate_submission_package.py
├── outputs/
│   └── v50_debug_log.md
└── data/  # managed via DVC/lakeFS
```

---

## 🛠️ Setup

```bash
# Install dependencies
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

> Requires: Python 3.10+, PyTorch 2.x, CUDA (optional for GPU)

---

## 🔧 Training

```bash
poetry run python scripts/train_v50.py
```

This will train the model using configs from `configs/config.yaml` and save `model.pt` to `outputs/`.

---

## 🧠 Inference

```bash
poetry run python scripts/predict_v50.py
```

Generates predicted `μ` and `σ` arrays for test planets.

---

## 📝 Submission File

```bash
poetry run python scripts/submission.py
```

Builds a CSV file with:
- 1 column: `planet_id`
- 283 columns: `mu_1` through `mu_283`
- 283 columns: `sigma_1` through `sigma_283`

---

## 📦 Packaging

```bash
poetry run python scripts/generate_submission_package.py
```

This script:
- Bundles `model.pt`, `submission.csv`, `config.yaml`
- Computes their SHA256 hashes
- Writes `run_hash_summary_v50.json`
- Creates a final `submission_bundle.zip`

---

## 🧪 Validation + Calibration

```bash
# Validate GLL score on a held-out validation set
poetry run python scripts/validate.py

# Calibrate σ using temperature scaling
poetry run python scripts/calibrate.py
```

---

## 🧬 Scientific Enhancements (Optional Modules)

- Spectral basis logic: `photonic_basis.yaml`
- Smoothness constraints via FFT
- Symbolic QA overlay tools
- Violation heatmaps, entropy maps, FFT overlays

---

## 🛰 Background

- Ariel is an ESA mission to study the atmospheres of 1,000+ exoplanets via transmission spectroscopy
- The NeurIPS 2025 challenge simulates this scenario using high-dimensional time-series detector data
- μ = mean transit depth; σ = predictive uncertainty
- GLL = log-likelihood metric comparing μ, σ to hidden ground truth

---

## 🔐 Reproducibility

- `spectramind.toml`: declares pipeline components + dependencies
- `manifest_v50.csv`: lists hashes of all critical files
- `run_hash_summary_v50.json`: snapshot of model+config+output for publication or competition
- `poetry.lock`: deterministic Python env

---

## 📜 License

[MIT License](./LICENSE)

---

## 🏁 Challenge Compliance

- ✅ Runs in <9h on Kaggle GPU
- ✅ Output: 567-column `submission.csv`
- ✅ Fully hashed & versioned
- ✅ Scientific, modular, inspectable

---

## ✨ Authors & Acknowledgements

Developed by **Andy Barta** and team, building upon the foundations of the ESA Ariel Mission and NeurIPS Exoplanet Research Community.

---

For questions, collaboration, or to build on V50+, contact us or fork this repo. We welcome scientific extension, symbolic reasoning, and anomaly detection collaboration.
