🪐 SpectraMind V50 – Scientific AI for the NeurIPS 2025 Ariel Data Challenge

SpectraMind V50 is a modular, reproducible, research-grade AI pipeline for exoplanetary atmosphere characterization, engineered for the ESA + NeurIPS 2025 Ariel Data Challenge.

This system performs full-stack inference from raw detector time-series data (AIRS-CH0 and FGS1) to final scientific predictions:
	•	Mean transit spectrum μ (ppm)
	•	Per-bin uncertainty σ (ppm)

SpectraMind V50 integrates deep learning, symbolic logic, physical constraints, and scientific diagnostics to extract latent planetary signals buried in noise, systematics, and stellar variability.

⸻

📡 Scientific & Mission Context
	•	Mission: ESA’s Ariel telescope (launching 2029) will observe transits of ~1,000 exoplanets.
	•	Goal: Recover molecular fingerprints and temperature profiles from infrared & visible spectra.
	•	Challenge: Planetary signals are weak (~10–150 ppm) and deeply entangled in non-Gaussian noise from optics, detectors, and host stars ￼.
	•	Solution: V50 applies machine learning models trained on high-fidelity simulations, augmented with physical priors and symbolic constraints.

⸻

🧪 Evaluation Metric: GLL (Gaussian Log-Likelihood)

For each planet p and wavelength bin i:

GLL_{p,i} = \log(\sigma_{p,i}) + \frac{(y_{p,i} - \mu_{p,i})^2}{2\sigma_{p,i}^2}

The total score is normalized:

Score = \frac{L_{ref} - L}{L_{ref} - L_{ideal}}

	•	L_ref: Mean + variance baseline
	•	L_ideal: Perfect prediction, 10 ppm σ for AIRS, 1 ppm for FGS1
	•	Weights: FGS1 gets 0.4; AIRS bins ≈ 0.0069 ￼

⸻

🚀 Core Pipeline Capabilities

Component	Description
🔁 Dual Encoder	Mamba SSM for FGS1, Spectral GNN for AIRS
🔬 Multi-scale Decoder	μ predicted in low/mid/high frequency bands
📉 Flow-based σ Head	Uncertainty modeling with invertible flows and Softplus activation
🧠 Symbolic Constraints	Smoothness, monotonicity, non-negativity, molecular absorption templates
🔬 Scientific Diagnostics	FFT variance, SHAP overlays, violation maps, symbolic explanation layers
⚙️ CLI Interface	Modular CLI powered by Typer + Hydra
🔐 Reproducibility	DVC + lakeFS, TOML + manifest, hash tracking
🛰 Submission Compliance	567-column CSV, ≤9h runtime, validator & hash integrity


⸻

🗂 Directory & File Structure

spectramind-v50/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml                # Poetry-managed Python project
├── poetry.lock
├── spectramind.toml             # Version info, metadata, reproducibility keys
├── manifest_v50.csv             # Full file registry and SHA256 hashes
├── run_hash_summary_v50.json    # Per-run hash and config log
├── constraint_violation_log.json
├── v50_debug_log.md             # Runtime anomaly notes
│
├── configs/                     # YAML config directory (Hydra-compatible)
│   ├── config.yaml
│   ├── science_constraints_v50.yaml
│   ├── photonic_basis.yaml
│   ├── fft_templates.yaml
│
├── src/spectramind/
│   ├── core/
│   │   ├── model_v50_ar.py
│   │   ├── multi_scale_decoder.py
│   │   ├── flow_uncertainty_head.py
│
│   ├── cli/
│   │   ├── cli_v50.py
│   │   ├── commands.py
│   │   ├── selftest.py
│
│   ├── models/
│   │   ├── fgs1_mamba.py
│   │   └── airs_gnn.py
│
│   ├── utils/
│   │   ├── calibrate.py
│   │   ├── gll_loss.py
│   │   └── dataloader.py
│
│   ├── symbolic/
│   │   ├── symbolic_loss.py
│   │   ├── symbolic_logic_engine.py
│   │   ├── photonic_alignment.py
│
│   ├── diagnostics/
│   │   ├── fft_variance_heatmap.py
│   │   ├── violation_heatmap.py
│   │   ├── coherence_curve_plot.py
│   │   └── generate_diagnostic_summary.py
│
│   ├── explain/
│   │   └── shap_overlay.py
│
│   ├── training/
│   │   └── train_v50.py
│
│   ├── inference/
│   │   └── predict_v50.py
│
│   ├── evaluation/
│   │   └── validate.py
│
├── scripts/
│   ├── submission.py
│   ├── submission_validator_v50.py
│   ├── generate_submission_package.py
│   ├── auto_ablate_v50.py
│   └── v50_pipeline_finalizer.py
│
├── outputs/
│   ├── model.pt
│   ├── submission.csv
│   └── logs/
│
└── data/
    ├── train/
    │   ├── fgs1_tensor.npy
    │   ├── airs_tensor.npy
    │   ├── gt_mu.npy
    │   └── gt_sigma.npy
    └── test/
        ├── fgs1_tensor.npy
        └── airs_tensor.npy


⸻

🛠 Installation

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone & install
git clone https://github.com/your-org/spectramind-v50.git
cd spectramind-v50
poetry install

Requires: Python 3.10+, CUDA 12.1+, PyTorch ≥ 2.1, Poetry, Git, PyEnv (optional)

⸻

🧪 CLI Usage Examples

# Train the model
poetry run python src/spectramind/training/train_v50.py

# Predict spectra
poetry run python src/spectramind/inference/predict_v50.py

# Validate GLL (on train or dev set)
poetry run python src/spectramind/evaluation/validate.py

# Calibrate uncertainties
poetry run python src/spectramind/utils/calibrate.py

# Generate 567-column submission
poetry run python scripts/submission.py

# Package for Kaggle
poetry run python scripts/generate_submission_package.py

# Self-test CLI
poetry run python src/spectramind/cli/selftest.py


⸻

🔬 Scientific Features
	•	Dual-Instrument Modeling:
	•	FGS1 (white-light): Mamba SSM for long-term trend learning
	•	AIRS-CH0 (infrared): GNN with spatial dispersion-aware edges
	•	Decoder Logic:
	•	Predicts μ via multi-resolution pathways
	•	σ from normalizing flows with uncertainty calibration
	•	Constraints:
	•	Physical smoothness
	•	Symbolic logic (e.g., CH₄ bands must not show negative μ)
	•	Molecular templates in photonic_basis.yaml
	•	Diagnostics Tools:
	•	FFT heatmaps of variance
	•	Constraint violation overlays
	•	SHAP + attention map fusion
	•	Reproducibility:
	•	Full manifest + config TOML
	•	DVC for large files
	•	lakeFS support (optional)
	•	Run hashes logged in run_hash_summary_v50.json

⸻

🧬 Symbolic Modules

Module	Function
symbolic_logic_engine	Core constraint interpreter
photonic_alignment	Spectral bin matcher against CH₄, CO₂, H₂O templates
symbolic_loss	Penalizes violations in smoothness, sign, molecular fit
fft_templates.yaml	Reference FFT shapes for emissions and absorption dips


⸻

📑 Submission Format

Each row in submission.csv:
	•	Column 0: planet_id
	•	Columns 1–283: mu_1 to mu_283
	•	Columns 284–566: sigma_1 to sigma_283

Total: 567 columns
Use submission_validator_v50.py to validate before upload.

⸻

📜 License

MIT License © 2025
Maintained by [Andy Barta / SpectraMind Research]

⸻

🙌 Contributing

We welcome PRs for:
	•	Additional symbolic rules
	•	New diagnostic tools (e.g., spectral event detection)
	•	Optimized inference heads
	•	UI & MLOps dashboards

⸻

🧠 Epilogue

SpectraMind V50 is not just a Kaggle submission. It is a scientific computing framework rooted in physics, logic, and modern AI. Whether used for challenge participation, academic research, or future mission preparation — its goal remains the same:

Reveal the unseen worlds orbiting distant stars.

Let the science begin. 🌌

⸻

If you would like a downloadable version, PDF export, GitHub upload, or want this README embedded into your pipeline, just say the word.