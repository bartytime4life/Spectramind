
⸻


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
	•	Challenge: Planetary signals are weak (~10–150 ppm) and deeply entangled in non-Gaussian noise from optics, detectors, and host stars .
	•	Solution: V50 applies machine learning models trained on high-fidelity simulations, augmented with physical priors and symbolic constraints.

⸻

🧪 Evaluation Metric: GLL (Gaussian Log-Likelihood)

For each planet p and wavelength bin i:

GLL_{p,i} = \log(\sigma_{p,i}) + \frac{(y_{p,i} - \mu_{p,i})^2}{2\sigma_{p,i}^2}

The total score is normalized:

Score = \frac{L_{ref} - L}{L_{ref} - L_{ideal}}

	•	L_ref: Mean + variance baseline
	•	L_ideal: Perfect prediction, 10 ppm σ for AIRS, 1 ppm for FGS1
	•	Weights: FGS1 gets 0.4; AIRS bins ≈ 0.0069 

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
├── pyproject.toml  
├── poetry.lock  
├── spectramind.toml                 # Project metadata + reproducibility hash  
├── manifest_v50.csv                 # All files, hashes, categories  
├── run_hash_summary_v50.json        # Hash summary of current run  
├── constraint_violation_log.json    # JSON log of symbolic constraint violations  
├── v50_debug_log.md                 # Developer notes and known issues  
│  
├── configs/                         # Hydra YAML configs  
│   ├── config.yaml  
│   ├── run/  
│   │   └── experiment.yaml  
│   ├── model/  
│   │   └── mamba_gnn.yaml  
│   ├── train/  
│   │   └── default.yaml  
│   ├── predict/  
│   │   └── submission.yaml  
│   ├── data/  
│   │   └── default.yaml  
│   ├── symbolic/  
│   │   └── full.yaml  
│   ├── ui/  
│   │   └── dashboard.yaml  
│   ├── science_constraints_v50.yaml  
│   ├── photonic_basis.yaml  
│   └── fft_templates.yaml  
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
│   │   ├── cli_dashboard_mini.py  
│   │   ├── cli_explain_util.py  
│   │   ├── execution_flow.py  
│   │   ├── error_humanizer.py  
│   │   └── selftest.py  
│  
│   ├── models/  
│   │   ├── fgs1_mamba.py  
│   │   ├── airs_gnn.py  
│   │   └── moe_decoder_head.py  
│  
│   ├── utils/  
│   │   ├── calibrate.py  
│   │   ├── gll_loss.py  
│   │   ├── dataloader.py  
│   │   ├── generate_html_report.py  
│   │   ├── generate_quantile_bands.py  
│   │   └── plot_quantiles_vs_target.py  
│  
│   ├── symbolic/  
│   │   ├── symbolic_logic_engine.py  
│   │   ├── symbolic_loss.py  
│   │   ├── photonic_alignment.py  
│   │   ├── symbolic_rule_scorer.py  
│   │   ├── symbolic_profile_switcher.py  
│   │   ├── symbolic_violation_predictor.py  
│   │   ├── auto_symbolic_rule_miner.py  
│   │   ├── neural_logic_graph.py  
│   │   └── symbolic_program_ensemble.py  
│  
│   ├── diagnostics/  
│   │   ├── fft_variance_heatmap.py  
│   │   ├── violation_heatmap.py  
│   │   ├── coherence_curve_plot.py  
│   │   ├── entropy_heatmap.py  
│   │   ├── latent_drift_overlay.py  
│   │   ├── anomaly_feedback_trainer.py  
│   │   └── generate_diagnostic_summary.py  
│  
│   ├── explain/  
│   │   ├── shap_overlay.py  
│   │   ├── shap_attention_overlay.py  
│   │   ├── latent_decomposer.py  
│   │   ├── symbolic_influence_map.py  
│   │   ├── latent_rule_attention_overlay.py  
│   │   └── posterior_explorer_dashboard.py  
│  
│   ├── simulators/  
│   │   ├── instrument_simulator.py  
│   │   ├── spectral_transfer_graph.py  
│   │   └── temporal_transit_simulator.py  
│  
│   ├── adaptation/  
│   │   ├── planet_memory_bank.py  
│   │   ├── hypercluster_adaptor.py  
│   │   └── planet_episode_summarizer.py  
│  
│   ├── training/  
│   │   └── train_v50.py  
│  
│   ├── inference/  
│   │   └── predict_v50.py  
│  
│   ├── evaluation/  
│   │   ├── validate.py  
│   │   ├── calibration_checker.py  
│   │   └── generate_uncertainty_report.py  
│  
├── scripts/  
│   ├── submission.py  
│   ├── submission_validator_v50.py  
│   ├── generate_submission_package.py  
│   ├── submission_diff_viewer.py  
│   ├── auto_ablate_v50.py  
│   └── v50_pipeline_finalizer.py  
│  
├── outputs/  
│   ├── submission.csv  
│   ├── logs/  
│   ├── model.pt  
│   ├── run_cfg.yaml  
│   └── diagnostics/  
│       ├── fft/  
│       ├── shap/  
│       └── html_report/  
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

```bash
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
photonic_alignment	Spectral bin matcher against CH₄, CO₂, H₂O
symbolic_loss	Penalizes violations in smoothness/sign/molecular
fft_templates.yaml	Reference FFT shapes for emission/absorption dips

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

---

Let me know if you'd like:
- This saved as an actual file you can download (`README.md`)
- A `.pdf` export
- Or pushed to a live repository structure.