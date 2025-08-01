SpectraMind V50 – CLI Help Documentation
This guide documents all available CLI commands in cli_v50.py for the SpectraMind V50 pipeline. Use python cli_v50.py COMMAND --help to view usage options.

🔧 Training & Calibration
train


Trains the model using YAML config.


--config: Path to config YAML file (default: configs/config_v50.yaml)


--decoder: moe, diffusion, or quantile


--auto-package: Whether to package after training (default: True)


retrain-symbolic


Retrains model using symbolic constraint violation feedback.


master-train


Runs unified training pipeline (static + temporal) and packages results.


calibrate


Runs full calibration pipeline on a specified planet.


planet_id: Planet identifier



📈 Inference & Diagnostics
inference


Runs full model inference on test data.


diagnose


Generates symbolic, SHAP, and residual diagnostic overlays.


diagnostics-html


Re-runs symbolic scoring and generates a full HTML diagnostic report.


html-report


Regenerates HTML diagnostics from current outputs.


explain


Re-runs diagnostic summary only (no plots or packaging).


simulate-lightcurve


Visualizes a transit lightcurve from μ + metadata CSVs.


rule-attention-overlay


Overlays symbolic rule activation map with decoder attention.



📦 Submission Packaging
submit


Creates a finalized ZIP package using all diagnostics and reports.


export


Same as submit; re-runs ZIP creation.


validate


Validates submission.csv formatting and scientific structure.


compare


Compares two submission.csv files side-by-side.



🧪 System Testing & Validation
selftest


Runs basic CLI smoke test (validate, export, diagnose, explain).


selftest-workflow


Executes full pipeline test: train → inference → conformalize → submit → validate.


health


Full system check (CLI registry, symbolic modules, DVC files, hash logs).



🛰 Scientific Tools
transfer-graph


Builds latent → μ transfer graph and saves as JSON.


plot-transfer-graph


Visualizes the transfer graph as a bipartite layout.


conformalize


Applies COREL GNN to refine μ and σ predictions.


Arguments:


--model-path


--mu-file


--sigma-file


--edge-file



🔍 Miscellaneous
version


Prints current CLI version.




