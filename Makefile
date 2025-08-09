PYTHON=python
CONFIG=configs/config_v50.yaml

train-all:
	$(PYTHON) src/spectramind/training/train_v50.py --config $(CONFIG)

predict:
	$(PYTHON) src/spectramind/inference/predict_v50.py --config $(CONFIG)

make-submission:
	$(PYTHON) scripts/generate_submission_package.py --config $(CONFIG)

diagnostics:
	$(PYTHON) src/spectramind/diagnostics/generate_diagnostic_summary.py

leaderboard:
	$(PYTHON) scripts/v50_pipeline_finalizer.py

clean:
	rm -rf outputs/* diagnostics/* *.zip *.pt *.npy
