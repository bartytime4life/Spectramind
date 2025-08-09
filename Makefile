.PHONY: train predict dashboard selftest kaggle-export

train:
	python -m spectramind train --epochs 2 --lr 3e-4

predict:
	python -m spectramind predict --out-csv outputs/submission.csv

dashboard:
	python -m spectramind dashboard --html outputs/diagnostics/diagnostic_report_v50.html

selftest:
	python -m src.spectramind.cli.selftest

kaggle-export:
	zip -r outputs/kaggle_bundle.zip . -x "*.git*" -x "*.dvc*" -x "__pycache__/*"
