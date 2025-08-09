# SpectraMind V50 – Quick Commands Makefile (with Kaggle presets + verify + bootstrap)
# ------------------------------------------------------------------------
# Run Hydra-powered stubs:
#   python -m spectramind.training.train_v50
#   python -m spectramind.inference.predict_v50
#
# Kaggle presets:
#   make kaggle-verify
#   make kaggle-install
#   make kaggle-bootstrap
#   make kaggle-train
#   make kaggle-predict
#
# Pass Hydra overrides with:
#   OVERRIDES='key=val key2=val'
#
# Override Kaggle mount paths or batch/epoch sizes inline:
#   make kaggle-train KAGGLE_FGS1=/kaggle/input/my-fgs1 EPOCHS=3

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# --- Parameters ---------------------------------------------------------------
PY            ?= python3
SRC           ?= src
CFG           ?= configs/config_v50.yaml
EXPORT_PYTHONPATH := $(abspath $(SRC))

export PYTHONPATH := $(EXPORT_PYTHONPATH)
export HYDRA_FULL_ERROR := 1

OVERRIDES ?=

TRAIN_CMD := $(PY) -m spectramind.training.train_v50
PRED_CMD  := $(PY) -m spectramind.inference.predict_v50

# --- Kaggle Presets ----------------------------------------------------------
KAGGLE_FGS1 ?= /kaggle/input/fgs1
KAGGLE_AIRS ?= /kaggle/input/airs-ch0
KAGGLE_CAL  ?= /kaggle/input/calibration

KAGGLE_WORK ?= /kaggle/working
KAGGLE_SUB  ?= $(KAGGLE_WORK)/submission.csv

EPOCHS      ?= 5
BATCH       ?= 8
WORKERS     ?= 2

KAGGLE_DATA_OVERRIDES = \
  data.fgs1_path=$(KAGGLE_FGS1) \
  data.airs_path=$(KAGGLE_AIRS) \
  data.calibration_dir=$(KAGGLE_CAL) \
  data.batch_size=$(BATCH) \
  data.num_workers=$(WORKERS)

KAGGLE_TRAIN_OVERRIDES = \
  training.scheduler.max_epochs=$(EPOCHS)

KAGGLE_PRED_OVERRIDES = \
  predict.export.submission_csv=$(KAGGLE_SUB)

# --- Helpers -----------------------------------------------------------------
define banner
	@echo "================================================================"
	@echo " $(1)"
	@echo "================================================================"
endef

# --- Targets -----------------------------------------------------------------

.PHONY: help
help:
	@echo "SpectraMind V50 – Make targets"
	@echo ""
	@echo "  make train              Run training stub"
	@echo "  make predict            Run inference stub"
	@echo "  make kaggle-verify      Check Kaggle mount paths exist"
	@echo "  make kaggle-install     Install pinned deps + PyG CUDA wheels on Kaggle"
	@echo "  make kaggle-bootstrap   Full bootstrap: fetch repo → install → verify → train → predict"
	@echo "  make kaggle-train       Train with Kaggle paths/epochs preset"
	@echo "  make kaggle-predict     Predict with Kaggle paths/output preset"
	@echo "  make cfg                Show active config path"
	@echo "  make dirs               Create artifact directories"
	@echo "  make clean              Remove build/cache/temp outputs"
	@echo ""
	@echo "Variables:"
	@echo "  OVERRIDES='key=val key2=val'"
	@echo "  PY=python3.11"
	@echo "  CFG=configs/config_v50.yaml"
	@echo ""
	@echo "Kaggle defaults:"
	@echo "  KAGGLE_FGS1=$(KAGGLE_FGS1)"
	@echo "  KAGGLE_AIRS=$(KAGGLE_AIRS)"
	@echo "  KAGGLE_CAL=$(KAGGLE_CAL)"
	@echo "  KAGGLE_WORK=$(KAGGLE_WORK)"
	@echo "  KAGGLE_SUB=$(KAGGLE_SUB)"
	@echo "  EPOCHS=$(EPOCHS)  BATCH=$(BATCH)  WORKERS=$(WORKERS)"
	@echo ""

.PHONY: train
train: dirs
	$(call banner,Running TRAIN)
	test -f "$(CFG)" || { echo "Config not found: $(CFG)"; exit 1; }
	$(TRAIN_CMD) $(OVERRIDES)

.PHONY: predict
predict: dirs
	$(call banner,Running PREDICT)
	test -f "$(CFG)" || { echo "Config not found: $(CFG)"; exit 1; }
	$(PRED_CMD) $(OVERRIDES)

.PHONY: kaggle-verify
kaggle-verify:
	$(call banner,Verifying Kaggle mount paths)
	@if [ -d "$(KAGGLE_FGS1)" ]; then echo "✅ FGS1: $(KAGGLE_FGS1) exists"; else echo "❌ Missing: $(KAGGLE_FGS1)"; fi
	@if [ -d "$(KAGGLE_AIRS)" ]; then echo "✅ AIRS: $(KAGGLE_AIRS) exists"; else echo "❌ Missing: $(KAGGLE_AIRS)"; fi
	@if [ -d "$(KAGGLE_CAL)" ]; then echo "✅ CAL:  $(KAGGLE_CAL) exists"; else echo "❌ Missing: $(KAGGLE_CAL)"; fi
	@echo "💡 Adjust paths with: make kaggle-verify KAGGLE_FGS1=/new/path"

# --- Kaggle environment setup -----------------------------------------------
.PHONY: kaggle-install
kaggle-install:
	$(call banner,Setting up Kaggle Python environment)
	@if [ -f install_kaggle.sh ]; then \
		chmod +x install_kaggle.sh; \
		./install_kaggle.sh; \
	else \
		echo "install_kaggle.sh not found; running inline install instead..."; \
		pip install -r requirements.txt --no-cache-dir; \
		pip install --no-cache-dir \
		  torch-scatter==2.1.2+pt21cu121 \
		  torch-sparse==0.6.18+pt21cu121 \
		  torch-cluster==1.6.3+pt21cu121 \
		  torch-spline-conv==1.2.2+pt21cu121 \
		  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html; \
	fi
	@echo "✅ Kaggle environment ready."

.PHONY: kaggle-bootstrap
kaggle-bootstrap:
	$(call banner,Full Kaggle bootstrap: fetch repo → install deps → verify → train → predict)
	@if [ ! -f kaggle/bootstrap_cell.sh ]; then \
		echo "❌ kaggle/bootstrap_cell.sh not found. Please add it to your repo at kaggle/bootstrap_cell.sh"; \
		exit 1; \
	fi
	chmod +x kaggle/bootstrap_cell.sh
	./kaggle/bootstrap_cell.sh $(OVERRIDES)

.PHONY: kaggle-train
kaggle-train: kaggle-verify
	$(call banner,Kaggle TRAIN)
	test -f "$(CFG)" || { echo "Config not found: $(CFG)"; exit 1; }
	$(TRAIN_CMD) $(KAGGLE_DATA_OVERRIDES) $(KAGGLE_TRAIN_OVERRIDES) $(OVERRIDES)

.PHONY: kaggle-predict
kaggle-predict: kaggle-verify
	$(call banner,Kaggle PREDICT)
	test -f "$(CFG)" || { echo "Config not found: $(CFG)"; exit 1; }
	$(PRED_CMD) $(KAGGLE_DATA_OVERRIDES) $(KAGGLE_PRED_OVERRIDES) $(OVERRIDES)

.PHONY: cfg
cfg:
	@echo "$(CFG)"

.PHONY: dirs
dirs:
	@mkdir -p artifacts/checkpoints
	@mkdir -p artifacts/corel
	@mkdir -p diagnostics
	@mkdir -p outputs

.PHONY: clean
clean:
	@echo "Removing caches and temporary outputs..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@find $(SRC) -name "__pycache__" -type d -exec rm -rf {} +
	@rm -rf outputs/* diagnostics/* artifacts/checkpoints/* artifacts/corel/*