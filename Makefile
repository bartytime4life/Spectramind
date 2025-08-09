# SpectraMind V50 – Quick Commands Makefile
# -----------------------------------------
# Runs the Hydra-powered stubs:
#   - python -m spectramind.training.train_v50
#   - python -m spectramind.inference.predict_v50
#
# Pass Hydra overrides via OVERRIDES, e.g.:
#   make train OVERRIDES='training.scheduler.max_epochs=3 data.fgs1_path=/kaggle/input/fgs1'
#   make predict OVERRIDES='predict.export.submission_csv=outputs/submission.csv'
#
# Optional convenience variables you can set on the command line:
#   PY=python3.11   (default: python3)
#   CFG=configs/config_v50.yaml
#   SRC=src

SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# --- Parameters ---------------------------------------------------------------
PY            ?= python3
SRC           ?= src
CFG           ?= configs/config_v50.yaml
EXPORT_PYTHONPATH := $(abspath $(SRC))

# Expose src/ as a package root
export PYTHONPATH := $(EXPORT_PYTHONPATH)

# Hydra verbosity (keep full tracebacks on)
export HYDRA_FULL_ERROR := 1

# Optional user-provided overrides (Hydra style: key=value key2=value2 ...)
OVERRIDES ?=

# --- Commands ----------------------------------------------------------------
TRAIN_CMD := $(PY) -m spectramind.training.train_v50
PRED_CMD  := $(PY) -m spectramind.inference.predict_v50

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
	@echo "  make train                Run training stub with Hydra config"
	@echo "  make predict              Run inference stub and write submission.csv"
	@echo "  make cfg                  Print the active root config path"
	@echo "  make dirs                 Create common artifact directories"
	@echo "  make clean                Remove typical build/cache folders"
	@echo ""
	@echo "Overrides:"
	@echo "  OVERRIDES='key=value key2=value2'  # Hydra overrides"
	@echo "  PY=python3.11                       # choose Python"
	@echo "  CFG=configs/config_v50.yaml         # choose config"
	@echo ""
	@echo "Examples:"
	@echo "  make train OVERRIDES='training.scheduler.max_epochs=3'"
	@echo "  make predict OVERRIDES='predict.export.submission_csv=outputs/submission.csv'"
	@echo ""

.PHONY: train
train: dirs
	$(call banner,Running TRAIN with $(PY) and $(CFG))
	test -f "$(CFG)" || { echo "Config not found: $(CFG)"; exit 1; }
	$(TRAIN_CMD) $(OVERRIDES)

.PHONY: predict
predict: dirs
	$(call banner,Running PREDICT with $(PY) and $(CFG))
	test -f "$(CFG)" || { echo "Config not found: $(CFG)"; exit 1; }
	$(PRED_CMD) $(OVERRIDES)

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