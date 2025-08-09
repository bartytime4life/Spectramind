#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

log() { echo "[$(date -u +%FT%TZ)] $*"; }
fail() { echo "❌ $*" >&2; exit 1; }

# ---------------------------
# 1) Defaults (overridable)
# ---------------------------
: "${USE_KAGGLE_DATASET:=true}"                      # true=copy from Kaggle Dataset; false=git clone
: "${REPO_DATASET:=bartytime4life/spectramind-v50-repo}"   # <owner>/<slug> of your repo dataset
: "${GIT_URL:=https://github.com/bartytime4life/SpectraMind.git}"
: "${BRANCH:=main}"

# Kaggle input mounts (override if your inputs differ)
: "${KAGGLE_FGS1:=/kaggle/input/fgs1}"
: "${KAGGLE_AIRS:=/kaggle/input/airs-ch0}"
: "${KAGGLE_CAL:=/kaggle/input/calibration}"

# Quick run knobs
: "${EPOCHS:=2}"
: "${BATCH:=8}"
: "${WORKERS:=2}"
: "${FAST_MODE:=false}"   # true => EPOCHS=1 BATCH=4 WORKERS=2

# Output
TS="$(date -u +%Y%m%d_%H%M%S)"
: "${SUB_CSV:=/kaggle/working/submission_${TS}.csv}"

# Allow CLI overrides as key=value pairs
for kv in "$@"; do
  if [[ "$kv" == *=* ]]; then
    # shellcheck disable=SC2163
    export "$kv"
  fi
done

if [[ "$FAST_MODE" == "true" ]]; then
  EPOCHS=1; BATCH=4; WORKERS=2
fi

# ---------------------------
# 2) Fetch repo
# ---------------------------
ROOT="/kaggle/working/SpectraMind"
mkdir -p /kaggle/working
if [[ "$USE_KAGGLE_DATASET" == "true" ]]; then
  SRC="/kaggle/input/${REPO_DATASET}"
  log "📦 Using Kaggle Dataset: ${REPO_DATASET}"
  if [[ ! -d "$SRC" ]]; then
    fail "Kaggle dataset not found at $SRC. Set REPO_DATASET=<owner>/<slug> or USE_KAGGLE_DATASET=false to git clone."
  fi
  rm -rf "$ROOT"
  cp -r "$SRC" "$ROOT" || true

  # Normalize if dataset extracted into nested folder
  if [[ ! -f "$ROOT/Makefile" ]]; then
    CAND=$(find "$ROOT" -maxdepth 1 -type d ! -path "$ROOT" | head -n 1 || true)
    if [[ -n "${CAND:-}" ]]; then
      log "📁 Normalizing nested dataset folder: $CAND"
      rm -rf "${ROOT}.tmp" && mv "$CAND" "${ROOT}.tmp"
      rm -rf "$ROOT" && mv "${ROOT}.tmp" "$ROOT"
    fi
  fi
else
  log "📦 Cloning from Git: $GIT_URL (branch: $BRANCH)"
  rm -rf "$ROOT"
  git clone -b "$BRANCH" --depth=1 "$GIT_URL" "$ROOT"
fi

cd "$ROOT"
log "📁 Repo root: $(pwd)"
ls -la | sed -n '1,80p'

# Make sure src/ is on PYTHONPATH for module runs
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
log "PYTHONPATH=${PYTHONPATH}"

# ---------------------------
# 3) Internet check
# ---------------------------
HAVE_NET=1
if ! python - <<'PY' >/dev/null 2>&1
import socket; 
s=socket.socket(); s.settimeout(3); 
try:
  s.connect(("pypi.org",443)); 
  print("ok")
finally:
  s.close()
PY
then
  HAVE_NET=0
  log "🌐 Internet OFF or blocked; will skip pip installs (assuming env already has deps)."
else
  log "🌐 Internet OK; pip installs enabled."
fi

# ---------------------------
# 4) Dependency install
#    - Prefer install_kaggle.sh if present
#    - Else pip install requirements + auto PyG CUDA wheels
# ---------------------------
install_pyg() {
  # Detect torch/cuda and choose correct PyG wheel index
  local torchv cudav major minor torch_tag url
  torchv="$(python - <<'PY'
try:
  import torch, re
  v=torch.__version__.split('+')[0]
  print(v)
except Exception:
  print("")
PY
)"
  cudav="$(python - <<'PY'
try:
  import torch
  print(torch.version.cuda or "cpu")
except Exception:
  print("cpu")
PY
)"
  if [[ -z "$torchv" ]]; then
    fail "PyTorch not found after install; cannot install torch-geometric wheels."
  fi

  major="${torchv%%.*}"            # '2'
  minor="$(echo "$torchv" | cut -d. -f2)" # '1'
  torch_tag="${major}.${minor}.0"  # e.g., 2.1.0 is what PyG indexes use

  # Map CUDA to tag
  local cutag="cpu"
  if [[ "$cudav" == "cpu" || -z "$cudav" ]]; then
    cutag="cpu"
  elif [[ "$cudav" == 12.* ]]; then
    # Most Kaggle GPUs with torch 2.1.x are cu121
    cutag="cu121"
  elif [[ "$cudav" == 11.8* || "$cudav" == 11.7* ]]; then
    cutag="cu118"
  else
    # Fallback guess: treat 12.x as cu121; else cu118
    if [[ "$cudav" == 12.* ]]; then cutag="cu121"; else cutag="cu118"; fi
  fi

  url="https://data.pyg.org/whl/torch-${torch_tag}+${cutag}.html"
  log "🔹 Installing PyG wheels for torch=${torchv}, cuda=${cudav} via: $url"

  # Only install if torch_geometric missing
  if ! python -c "import torch_geometric" >/dev/null 2>&1; then
    pip install --no-cache-dir \
      torch-scatter==2.1.2 \
      torch-sparse==0.6.18 \
      torch-cluster==1.6.3 \
      torch-spline-conv==1.2.2 \
      -f "$url"
    pip install --no-cache-dir torch-geometric==2.5.3
  else
    log "✅ torch-geometric already present; skipping."
  fi
}

if [[ -f "install_kaggle.sh" ]]; then
  if [[ $HAVE_NET -eq 1 ]]; then
    log "🛠 Running install_kaggle.sh"
    chmod +x install_kaggle.sh
    ./install_kaggle.sh
  else
    log "⚠️ Internet off; skipping install_kaggle.sh"
  fi
else
  if [[ $HAVE_NET -eq 1 ]]; then
    if [[ -f "requirements.txt" ]]; then
      log "🛠 Installing requirements.txt"
      pip install -r requirements.txt --no-cache-dir
    else
      log "ℹ️ No requirements.txt found; skipping base pip install."
    fi
    install_pyg
  else
    log "⚠️ Internet off; assuming deps preinstalled."
  fi
fi

# ---------------------------
# 5) Env checksum → v50_debug_log.md
# ---------------------------
python - <<'PY' >> v50_debug_log.md
from datetime import datetime
def ver(mod):
    try:
        m=__import__(mod); return getattr(m,'__version__','?')
    except Exception:
        return 'missing'
import json, platform, sys
info = {
    "ts_utc": datetime.utcnow().isoformat()+"Z",
    "python": sys.version.split()[0],
    "platform": f"{platform.system()} {platform.release()}",
    "torch": ver("torch"),
    "cuda_available": False,
    "cuda_version": None,
    "torch_geometric": ver("torch_geometric"),
    "numpy": ver("numpy"),
    "pandas": ver("pandas"),
    "scipy": ver("scipy"),
    "sklearn": ver("sklearn"),
}
try:
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
except Exception: pass
print(json.dumps(info))
PY
log "🧾 Appended env checksum to v50_debug_log.md"

# ---------------------------
# 6) Verify mounts (optional)
# ---------------------------
if [[ -f Makefile ]]; then
  make kaggle-verify KAGGLE_FGS1="$KAGGLE_FGS1" KAGGLE_AIRS="$KAGGLE_AIRS" KAGGLE_CAL="$KAGGLE_CAL" || true
else
  log "ℹ️ Makefile not found; skipping kaggle-verify."
fi

# ---------------------------
# 7) Train + Predict
# ---------------------------
if [[ -f Makefile ]]; then
  log "🚂 Training (epochs=${EPOCHS}, batch=${BATCH}, workers=${WORKERS})"
  make kaggle-train \
    KAGGLE_FGS1="$KAGGLE_FGS1" \
    KAGGLE_AIRS="$KAGGLE_AIRS" \
    KAGGLE_CAL="$KAGGLE_CAL" \
    EPOCHS="$EPOCHS" BATCH="$BATCH" WORKERS="$WORKERS"

  log "🔮 Predicting → ${SUB_CSV}"
  make kaggle-predict \
    KAGGLE_FGS1="$KAGGLE_FGS1" \
    KAGGLE_AIRS="$KAGGLE_AIRS" \
    KAGGLE_CAL="$KAGGLE_CAL" \
    KAGGLE_SUB="$SUB_CSV"
else
  log "ℹ️ Makefile not found; running modules directly."
  export HYDRA_FULL_ERROR=1
  python -m spectramind.training.train_v50 "training.scheduler.max_epochs=${EPOCHS}" || true
  python -m spectramind.inference.predict_v50 "predict.export.submission_csv=${SUB_CSV}" || true
fi

# Show a peek at the submission
if [[ -f "$SUB_CSV" ]]; then
  log "📄 Submission at: $SUB_CSV"
  head -n 2 "$SUB_CSV" || true
else
  log "⚠️ submission.csv not found at: $SUB_CSV"
fi

log "✅ Bootstrap complete."