#!/usr/bin/env bash
set -euo pipefail
pip install --upgrade pip wheel
pip install -r requirements.txt
echo "✅ Kaggle env ready."
