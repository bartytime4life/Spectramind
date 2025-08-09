#!/bin/bash
set -e

echo "🔹 Installing core requirements from requirements.txt..."
pip install -r requirements.txt --no-cache-dir

echo "🔹 Installing PyTorch Geometric CUDA 12.1 wheels for PyTorch 2.1.x..."
pip install --no-cache-dir \
  torch-scatter==2.1.2+pt21cu121 \
  torch-sparse==0.6.18+pt21cu121 \
  torch-cluster==1.6.3+pt21cu121 \
  torch-spline-conv==1.2.2+pt21cu121 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo "✅ Kaggle environment ready."