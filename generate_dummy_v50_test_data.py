"""
SpectraMind V50 – Dummy Test Data Generator
-------------------------------------------
Creates synthetic test data for μ, σ, edge_index, COREL model, and submission.csv,
ensuring compatibility with the full V50 pipeline (train, inference, diagnostics, CLI).
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd

# --- Create directories ---
dirs = ["outputs", "models", "calibration_data", "diagnostics"]
for d in dirs:
    Path(d).mkdir(exist_ok=True)

# --- Dummy μ and σ ---
N = 4  # number of test samples
B = 283  # spectral bins

mu = torch.randn(N, B).clamp(min=0.0) * 1000  # ppm scale
sigma = torch.abs(torch.randn(N, B)) + 10.0   # positive σ with baseline

torch.save(mu, "outputs/mu.pt")
torch.save(sigma, "outputs/sigma.pt")

# --- Dummy y (true values) for diagnostics ---
y = mu + torch.randn_like(mu) * 0.5
torch.save(y, "outputs/y.pt")

# --- Dummy edge_index (linear graph) ---
edge_list = [[i, i+1] for i in range(B-1)] + [[i+1, i] for i in range(B-1)]
edge_index = torch.tensor(edge_list).T  # shape: [2, E]
torch.save(edge_index, "calibration_data/edge_index.pt")

# --- Dummy COREL model weights ---
corel_state_dict = {
    'conv1.weight': torch.randn(64, 1),
    'conv1.bias': torch.randn(64),
    'conv2.weight': torch.randn(64, 64),
    'conv2.bias': torch.randn(64),
    'out_mean.weight': torch.randn(1, 64),
    'out_mean.bias': torch.randn(1),
    'out_radius.weight': torch.randn(1, 64),
    'out_radius.bias': torch.randn(1)
}
torch.save(corel_state_dict, "models/corel_gnn.pt")

# --- Dummy submission.csv ---
planet_ids = [f"test_planet_{i}" for i in range(N)]
mu_np = mu.numpy()
sigma_np = sigma.numpy()
columns = ["planet_id"] + [f"mu_{i}" for i in range(B)] + [f"sigma_{i}" for i in range(B)]

rows = []
for i in range(N):
    row = [planet_ids[i]] + list(mu_np[i]) + list(sigma_np[i])
    rows.append(row)

df = pd.DataFrame(rows, columns=columns)
df.to_csv("submission.csv", index=False)

# --- Dummy overlay (symbolic class) file ---
symbolic_df = pd.DataFrame({
    "planet_id": planet_ids,
    "symbolic_class": ["water"] * N
})
symbolic_df.to_csv("diagnostics/symbolic_clusters.csv", index=False)

print("✅ Dummy V50 test data generated and saved.")