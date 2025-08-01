"""
SpectraMind V50 – Uncertainty Calibration (COREL + Temp + Logging)
-------------------------------------------------------------------
Applies temperature scaling and COREL GNN calibration to μ/σ.
Generates calibrated outputs, delta diagnostics, logs symbolic shifts,
and saves all to outputs/calibrated/.
"""

import torch
import os
from pathlib import Path
import numpy as np
import json
from corel_inference import load_corel_model, apply_corel


def temperature_scale(sigma_tensor: torch.Tensor, temperature: float = 1.2) -> torch.Tensor:
    return sigma_tensor * temperature

def compute_deltas(before: torch.Tensor, after: torch.Tensor) -> dict:
    delta = after - before
    return {
        "mean_delta": delta.mean().item(),
        "std_delta": delta.std().item(),
        "max_delta": delta.max().item(),
        "min_delta": delta.min().item()
    }

def run_calibration(
    mu_file="outputs/mu.pt",
    sigma_file="outputs/sigma.pt",
    edge_file="calibration_data/edge_index.pt",
    corel_weights="models/corel_gnn.pt",
    temperature=1.2,
    save_path="outputs/calibrated"
):
    os.makedirs(save_path, exist_ok=True)

    mu = torch.load(mu_file)
    sigma = torch.load(sigma_file)
    edge_index = torch.load(edge_file)

    print("⚙️  Applying temperature scaling to σ...")
    sigma_scaled = temperature_scale(sigma, temperature)
    torch.save(sigma_scaled, f"{save_path}/sigma_temp.pt")

    print("🧠 Running COREL calibration on μ and σ...")
    model = load_corel_model(corel_weights)
    mu_corel, sigma_corel = apply_corel(model, mu, sigma_scaled, edge_index)

    torch.save(mu_corel, f"{save_path}/mu_corel.pt")
    torch.save(sigma_corel, f"{save_path}/sigma_corel.pt")

    print("✅ Calibration complete. Results saved.")

    # Compute delta diagnostics
    mu_stats = compute_deltas(mu, mu_corel)
    sigma_stats = compute_deltas(sigma_scaled, sigma_corel)
    summary = {
        "temperature": temperature,
        "mu_delta_stats": mu_stats,
        "sigma_delta_stats": sigma_stats,
        "input_mu_file": str(mu_file),
        "input_sigma_file": str(sigma_file),
        "output_dir": str(save_path)
    }

    with open(Path(save_path) / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("v50_debug_log.md", "a") as log:
        log.write("\n### Calibration Summary\n")
        for k, v in summary.items():
            log.write(f"- {k}: {v}\n")

    print("📊 Calibration deltas:")
    print(json.dumps(summary, indent=2))

    return mu_corel, sigma_corel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--mu", default="outputs/mu.pt")
    parser.add_argument("--sigma", default="outputs/sigma.pt")
    parser.add_argument("--edge", default="calibration_data/edge_index.pt")
    parser.add_argument("--corel", default="models/corel_gnn.pt")
    parser.add_argument("--out", default="outputs/calibrated")
    args = parser.parse_args()

    run_calibration(
        mu_file=args.mu,
        sigma_file=args.sigma,
        edge_file=args.edge,
        corel_weights=args.corel,
        temperature=args.temperature,
        save_path=args.out
    )
