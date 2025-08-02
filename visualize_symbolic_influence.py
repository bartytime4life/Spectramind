# visualize_symbolic_influence.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from flow_uncertainty_head import FlowUncertaintyHead


def visualize_sigma_with_symbolic():
    model = FlowUncertaintyHead()
    z = torch.randn(1, 128)
    mu = torch.randn(1, 283)
    symbolic = torch.rand(1, 283)

    out = model(z=z, mu=mu, symbolic=symbolic, return_diagnostics=True)
    sigma = out["sigma"].squeeze().cpu().numpy()
    logits = out["logits"].squeeze().cpu().numpy()
    symbolic = out["symbolic"].squeeze().cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(sigma, label="σ output", linewidth=2)
    plt.plot(logits, label="pre-softplus logits", alpha=0.7)
    plt.plot(symbolic, label="symbolic overlay", linestyle="--", alpha=0.6)
    plt.legend()
    plt.title("FlowUncertaintyHead Output vs Symbolic Overlay")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("diagnostics/sigma_symbolic_overlay.png")
    print("✅ Saved plot: diagnostics/sigma_symbolic_overlay.png")


if __name__ == "__main__":
    visualize_sigma_with_symbolic()