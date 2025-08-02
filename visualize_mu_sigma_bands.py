"""
SpectraMind V50 – Multi-Scale μ and σ Overlay with Symbolic Violations
-----------------------------------------------------------------------
Visualizes the μ output from multi-scale decoding (low/mid/high/fused),
the predicted σ from the uncertainty decoder, and symbolic violation overlays.

Helps correlate uncertainty with physical constraint violations and band structure.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from flow_uncertainty_head import FlowUncertaintyHead
from multi_scale_decoder import MultiScaleDecoder


def visualize_mu_sigma_fusion_overlay(
    latent_dim: int = 128,
    mu_dim: int = 283,
    outdir: str = "diagnostics",
    save_name: str = "mu_sigma_overlay.png",
    seed: int = 42
):
    """
    Generates joint overlay of μ (multi-band) and σ prediction across spectral bins.
    Overlays symbolic violation magnitude for interpretability.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    z = torch.randn(1, latent_dim)

    mu_decoder = MultiScaleDecoder(
        latent_dim=latent_dim,
        output_bins=mu_dim,
        return_diagnostics=True
    )

    sigma_decoder = FlowUncertaintyHead(
        latent_dim=latent_dim,
        mu_dim=mu_dim,
        hidden_dim=256,
        use_mu=True,
        residual_sigma=False,
        output_type="sigma",
        fusion_mode="none"
    )

    mu_outputs = mu_decoder(z)
    mu = mu_outputs["mu"]
    sigma = sigma_decoder(z=z, mu=mu)

    # Simulated symbolic violations (placeholder; replace with real symbolic_loss)
    symbolic_violation = (mu - mu.mean(dim=1, keepdim=True)).abs()

    # Convert to numpy for plotting
    mu_np = mu.squeeze().detach().cpu().numpy()
    mu_low = mu_outputs["mu_low"].squeeze().detach().cpu().numpy()
    mu_mid = mu_outputs["mu_mid"].squeeze().detach().cpu().numpy()
    mu_high = mu_outputs["mu_high"].squeeze().detach().cpu().numpy()
    sigma_np = sigma.squeeze().detach().cpu().numpy()
    symbolic_np = symbolic_violation.squeeze().detach().cpu().numpy()
    fusion_weights = mu_outputs["fusion_weights"].detach().cpu().numpy()

    x = np.arange(mu_dim)

    plt.figure(figsize=(14, 6))
    plt.plot(x, mu_np, label=f"μ final", linewidth=2.2)
    plt.plot(x, mu_low, label=f"μ low (w={fusion_weights[0]:.2f})", linestyle="--")
    plt.plot(x, mu_mid, label=f"μ mid (w={fusion_weights[1]:.2f})", linestyle="--")
    plt.plot(x, mu_high, label=f"μ high (w={fusion_weights[2]:.2f})", linestyle="--")
    plt.plot(x, sigma_np, label="σ prediction", color="orange", alpha=0.6)
    plt.fill_between(x, 0, symbolic_np, alpha=0.25, label="symbolic violations", color="red")

    plt.legend()
    plt.title("Multi-Scale μ Decoder and σ Prediction Overlay with Symbolic Violations")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Predicted Value")
    plt.tight_layout()

    path = os.path.join(outdir, save_name)
    plt.savefig(path)
    print(f"✅ Saved joint μ/σ + symbolic overlay plot to: {path}")


if __name__ == "__main__":
    visualize_mu_sigma_fusion_overlay()