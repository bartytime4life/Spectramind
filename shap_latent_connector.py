import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from flow_uncertainty_head import FlowUncertaintyHead
import os


def shap_connect_symbolic_to_latent(
    B: int = 10,
    latent_dim: int = 128,
    mu_dim: int = 283,
    seed: int = 42,
    outdir: str = "diagnostics",
    save_name: str = "shap_symbolic_to_sigma_latent.png",
    max_display: int = 25
):
    """
    Computes SHAP contribution from symbolic bins to σ predictions.
    Saves a diagnostic bar plot of symbolic bin importance.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    z = torch.randn(B, latent_dim)
    mu = torch.randn(B, mu_dim)
    symbolic = torch.randn(B, mu_dim)

    model = FlowUncertaintyHead(
        latent_dim=latent_dim,
        mu_dim=mu_dim,
        hidden_dim=256,
        use_mu=True,
        residual_sigma=False,
        output_type="sigma",
        fusion_mode="concat",
        dropout=0.1,
        layer_norm=True
    )
    model.eval()

    # Only vary symbolic while holding z and mu fixed
    def model_wrapper(symbolic_input):
        symbolic_input = symbolic_input.float()
        batch = symbolic_input.shape[0]
        return model(
            z=z[:batch],
            mu=mu[:batch],
            symbolic=symbolic_input
        )

    explainer = shap.Explainer(model_wrapper, symbolic)
    shap_values = explainer(symbolic[:min(5, B)])

    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_values[:, :], max_display=max_display)
    plt.title("Symbolic Bin SHAP Influence on σ (Latent Pathway)")
    plt.tight_layout()
    filepath = os.path.join(outdir, save_name)
    plt.savefig(filepath)
    print(f"✅ SHAP symbolic→σ latent overlay saved to: {filepath}")


if __name__ == "__main__":
    shap_connect_symbolic_to_latent()