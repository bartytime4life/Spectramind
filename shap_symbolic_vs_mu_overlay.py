import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from flow_uncertainty_head import FlowUncertaintyHead
import os


def shap_compare_symbolic_vs_mu(
    B: int = 8,
    latent_dim: int = 128,
    mu_dim: int = 283,
    seed: int = 42,
    outdir: str = "diagnostics",
    symbolic_name: str = "shap_symbolic_vs_sigma.png",
    mu_name: str = "shap_mu_vs_sigma.png",
    max_display: int = 25
):
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

    # Wrapper for symbolic SHAP
    def wrapper_symbolic(symbolic_input):
        symbolic_input = symbolic_input.float()
        batch = symbolic_input.shape[0]
        return model(
            z=z[:batch],
            mu=mu[:batch],
            symbolic=symbolic_input
        )

    explainer_symbolic = shap.Explainer(wrapper_symbolic, symbolic)
    shap_vals_sym = explainer_symbolic(symbolic[:min(5, B)])

    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_vals_sym[:, :], max_display=max_display)
    plt.title("Symbolic Bin SHAP → σ")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, symbolic_name))

    # Wrapper for μ SHAP
    def wrapper_mu(mu_input):
        mu_input = mu_input.float()
        batch = mu_input.shape[0]
        return model(
            z=z[:batch],
            mu=mu_input,
            symbolic=symbolic[:batch]
        )

    explainer_mu = shap.Explainer(wrapper_mu, mu)
    shap_vals_mu = explainer_mu(mu[:min(5, B)])

    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_vals_mu[:, :], max_display=max_display)
    plt.title("μ SHAP → σ")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, mu_name))

    print(f"✅ SHAP comparisons saved to:")
    print(f"   • {os.path.join(outdir, symbolic_name)}")
    print(f"   • {os.path.join(outdir, mu_name)}")


if __name__ == "__main__":
    shap_compare_symbolic_vs_mu()