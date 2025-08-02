import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from flow_uncertainty_head import FlowUncertaintyHead


def compute_shap_sigma_overlay():
    B, latent_dim, mu_dim = 10, 128, 283
    z = torch.randn(B, latent_dim)
    mu = torch.randn(B, mu_dim)
    symbolic = torch.randn(B, mu_dim)

    # Forward hook wrapper
    def model_fn(x):
        z_in = x[:, :latent_dim]
        mu_in = x[:, latent_dim:latent_dim+mu_dim]
        sym_in = x[:, latent_dim+mu_dim:]
        return model(z=z_in, mu=mu_in, symbolic=sym_in)

    model = FlowUncertaintyHead(
        latent_dim=latent_dim,
        mu_dim=mu_dim,
        hidden_dim=256,
        use_mu=True,
        residual_sigma=False,
        output_type="sigma",
        fusion_mode="concat",
        dropout=0.1,
        layer_norm=False
    )
    model.eval()

    # Concatenate into SHAP-ready input
    x_full = torch.cat([z, mu, symbolic], dim=1)

    explainer = shap.Explainer(model_fn, x_full)
    shap_values = explainer(x_full[:4])  # Small subset for speed

    # Plot summary for μ, symbolic contributions
    shap.plots.bar(shap_values[:, latent_dim:latent_dim+mu_dim], max_display=20)
    plt.title("SHAP Contribution from μ to σ")
    plt.tight_layout()
    plt.savefig("diagnostics/shap_mu_to_sigma.png")

    shap.plots.bar(shap_values[:, latent_dim+mu_dim:], max_display=20)
    plt.title("SHAP Contribution from Symbolic to σ")
    plt.tight_layout()
    plt.savefig("diagnostics/shap_symbolic_to_sigma.png")

    print("✅ Saved SHAP overlay plots for σ decoder")

if __name__ == "__main__":
    compute_shap_sigma_overlay()