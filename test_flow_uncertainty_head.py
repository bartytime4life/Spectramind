# test_flow_uncertainty_head.py

import torch
from flow_uncertainty_head import FlowUncertaintyHead


def test_all_modes():
    B, latent_dim, mu_dim = 2, 128, 283
    z = torch.randn(B, latent_dim)
    mu = torch.randn(B, mu_dim)
    symbolic = torch.randn(B, mu_dim)
    base_sigma = torch.rand(B, mu_dim)

    model = FlowUncertaintyHead(
        latent_dim=latent_dim,
        mu_dim=mu_dim,
        hidden_dim=256,
        use_mu=True,
        residual_sigma=True,
        output_type="sigma",
        fusion_mode="concat",
        symbolic_dim=mu_dim,
        dropout=0.1,
        layer_norm=True
    )

    model.eval()
    with torch.no_grad():
        out = model(z=z, mu=mu, symbolic=symbolic, base_sigma=base_sigma)
        diag = model(z=z, mu=mu, symbolic=symbolic, base_sigma=base_sigma, return_diagnostics=True)

    assert out.shape == (B, mu_dim)
    assert torch.all(out > 0)
    assert "sigma" in diag and diag["sigma"].shape == (B, mu_dim)
    assert "logits" in diag and diag["logits"].shape == (B, mu_dim)

    print("✅ All tests passed for FlowUncertaintyHead!")


if __name__ == "__main__":
    test_all_modes()