import typer
import torch
import matplotlib.pyplot as plt
import numpy as np
from flow_uncertainty_head import FlowUncertaintyHead
import os

app = typer.Typer(help="SpectraMind V50 – σ Diagnostic CLI")

@app.command()
def overlay(
    latent_dim: int = 128,
    mu_dim: int = 283,
    seed: int = 42,
    outdir: str = "diagnostics",
    save_name: str = "sigma_symbolic_overlay.png"
):
    """Plot σ vs logits vs symbolic overlay for a random batch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    z = torch.randn(1, latent_dim)
    mu = torch.randn(1, mu_dim)
    symbolic = torch.rand(1, mu_dim)

    model = FlowUncertaintyHead(
        latent_dim=latent_dim,
        mu_dim=mu_dim,
        hidden_dim=256,
        use_mu=True,
        residual_sigma=False,
        output_type="sigma",
        fusion_mode="concat",
        symbolic_dim=mu_dim,
        dropout=0.1,
        layer_norm=True
    )

    model.eval()
    with torch.no_grad():
        diagnostics = model(z=z, mu=mu, symbolic=symbolic, return_diagnostics=True)

    sigma = diagnostics["sigma"].squeeze().cpu().numpy()
    logits = diagnostics["logits"].squeeze().cpu().numpy()
    symbolic_vals = diagnostics["symbolic"].squeeze().cpu().numpy()

    os.makedirs(outdir, exist_ok=True)
    x = np.arange(mu_dim)

    plt.figure(figsize=(12, 5))
    plt.plot(x, sigma, label="σ output", linewidth=2)
    plt.plot(x, logits, label="pre-softplus logits", alpha=0.7)
    plt.plot(x, symbolic_vals, label="symbolic overlay", linestyle="--", alpha=0.6)
    plt.legend()
    plt.title("FlowUncertaintyHead Output vs Symbolic Overlay")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Value")
    plt.tight_layout()
    outfile = os.path.join(outdir, save_name)
    plt.savefig(outfile)
    typer.echo(f"✅ Saved: {outfile}")

if __name__ == "__main__":
    app()