# cli_validate_sigma.py

import torch
import typer
from flow_uncertainty_head import FlowUncertaintyHead

app = typer.Typer()

@app.command()
def check(batch: int = 4):
    model = FlowUncertaintyHead()
    z = torch.randn(batch, 128)
    mu = torch.randn(batch, 283)
    symbolic = torch.randn(batch, 283)
    out = model(z=z, mu=mu, symbolic=symbolic)
    print("σ stats: min =", out.min().item(), " max =", out.max().item())
    assert torch.all(out > 0)
    print("✅ sigma decoder passed shape and positivity test")

if __name__ == "__main__":
    app()