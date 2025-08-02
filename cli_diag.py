import typer
from pathlib import Path
import numpy as np
from diagnostics.constraint_violation_overlay import plot_violation_overlay

app = typer.Typer()

@app.command("overlay-violations")
def overlay_violations(
    mu_path: Path = typer.Option(..., help="Path to .npy or .npz file containing μ predictions"),
    violation_path: Path = typer.Option(..., help="Path to .npy or .npz file containing symbolic violations"),
    outdir: Path = typer.Option("outputs/diagnostics", help="Directory to save output visualizations"),
    normalize: bool = typer.Option(False, help="Normalize violations relative to μ"),
    csv: bool = typer.Option(True, help="Save overlay values as CSV")
):
    """
    Overlay symbolic violations on predicted μ spectrum.
    """
    def load_tensor(p):
        if p.suffix == ".npz":
            return np.load(p)["mu"] if "mu" in np.load(p) else list(np.load(p).values())[0]
        return np.load(p)

    mu = load_tensor(mu_path)
    vio = load_tensor(violation_path)

    plot_violation_overlay(mu, vio, outdir=str(outdir), normalize=normalize, save_csv=csv)