"""
SpectraMind V50 – CLI Diagnostics Suite
----------------------------------------
Runs violation overlays, SHAP diagnostics, and gallery renderings.
"""

import typer
import numpy as np
from pathlib import Path
from typing import Optional
from constraint_violation_overlay import batch_violation_overlay
from overlay_gallery import overlay_gallery

app = typer.Typer(help="SpectraMind V50 – Diagnostic Tools CLI")


@app.command("overlay-batch")
def overlay_batch(
    mu_path: Path = typer.Option(..., help="Path to .npy file of μ values (N, 283)"),
    violation_path: Path = typer.Option(..., help="Path to .npy file of symbolic violations (N, 283)"),
    planet_ids_path: Path = typer.Option(..., help="Path to .txt or .npy file of planet IDs (N,)"),
    shap_path: Optional[Path] = typer.Option(None, help="Optional SHAP array (N, 283)"),
    outdir: Path = typer.Option("outputs/diagnostics/planet_overlays", help="Output directory for plot files"),
    normalize: bool = typer.Option(False, help="Normalize violations and SHAP relative to μ"),
    save_csv: bool = typer.Option(True, help="Save .csv files with overlay values")
):
    """
    Batch overlay of μ + symbolic violations (and SHAP) across all planets.
    """
    typer.echo(f"📂 Loading: μ → {mu_path}, violations → {violation_path}, IDs → {planet_ids_path}")
    mu = np.load(mu_path)
    vio = np.load(violation_path)
    shap = np.load(shap_path) if shap_path else None

    if planet_ids_path.suffix == ".npy":
        ids = np.load(planet_ids_path).tolist()
    else:
        with open(planet_ids_path) as f:
            ids = [line.strip() for line in f if line.strip()]

    if len(ids) != len(mu):
        raise ValueError(f"❌ Mismatch: {len(ids)} planet IDs vs {len(mu)} μ rows.")

    batch_violation_overlay(
        mu_matrix=mu,
        violation_matrix=vio,
        planet_ids=ids,
        shap_matrix=shap,
        outdir=str(outdir),
        normalize=normalize,
        save_csv=save_csv
    )


@app.command("gallery")
def generate_gallery(
    overlay_dir: Path = typer.Option("outputs/diagnostics/planet_overlays", help="Directory containing *_violation_overlay.png"),
    save_path: Path = typer.Option("outputs/diagnostics/overlay_gallery.png", help="Output file for mosaic"),
    grid_cols: int = typer.Option(3, help="Columns in mosaic"),
    max_images: int = typer.Option(12, help="Max tiles to include"),
    image_width: float = typer.Option(4.0, help="Single plot width"),
    image_height: float = typer.Option(3.0, help="Single plot height")
):
    """
    Render a mosaic of individual planet overlay plots.
    """
    typer.echo(f"🧩 Rendering gallery from: {overlay_dir}")
    overlay_gallery(
        overlay_dir=str(overlay_dir),
        save_path=str(save_path),
        grid_cols=grid_cols,
        max_images=max_images,
        image_size=(image_width, image_height)
    )


if __name__ == "__main__":
    app()