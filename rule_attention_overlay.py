"""
SpectraMind V50 – Latent Rule Attention Overlay
------------------------------------------------
Overlays symbolic rule influence onto latent space projections (UMAP/PCA/TSNE).
Visualizes how symbolic constraints affect latent representations.
Supports summary stats, normalized coloring, and customizable projection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import json
import os
from sklearn.preprocessing import StandardScaler

app = typer.Typer(help="Visualize symbolic rule attention in latent space")

@app.command()
def run_overlay(
    latent_file: Path = typer.Option("outputs/latents.npy", help="Latent embedding array (N, D)"),
    rule_mask_file: Path = typer.Option("outputs/rule_mask.json", help="Symbolic rule bin mask (planet_id → bin list)"),
    id_file: Path = typer.Option("outputs/planet_ids.txt", help="File with ordered planet IDs (one per line)"),
    projection: str = typer.Option("pca", help="Projection method: pca | tsne"),
    normalize: bool = typer.Option(True, help="Normalize latent vectors before projection"),
    out_file: Path = typer.Option("outputs/diagnostics/latent_rule_overlay.png", help="Path to save visualization"),
    save_csv: bool = typer.Option(True, help="Save 2D projected coordinates with rule influence")
):
    """
    Plots 2D projection of latent space, overlaying symbolic rule influence per point.
    """
    latents = np.load(latent_file)  # shape (N, D)
    with open(rule_mask_file) as f:
        rule_mask = json.load(f)
    ids = [line.strip() for line in open(id_file)]

    if latents.shape[0] != len(ids):
        raise ValueError("Latents and ID file must align in row count")

    # Create binary violation map
    violators = np.array([1 if pid in rule_mask and len(rule_mask[pid]) > 0 else 0 for pid in ids])

    if normalize:
        latents = StandardScaler().fit_transform(latents)

    if projection == "pca":
        reducer = PCA(n_components=2)
    elif projection == "tsne":
        reducer = TSNE(n_components=2, perplexity=30)
    else:
        raise ValueError("Projection must be: pca | tsne")

    coords = reducer.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=violators, cmap="coolwarm", alpha=0.7)
    plt.title("Latent Space with Symbolic Rule Overlay")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Symbolic Violation")
    os.makedirs(out_file.parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"✅ Saved overlay visualization to {out_file}")

    if save_csv:
        df = pd.DataFrame(coords, columns=["x", "y"])
        df["planet_id"] = ids
        df["rule_violation"] = violators
        df.to_csv(out_file.with_suffix(".csv"), index=False)
        print(f"📄 Saved projection CSV to {out_file.with_suffix('.csv')}")

if __name__ == "__main__":
    app()
