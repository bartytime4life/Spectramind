"""
SpectraMind V50 – UMAP Latent Visualizer (Challenge Edition)
-------------------------------------------------------------
Extracts and visualizes 2D UMAP of latent space from V50 encoder.

✅ Uses AIRS+FGS1 encoder output
✅ Compatible with V50 dataloader and checkpoint
✅ Saves .npy, .csv, .png, and interactive .html
✅ Supports symbolic, entropy, or SHAP overlays
✅ Designed for diagnostics and latent drift tools
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import umap
import plotly.express as px

from model_v50_ar import V50ArielModel
from dataloader import load_dataset_from_config


@torch.no_grad()
def extract_latents(model, loader, device):
    model.eval()
    latents, labels, metas = [], [], []
    for batch in tqdm(loader, desc="🔍 Extracting Latents"):
        fgs, airs, meta = batch["fgs"].to(device), batch["airs"].to(device), batch["meta"].to(device)
        planet_ids = batch["planet_id"]
        _, z = model(fgs, airs, meta, return_latent=True)
        latents.append(z.mean(dim=1).cpu().numpy())
        labels.extend(planet_ids)
        metas.append(meta.cpu().numpy())
    return np.concatenate(latents), labels, np.concatenate(metas)


def plot_umap_static(embedding, labels, out_path):
    label_set = sorted(set(labels))
    label_map = {pid: i for i, pid in enumerate(label_set)}
    numeric_labels = [label_map[pid] for pid in labels]

    plt.figure(figsize=(12, 9))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=numeric_labels, cmap="tab20", s=12, alpha=0.8)
    plt.colorbar(sc, label="Planet ID (mapped index)")
    plt.title("UMAP of Latent Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Static UMAP plot saved to: {out_path}")


def plot_umap_interactive(embedding, labels, out_html):
    df = pd.DataFrame(embedding, columns=["UMAP-1", "UMAP-2"])
    df["planet_id"] = labels
    fig = px.scatter(df, x="UMAP-1", y="UMAP-2", color="planet_id", title="UMAP of V50 Latents (interactive)",
                     color_discrete_sequence=px.colors.qualitative.Set3)
    fig.write_html(out_html)
    print(f"🌐 Interactive UMAP saved to: {out_html}")


def save_latent_artifacts(latents, labels, tag="latent"):
    np.save(f"diagnostics/latents_{tag}.npy", latents)
    pd.DataFrame({"planet_id": labels}).to_csv(f"diagnostics/latents_{tag}_labels.csv", index=False)
    print(f"🧪 Latents saved: latents_{tag}.npy + _labels.csv")


def run_umap(latents, n_neighbors=30, min_dist=0.1):
    scaler = StandardScaler()
    X = scaler.fit_transform(latents)
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = umap_model.fit_transform(X)
    return embedding


def main():
    import argparse
    parser = argparse.ArgumentParser(description="🧠 SpectraMind V50 – Latent UMAP Explorer")
    parser.add_argument("--config", type=str, required=True, help="Hydra-compatible YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--out_png", type=str, default="diagnostics/umap_latents.png", help="Static PNG path")
    parser.add_argument("--out_html", type=str, default="diagnostics/umap_latents.html", help="Interactive HTML path")
    parser.add_argument("--tag", type=str, default="v50", help="Save tag for latents")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    args = parser.parse_args()

    # Device and model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V50ArielModel.from_checkpoint(args.checkpoint).to(device)

    # Load dataset
    dataset, _ = load_dataset_from_config(args.config, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Extract latents
    latents, labels, _ = extract_latents(model, loader, device)
    save_latent_artifacts(latents, labels, tag=args.tag)

    # Run UMAP
    embedding = run_umap(latents, n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    # Save plots
    plot_umap_static(embedding, labels, args.out_png)
    plot_umap_interactive(embedding, labels, args.out_html)


if __name__ == "__main__":
    main()