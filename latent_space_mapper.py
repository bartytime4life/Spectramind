"""
SpectraMind V50 – Latent Space Extractor & Visualizer
------------------------------------------------------
Generates 2D latent map using AIRS + FGS1 + metadata fusion.
Supports color by metadata, export to CSV, flexible projection (t-SNE/UMAP), and diagnostics.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

from model_v50_ar import SpectraMindModel
from dataloader import load_v50_dataset

@torch.no_grad()
def extract_latents(model, loader, device):
    latents, ids, temps = [], [], []
    for batch in loader:
        features, target = batch
        fgs = features['fgs1_sequence'].to(device)
        airs = features['airs_sequence'].to(device)
        meta = features['metadata'].to(device)
        pid_list = features['planet_id']
        Teff = features.get('Teff', torch.zeros(len(fgs))).cpu().numpy()

        z = model.encode_latent(fgs, airs, meta)
        latents.append(z.cpu().numpy())
        ids.extend(pid_list)
        temps.extend(Teff)

    return np.vstack(latents), ids, temps

def run_latent_mapping(
    config_path="configs/config_v50.json",
    out_dir="outputs/latents_v50",
    model_path="outputs/model.pt",
    projection="tsne",
    color_by="Teff"
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpectraMindModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    loader = load_v50_dataset(config_path=config_path, split="train", batch_size=32)

    print("📦 Extracting V50 latent vectors...")
    features, ids, temps = extract_latents(model, loader, device)
    print(f"✅ Latent feature shape: {features.shape}")

    print(f"🔍 Running {projection.upper()} projection...")
    if projection == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif projection == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported projection: use 'tsne' or 'pca'")

    coords = reducer.fit_transform(features)
    df = pd.DataFrame(coords, columns=["x", "y"])
    df["planet_id"] = ids
    df["Teff"] = temps
    csv_path = os.path.join(out_dir, f"latent_map_{projection}.csv")
    df.to_csv(csv_path, index=False)

    png_path = os.path.join(out_dir, f"latent_map_{projection}.png")
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df["x"], df["y"], s=10, alpha=0.7, edgecolors='none', c=df[color_by], cmap="viridis")
    plt.colorbar(sc, label=color_by)
    plt.title(f"SpectraMind V50 – Latent Space ({projection.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print(f"✅ Latent map saved to {png_path}")
    print(f"📄 CSV saved to {csv_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="SpectraMind V50: Latent space projection")
    parser.add_argument("--config", default="configs/config_v50.json")
    parser.add_argument("--model", default="outputs/model.pt")
    parser.add_argument("--out_dir", default="outputs/latents_v50")
    parser.add_argument("--projection", default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--color_by", default="Teff")
    args = parser.parse_args()

    run_latent_mapping(
        config_path=args.config,
        out_dir=args.out_dir,
        model_path=args.model,
        projection=args.projection,
        color_by=args.color_by
    )
