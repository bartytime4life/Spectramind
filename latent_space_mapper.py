"""
SpectraMind V50 – Latent Space Extractor & Cluster Visualizer
-------------------------------------------------------------
Projects latent space using t-SNE or PCA and overlays KMeans cluster labels or metadata colors.
Saves annotated CSV and visualization plot.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
    color_by="Teff",
    clusters: int = 0
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
    print(f"✅ Latent shape: {features.shape}")

    if projection == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif projection == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("projection must be 'tsne' or 'pca'")

    print(f"🔍 Running {projection.upper()} projection...")
    coords = reducer.fit_transform(features)

    df = pd.DataFrame(coords, columns=["x", "y"])
    df["planet_id"] = ids
    df["Teff"] = temps

    if clusters > 0:
        print(f"📊 Running KMeans clustering (k={clusters})...")
        km = KMeans(n_clusters=clusters, random_state=42)
        df["cluster_id"] = km.fit_predict(coords)
    else:
        df["cluster_id"] = -1

    # Save CSV
    csv_path = os.path.join(out_dir, f"latent_map_{projection}.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    png_path = os.path.join(out_dir, f"latent_map_{projection}.png")
    plt.figure(figsize=(10, 6))

    if clusters > 0:
        scatter = plt.scatter(df["x"], df["y"], c=df["cluster_id"], cmap="tab10", s=10, alpha=0.8)
        plt.colorbar(scatter, label="Cluster ID")
    else:
        scatter = plt.scatter(df["x"], df["y"], c=df[color_by], cmap="viridis", s=10, alpha=0.8)
        plt.colorbar(scatter, label=color_by)

    plt.title(f"SpectraMind V50 – Latent Map ({projection.upper()})")
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
    parser = argparse.ArgumentParser(description="SpectraMind V50 – Latent Space Projection")
    parser.add_argument("--config", default="configs/config_v50.json")
    parser.add_argument("--model", default="outputs/model.pt")
    parser.add_argument("--out_dir", default="outputs/latents_v50")
    parser.add_argument("--projection", default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--color_by", default="Teff")
    parser.add_argument("--clusters", type=int, default=0, help="Number of KMeans clusters (0 = disable)")
    args = parser.parse_args()

    run_latent_mapping(
        config_path=args.config,
        out_dir=args.out_dir,
        model_path=args.model,
        projection=args.projection,
        color_by=args.color_by,
        clusters=args.clusters
    )
