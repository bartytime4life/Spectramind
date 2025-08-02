"""
SpectraMind V50 – UMAP Latent Visualizer
----------------------------------------
Extracts latent vectors from model encoder and visualizes 2D UMAP.

✅ Compatible with model_v50_ar
✅ Saves static .png and interactive .html
✅ Accepts overlay labels (e.g. symbolic clusters)
✅ Saves .npy and .csv for other tools
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    latents, planet_ids = [], []
    for batch in tqdm(loader, desc="🔍 Extracting Latents"):
        fgs, airs, meta = batch["fgs"].to(device), batch["airs"].to(device), batch["meta"].to(device)
        planet_id_batch = batch["planet_id"]
        _, z = model(fgs, airs, meta, return_latent=True)
        latents.append(z.mean(dim=1).cpu().numpy())
        planet_ids.extend(planet_id_batch)
    return np.concatenate(latents), planet_ids


def plot_umap_static(embedding, labels, out_path):
    unique = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique)}
    numeric_labels = [label_map[l] for l in labels]

    plt.figure(figsize=(12, 8))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=numeric_labels, cmap="tab20", s=10, alpha=0.8)
    plt.colorbar(sc, label="Label index")
    plt.title("UMAP of Latent Embeddings (Static)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Static UMAP saved to: {out_path}")


def plot_umap_interactive(embedding, labels, out_html, overlay_label="Label"):
    df = pd.DataFrame(embedding, columns=["UMAP-1", "UMAP-2"])
    df["label"] = labels
    fig = px.scatter(df, x="UMAP-1", y="UMAP-2", color="label",
                     hover_name="label", opacity=0.75, template="plotly_white",
                     title="UMAP of SpectraMind V50 Latents")
    fig.update_traces(marker=dict(size=6))
    fig.write_html(out_html)
    print(f"🌐 Interactive UMAP saved to: {out_html}")


def run_umap(latents, n_neighbors=30, min_dist=0.1):
    scaled = StandardScaler().fit_transform(latents)
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit_transform(scaled)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UMAP Latent Visualizer (V50)")
    parser.add_argument("--config", type=str, required=True, help="Hydra YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--tag", type=str, default="v50", help="Prefix for saved outputs")
    parser.add_argument("--out_png", type=str, default="diagnostics/umap_latents.png")
    parser.add_argument("--out_html", type=str, default="diagnostics/umap_latents.html")
    parser.add_argument("--overlay_csv", type=str, help="CSV with planet_id and label")
    parser.add_argument("--overlay_column", type=str, default="symbolic_class")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V50ArielModel.from_checkpoint(args.checkpoint).to(device)

    dataset, _ = load_dataset_from_config(args.config, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    latents, planet_ids = extract_latents(model, loader, device)

    # Save for other modules
    np.save(f"diagnostics/latents_{args.tag}.npy", latents)
    pd.DataFrame({"planet_id": planet_ids}).to_csv(f"diagnostics/latents_{args.tag}_labels.csv", index=False)

    # Overlay support
    labels_to_plot = planet_ids
    if args.overlay_csv and os.path.exists(args.overlay_csv):
        overlay_df = pd.read_csv(args.overlay_csv)
        overlay_map = dict(zip(overlay_df["planet_id"], overlay_df[args.overlay_column]))
        labels_to_plot = [overlay_map.get(pid, "unlabeled") for pid in planet_ids]
        print(f"🎨 Using overlay column: {args.overlay_column}")

    # UMAP
    embedding = run_umap(latents, args.n_neighbors, args.min_dist)
    plot_umap_static(embedding, labels_to_plot, args.out_png)
    plot_umap_interactive(embedding, labels_to_plot, args.out_html, overlay_label=args.overlay_column)


if __name__ == "__main__":
    main()