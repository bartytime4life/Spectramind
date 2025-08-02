"""
SpectraMind V50 – t-SNE Latent Visualizer
-----------------------------------------
Generates a 2D t-SNE projection of V50 latent space.

✅ Supports symbolic/class overlay
✅ Saves static .png and .npy
✅ Compatible with config_v50.yaml
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from model_v50_ar import V50ArielModel
from dataloader import load_dataset_from_config


@torch.no_grad()
def extract_latents(model, loader, device):
    model.eval()
    latents, labels = [], []
    for batch in tqdm(loader, desc="🔍 Extracting Latents"):
        fgs, airs, meta = batch["fgs"].to(device), batch["airs"].to(device), batch["meta"].to(device)
        planet_ids = batch["planet_id"]
        _, z = model(fgs, airs, meta, return_latent=True)
        latents.append(z.mean(dim=1).cpu().numpy())
        labels.extend(planet_ids)
    return np.concatenate(latents), labels


def plot_tsne(latents, labels, out_path, perplexity=30, n_iter=1000, learning_rate='auto'):
    print(f"🔍 Running t-SNE: perplexity={perplexity}, n_iter={n_iter}, learning_rate={learning_rate}")
    scaled = StandardScaler().fit_transform(latents)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                init='pca', learning_rate=learning_rate, random_state=42)
    embed = tsne.fit_transform(scaled)

    unique = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique)}
    numeric_labels = [label_map[l] for l in labels]

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(embed[:, 0], embed[:, 1], c=numeric_labels, cmap='tab20', s=10, alpha=0.8)
    plt.colorbar(scatter, label="Label (e.g. planet/class)")
    plt.title(f"t-SNE of V50 Latents (perp={perplexity}, n_iter={n_iter})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved t-SNE plot to: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SpectraMind V50 – t-SNE Latent Visualizer")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--out', type=str, default='diagnostics/tsne_latents.png', help='Output plot path')
    parser.add_argument('--overlay_csv', type=str, help='CSV with planet_id + overlay label')
    parser.add_argument('--overlay_column', type=str, default='symbolic_class', help='Overlay column')
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--learning_rate', type=str, default='auto')
    parser.add_argument('--tag', type=str, default="tsne_v50")
    args = parser.parse_args()

    # Load config
    import yaml
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"❌ Config not found: {args.config}")
    cfg = yaml.safe_load(open(args.config))

    # Load model + data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, _ = load_dataset_from_config(cfg, split="train")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = V50ArielModel(cfg["model_target"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Extract
    latents, planet_ids = extract_latents(model, loader, device)
    np.save(f"diagnostics/tsne_latents_{args.tag}.npy", latents)
    pd.DataFrame({"planet_id": planet_ids}).to_csv(f"diagnostics/tsne_labels_{args.tag}.csv", index=False)

    # Overlay support
    labels_to_plot = planet_ids
    if args.overlay_csv:
        overlay_df = pd.read_csv(args.overlay_csv)
        overlay_map = dict(zip(overlay_df["planet_id"], overlay_df[args.overlay_column]))
        labels_to_plot = [overlay_map.get(pid, "unlabeled") for pid in planet_ids]
        print(f"🎨 Overlaying t-SNE by: {args.overlay_column}")

    plot_tsne(latents, labels_to_plot, args.out, args.perplexity, args.n_iter, args.learning_rate)


if __name__ == "__main__":
    main()