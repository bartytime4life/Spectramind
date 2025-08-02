"""
SpectraMind V50 – Interactive t-SNE Latent Visualizer
-------------------------------------------------------
Generates a Plotly HTML t-SNE projection of latent space.

✅ Overlay symbolic clusters
✅ Supports --html-out for dashboard use
✅ Planet_id tooltips and color coding
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import plotly.express as px
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


def run_tsne(latents, perplexity=30.0, n_iter=1000, learning_rate='auto'):
    scaled = StandardScaler().fit_transform(latents)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate=learning_rate,
        random_state=42
    )
    return tsne.fit_transform(scaled)


def plot_interactive_tsne(embed, labels, planet_ids, out_html, overlay_label="Label"):
    df = pd.DataFrame(embed, columns=["TSNE-1", "TSNE-2"])
    df["label"] = labels
    df["planet_id"] = planet_ids

    fig = px.scatter(
        df,
        x="TSNE-1",
        y="TSNE-2",
        color="label",
        hover_data=["planet_id"],
        title="t-SNE of SpectraMind V50 Latents",
        template="plotly_white",
        height=700
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.3)))
    fig.write_html(out_html)
    print(f"🌐 Interactive t-SNE saved to: {out_html}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive t-SNE for V50 Latent Space")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--overlay_csv", type=str)
    parser.add_argument("--overlay_column", type=str, default="symbolic_class")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--learning_rate", type=str, default="auto")
    parser.add_argument("--html_out", type=str, default="diagnostics/tsne_latents.html")
    parser.add_argument("--tag", type=str, default="tsne_v50")
    args = parser.parse_args()

    # Load config
    import yaml
    cfg = yaml.safe_load(open(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, _ = load_dataset_from_config(cfg, split="train")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model = V50ArielModel(cfg["model_target"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    latents, planet_ids = extract_latents(model, loader, device)

    # Overlay support
    labels = planet_ids
    if args.overlay_csv:
        overlay_df = pd.read_csv(args.overlay_csv)
        overlay_map = dict(zip(overlay_df["planet_id"], overlay_df[args.overlay_column]))
        labels = [overlay_map.get(pid, "unlabeled") for pid in planet_ids]
        print(f"🎨 Using overlay column: {args.overlay_column}")

    # t-SNE + plot
    embed = run_tsne(latents, args.perplexity, args.n_iter, args.learning_rate)
    plot_interactive_tsne(embed, labels, planet_ids, args.html_out, overlay_label=args.overlay_column)


if __name__ == "__main__":
    main()