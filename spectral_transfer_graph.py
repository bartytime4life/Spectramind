"""
SpectraMind V50 – Spectral Transfer Graph
----------------------------------------
Models latent-to-spectrum signal flow using edge-weighted graphs.
Used for cross-instrument coherence, interpretability overlays,
or transfer-aware rule mining.
"""

import torch
import networkx as nx
from typing import Tuple, Union
import json
import os
from networkx.readwrite import json_graph

def build_transfer_graph(
    z: torch.Tensor,
    mu: torch.Tensor,
    threshold: float = 0.1,
    corr_type: str = "pearson"
) -> nx.DiGraph:
    """
    Computes correlation graph from latent z-dimensions to μ bins.
    Returns a directed graph with edge weights equal to correlation value.

    Args:
        z: (N, D) latent representation
        mu: (N, 283) predicted mean spectra
        threshold: abs(correlation) threshold to include edge
        corr_type: 'pearson' or 'cosine'

    Returns:
        nx.DiGraph with edges from z_i to mu_j
    """
    N, D = z.shape
    assert mu.shape[0] == N and mu.shape[1] == 283, "mu shape must be (N, 283)"

    G = nx.DiGraph()
    zc = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-6)
    muc = (mu - mu.mean(dim=0)) / (mu.std(dim=0) + 1e-6)

    for i in range(D):
        for j in range(283):
            if corr_type == "pearson":
                corr = torch.dot(zc[:, i], muc[:, j]) / N
            elif corr_type == "cosine":
                corr = torch.nn.functional.cosine_similarity(z[:, i], mu[:, j], dim=0)
            else:
                raise ValueError("corr_type must be 'pearson' or 'cosine'")

            weight = corr.item()
            if abs(weight) >= threshold:
                G.add_edge(f"z_{i}", f"mu_{j}", weight=weight)

    return G

def extract_important_latents(G: nx.DiGraph, threshold: float = 0.5) -> Tuple[int]:
    """
    Extracts z-dim indices with at least one edge to a μ bin above threshold.

    Returns:
        Tuple of int indices for latent dimensions
    """
    return tuple(sorted({int(u.split("_")[1]) for u, v, d in G.edges(data=True)
                         if u.startswith("z_") and abs(d['weight']) >= threshold}))

def save_graph_json(G: nx.DiGraph, path: str = "outputs/z_to_mu_transfer_graph.json"):
    """
    Saves graph as node-link JSON for visualization.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json_graph.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\u2705 Saved latent→μ graph to {path}")

if __name__ == "__main__":
    torch.manual_seed(42)
    z = torch.randn(64, 12)
    mu = torch.randn(64, 283)
    G = build_transfer_graph(z, mu, threshold=0.15)
    save_graph_json(G)
    important_z = extract_important_latents(G, threshold=0.4)
    print("Important z dims:", important_z)