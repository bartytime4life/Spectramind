"""
SpectraMind V50 – Transfer Graph Visualizer
--------------------------------------------
Plots latent → spectral bin correlation graph as a bipartite layout.
Color/width encodes edge weight. Symbolic violations and SHAP importance shown.
"""

import matplotlib.pyplot as plt
import networkx as nx
import json
from networkx.readwrite import json_graph
import pandas as pd
import os
import numpy as np
import matplotlib.colors as mcolors

def load_violation_scores(log_path="constraint_violation_log.json"):
    if not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        data = json.load(f)
    agg = {}
    for pid, info in data.items():
        for bin_idx, val in info.get("bin_scores", {}).items():
            key = f"mu_{bin_idx}"
            agg[key] = agg.get(key, 0.0) + val
    return agg

def load_shap_scores(shap_csv="outputs/symbolic_rule_scores.csv"):
    if not os.path.exists(shap_csv):
        return {}
    df = pd.read_csv(shap_csv)
    return {f"z_{int(row['latent_dim'])}": row["shap_score"] for _, row in df.iterrows()}

def get_node_colors(G, shap_scores, violation_scores):
    colors = []
    for node in G.nodes():
        if node.startswith("z_"):
            score = shap_scores.get(node, 0.0)
            colors.append(plt.cm.Greens(score))
        elif node.startswith("mu_"):
            v = violation_scores.get(node, 0.0)
            if v > 5:
                colors.append("red")
            elif v > 1:
                colors.append("yellow")
            else:
                colors.append("lightgreen")
        else:
            colors.append("gray")
    return colors

def plot_transfer_graph(json_path="outputs/z_to_mu_transfer_graph.json", 
                         output="outputs/transfer_graph.png",
                         shap_csv="outputs/symbolic_rule_scores.csv",
                         violation_json="constraint_violation_log.json",
                         html_embed_path="outputs/transfer_graph_snippet.html"):
    with open(json_path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)

    pos = {}
    z_nodes = sorted([n for n in G.nodes if n.startswith("z_")], key=lambda x: int(x.split("_")[1]))
    mu_nodes = sorted([n for n in G.nodes if n.startswith("mu_")], key=lambda x: int(x.split("_")[1]))

    for i, n in enumerate(z_nodes):
        pos[n] = (0, -i)
    for i, n in enumerate(mu_nodes):
        pos[n] = (1, -i)

    plt.figure(figsize=(14, 12))
    edge_weights = [abs(G[u][v]['weight']) for u, v in G.edges]

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_weights,
        edge_cmap=plt.cm.plasma,
        edge_vmin=0,
        edge_vmax=max(edge_weights),
        width=[2.5 * abs(G[u][v]['weight']) for u, v in G.edges]
    )

    shap_scores = load_shap_scores(shap_csv)
    violation_scores = load_violation_scores(violation_json)
    node_colors = get_node_colors(G, shap_scores, violation_scores)

    nx.draw_networkx_nodes(G, pos, node_size=60, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.title("Latent → μ Spectral Transfer Graph (SHAP + Symbolic Overlay)")
    plt.axis("off")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output)
    print(f"✅ Transfer graph saved to {output}")

    with open(html_embed_path, "w") as html:
        html.write("<h3>Latent → μ Spectral Transfer Graph</h3>\n")
        html.write("<p>Overlay of symbolic violations (μ nodes) and SHAP contributions (z nodes)</p>\n")
        html.write(f'<img src="{output}" style="max-width: 100%; border:1px solid #ccc"/>\n')
    print(f"📜 Embedded HTML snippet saved to {html_embed_path}")

if __name__ == "__main__":
    plot_transfer_graph()