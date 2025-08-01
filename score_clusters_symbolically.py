"""
SpectraMind V50 – Cluster-Wise Symbolic Score Analyzer
-------------------------------------------------------
Computes mean symbolic constraint scores across each cluster.
Generates per-cluster summaries and diagnostic tables.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import typer

from symbolic_loss import compute_symbolic_losses
import torch

app = typer.Typer(help="Score symbolic constraints cluster-wise")

@app.command()
def score_by_cluster(
    submission_csv: Path = typer.Option("submission.csv", help="Predicted μ values"),
    cluster_csv: Path = typer.Option(..., help="CSV with planet_id and cluster_id"),
    output_dir: Path = typer.Option("outputs/diagnostics/cluster_symbolic_scores", help="Output directory")
):
    os.makedirs(output_dir, exist_ok=True)

    sub_df = pd.read_csv(submission_csv)
    cluster_df = pd.read_csv(cluster_csv)
    sub_df = sub_df.set_index("planet_id")
    cluster_df = cluster_df.set_index("planet_id")

    symbolic_config = {
        "smoothness": True,
        "nonnegativity": True,
        "variance_shaping": True,
        "enable_fft": False,
        "enable_asymmetry": False,
        "enable_photonic_templates": False
    }

    cluster_scores = defaultdict(list)

    for pid in cluster_df.index:
        if pid not in sub_df.index:
            continue
        mu = torch.tensor(sub_df.loc[pid].values[:283], dtype=torch.float32).unsqueeze(0)  # (1, 283)
        loss_dict = compute_symbolic_losses(mu, symbolic_config)
        loss_np = {k: float(v) for k, v in loss_dict.items()}
        cluster_id = cluster_df.loc[pid, "cluster_id"]
        cluster_scores[cluster_id].append(loss_np)

    # Aggregate
    summary = {}
    for cluster_id, score_list in cluster_scores.items():
        agg = defaultdict(list)
        for row in score_list:
            for k, v in row.items():
                agg[k].append(v)
        mean_scores = {k: float(np.mean(v)) for k, v in agg.items()}
        summary[cluster_id] = {
            "num_planets": len(score_list),
            "mean_scores": mean_scores
        }

    # Save CSV
    csv_rows = []
    for cid, val in summary.items():
        row = {"cluster_id": cid, "num_planets": val["num_planets"]}
        row.update(val["mean_scores"])
        csv_rows.append(row)

    df_out = pd.DataFrame(csv_rows).sort_values("cluster_id")
    csv_path = output_dir / "cluster_symbolic_scores.csv"
    df_out.to_csv(csv_path, index=False)

    json_path = output_dir / "cluster_symbolic_scores.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Saved symbolic cluster scores:")
    print(f"   CSV → {csv_path}")
    print(f"   JSON → {json_path}")

if __name__ == "__main__":
    app()
