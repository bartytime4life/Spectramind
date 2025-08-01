"""
SpectraMind V50 – SHAP × Symbolic Overlay by Cluster
-----------------------------------------------------
Averages SHAP importance and symbolic violation frequency for each latent cluster.
Generates per-cluster overlay plots and saves summary stats.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import typer

app = typer.Typer(help="Generate SHAP × Symbolic overlays by latent cluster")

@app.command()
def overlay_per_cluster(
    cluster_csv: Path = typer.Option(..., help="Path to latent_map_tsne.csv with cluster_id and planet_id"),
    shap_file: Path = typer.Option("shap_overlay.json", help="SHAP values per planet_id"),
    violation_file: Path = typer.Option("constraint_violation_log.json", help="Symbolic violation log"),
    outdir: Path = typer.Option("outputs/diagnostics/cluster_shap_overlay", help="Directory to save cluster overlays")
):
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(cluster_csv)
    with open(shap_file) as f:
        shap_data = json.load(f)
    with open(violation_file) as f:
        violation_data = json.load(f)

    cluster_dict = defaultdict(list)
    for _, row in df.iterrows():
        cluster_id = row['cluster_id']
        pid = row['planet_id']
        cluster_dict[cluster_id].append(pid)

    summary = {}
    for cid, pids in cluster_dict.items():
        shap_sum = np.zeros(283)
        violation_freq = np.zeros(283)

        for pid in pids:
            if pid not in shap_data:
                continue
            shap_sum += np.array(shap_data[pid])
            if pid in violation_data:
                for bin_idx in violation_data[pid]:
                    if 0 <= int(bin_idx) < 283:
                        violation_freq[int(bin_idx)] += 1

        shap_mean = shap_sum / len(pids)
        violation_norm = violation_freq / len(pids)
        fusion = shap_mean * violation_norm

        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(shap_mean, label="SHAP", color="blue")
        plt.plot(violation_norm, label="Violations", color="red")
        plt.plot(fusion, label="SHAP × Violation", color="black")
        plt.title(f"Cluster {cid}: SHAP × Symbolic Overlay")
        plt.xlabel("Spectral Bin")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        png_path = outdir / f"shap_overlay_cluster_{cid}.png"
        plt.savefig(png_path)
        plt.close()
        print(f"✅ Saved: {png_path}")

        summary[str(cid)] = {
            "num_planets": len(pids),
            "top_fused_bins": list(np.argsort(-fusion)[:5].astype(int)),
            "fusion_mean": float(np.mean(fusion)),
            "fusion_std": float(np.std(fusion))
        }

    with open(outdir / "cluster_overlay_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("📊 Summary saved to cluster_overlay_summary.json")

if __name__ == "__main__":
    app()
