"""
SpectraMind V50 – GLL Score Heatmap (Ultimate Version)
-------------------------------------------------------
Visualizes Gaussian Log-Likelihood loss across all planets and bins.
Features:
- Per-planet per-bin GLL matrix computation
- Heatmap visualization
- CSV + NPY + Markdown export
- Planet ID labeling
- Auto-call from diagnostics dashboard or CLI
- Outlier bin masking (optional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional


def compute_gll_matrix(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    var = sigma**2 + 1e-6
    gll = 0.5 * (np.log(2 * np.pi * var) + (y - mu)**2 / var)
    return gll  # shape (N, 283)


def export_gll_matrix(
    gll_matrix: np.ndarray,
    planet_ids: list,
    outdir: str = "diagnostics/gll",
    name: str = "gll_matrix"
):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(gll_matrix, columns=[f"bin_{i}" for i in range(gll_matrix.shape[1])])
    df.insert(0, "planet_id", planet_ids)
    df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    np.save(os.path.join(outdir, f"{name}.npy"), gll_matrix)
    print(f"✅ GLL matrix saved: {name}.csv / .npy")


def plot_gll_heatmap(
    gll_matrix: np.ndarray,
    planet_ids: Optional[list] = None,
    outdir: str = "diagnostics/gll",
    save_name: str = "gll_score_heatmap.png",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = False
):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    im = plt.imshow(gll_matrix, aspect="auto", cmap="hot", interpolation="nearest",
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="GLL Loss")
    plt.xlabel("Spectral Bin")
    plt.ylabel("Planet" if planet_ids is None else "Planet ID")
    plt.title("GLL Score Heatmap (μ, σ vs y)")

    if planet_ids is not None and len(planet_ids) <= 50:
        plt.yticks(ticks=np.arange(len(planet_ids)), labels=planet_ids, fontsize=7)

    if annotate and gll_matrix.shape[0] <= 25:
        for i in range(gll_matrix.shape[0]):
            for j in range(gll_matrix.shape[1]):
                val = f"{gll_matrix[i,j]:.1f}"
                plt.text(j, i, val, fontsize=6, ha='center', va='center', color='white')

    plt.tight_layout()
    path = os.path.join(outdir, save_name)
    plt.savefig(path)
    plt.close()
    print(f"📊 GLL heatmap saved: {path}")


def generate_gll_diagnostics(
    submission_path: str = "submission.csv",
    ground_truth_path: str = "ground_truth.csv",
    outdir: str = "diagnostics/gll",
    mask_threshold: Optional[float] = None
):
    sub = pd.read_csv(submission_path)
    gt = pd.read_csv(ground_truth_path)

    mu_cols = [f"mu_{i}" for i in range(283)]
    sigma_cols = [f"sigma_{i}" for i in range(283)]
    mu = sub[mu_cols].values
    sigma = sub[sigma_cols].values
    y = gt[mu_cols].values
    planet_ids = sub["planet_id"].tolist()

    gll_matrix = compute_gll_matrix(y, mu, sigma)

    if mask_threshold is not None:
        gll_matrix = np.where(gll_matrix > mask_threshold, mask_threshold, gll_matrix)

    export_gll_matrix(gll_matrix, planet_ids, outdir)
    plot_gll_heatmap(gll_matrix, planet_ids=planet_ids, outdir=outdir)

    # Optional summary
    avg_loss = gll_matrix.mean()
    max_loss = gll_matrix.max()
    print(f"📈 GLL avg: {avg_loss:.4f} | max: {max_loss:.2f}")


if __name__ == "__main__":
    generate_gll_diagnostics()