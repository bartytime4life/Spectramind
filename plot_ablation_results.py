"""
SpectraMind V50 – Ablation Result Visualizer
--------------------------------------------
Plots ablation trial results for GLL score exploration.

Supports heatmaps, scatter plots, and trial introspection.
Integrates with the symbolic-aware diagnostics layer.

Usage:
    python plot_ablation_results.py --csv ablation/trial_results.csv
"""

import os
import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

app = typer.Typer()
sns.set_context("notebook")
sns.set_style("whitegrid")

def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"[ERROR] No ablation results found at: {csv_path}")
    df = pd.read_csv(csv_path)
    df["gll"] = pd.to_numeric(df["gll"], errors="coerce")
    df = df.dropna(subset=["gll"])
    return df

def normalize_gll(df: pd.DataFrame, l_baseline: float = 1.2, l_ideal: float = 0.782):
    df["gll_norm"] = (df["gll"] - l_ideal) / (l_baseline - l_ideal)
    return df

def plot_heatmap(df: pd.DataFrame, x: str, y: str, value: str = "gll", title: str = "", save_prefix: str = "heatmap"):
    pivot = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': value.upper()})
    plt.title(title, fontsize=14)
    plt.xlabel(x)
    plt.ylabel(y)
    pdf_path = f"ablation/{save_prefix}.pdf"
    png_path = f"ablation/{save_prefix}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    typer.echo(f"[Saved] {pdf_path}, {png_path}")
    plt.show()

def plot_scatter(df: pd.DataFrame, x: str, y: str = "gll", hue: Optional[str] = None, title: str = "", save_prefix: str = "scatter"):
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="Spectral", s=100)
    plt.title(title, fontsize=14)
    plt.xlabel(x)
    plt.ylabel("GLL Score")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    pdf_path = f"ablation/{save_prefix}.pdf"
    png_path = f"ablation/{save_prefix}.png"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    typer.echo(f"[Saved] {pdf_path}, {png_path}")
    plt.show()

@app.command()
def visualize(
    csv: Path = typer.Option(..., help="Path to ablation trial_results.csv"),
    normalize: bool = typer.Option(True, help="Normalize GLL score using baseline/ideal"),
):
    df = load_results(csv)

    if normalize:
        df = normalize_gll(df)
        gll_col = "gll_norm"
    else:
        gll_col = "gll"

    plot_heatmap(df, x="nonneg", y="smooth", value=gll_col,
                 title="GLL vs Smoothness & Nonnegativity",
                 save_prefix="heatmap_smooth_nonneg")

    plot_scatter(df, x="mc_dropout", hue="smooth", y=gll_col,
                 title="MC Dropout Passes vs GLL (colored by smoothness)",
                 save_prefix="scatter_mc_dropout")

    plot_scatter(df, x="entropy", hue="nonneg", y=gll_col,
                 title="Entropy Penalty vs GLL (colored by nonneg)",
                 save_prefix="scatter_entropy")

    if "trial_id" in df.columns:
        df_sorted = df.sort_values(by=gll_col).reset_index(drop=True)
        print("\n[Top Trials by GLL]:")
        print(df_sorted[[gll_col, "mc_dropout", "smooth", "nonneg", "entropy", "trial_id"]].head(10).to_markdown())

if __name__ == "__main__":
    app()
