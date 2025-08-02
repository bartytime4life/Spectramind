"""
SpectraMind V50 – SHAP / PCA Cluster Explorer
---------------------------------------------
Generates interactive PCA visualizations of SHAP/gradient clusters across planets,
allowing color-mapped overlays by cluster, entropy, GLL, or symbolic violations.

✅ Interactive Plotly output
✅ CLI-compatible and diagnostics directory-integrated
✅ Hover tooltips with planet ID, GLL, entropy
✅ Auto scaling for large cluster counts
✅ Color-continuous and discrete modes
"""

import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
from typing import Optional

DEFAULT_OUTPUT = "diagnostics/plotly_pca_shap_clusters.html"
pio.renderers.default = "notebook_connected"  # CLI safety fallback

def plot_pca_shap_clusters(
    csv_path: str,
    output_html: str = DEFAULT_OUTPUT,
    color_col: str = "cluster",
    size_col: Optional[str] = None,
    hover_cols: Optional[list] = None,
    title: Optional[str] = "PCA of SHAP/Gradient Clusters",
    color_map: Optional[str] = None,
    opacity: float = 0.85,
    size_range: tuple = (5, 20),
    max_points: int = 30000,
):
    """
    Create an interactive 2D PCA cluster plot with Plotly.

    Args:
        csv_path (str): Path to CSV containing 'PC1', 'PC2', 'planet_id', etc.
        output_html (str): Output path for HTML file.
        color_col (str): Column to use for coloring the points.
        size_col (Optional[str]): Column to use for marker size (optional).
        hover_cols (list): Columns to display on hover. Default: ['planet_id', color_col, 'gll', 'entropy']
        title (str): Title of the plot.
        color_map (Optional[str]): Optional colormap (e.g., 'Viridis', 'Jet').
        opacity (float): Marker opacity.
        size_range (tuple): Min/max point size.
        max_points (int): Limit points for rendering performance.

    Returns:
        str: Path to saved HTML plot.
    """
    if not os.path.exists(csv_path) or not csv_path.endswith(".csv"):
        raise FileNotFoundError(f"CSV not found or invalid: {csv_path}")

    df = pd.read_csv(csv_path)

    if "PC1" not in df.columns or "PC2" not in df.columns:
        raise ValueError("CSV must contain 'PC1' and 'PC2' columns for plotting.")

    if hover_cols is None:
        hover_cols = ["planet_id"]
        if color_col not in hover_cols:
            hover_cols.append(color_col)
        for col in ("gll", "entropy", "violation_score"):
            if col in df.columns:
                hover_cols.append(col)

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42).reset_index(drop=True)

    color_is_numeric = np.issubdtype(df[color_col].dtype, np.number)

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=df[color_col] if color_is_numeric else df[color_col].astype(str),
        size=df[size_col] if size_col and size_col in df.columns else None,
        hover_data=hover_cols,
        title=title,
        template="plotly_white",
        color_continuous_scale=color_map if color_is_numeric else None,
    )

    fig.update_traces(marker=dict(opacity=opacity), selector=dict(mode='markers'))
    fig.update_layout(
        legend_title=color_col,
        title_font_size=20,
        title_x=0.5,
    )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"✅ Saved PCA cluster plot to: {output_html}")
    return output_html


# CLI Entrypoint
if __name__ == "__main__":
    import typer
    app = typer.Typer(help="SpectraMind – SHAP PCA Cluster Plotter")

    @app.command()
    def run(
        csv_path: str = typer.Argument(..., help="Path to PCA/SHAP CSV file."),
        output_html: str = typer.Option(DEFAULT_OUTPUT, help="Path to save output HTML."),
        color_col: str = typer.Option("cluster", help="Column name to color by."),
        size_col: Optional[str] = typer.Option(None, help="Optional column name for point size."),
        opacity: float = typer.Option(0.85, help="Point opacity."),
        color_map: Optional[str] = typer.Option(None, help="Optional colormap name."),
    ):
        plot_pca_shap_clusters(
            csv_path=csv_path,
            output_html=output_html,
            color_col=color_col,
            size_col=size_col,
            opacity=opacity,
            color_map=color_map,
        )

    app()