"""
SpectraMind V50 – Overlay Gallery Generator
-------------------------------------------
Combines individual μ + violation plots into a mosaic image for inspection or reporting.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from glob import glob
from math import ceil
from typing import Tuple


def overlay_gallery(
    overlay_dir: str = "outputs/diagnostics/planet_overlays",
    save_path: str = "outputs/diagnostics/overlay_gallery.png",
    grid_cols: int = 3,
    image_size: Tuple[float, float] = (4.0, 3.0),
    max_images: int = 12,
    sort_by: str = "name",  # "name", "size", or "mtime"
    dpi: int = 200,
    annotate: bool = True,
    tight: bool = True
):
    """
    Combines individual overlay plots into a gallery mosaic.

    Args:
        overlay_dir: directory containing *_violation_overlay.png images
        save_path: path to save the mosaic PNG
        grid_cols: number of columns in the grid
        image_size: (width, height) per subplot in inches
        max_images: maximum number of images to include
        sort_by: how to sort images ("name", "size", or "mtime")
        dpi: resolution of final saved image
        annotate: whether to show planet_id as title
        tight: whether to use tight_layout
    """
    overlay_dir = Path(overlay_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(overlay_dir.glob("*_violation_overlay.png"))
    if not files:
        print(f"⚠️ No overlay images found in {overlay_dir}")
        return

    if sort_by == "size":
        files.sort(key=lambda f: f.stat().st_size, reverse=True)
    elif sort_by == "mtime":
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    else:
        files.sort()

    files = files[:max_images]
    n = len(files)
    rows = ceil(n / grid_cols)

    fig, axs = plt.subplots(rows, grid_cols, figsize=(grid_cols * image_size[0], rows * image_size[1]))
    axs = np.atleast_2d(axs).flatten()

    for ax in axs[n:]:
        ax.axis("off")

    for i, file in enumerate(files):
        img = plt.imread(file)
        axs[i].imshow(img)
        axs[i].axis("off")
        if annotate:
            axs[i].set_title(file.stem.replace("_violation_overlay", ""), fontsize=8)

    if tight:
        plt.tight_layout()

    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"🖼️ Saved overlay gallery: {save_path}")
    print(f"📦 Included: {n} overlay images from {overlay_dir}")


# Example demo run (safe to remove in production use)
if __name__ == "__main__":
    overlay_gallery()