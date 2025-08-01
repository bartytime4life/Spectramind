"""
SpectraMind V50 – Violation Heatmap
-----------------------------------
Visualizes symbolic constraint violations per bin across the dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import typer
from pathlib import Path
from rich import print

app = typer.Typer()

def aggregate_violations(violation_tensor: np.ndarray) -> np.ndarray:
    """
    Args:
        violation_tensor: shape (N, 283), binary or float violation flags per bin

    Returns:
        mean_violation: (283,) average per-bin violation frequency or score
    """
    return np.mean(violation_tensor, axis=0)


def plot_violation_heatmap(mean_violation: np.ndarray, outdir="outputs/diagnostics", save_name="symbolic_violation_heatmap.png"):
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(12, 3))
    plt.imshow(mean_violation[None, :], aspect="auto", cmap="hot", extent=[0, 282, 0, 1])
    plt.colorbar(label="Avg Violation Score")
    plt.title("Symbolic Constraint Violations per Bin")
    plt.xlabel("Spectral Bin")
    plt.yticks([])
    plt.tight_layout()
    save_path = os.path.join(outdir, save_name)
    plt.savefig(save_path)
    print(f"✅ Saved symbolic violation heatmap: {save_path}")

    # Also save CSV of values
    csv_path = os.path.join(outdir, "violation_scores.csv")
    pd.DataFrame({"bin": np.arange(283), "violation_score": mean_violation}).to_csv(csv_path, index=False)
    print(f"📄 Saved CSV of bin scores: {csv_path}")


def load_violation_tensor(submission_path: Path) -> np.ndarray:
    """
    Attempts to find violation data given a submission path.

    Checks:
    - submission_dir/violation_tensor.npy
    - constraint_violation_log.json → aggregated binary tensor
    """
    subdir = submission_path.parent

    npy_path = subdir / "violation_tensor.npy"
    json_path = subdir / "constraint_violation_log.json"

    if npy_path.exists():
        print(f"[blue]📂 Found tensor:[/] {npy_path}")
        return np.load(npy_path)

    elif json_path.exists():
        print(f"[blue]📂 Parsing JSON log:[/] {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        # Assumes format: {planet_id: [0/1 per bin]}
        tensor = np.stack(list(data.values()))
        return tensor

    else:
        raise FileNotFoundError(f"Could not locate violation_tensor.npy or constraint_violation_log.json in: {subdir}")


@app.command()
def main(
    submission: Path = typer.Option("submission.csv", help="Submission file to infer diagnostic folder"),
    outdir: Path = typer.Option("outputs/diagnostics", help="Directory to save plots and CSV")
):
    """
    Aggregates symbolic violations and visualizes per-bin heatmap.
    """
    tensor = load_violation_tensor(submission)
    mean_violation = aggregate_violations(tensor)
    plot_violation_heatmap(mean_violation, outdir=outdir)

if __name__ == "__main__":
    app()
