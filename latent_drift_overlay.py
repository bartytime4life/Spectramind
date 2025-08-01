"""
SpectraMind V50 – Latent Drift Overlay
---------------------------------------
Visualizes how latent representations drift across time, versions, or retraining cycles.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich import print


class LatentDriftOverlay:
    def __init__(self, save_path="outputs/diagnostics/latent_drift_overlay.png", save_csv=True):
        self.save_path = save_path
        self.save_csv = save_csv

    def compute_drift(self, z1: torch.Tensor, z2: torch.Tensor):
        """
        Args:
            z1: (B, D) latent snapshot at t1
            z2: (B, D) latent snapshot at t2

        Returns:
            drift_per_sample: (B,) L2 norm between z1 and z2
            drift_per_dim: (D,) mean absolute drift per latent dimension
        """
        drift_per_sample = torch.norm(z2 - z1, dim=1)
        drift_per_dim = (z2 - z1).abs().mean(dim=0)
        return drift_per_sample.cpu(), drift_per_dim.cpu()

    def plot(self, drift_per_sample: torch.Tensor, drift_per_dim: torch.Tensor):
        outdir = os.path.dirname(self.save_path)
        os.makedirs(outdir, exist_ok=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist(drift_per_sample.numpy(), bins=30, color="tomato", alpha=0.8)
        axs[0].set_title("Latent Drift Distribution (L2 per Sample)")
        axs[0].set_xlabel("L2 Distance")
        axs[0].set_ylabel("Sample Count")
        axs[0].grid(True)

        axs[1].bar(np.arange(len(drift_per_dim)), drift_per_dim.numpy(), color="slateblue")
        axs[1].set_title("Per-Dimension Drift (Mean Absolute)")
        axs[1].set_xlabel("Latent Dimension")
        axs[1].set_ylabel("Avg Drift")
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
        print(f"🖼️ Saved latent drift plot to [green]{self.save_path}[/]")

        if self.save_csv:
            csv1 = self.save_path.replace(".png", "_per_sample.csv")
            csv2 = self.save_path.replace(".png", "_per_dim.csv")
            pd.DataFrame({"drift_l2": drift_per_sample.numpy()}).to_csv(csv1, index=False)
            pd.DataFrame({"latent_dim": np.arange(len(drift_per_dim)), "drift": drift_per_dim.numpy()}).to_csv(csv2, index=False)
            print(f"📄 Saved drift CSVs to [green]{csv1}[/] and [green]{csv2}[/]")


# Optional test/demo:
if __name__ == "__main__":
    z_t1 = torch.randn(100, 64)
    z_t2 = z_t1 + 0.1 * torch.randn(100, 64)
    overlay = LatentDriftOverlay()
    sample_drift, dim_drift = overlay.compute_drift(z_t1, z_t2)
    overlay.plot(sample_drift, dim_drift)