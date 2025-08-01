"""
SpectraMind V50 – Latent Decomposer
------------------------------------
Maps latent vector z to spectral bins, attributing influence per z-dimension.
"""

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from rich import print

class LatentDecomposer:
    def __init__(self, model_decoder, save_path="outputs/diagnostics/latent_decomposition.png", save_csv=True):
        """
        Args:
            model_decoder: a callable that maps (B, D) → (B, 283) or (B, μ, σ)
            save_path: path to .png plot
            save_csv: whether to also write .csv
        """
        self.decoder = model_decoder
        self.save_path = save_path
        self.save_csv = save_csv

    def decompose(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) latent vectors
        Returns:
            influence: (D, output_dim) mean contribution from each z dim
        """
        B, D = z.shape
        z_base = torch.zeros_like(z)
        outputs = []

        for i in range(D):
            z_perturb = z_base.clone()
            z_perturb[:, i] = z[:, i]  # isolate one dim
            with torch.no_grad():
                mu = self.decoder(z_perturb)  # shape: (B, 283) or (B, μ, σ)
                if isinstance(mu, tuple) and isinstance(mu[0], torch.Tensor):
                    mu = mu[0]  # extract μ
                if mu.ndim != 2:
                    raise ValueError(f"Decoder returned shape {mu.shape}, expected (B, 283)")
            outputs.append(mu.cpu())

        influence = torch.stack(outputs, dim=0)  # (D, B, 283)
        mean_influence = influence.mean(dim=1)   # (D, 283)
        return mean_influence

    def plot(self, influence: torch.Tensor):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.imshow(influence, aspect="auto", cmap="viridis")
        plt.colorbar(label="Contribution to μ")
        plt.title("Latent Decomposition Map (z-dim → μ bins)")
        plt.xlabel("Spectral Bin")
        plt.ylabel("Latent Dimension")
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()
        print(f"✅ Saved latent decomposition heatmap: {self.save_path}")

        if self.save_csv:
            csv_path = self.save_path.replace(".png", ".csv")
            pd.DataFrame(influence.numpy()).to_csv(csv_path, index=False)
            print(f"📄 Saved CSV of influence map: {csv_path}")


# --- CLI Entry ---
if __name__ == "__main__":
    dummy_decoder = lambda z: torch.randn(z.size(0), 283)  # Mock decoder
    z = torch.randn(8, 64)
    decomposer = LatentDecomposer(model_decoder=dummy_decoder)
    influence = decomposer.decompose(z)
    decomposer.plot(influence)