"""
SpectraMind V50 – Latent Rule Attention Overlay (SHAP × Symbolic Diagnostic)
-------------------------------------------------------------------------------
Combines symbolic rule activation maps with decoder or MoE attention vectors.
Generates overlay diagnostic plots and saves metrics/logs for visual inspection.
"""

import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import json


class LatentRuleAttentionOverlay:
    def __init__(self, rule_map: torch.Tensor, save_path="outputs/diagnostics/latent_rule_overlay.png"):
        """
        Args:
            rule_map: (B, 283) symbolic activation tensor (float or binary)
        """
        self.rule_map = rule_map  # (B, 283)
        self.save_path = Path(save_path)

    def overlay(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_weights: (B, 283) tensor from decoder or SHAP
        Returns:
            overlay: (B, 283) product of rule_map and attention_weights
        """
        return self.rule_map * attention_weights

    def plot(self, overlay: torch.Tensor, title="Latent × Rule Attention Overlay"):
        os.makedirs(self.save_path.parent, exist_ok=True)
        plt.figure(figsize=(10, 4))
        mean_overlay = overlay.mean(dim=0).cpu().numpy()
        plt.plot(mean_overlay)
        plt.title(title)
        plt.xlabel("Spectral Bin")
        plt.ylabel("Weighted Rule Activation")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()

        summary = {
            "mean": float(np.mean(mean_overlay)),
            "std": float(np.std(mean_overlay)),
            "max": float(np.max(mean_overlay)),
            "min": float(np.min(mean_overlay))
        }
        with open(self.save_path.with_suffix(".json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("✅ Overlay plot saved to", self.save_path)
        print("📊 Stats:", summary)


if __name__ == "__main__":
    rule_mask_path = Path("outputs/rule_mask.pt")
    attention_path = Path("outputs/decoder_attention.pt")

    if not rule_mask_path.exists() or not attention_path.exists():
        print("❌ Missing rule_mask.pt or decoder_attention.pt in outputs/")
    else:
        rules = torch.load(rule_mask_path)  # shape: (B, 283)
        attn = torch.load(attention_path)   # shape: (B, 283)
        overlay = LatentRuleAttentionOverlay(rule_map=rules)
        masked = overlay.overlay(attn)
        overlay.plot(masked)
