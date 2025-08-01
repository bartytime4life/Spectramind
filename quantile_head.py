"""
SpectraMind V50 – Quantile Regression Decoder Head
---------------------------------------------------
Predicts lower and upper quantiles (e.g. 10% and 90%) for μ bands.
Used for uncertainty estimation and envelope confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileHead(nn.Module):
    def __init__(self, input_dim=272, output_dim=283):
        super().__init__()
        self.q10 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.q50 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.q90 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return {
            "q10": self.q10(x),
            "q50": self.q50(x),
            "q90": self.q90(x)
        }

def quantile_pinball_loss(y_true, q10, q50, q90, alpha=0.1):
    """
    Computes asymmetric pinball loss for 10%, 50%, 90% bands.
    """
    delta_10 = y_true - q10
    delta_90 = q90 - y_true
    delta_50 = y_true - q50

    loss_10 = torch.maximum(alpha * delta_10, (alpha - 1) * delta_10)
    loss_90 = torch.maximum(alpha * delta_90, (alpha - 1) * delta_90)
    loss_50 = 0.5 * torch.abs(delta_50)  # Optional: symmetric penalty for median

    return loss_10.mean() + loss_90.mean() + loss_50.mean()

if __name__ == "__main__":
    model = QuantileHead(input_dim=272)
    x = torch.randn(4, 272)
    y = torch.randn(4, 283)
    out = model(x)
    loss = quantile_pinball_loss(y, out["q10"], out["q50"], out["q90"])
    print(f"Quantile Loss: {loss.item():.4f}")
