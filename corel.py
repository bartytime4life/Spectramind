"""
corel.py – SpectralCOREL with Edge Features and Positional Encoding
-------------------------------------------------------------------
GNN-based uncertainty refinement model using edge-conditioned message passing.
Supports:
• Learned or sinusoidal positional encodings
• Edge features: bin distance, molecule, detector region, etc.
• NNConv for edge-aware bin interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import math


class SpectralCOREL(nn.Module):
    def __init__(
        self,
        input_dim=1,
        edge_feat_dim=3,       # e.g., [bin_dist, molecule_id, detector_region]
        posenc_dim=16,
        hidden_dim=64,
        num_bins=283,
        coverage=0.90,
        posenc_type="learned"  # or "sinusoidal"
    ):
        super().__init__()
        self.num_bins = num_bins
        self.coverage = coverage
        self.posenc_type = posenc_type
        self.posenc_dim = posenc_dim

        # --- Positional encoding for bin index ---
        if posenc_type == "learned":
            self.positional_encoding = nn.Embedding(num_bins, posenc_dim)
        elif posenc_type == "sinusoidal":
            self.register_buffer("pe_sin", self._build_sinusoidal(num_bins, posenc_dim))
        else:
            raise ValueError("posenc_type must be 'learned' or 'sinusoidal'")

        # --- NNConv requires an edge network to compute weights from edge features ---
        edge_net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim)
        )

        self.nnconv1 = NNConv(input_dim + posenc_dim, hidden_dim, edge_net, aggr='mean')
        self.nnconv2 = NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean')

        self.out_mean = nn.Linear(hidden_dim, 1)
        self.out_radius = nn.Linear(hidden_dim, 1)

    def _build_sinusoidal(self, num_pos, dim):
        pe = torch.zeros(num_pos, dim)
        position = torch.arange(0, num_pos).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # shape: (num_bins, posenc_dim)

    def forward(
        self,
        mu_pred: torch.Tensor,            # (B, 283)
        sigma_pred: torch.Tensor,         # (B, 283)
        edge_index: torch.Tensor,         # (2, E)
        edge_attr: torch.Tensor           # (E, edge_feat_dim)
    ):
        B, N = mu_pred.shape
        device = mu_pred.device

        if self.posenc_type == "learned":
            posenc = self.positional_encoding(torch.arange(N, device=device))  # (N, posenc_dim)
        else:
            posenc = self.pe_sin[:N].to(device)  # (N, posenc_dim)

        mu_out, r_out = [], []

        for b in range(B):
            x = sigma_pred[b].unsqueeze(-1)  # (N, 1)
            x = torch.cat([x, posenc], dim=-1)  # (N, 1 + posenc_dim)

            h = F.relu(self.nnconv1(x, edge_index, edge_attr))
            h = F.relu(self.nnconv2(h, edge_index, edge_attr))

            mu_corr = mu_pred[b] + self.out_mean(h).squeeze(-1)
            r_corr = F.softplus(self.out_radius(h).squeeze(-1))

            mu_out.append(mu_corr)
            r_out.append(r_corr)

        return torch.stack(mu_out), torch.stack(r_out)


@torch.no_grad()
def conformalize_from_validation(val_errors: torch.Tensor, alpha: float = 0.10) -> float:
    """Returns q̂ such that Pr(|y - μ| ≤ q̂) ≥ 1 - α"""
    return torch.quantile(val_errors, 1 - alpha).item()


# --- Smoke test ---
if __name__ == "__main__":
    print("🧪 Running SpectralCOREL edge-feature GNN test...")

    model = SpectralCOREL(posenc_type="learned")
    B, N = 2, 283
    mu = torch.randn(B, N)
    sigma = torch.abs(torch.randn(B, N))

    # Linear bin chain edges
    edge_index = torch.tensor([[i, i+1] for i in range(N-1)] + [[i+1, i] for i in range(N-1)]).t()
    E = edge_index.shape[1]

    # Dummy edge features: [bin_dist, mol_type, detector_region]
    edge_attr = torch.randn(E, 3)

    mu_corr, r_corr = model(mu, sigma, edge_index, edge_attr)
    print("✅ Edge-feature SpectralCOREL test passed.")
    print("μ shape:", mu_corr.shape, "| radius shape:", r_corr.shape)