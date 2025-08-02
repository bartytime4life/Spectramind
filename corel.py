"""
corel.py – SpectralCOREL with Temporal Attention + Positional Encoding
-----------------------------------------------------------------------
GAT-based conformal uncertainty model using:
- Learned or sinusoidal bin position embeddings
- Attention over spectral bins
- Residual-corrected μ and conformal radii
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math


class SpectralCOREL(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        num_bins=283,
        posenc_dim=16,
        posenc_type="learned",  # "learned" or "sinusoidal"
        coverage=0.90,
        heads=4
    ):
        super().__init__()
        self.num_bins = num_bins
        self.coverage = coverage
        self.posenc_type = posenc_type
        self.posenc_dim = posenc_dim

        if posenc_type == "learned":
            self.positional_encoding = nn.Embedding(num_bins, posenc_dim)
        elif posenc_type == "sinusoidal":
            self.register_buffer("pe_sin", self._build_sinusoidal(num_bins, posenc_dim))
        else:
            raise ValueError("posenc_type must be 'learned' or 'sinusoidal'")

        self.gnn1 = GATConv(input_dim + posenc_dim, hidden_dim, heads=heads, concat=True)
        self.gnn2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

        self.out_mean = nn.Linear(hidden_dim, 1)
        self.out_radius = nn.Linear(hidden_dim, 1)

    def _build_sinusoidal(self, num_pos, dim):
        pe = torch.zeros(num_pos, dim)
        position = torch.arange(0, num_pos).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # shape: (num_bins, posenc_dim)

    def forward(self, mu_pred: torch.Tensor, sigma_pred: torch.Tensor, edge_index: torch.Tensor):
        B, N = mu_pred.shape
        assert N == self.num_bins

        if self.posenc_type == "learned":
            pos_enc = self.positional_encoding(torch.arange(N, device=sigma_pred.device))  # (N, posenc_dim)
        else:  # sinusoidal
            pos_enc = self.pe_sin[:N].to(sigma_pred.device)

        mu_out, r_out = [], []

        for b in range(B):
            x = sigma_pred[b].unsqueeze(-1)  # (N, 1)
            x = torch.cat([x, pos_enc], dim=-1)

            h = F.elu(self.gnn1(x, edge_index))
            h = F.elu(self.gnn2(h, edge_index))

            mu_corr = mu_pred[b] + self.out_mean(h).squeeze(-1)
            r_corr = F.softplus(self.out_radius(h).squeeze(-1))

            mu_out.append(mu_corr)
            r_out.append(r_corr)

        return torch.stack(mu_out), torch.stack(r_out)


@torch.no_grad()
def conformalize_from_validation(val_errors: torch.Tensor, alpha: float = 0.10) -> float:
    """Returns q̂ such that Pr(|y - μ| ≤ q̂) ≥ 1 - α"""
    return torch.quantile(val_errors, 1 - alpha).item()


# --- Smoke Test ---
if __name__ == "__main__":
    for mode in ["learned", "sinusoidal"]:
        print(f"\n🧪 Testing SpectralCOREL with posenc_type='{mode}'")
        model = SpectralCOREL(posenc_type=mode)
        mu = torch.randn(2, 283)
        sigma = torch.abs(torch.randn(2, 283))

        edge_list = [[i, i+1] for i in range(282)] + [[i+1, i] for i in range(282)]
        edge_index = torch.tensor(edge_list).t().contiguous()

        mu_corr, r_corr = model(mu, sigma, edge_index)
        print(f"μ shape: {mu_corr.shape} | radius shape: {r_corr.shape}")