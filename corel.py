"""
corel.py – SpectralCOREL with Temporal Correlation
--------------------------------------------------
GNN-based conformal uncertainty model for spectral bins with:
- Learned positional encodings for bin index
- GATConv for attention over bin neighbors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class SpectralCOREL(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_bins=283, posenc_dim=16, coverage=0.90, heads=4):
        """
        Args:
            input_dim: input node dim (e.g., σ only)
            hidden_dim: hidden GNN dim
            num_bins: number of spectral bins
            posenc_dim: positional encoding dim
            coverage: confidence level (1 - α)
            heads: number of attention heads for GATConv
        """
        super().__init__()
        self.num_bins = num_bins
        self.coverage = coverage

        # Learned positional encoding (bin index → embedding)
        self.positional_encoding = nn.Embedding(num_bins, posenc_dim)

        self.gnn1 = GATConv(input_dim + posenc_dim, hidden_dim, heads=heads, concat=True)
        self.gnn2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)

        self.out_mean = nn.Linear(hidden_dim, 1)
        self.out_radius = nn.Linear(hidden_dim, 1)

    def forward(self, mu_pred: torch.Tensor, sigma_pred: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            mu_pred: (B, 283) predicted μ
            sigma_pred: (B, 283) predicted σ
            edge_index: (2, E) graph of bin connectivity (shared across batch)

        Returns:
            mu_corr: (B, 283) refined μ
            r_corr: (B, 283) conformal radii
        """
        B, N = mu_pred.shape
        assert N == self.num_bins, f"Expected {self.num_bins} bins, got {N}"

        device = sigma_pred.device
        pos_idx = torch.arange(N, device=device)

        # Expand positional encodings
        pe = self.positional_encoding(pos_idx)  # (N, posenc_dim)

        mu_out, r_out = [], []

        for b in range(B):
            x = sigma_pred[b].unsqueeze(-1)  # (N, 1)
            x = torch.cat([x, pe], dim=-1)   # (N, 1 + posenc)

            h = F.elu(self.gnn1(x, edge_index))
            h = F.elu(self.gnn2(h, edge_index))

            mu_corr = mu_pred[b] + self.out_mean(h).squeeze(-1)
            r_corr = F.softplus(self.out_radius(h).squeeze(-1))

            mu_out.append(mu_corr)
            r_out.append(r_corr)

        return torch.stack(mu_out), torch.stack(r_out)


@torch.no_grad()
def conformalize_from_validation(val_errors: torch.Tensor, alpha: float = 0.10) -> float:
    """Quantile thresholding for global conformal radius"""
    return torch.quantile(val_errors, 1 - alpha).item()


# --- Smoke Test ---
if __name__ == "__main__":
    model = SpectralCOREL()
    mu = torch.randn(4, 283)
    sigma = torch.abs(torch.randn(4, 283))

    edge_list = [[i, i+1] for i in range(282)] + [[i+1, i] for i in range(282)]
    edge_index = torch.tensor(edge_list).t().contiguous()

    mu_corr, r_corr = model(mu, sigma, edge_index)
    print("✅ Temporal SpectralCOREL test passed.")
    print("μ shape:", mu_corr.shape, "| radius shape:", r_corr.shape)