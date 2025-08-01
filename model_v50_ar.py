"""
SpectraMind V50 – Modular Dual-Branch Model
-------------------------------------------
FGS1 branch: GRU-based structured time-series encoder
AIRS branch: Simple GNN-style feedforward projection
Decoder: MoE | Diffusion | Quantile
Output: μ (mean spectrum), σ (uncertainty)
Supports symbolic constraint losses + MetaLossScheduler
"""

import torch
import torch.nn as nn
from symbolic_loss import compute_symbolic_losses
from meta_loss_scheduler import MetaLossScheduler
from moe_decoder_head import MoEDecoderHead
from diffusion_decoder import DiffusionDecoder
from quantile_head import QuantileDecoderHead

class MambaBranch(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, layers=4):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=layers)
        self.project = nn.Linear(hidden_dim, 128)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, T, 1)
        out, _ = self.rnn(x)
        return self.project(out.mean(dim=1))  # (B, 128)

class AIRSGraphBranch(nn.Module):
    def __init__(self, input_dim=282, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x):
        x = self.input_proj(x)  # (T, hidden)
        return self.gnn_layers(x).mean(dim=0)  # (128,)

class SpectraMindModel(nn.Module):
    def __init__(self, decoder_type="moe"):
        """Instantiate full model with selectable decoder"""
        super().__init__()
        self.fgs_branch = MambaBranch()
        self.airs_branch = AIRSGraphBranch()
        self.decoder_type = decoder_type

        if decoder_type == "moe":
            self.head = MoEDecoderHead(input_dim=264 + 8, output_dim=283, num_experts=4, return_sigma=True)
        elif decoder_type == "diffusion":
            self.head = DiffusionDecoder(input_dim=264 + 8, output_dim=283)
        elif decoder_type == "quantile":
            self.head = QuantileDecoderHead(input_dim=264 + 8, output_dim=283)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")

        self.scheduler = MetaLossScheduler(schedule_type="linear_ramp", max_weight=1.0, warmup_epochs=5)

    def encode_latent(self, fgs_seq, airs_seq, metadata):
        """Return latent fused representation prior to decoding"""
        fgs_feat = self.fgs_branch(fgs_seq)
        airs_feat = self.airs_branch(airs_seq).unsqueeze(0).expand(fgs_feat.size(0), -1)
        return torch.cat([fgs_feat, airs_feat, metadata], dim=1)

    def forward(self, fgs_seq, airs_seq, metadata):
        x = self.encode_latent(fgs_seq, airs_seq, metadata)

        if self.decoder_type == "moe":
            return self.head(x)
        elif self.decoder_type == "diffusion":
            mu = self.head(x)
            sigma = torch.full_like(mu, 0.05)
            return mu, sigma
        elif self.decoder_type == "quantile":
            return self.head(x)  # returns mu, sigma

    def compute_total_loss(self, mu, sigma, y_true, config, epoch=0):
        """Compute GLL + symbolic weighted composite loss"""
        mse = nn.functional.mse_loss(mu, y_true)
        symbolic_losses = compute_symbolic_losses(mu, config)
        symbolic_weight = self.scheduler.get_weight(epoch)
        total = mse + symbolic_weight * sum(symbolic_losses.values())

        return total, {
            "mse": mse.item(),
            "symbolic_weight": symbolic_weight,
            **{k: v.item() for k, v in symbolic_losses.items()}
        }
