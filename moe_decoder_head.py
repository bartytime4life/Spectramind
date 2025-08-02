import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class MoEDecoder(nn.Module):
    """
    SpectraMind V50 – Mixture of Experts Decoder (Fully Upgraded)
    -------------------------------------------------------------
    Produces μ (and optionally σ) via weighted expert decoding
    from latent + meta input using meta-based routing.

    Features:
    - Dropout regularization
    - Per-sample routing entropy
    - Expert selection trace (SHAP/diagnostics)
    - Expert weight visualization plot
    - Optional return of raw expert outputs
    - Routing buffers for analysis modules
    """

    def __init__(self,
                 latent_dim=64,
                 output_dim=283,
                 meta_dim=3,
                 n_experts=4,
                 moe_mode="mixture",
                 dropout=0.1,
                 predict_sigma=True):
        super().__init__()
        self.n_experts = n_experts
        self.output_dim = output_dim
        self.moe_mode = moe_mode
        self.predict_sigma = predict_sigma

        self.gate = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_experts)
        )
        self.softmax = nn.Softmax(dim=-1)

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim)
            ) for _ in range(n_experts)
        ])

        if self.predict_sigma:
            self.sigma_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, output_dim),
                    nn.Softplus()
                ) for _ in range(n_experts)
            ])

        self.last_weights = None
        self.last_logits = None

    def forward(self, z, meta, return_expert_outputs=False, plot_weights=False):
        gate_logits = self.gate(meta)
        gate_weights = self.softmax(gate_logits)
        gate_weights_exp = gate_weights.unsqueeze(-1)

        self.last_weights = gate_weights.detach()
        self.last_logits = gate_logits.detach()

        mu_experts = torch.stack([dec(z) for dec in self.decoders], dim=1)
        mu = (gate_weights_exp * mu_experts).sum(dim=1)

        if self.predict_sigma:
            sigma_experts = torch.stack([head(z) for head in self.sigma_heads], dim=1)
            sigma = (gate_weights_exp * sigma_experts).sum(dim=1)
            if plot_weights:
                self.plot_expert_weights(gate_weights)
            if return_expert_outputs:
                return mu, sigma, mu_experts, sigma_experts, gate_weights
            return mu, sigma

        if plot_weights:
            self.plot_expert_weights(gate_weights)

        if return_expert_outputs:
            return mu, mu_experts, gate_weights
        return mu

    def routing_entropy(self, gate_weights):
        eps = 1e-8
        return -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=1)

    def get_last_routing_weights(self):
        return self.last_weights

    def get_last_logits(self):
        return self.last_logits

    def plot_expert_weights(self, weights):
        if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
            return
        os.makedirs("outputs/diagnostics/plots", exist_ok=True)
        avg_weights = weights.mean(dim=0).cpu().numpy()
        labels = [f"Expert {i}" for i in range(self.n_experts)]

        plt.figure(figsize=(8, 4))
        plt.bar(labels, avg_weights, color="royalblue")
        plt.ylim(0, 1)
        plt.ylabel("Mean Routing Weight")
        plt.title("SpectraMind V50 – MoE Routing Distribution")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("outputs/diagnostics/plots/moe_routing_bar.png")
        plt.close()

if __name__ == "__main__":
    z = torch.randn(8, 64)
    meta = torch.randn(8, 3)
    model = MoEDecoder()
    mu, sigma, mu_all, sigma_all, weights = model(z, meta, return_expert_outputs=True, plot_weights=True)
    entropy = model.routing_entropy(weights)
    print("mu:", mu.shape, "sigma:", sigma.shape)
    print("routing entropy:", entropy)