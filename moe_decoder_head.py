import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEDecoder(nn.Module):
    """
    SpectraMind V50 – Mixture of Experts Decoder
    --------------------------------------------
    Produces μ (and optionally σ) via weighted expert decoding from latent + meta input.
    """

    def __init__(self,
                 latent_dim=64,
                 output_dim=283,
                 meta_dim=3,
                 n_experts=3,
                 moe_mode="mean",         # "mean" or "mixture"
                 dropout=0.1,
                 predict_sigma=False):
        super().__init__()
        self.n_experts = n_experts
        self.output_dim = output_dim
        self.moe_mode = moe_mode
        self.predict_sigma = predict_sigma

        # Gating function over meta input
        self.gate = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_experts),
            nn.Softmax(dim=-1)
        )

        # Expert networks (μ predictors)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim)
            ) for _ in range(n_experts)
        ])

        # Optional σ (uncertainty) predictors per expert
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

    def forward(self, z, meta, return_expert_outputs=False):
        """
        Args:
            z: (B, latent_dim)
            meta: (B, meta_dim)
            return_expert_outputs: if True, returns raw expert outputs for inspection

        Returns:
            μ: (B, output_dim)
            σ: (B, output_dim) if predict_sigma
        """
        B = z.size(0)
        gate_weights = self.gate(meta)              # (B, K)
        gate_weights_exp = gate_weights.unsqueeze(-1)  # (B, K, 1)

        mu_experts = torch.stack([dec(z) for dec in self.decoders], dim=1)  # (B, K, D)

        mu = (gate_weights_exp * mu_experts).sum(dim=1)  # (B, D)

        if self.predict_sigma:
            sigma_experts = torch.stack([head(z) for head in self.sigma_heads], dim=1)  # (B, K, D)
            sigma = (gate_weights_exp * sigma_experts).sum(dim=1)  # (B, D)
            return (mu, sigma) if not return_expert_outputs else (mu, sigma, mu_experts, sigma_experts, gate_weights)

        return mu if not return_expert_outputs else (mu, mu_experts, gate_weights)
