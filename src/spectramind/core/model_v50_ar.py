import torch, torch.nn as nn
from .multi_scale_decoder import MultiScaleDecoder
from .flow_uncertainty_head import FlowUncertaintyHead

class SpectraMindModelV50(nn.Module):
    def __init__(self, latent_dim=64, out_bins=283):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = MultiScaleDecoder(latent_dim, out_bins)
        self.uq = FlowUncertaintyHead(latent_dim, out_bins)

    def forward(self, z):
        mu = self.decoder(z)
        sigma = self.uq(z)
        return mu, sigma
