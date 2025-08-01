"""
SpectraMind V50 – Diffusion-Based Decoder
------------------------------------------
Decoder head that denoises latent input representations into μ (mean) predictions
via a learned diffusion process. Inspired by denoising score matching and DDIM.
Includes optional noise-aware sampling and time embedding.
"""

import torch
import torch.nn as nn
import math

class DiffusionDecoder(nn.Module):
    def __init__(
        self,
        input_dim=256,
        output_dim=283,
        timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        step_stride=20,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.step_stride = step_stride

        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.denoise_fn = nn.Sequential(
            nn.Linear(input_dim + 1, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Sample noisy input from x_start at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_ac = self.sqrt_alpha_cumprod[t].view(-1, 1)
        sqrt_om = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1)

        return sqrt_ac * x_start + sqrt_om * noise

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Predicts final μ spectrum via coarse denoising loop.
        """
        B, _ = latents.shape
        x = latents

        for t in reversed(range(0, self.timesteps, self.step_stride)):
            t_scaled = torch.full((B, 1), t / self.timesteps, device=latents.device)
            denoise_input = torch.cat([x, t_scaled], dim=-1)
            x = self.denoise_fn(denoise_input)

        return x

    def sample(self, shape):
        """
        Generate μ spectra starting from pure noise.
        """
        x = torch.randn(shape, device=self.betas.device)
        return self.forward(x)

if __name__ == "__main__":
    model = DiffusionDecoder()
    dummy_latents = torch.randn(4, 256)
    out = model(dummy_latents)
    print("\u2705 DiffusionDecoder output shape:", out.shape)

    print("\nSampling from noise:")
    sample = model.sample((4, 256))
    print("\u2705 Sampled output shape:", sample.shape)