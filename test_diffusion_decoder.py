"""
Unit Test for SpectraMind V50 – DiffusionDecoder
--------------------------------------------------
Verifies inference and sampling for diffusion-based decoder.
"""

import torch
import pytest
from diffusion_decoder import DiffusionDecoder

def test_diffusion_decoder_forward():
    model = DiffusionDecoder(input_dim=256, output_dim=283)
    latents = torch.randn(4, 256)
    output = model(latents)
    assert output.shape == (4, 283), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaNs"
    print("\u2705 test_diffusion_decoder_forward passed")

def test_diffusion_decoder_sampling():
    model = DiffusionDecoder(input_dim=256, output_dim=283)
    sampled = model.sample((4, 256))
    assert sampled.shape == (4, 283), "Sampled output shape mismatch"
    assert not torch.isnan(sampled).any(), "Sampled output contains NaNs"
    print("\u2705 test_diffusion_decoder_sampling passed")

if __name__ == "__main__":
    test_diffusion_decoder_forward()
    test_diffusion_decoder_sampling()

