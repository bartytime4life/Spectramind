"""
Unit Tests for SpectraMind V50 Core Modules
-------------------------------------------
Covers:
- symbolic_loss.py
- moe_decoder_head.py
- meta_loss_scheduler.py
"""

import pytest
import torch
import numpy as np

from symbolic_loss import compute_symbolic_losses
from moe_decoder_head import MoEDecoderHead
from meta_loss_scheduler import MetaLossScheduler


@pytest.fixture
def dummy_mu():
    return torch.rand(4, 283)


def test_symbolic_loss(dummy_mu):
    config = {
        "smoothness": True,
        "monotonicity": "none",
        "nonnegativity": True,
        "variance_shaping": True,
        "target_std": 0.05,
    }
    losses = compute_symbolic_losses(dummy_mu, config)

    required_keys = {"smoothness", "nonnegative", "variance"}
    missing = required_keys - losses.keys()
    assert not missing, f"Missing loss keys: {missing}"

    for name, val in losses.items():
        assert isinstance(val, (float, torch.Tensor)), f"Loss {name} is not numeric"
        assert val >= 0, f"Loss {name} returned negative value: {val}"


def test_moe_decoder_output_shape_and_sanity():
    B, D_in, D_out = 2, 272, 283  # 264 + 8 latent + meta
    x = torch.randn(B, D_in)
    model = MoEDecoderHead(input_dim=D_in, output_dim=D_out, num_experts=4, return_sigma=True)
    mu, sigma = model(x)

    assert isinstance(mu, torch.Tensor), "mu is not a tensor"
    assert isinstance(sigma, torch.Tensor), "sigma is not a tensor"
    assert mu.shape == (B, D_out), f"mu shape mismatch: {mu.shape}"
    assert sigma.shape == (B, D_out), f"sigma shape mismatch: {sigma.shape}"
    assert (sigma > 0).all(), "sigma must be strictly positive"
    assert not torch.isnan(mu).any(), "mu contains NaNs"
    assert not torch.isnan(sigma).any(), "sigma contains NaNs"


def test_meta_scheduler_linear_ramp():
    sched = MetaLossScheduler(schedule_type="linear_ramp", max_weight=1.0, warmup_epochs=5)
    weights = [sched.get_weight(epoch) for epoch in range(10)]

    assert np.isclose(weights[0], 0.0), f"Expected 0.0, got {weights[0]}"
    assert np.isclose(weights[5], 1.0), f"Expected 1.0 at epoch 5, got {weights[5]}"
    assert all(np.isclose(w, 1.0) for w in weights[5:]), "Weights should remain at 1.0 after warmup"
