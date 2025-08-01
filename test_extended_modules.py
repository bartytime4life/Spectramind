"""
Extended Unit Tests for SpectraMind V50 Modules
-----------------------------------------------
Covers:
- quantile_head.py
- corel_inference.py
- shap_overlay.py
"""

import torch
import pytest
from quantile_head import QuantileHead
from corel_inference import apply_corel, load_corel_model
from shap_overlay import compute_shap_overlay

def test_quantile_head():
    model = QuantileHead(input_dim=264 + 8, output_dim=283, quantiles=[0.1, 0.5, 0.9])
    x = torch.randn(3, 272)
    outputs = model(x)
    assert isinstance(outputs, dict), "Output must be a dictionary"
    for q, vals in outputs.items():
        assert vals.shape == (3, 283), f"Quantile {q} shape incorrect: {vals.shape}"
        assert not torch.isnan(vals).any(), f"Quantile {q} contains NaNs"

def test_corel_inference():
    model = load_corel_model("models/corel_gnn.pt")
    mu = torch.rand(4, 283)
    sigma = torch.rand(4, 283).clamp(min=1e-3)
    edge_index = torch.randint(0, 4, (2, 100))
    mu_corr, sigma_corr = apply_corel(model, mu, sigma, edge_index)
    assert mu_corr.shape == mu.shape, "Corrected μ shape mismatch"
    assert sigma_corr.shape == sigma.shape, "Corrected σ shape mismatch"
    assert (sigma_corr > 0).all(), "σ after COREL must be positive"

def test_shap_overlay():
    shap_values = torch.randn(2, 283)
    rule_violations = torch.randint(0, 2, (2, 283)).float()
    overlay = compute_shap_overlay(shap_values, rule_violations)
    assert overlay.shape == (2, 283), "SHAP overlay shape mismatch"
    assert not torch.isnan(overlay).any(), "SHAP overlay contains NaNs"
