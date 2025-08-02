"""
SpectraMind V50 – Photonic Band Alignment (Ultimate Version)
-------------------------------------------------------------
Encourages μ spectra to align with known molecular absorption dips.
Includes:
- Cosine-based shape alignment
- Directional slope penalty
- YAML template loading
- Optional diagnostics dump
"""

import torch
import yaml
import os
import matplotlib.pyplot as plt

def load_photonic_templates(template_path: str) -> dict:
    with open(template_path, 'r') as f:
        templates = yaml.safe_load(f)
    return templates  # {"CH4": [indices], ...}

def enforce_photonic_template(mu: torch.Tensor, metadata: dict, template_path: str, 
                               outdir: str = "outputs/diagnostics/photonic", 
                               save_plots: bool = True) -> dict:
    """
    Args:
        mu: Tensor (B, 283)
        metadata: dict (unused)
        template_path: path to photonic_basis.yaml
        outdir: optional directory to save visual diagnostics
        save_plots: if True, saves overlay plots per molecule

    Returns:
        dict[str, Tensor] of per-molecule alignment losses
    """
    os.makedirs(outdir, exist_ok=True)
    templates = load_photonic_templates(template_path)
    losses = {}

    for molecule, bins in templates.items():
        bin_tensor = torch.tensor(bins, device=mu.device)
        subregion = mu[:, bin_tensor]  # (B, len(bins))

        dip_vector = torch.linspace(1, 0, steps=subregion.size(1), device=mu.device).unsqueeze(0)
        norm_mu = torch.nn.functional.normalize(subregion, dim=1)
        norm_dip = torch.nn.functional.normalize(dip_vector, dim=1)
        cosine_sim = (norm_mu * norm_dip).sum(dim=1)
        cosine_loss = (1 - cosine_sim).mean()

        slope_loss = torch.mean(torch.relu(subregion[:, 1:] - subregion[:, :-1]))

        losses[f"photonic_{molecule}_cosine"] = cosine_loss
        losses[f"photonic_{molecule}_dip"] = slope_loss

        if save_plots:
            try:
                overlay_path = os.path.join(outdir, f"photonic_overlay_{molecule}.png")
                plot_mu_overlay(subregion[0].detach().cpu(), dip_vector[0].detach().cpu(), overlay_path, molecule)
            except Exception as e:
                print(f"Warning: Failed to plot overlay for {molecule}: {e}")

    return losses

def plot_mu_overlay(spectrum, dip_proto, save_path, molecule_name):
    plt.figure(figsize=(6, 3))
    plt.plot(spectrum, label="μ prediction", linewidth=2)
    plt.plot(dip_proto, label="dip prototype", linestyle="--")
    plt.title(f"Photonic Band Alignment: {molecule_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    dummy_mu = torch.rand(2, 283) * 1000
    fake_meta = {}
    dummy_yaml = "photonic_basis.yaml"
    print(enforce_photonic_template(dummy_mu, fake_meta, dummy_yaml, save_plots=False))