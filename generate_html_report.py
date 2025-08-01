"""
SpectraMind V50 – HTML Diagnostic Report Generator
---------------------------------------------------
Creates a unified HTML report summarizing diagnostics:
- Predicted μ and σ curves
- SHAP overlays
- Symbolic violations
- FFT noise signatures
- Rule scoring table
- Quantile bands (if available)
- Debug log and config hash
"""

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from dominate import document
from dominate.tags import h1, h2, p, img, hr, div, table, tr, td, th, pre, style


def plot_array(arr, title, ylabel, path):
    plt.figure(figsize=(10, 3))
    plt.plot(arr)
    plt.title(title)
    plt.xlabel("Spectral Bin")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def generate_html_report(output_dir="outputs", decoder_type=None, config_hash=None):
    output_dir = Path(output_dir)
    report_path = output_dir / "report.html"

    mu_path = output_dir / "mu.pt"
    sigma_path = output_dir / "sigma.pt"
    shap_img = output_dir / "shap_overlay.png"
    fft_img = output_dir / "fft_overlay.png"
    quantile_img = output_dir / "quantile_band_plot.png"
    rule_scores = output_dir / "symbolic_rule_scores.csv"
    violations_json = Path("constraint_violation_log.json")
    debug_log = Path("v50_debug_log.md")

    doc = document(title="SpectraMind V50 – Diagnostic Report")

    with doc:
        h1("SpectraMind V50 Diagnostic Report")
        p("This report summarizes the μ, σ predictions, symbolic violations, SHAP overlays, FFT residuals, rule attributions, and quantile intervals.")

        if decoder_type or config_hash:
            with div():
                if decoder_type: p(f"Decoder: {decoder_type}")
                if config_hash: p(f"Config Hash: {config_hash}")
        hr()

        # μ
        if mu_path.exists():
            mu = torch.load(mu_path)[0].cpu().numpy()
            plot_array(mu, "Predicted Mean Spectrum (μ)", "μ (ppm)", output_dir / "mu_plot.png")
            h2("Mean Spectrum μ")
            img(src="mu_plot.png", width="100%")
            hr()

        # σ
        if sigma_path.exists():
            sigma = torch.load(sigma_path)[0].cpu().numpy()
            plot_array(sigma, "Predicted Uncertainty (σ)", "σ (ppm)", output_dir / "sigma_plot.png")
            h2("Uncertainty Spectrum σ")
            img(src="sigma_plot.png", width="100%")
            hr()

        # Quantile Bands
        if quantile_img.exists():
            h2("Quantile Band Coverage (q10–q90)")
            img(src=quantile_img.name, width="100%")
            hr()

        # SHAP
        if shap_img.exists():
            h2("SHAP Attribution Overlay")
            img(src=shap_img.name, width="100%")
            hr()

        # FFT
        if fft_img.exists():
            h2("FFT Power Spectrum of Residuals")
            img(src=fft_img.name, width="100%")
            hr()

        # Violations
        if violations_json.exists():
            h2("Symbolic Constraint Violations")
            with open(violations_json) as f:
                violations = json.load(f)
            for pid, v in violations.items():
                div(f"{pid}: {v}")
            hr()

        # Rule Scoring
        if rule_scores.exists():
            h2("Top Symbolic Rule Influences (SHAP × Violations)")
            df = pd.read_csv(rule_scores, index_col=0)
            if "mean" in df.index:
                mean_scores = df.loc["mean"].sort_values(ascending=False).head(10)
                with table():
                    with tr():
                        th("Rule ID")
                        th("Mean Influence Score")
                    for rule_id, score in mean_scores.items():
                        with tr():
                            td(rule_id)
                            td(f"{score:.4f}")
            hr()

        # Debug Log
        if debug_log.exists():
            h2("Debug Execution Log")
            with open(debug_log) as f:
                pre(f.read())
            hr()

    with open(report_path, "w") as f:
        f.write(doc.render())

    print(f"✅ HTML report saved to {report_path}")


if __name__ == "__main__":
    generate_html_report()
