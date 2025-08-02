"""
SpectraMind V50 – Unified HTML Diagnostics Report
--------------------------------------------------
Generates a dashboard-style HTML page including:
- GLL bin heatmap
- Symbolic violation overlays
- UMAP latent plots (static and interactive)
- Quantile violation bands
- SHAP or entropy overlays
- Diagnostic summary + anomalies

✅ For use in CLI and auto-reports
"""

import os
import json
from pathlib import Path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SpectraMind V50 – Diagnostic Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9f9f9;
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        h1, h2 {{
            color: #444;
        }}
        img {{
            max-width: 100%;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            margin-bottom: 24px;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
            margin-top: 10px;
            margin-bottom: 24px;
        }}
        pre {{
            background: #eee;
            padding: 10px;
            overflow-x: auto;
            border-radius: 6px;
        }}
    </style>
</head>
<body>

<h1>SpectraMind V50 – Diagnostics Dashboard</h1>

{sections}

</body>
</html>
"""

SECTION_TEMPLATE = """
<h2>{title}</h2>
{content}
"""

def embed_image(img_path):
    return f'<img src="{img_path}" alt="{img_path}" loading="lazy"/>'

def embed_iframe(html_path):
    return f'<iframe src="{html_path}"></iframe>'

def embed_json(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
        formatted = json.dumps(data, indent=2)
        return f"<pre>{formatted}</pre>"
    except Exception as e:
        return f"<pre>Failed to load {json_path}: {e}</pre>"

def generate_html_report(
    out_path="diagnostics/diagnostic_report.html",
    diagnostics_dir="diagnostics"
):
    diagnostics_dir = Path(diagnostics_dir)
    out_path = Path(out_path)
    sections = []

    def section(title, content):
        sections.append(SECTION_TEMPLATE.format(title=title, content=content))

    if (diagnostics_dir / "gll_heatmap_per_bin.png").exists():
        section("GLL Heatmap per Bin", embed_image("gll_heatmap_per_bin.png"))

    if (diagnostics_dir / "mu_violation_overlay.png").exists():
        section("μ Violation Overlay", embed_image("mu_violation_overlay.png"))

    if (diagnostics_dir / "quantile_band_check.png").exists():
        section("Quantile Band Constraint Check", embed_image("quantile_band_check.png"))

    if (diagnostics_dir / "umap_latents.png").exists():
        section("UMAP Projection (Static)", embed_image("umap_latents.png"))

    if (diagnostics_dir / "umap_latents.html").exists():
        section("UMAP Projection (Interactive)", embed_iframe("umap_latents.html"))

    if (diagnostics_dir / "shap_overlay.png").exists():
        section("SHAP Overlay", embed_image("shap_overlay.png"))

    if (diagnostics_dir / "anomalous_bins.json").exists():
        section("Anomalous Bins", embed_json(diagnostics_dir / "anomalous_bins.json"))

    if (diagnostics_dir / "diagnostic_summary.json").exists():
        section("Summary", embed_json(diagnostics_dir / "diagnostic_summary.json"))

    html = HTML_TEMPLATE.format(sections="\n".join(sections))
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    print(f"✅ HTML diagnostic report saved to: {out_path}")

if __name__ == "__main__":
    generate_html_report()