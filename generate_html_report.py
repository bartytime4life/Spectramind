"""
SpectraMind V50 – Unified HTML Diagnostics Dashboard
-----------------------------------------------------
Generates a single HTML file with embedded diagnostics:

✅ GLL heatmap
✅ Violation overlays
✅ Quantile band constraints
✅ Latent UMAP (static + interactive)
✅ SHAP overlays
✅ Symbolic anomaly reports
✅ Confidence-based links and styling

Output: diagnostics/diagnostic_report.html or versioned file
"""

import os
import json
from pathlib import Path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SpectraMind V50 – Diagnostic Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #f7f8fa;
            color: #222;
            padding: 30px;
            max-width: 1200px;
            margin: auto;
        }}
        h1, h2 {{
            color: #222;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }}
        img {{
            max-width: 100%;
            margin: 20px 0;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}
        iframe {{
            border: 1px solid #ccc;
            margin: 20px 0;
            width: 100%;
            height: 640px;
        }}
        pre {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        .caption {{
            font-style: italic;
            color: #555;
            margin-top: -10px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

<h1>SpectraMind V50 – Diagnostics Report</h1>
{sections}

</body>
</html>
"""

SECTION_TEMPLATE = """
<h2>{title}</h2>
{content}
"""

def embed_image(img_path):
    return f'<img src="{img_path}" alt="{img_path}"/>'

def embed_iframe(html_path, height=640):
    return f'<iframe src="{html_path}" height="{height}"></iframe>'

def embed_json(json_path):
    try:
        with open(json_path) as f:
            data = json.load(f)
        return f"<pre>{json.dumps(data, indent=2)}</pre>"
    except Exception as e:
        return f"<pre>⚠️ Failed to load {json_path}: {e}</pre>"

def generate_html_report(
    out_path="diagnostics/diagnostic_report.html",
    diagnostics_dir="diagnostics"
):
    diagnostics_dir = Path(diagnostics_dir)
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    sections = []

    def section(title, content):
        sections.append(SECTION_TEMPLATE.format(title=title, content=content))

    # GLL heatmap
    if (diagnostics_dir / "gll_heatmap_per_bin.png").exists():
        section("GLL Heatmap per Bin", embed_image("gll_heatmap_per_bin.png"))

    # Violation overlay
    if (diagnostics_dir / "mu_violation_overlay.png").exists():
        section("Symbolic Violation Overlay", embed_image("mu_violation_overlay.png"))

    # Quantile constraints
    if (diagnostics_dir / "quantile_band_check.png").exists():
        section("Quantile Constraint Check", embed_image("quantile_band_check.png"))

    # SHAP overlay (optional)
    if (diagnostics_dir / "shap_overlay.png").exists():
        section("SHAP Overlay", embed_image("shap_overlay.png"))

    # UMAP (static)
    if (diagnostics_dir / "umap_latents.png").exists():
        section("UMAP Projection (Static)", embed_image("umap_latents.png"))

    # UMAP (interactive w/ links and confidence)
    if (diagnostics_dir / "umap_latents.html").exists():
        caption = "<div class='caption'>Interactive UMAP: hover for symbolic label, size = confidence, click to open planet page.</div>"
        section("UMAP Projection (Interactive)", caption + embed_iframe("umap_latents.html"))

    # Anomalous bin list
    if (diagnostics_dir / "anomalous_bins.json").exists():
        section("Anomalous Bins (Symbolic + SHAP + Entropy)", embed_json(diagnostics_dir / "anomalous_bins.json"))

    # Diagnostic summary
    if (diagnostics_dir / "diagnostic_summary.json").exists():
        section("Full Diagnostic Summary", embed_json(diagnostics_dir / "diagnostic_summary.json"))

    # GLL score JSON
    if (diagnostics_dir / "gll_score_submission.json").exists():
        section("GLL Score Metadata", embed_json(diagnostics_dir / "gll_score_submission.json"))

    html = HTML_TEMPLATE.format(sections="\n".join(sections))
    with open(out_path, "w") as f:
        f.write(html)

    print(f"✅ Diagnostic HTML report written to: {out_path.resolve()}")

if __name__ == "__main__":
    generate_html_report()