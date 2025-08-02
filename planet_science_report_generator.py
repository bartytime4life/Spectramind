"""
SpectraMind V50 – Planet Science Report Generator (Ultimate Version)
---------------------------------------------------------------------
Generates structured, readable symbolic + scientific summaries per planet.
Hooks into:
- PlanetMemoryBank (latent + symbolic stats)
- SHAP overlays (if present)
- Dashboard & CLI output
- Diagnostics markdown export
"""

import json
import os
from typing import Dict, Optional
from pathlib import Path


def generate_report(
    planet_id: str,
    memory_path: str = "outputs/memory/planet_memory.json",
    write_md: bool = True,
    outdir: str = "diagnostics/reports"
) -> str:
    """
    Generates a symbolic + scientific summary for a given planet.

    Args:
        planet_id: planet to report on
        memory_path: path to planet memory JSON
        write_md: save report to Markdown file
        outdir: directory to write Markdown report

    Returns:
        str: report content (plain text)
    """
    if not os.path.exists(memory_path):
        return f"[Memory Missing] No memory found for {planet_id}."

    with open(memory_path, "r") as f:
        mem = json.load(f)
    p = mem.get(planet_id)
    if not p:
        return f"[No Entry] Planet {planet_id} not found in memory."

    mu_mean = p.get("mu_mean", 0.0)
    mu_std = p.get("mu_std", 0.0)
    z_dim = len(p.get("z", []))
    violations = p.get("violations", {})
    max_rule, max_score = ("none", 0.0)
    if violations:
        max_rule, max_score = max(violations.items(), key=lambda kv: kv[1])

    lines = [
        f"# 🪐 Planet Science Report: {planet_id}",
        "",
        f"**μ mean depth**: {mu_mean:.1f} ppm",
        f"**μ std deviation**: {mu_std:.1f} ppm",
        f"**Latent embedding size**: {z_dim} dimensions",
        f"**Top symbolic violation**: `{max_rule}` = {max_score:.3f}",
        ""
    ]

    if max_score > 0.2:
        lines.append("⚠️ High violation detected → symbolic tuning recommended.")
    elif max_score > 0.1:
        lines.append("ℹ️ Moderate symbolic deviation present.")
    else:
        lines.append("✅ Stable symbolic alignment.")

    report_text = "\n".join(lines)

    if write_md:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        md_path = Path(outdir) / f"report_{planet_id}.md"
        with open(md_path, "w") as f:
            f.write(report_text)
        print(f"📄 Report saved to {md_path}")

    return report_text


if __name__ == "__main__":
    print(generate_report("planet-001"))