"""
cli_explain_util.py – SpectraMind V50 Command-to-File Mapping & CLI Help Panel
-------------------------------------------------------------------------------
Provides:
• Mapping of CLI commands to underlying pipeline files
• CLI help table for `spectramind help` or diagnostics
• Self-test integration to ensure command-file links are valid
• Markdown and JSON export support for docs and validation
"""

import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Dict, Union, List

# --- Command → File(s) mapping ---
COMMAND_MAP: Dict[str, Union[str, List[str]]] = {
    "train": "train_v50.py",
    "inference": "predict_v50.py",
    "validate": "submission_validator_v50.py",
    "calibrate": "calibration_pipeline.py",
    "submit": ["generate_submission_package.py", "generate_html_report.py"],
    "tune": "auto_ablate_v50.py",
    "ablate": "auto_ablate_v50.py",
    "diagnose": "generate_diagnostic_summary.py",
    "explain": "generate_diagnostic_summary.py",
    "monitor": "logs/train.log",  # external
    "compare": "submission_diff_viewer.py",
    "export": "generate_submission_package.py",
    "cluster_gradients": "gradient_cluster_analyzer.py",
    "shap_cluster": "shap_cluster_overlay.py",
    "shap_metadata": "shap_overlay.py",
    "shap_gradient": "spectral_shap_gradient.py",
    "attention_pca": "attention_summary.py",
    "attention_umap": "attention_summary.py",
    "latent_tsne": "latent_space_mapper.py",
    "latent_umap": "latent_space_mapper.py",
    "latent_decompose": "latent_decomposer.py",
    "latent_drift": "latent_drift_overlay.py",
    "fft_overlay": "fft_overlay.py",
    "selftest": "cli_v50.py",
    "version": "cli_v50.py",
    "corel-train": "train_corel.py",
    "generate-dummy-data": "generate_dummy_v50_test_data.py",
    "analyze-log": "spectramind.py",
}

def show_help_table():
    console = Console()
    table = Table(title="📚 SpectraMind V50 – CLI Command Map", show_lines=False)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Linked File(s)", style="magenta")

    for cmd, files in COMMAND_MAP.items():
        if isinstance(files, list):
            table.add_row(cmd, ", ".join(files))
        else:
            table.add_row(cmd, files)

    console.print(table)

def export_to_markdown(md_path: str = "docs/command_map.md"):
    lines = ["# SpectraMind V50 – Command-to-File Map", "", "| Command | File(s) |", "|---------|---------|"]
    for cmd, files in COMMAND_MAP.items():
        file_str = ", ".join(files) if isinstance(files, list) else files
        lines.append(f"| `{cmd}` | `{file_str}` |")
    Path(md_path).write_text("\n".join(lines))
    print(f"📄 Markdown exported to {md_path}")

def export_to_json(json_path: str = "diagnostics/command_map.json"):
    Path(json_path).write_text(json.dumps(COMMAND_MAP, indent=2))
    print(f"🧾 JSON exported to {json_path}")

def validate_files_exist(verbose: bool = True) -> Dict[str, List[str]]:
    """Returns dict of {missing: [commands]}"""
    missing = []
    for cmd, files in COMMAND_MAP.items():
        for f in (files if isinstance(files, list) else [files]):
            if not Path(f).exists():
                missing.append((cmd, f))
                if verbose:
                    print(f"❌ Missing file for `{cmd}`: {f}")
    if not missing and verbose:
        print("✅ All mapped files found.")
    return {"missing": missing}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SpectraMind CLI Help + File Link Validator")
    parser.add_argument("--show", action="store_true", help="Show CLI table")
    parser.add_argument("--md", type=str, help="Export Markdown")
    parser.add_argument("--json", type=str, help="Export JSON")
    parser.add_argument("--validate", action="store_true", help="Validate all linked files exist")
    args = parser.parse_args()

    if args.show:
        show_help_table()
    if args.md:
        export_to_markdown(args.md)
    if args.json:
        export_to_json(args.json)
    if args.validate:
        validate_files_exist()