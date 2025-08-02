"""
SpectraMind V50 – Submission Packaging Script
---------------------------------------------
Bundles submission.csv, diagnostics, symbolic logs, COREL outputs,
transfer graphs, and HTML reports into a reproducible ZIP archive.

Supports CLI flags:
- --symbolic-only      Include only symbolic and reproducibility files
- --diagnostics-only   Run only diagnostics (skip finalizer)
- --finalize-only      Run only finalizer (skip diagnostics)
"""

import zipfile
import os
import datetime
import argparse
import torch
from pathlib import Path

from v50_pipeline_finalizer import finalize_submission
from generate_diagnostic_summary import generate_diagnostic_summary
from generate_html_report import generate_html_report
from symbolic_rule_scorer import score_symbolic_rules
from corel_inference import load_corel_model, apply_corel
from plot_transfer_graph import plot_transfer_graph

FULL_FILES = [
    "submission.csv",
    "manifest_v50.csv",
    "manifest_v50.json",
    "run_hash_summary_v50.json",
    "config_v50.yaml",
    "spectramind.toml",
    "constraint_violation_log.json",
    "outputs/diagnostics/summary.json",
    "diagnostics/zscore_distribution.png",
    "diagnostics/sigma_vs_residual_std.png",
    "diagnostics/fft_residuals.png",
    "outputs/transfer_graph.png",
    "outputs/report.html",
    "outputs/symbolic_rule_scores.csv",
    "outputs/mu_corel.pt",
    "outputs/sigma_corel.pt"
]

SYMBOLIC_ONLY_FILES = [
    "submission.csv",
    "spectramind.toml",
    "constraint_violation_log.json",
    "outputs/diagnostics/summary.json",
    "outputs/symbolic_rule_scores.csv",
    "outputs/transfer_graph.png",
    "outputs/report.html"
]

def run_corel_if_available():
    mu_path = Path("outputs/mu.pt")
    sigma_path = Path("outputs/sigma.pt")
    edge_path = Path("calibration_data/edge_index.pt")
    model_path = Path("models/corel_gnn.pt")

    if all(p.exists() for p in [mu_path, sigma_path, edge_path, model_path]):
        print("🧠 Refining μ and σ with COREL...")
        mu = torch.load(mu_path)
        sigma = torch.load(sigma_path)
        edge_index = torch.load(edge_path)
        model = load_corel_model(str(model_path))
        mu_corr, sigma_corr = apply_corel(model, mu, sigma, edge_index)
        torch.save(mu_corr, "outputs/mu_corel.pt")
        torch.save(sigma_corr, "outputs/sigma_corel.pt")
        print(f"✅ COREL complete: μ{tuple(mu_corr.shape)}, σ{tuple(sigma_corr.shape)}")
    else:
        print("⚠️ Skipping COREL: required files missing.")

def generate_zip(symbolic_only=False, finalize_only=False, diagnostics_only=False):
    if not diagnostics_only:
        print("🚀 Finalizing pipeline...")
        finalize_submission()

    if not finalize_only:
        print("📊 Running diagnostics...")
        dummy = torch.zeros((3, 283))
        generate_diagnostic_summary(dummy, dummy + 1, dummy + 0.5, symbolic_config={"smoothness": True})
        run_corel_if_available()
        score_symbolic_rules()
        plot_transfer_graph()
        generate_html_report()

    print("📦 Archiving ZIP package...")
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zipname = f"spectramind_v50_submission_{timestamp}.zip"

    selected_files = SYMBOLIC_ONLY_FILES if symbolic_only else FULL_FILES
    missing = []

    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in selected_files:
            if os.path.exists(path):
                zipf.write(path, arcname=os.path.relpath(path))
                print(f"✅ Included: {path}")
            else:
                print(f"⚠️ Missing: {path}")
                missing.append(path)

    print(f"\n✅ Submission ZIP created: {zipname}")

    with open("v50_debug_log.md", "a") as log:
        log.write(f"\n## 📦 Submission Package: `{zipname}`\n")
        log.write(f"⏱ Timestamp: {timestamp}\n")
        for f in selected_files:
            log.write(f"- {'[MISSING] ' if f in missing else ''}{f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbolic-only", action="store_true", help="Include only symbolic and reproducibility files")
    parser.add_argument("--diagnostics-only", action="store_true", help="Run only diagnostics (skip finalizer)")
    parser.add_argument("--finalize-only", action="store_true", help="Run only finalizer (skip diagnostics)")
    args = parser.parse_args()

    if args.finalize_only and args.diagnostics_only:
        raise ValueError("❌ Cannot use --finalize-only and --diagnostics-only together.")

    generate_zip(
        symbolic_only=args.symbolic_only,
        finalize_only=args.finalize_only,
        diagnostics_only=args.diagnostics_only
    )