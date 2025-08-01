"""
SpectraMind V50 – Unified Command Line Interface
-------------------------------------------------
Modular Typer-based CLI for full pipeline orchestration across:

• Training & retraining        • Calibration & inference
• Validation & submission      • Symbolic QA & SHAP overlays
• Transfer graph diagnostics   • COREL & conformal uncertainty refinement
• Simulation, packaging, and   • Self-test workflows & HTML reports
  spectrum visualizations
"""

import typer
import torch
from pathlib import Path

from selftest import check_system_health
from generate_submission_package import generate_zip

app = typer.Typer(help="SpectraMind V50 – Exoplanetary Spectrum Inference CLI")


@app.command("calibrate", help="Run full calibration on a single planet")
def calibrate(planet_id: str):
    from calibration_pipeline import run_full_calibration
    run_full_calibration(planet_id)


@app.command("train", help="Train model from Hydra YAML config and package")
def train(
    config: Path = typer.Option("configs/config_v50.yaml"),
    decoder: str = typer.Option("moe", help="Decoder: moe | diffusion | quantile"),
    auto_package: bool = typer.Option(True, help="Package results post-training")
):
    import yaml
    from train_v50 import train_from_config
    cfg = yaml.safe_load(open(config))
    cfg.setdefault("model", {})["decoder_type"] = decoder
    train_from_config(cfg)
    if auto_package:
        generate_zip()


@app.command("retrain-symbolic", help="Retrain model on symbolic constraint violations")
def retrain_symbolic():
    from symbolic_feedback_trainer import retrain_symbolic
    retrain_symbolic()


@app.command("master-train", help="Run full training pipeline + package")
def master_train():
    from master import main as master_main
    master_main()
    generate_zip()


@app.command("transfer-graph", help="Build and save latent→μ transfer graph")
def transfer_graph():
    from spectral_transfer_graph import build_transfer_graph, save_graph_json
    z = torch.load("outputs/z_tensor.pt")
    mu = torch.load("outputs/mu.pt")
    G = build_transfer_graph(z, mu, threshold=0.15)
    save_graph_json(G)


@app.command("plot-transfer-graph", help="Plot transfer graph as bipartite layout")
def plot_transfer_graph():
    from plot_transfer_graph import plot_transfer_graph as viz
    viz()


@app.command("diagnostics-html", help="COREL + symbolic + SHAP + HTML diagnostics")
def diagnostics_html():
    from generate_diagnostic_summary import generate_diagnostic_summary
    from symbolic_rule_scorer import score_symbolic_rules
    from generate_html_report import generate_html_report
    dummy = torch.zeros((3, 283))
    print("📊 Generating diagnostic summary...")
    generate_diagnostic_summary(dummy, dummy + 1, dummy + 0.5, symbolic_config={"smoothness": True})
    print("📐 Scoring symbolic rule influence...")
    score_symbolic_rules()
    print("📝 Creating HTML report...")
    generate_html_report()


@app.command("validate", help="Validate submission.csv format and contents")
def validate(submission: Path = Path("submission.csv")):
    from submission_validator_v50 import validate_submission
    validate_submission(str(submission))


@app.command("inference", help="Run inference on test set")
def inference():
    from predict_v50 import run
    run()


@app.command("diagnose", help="Symbolic + SHAP + residual diagnostic overlays")
def diagnose():
    from generate_diagnostic_summary import generate_diagnostic_summary
    import numpy as np
    dummy = np.zeros((3, 283))
    generate_diagnostic_summary(dummy, dummy + 1, dummy + 0.5)


@app.command("simulate-lightcurve", help="Simulate lightcurve from μ spectrum + metadata")
def simulate_lightcurve(
    mu_csv: Path = typer.Option(...),
    meta_csv: Path = typer.Option(...),
    output: Path = typer.Option("outputs/simulated_lightcurve.png")
):
    from simulate_lightcurve_from_mu import simulate_lightcurve as run
    run(mu_csv=mu_csv, meta_csv=meta_csv, output=output)


@app.command("rule-attention-overlay", help="Visualize symbolic rule × attention overlay")
def rule_attention_overlay():
    from latent_rule_attention_overlay import LatentRuleAttentionOverlay
    attn = torch.rand(4, 283)
    rules = (torch.rand(4, 283) > 0.7).float()
    overlay = LatentRuleAttentionOverlay(rule_map=rules)
    masked = overlay.overlay(attn)
    overlay.plot(masked)


@app.command("submit", help="Package submission.zip (submission.csv + logs + manifest)")
def submit():
    generate_zip()


@app.command("html-report", help="Regenerate diagnostics HTML from existing logs")
def html_report():
    from generate_html_report import generate_html_report
    generate_html_report()


@app.command("explain", help="Re-run SHAP + symbolic diagnostic suite")
def explain():
    from generate_diagnostic_summary import generate_diagnostic_summary
    generate_diagnostic_summary()


@app.command("export", help="Export zipped package (alias for submit)")
def export():
    generate_zip()


@app.command("compare", help="Compare two submission files (side-by-side heatmaps)")
def compare(a: Path, b: Path):
    from submission_diff_viewer import compare_submissions
    compare_submissions(a, b)


@app.command("selftest", help="Run CLI integrity and file presence tests")
def selftest():
    from selftest import run_selftest
    run_selftest()


@app.command("selftest-workflow", help="Test train → inference → submit pipeline")
def selftest_workflow():
    from selftest import validate_submission_workflow
    validate_submission_workflow()


@app.command("health", help="Run system-wide pipeline health check")
def health():
    check_system_health()


@app.command("version", help="Print CLI version")
def version():
    print("SpectraMind V50 CLI – version 0.1.0")


@app.command("conformalize", help="Refine μ and σ using COREL GNN conformal model")
def conformalize(
    model_path: Path = Path("models/corel_gnn.pt"),
    mu_file: Path = Path("outputs/mu.pt"),
    sigma_file: Path = Path("outputs/sigma.pt"),
    edge_file: Path = Path("calibration_data/edge_index.pt"),
):
    from corel_inference import load_corel_model, apply_corel
    mu = torch.load(mu_file)
    sigma = torch.load(sigma_file)
    edge_index = torch.load(edge_file)
    model = load_corel_model(str(model_path))
    mu_corr, radius_corr = apply_corel(model, mu, sigma, edge_index)
    torch.save(mu_corr, "outputs/mu_corel.pt")
    torch.save(radius_corr, "outputs/sigma_corel.pt")
    print("✅ COREL refinement complete. Output saved to outputs/")


if __name__ == "__main__":
    app()
