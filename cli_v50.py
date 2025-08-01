"""
SpectraMind V50 – Unified Command Line Interface
-------------------------------------------------
Typer-based CLI for full-stack orchestration across:

• Training & calibration           • Inference & validation
• Symbolic diagnostics             • SHAP overlays & submission auditing
• COREL refinement                 • Transfer graph visualization
• Lightcurve simulation            • Self-test workflows & health checks
• ZIP packaging & report generation
"""

import typer
import torch
from pathlib import Path

from selftest import check_system_health
from generate_submission_package import generate_zip

app = typer.Typer(help="SpectraMind V50 – CLI entry point for scientific pipeline operations")

@app.command("calibrate")
def calibrate(planet_id: str):
    """Run full calibration pipeline for a single planet"""
    from calibration_pipeline import run_full_calibration
    run_full_calibration(planet_id)

@app.command("train")
def train(
    config: Path = typer.Option("configs/config_v50.yaml"),
    decoder: str = typer.Option("moe", help="Decoder: moe | diffusion | quantile"),
    auto_package: bool = typer.Option(True, help="Package results post-training")
):
    """Train model using Hydra config and package results"""
    import yaml
    from train_v50 import train_from_config
    cfg = yaml.safe_load(open(config))
    cfg.setdefault("model", {})["decoder_type"] = decoder
    train_from_config(cfg)
    if auto_package:
        generate_zip()

@app.command("retrain-symbolic")
def retrain_symbolic():
    """Retrain using symbolic violation feedback log"""
    from symbolic_feedback_trainer import retrain_symbolic
    retrain_symbolic()

@app.command("master-train")
def master_train():
    """Run master training script and package outputs"""
    from master import main as master_main
    master_main()
    generate_zip()

@app.command("transfer-graph")
def transfer_graph():
    """Generate and save latent→μ transfer graph from Z and μ tensors"""
    from spectral_transfer_graph import build_transfer_graph, save_graph_json
    z = torch.load("outputs/z_tensor.pt")
    mu = torch.load("outputs/mu.pt")
    G = build_transfer_graph(z, mu, threshold=0.15)
    save_graph_json(G)

@app.command("plot-transfer-graph")
def plot_transfer_graph():
    """Visualize bipartite latent→μ graph with overlays"""
    from plot_transfer_graph import plot_transfer_graph as viz
    viz()

@app.command("diagnostics-html")
def diagnostics_html():
    """Run SHAP + symbolic + COREL diagnostics and generate HTML report"""
    from generate_diagnostic_summary import generate_diagnostic_summary
    from symbolic_rule_scorer import score_symbolic_rules
    from generate_html_report import generate_html_report
    dummy = torch.zeros((3, 283))
    generate_diagnostic_summary(dummy, dummy + 1, dummy + 0.5, symbolic_config={"smoothness": True})
    score_symbolic_rules()
    generate_html_report()

@app.command("validate")
def validate(submission: Path = Path("submission.csv")):
    """Validate submission.csv format, size, and contents"""
    from submission_validator_v50 import validate_submission
    validate_submission(str(submission))

@app.command("inference")
def inference():
    """Run inference using latest model checkpoint on test set"""
    from predict_v50 import run
    run()

@app.command("diagnose")
def diagnose():
    """Re-run symbolic and SHAP diagnostics using dummy overlays"""
    from generate_diagnostic_summary import generate_diagnostic_summary
    import numpy as np
    dummy = np.zeros((3, 283))
    generate_diagnostic_summary(dummy, dummy + 1, dummy + 0.5)

@app.command("simulate-lightcurve")
def simulate_lightcurve(mu_csv: Path, meta_csv: Path, output: Path = Path("outputs/simulated_lightcurve.png")):
    """Simulate toy transit lightcurve from μ + metadata"""
    from simulate_lightcurve_from_mu import simulate_lightcurve as run
    run(mu_csv=mu_csv, meta_csv=meta_csv, output=output)

@app.command("rule-attention-overlay")
def rule_attention_overlay():
    """Overlay symbolic rule mask with decoder attention"""
    from latent_rule_attention_overlay import LatentRuleAttentionOverlay
    attn = torch.rand(4, 283)
    rules = (torch.rand(4, 283) > 0.7).float()
    overlay = LatentRuleAttentionOverlay(rule_map=rules)
    masked = overlay.overlay(attn)
    overlay.plot(masked)

@app.command("submit")
def submit():
    """Package submission.zip with all required artifacts"""
    generate_zip()

@app.command("html-report")
def html_report():
    """Generate diagnostic HTML report from logs + outputs"""
    from generate_html_report import generate_html_report
    generate_html_report()

@app.command("explain")
def explain():
    """Re-run SHAP + symbolic overlays using current outputs"""
    from generate_diagnostic_summary import generate_diagnostic_summary
    generate_diagnostic_summary()

@app.command("export")
def export():
    """Alias for submit() – saves submission.zip"""
    generate_zip()

@app.command("compare")
def compare(a: Path, b: Path):
    """Compare two submission.csv files visually"""
    from submission_diff_viewer import compare_submissions
    compare_submissions(a, b)

@app.command("selftest")
def selftest():
    """Run CLI registry + file validation tests"""
    from selftest import run_selftest
    run_selftest()

@app.command("selftest-workflow")
def selftest_workflow():
    """Run train → inference → submit test pipeline"""
    from selftest import validate_submission_workflow
    validate_submission_workflow()

@app.command("health")
def health():
    """Run full system pipeline integrity check"""
    check_system_health()

@app.command("version")
def version():
    """Print CLI version info"""
    print("SpectraMind V50 CLI – version 0.1.0")

@app.command("conformalize")
def conformalize(
    model_path: Path = Path("models/corel_gnn.pt"),
    mu_file: Path = Path("outputs/mu.pt"),
    sigma_file: Path = Path("outputs/sigma.pt"),
    edge_file: Path = Path("calibration_data/edge_index.pt")
):
    """Apply COREL conformal prediction to refine μ + σ"""
    from corel_inference import load_corel_model, apply_corel
    mu = torch.load(mu_file)
    sigma = torch.load(sigma_file)
    edge_index = torch.load(edge_file)
    model = load_corel_model(str(model_path))
    mu_corr, radius_corr = apply_corel(model, mu, sigma, edge_index)
    torch.save(mu_corr, "outputs/mu_corel.pt")
    torch.save(radius_corr, "outputs/sigma_corel.pt")
    print("✅ COREL refinement complete → outputs/mu_corel.pt")

if __name__ == "__main__":
    app()
