import typer, yaml, json, torch, numpy as np
from pathlib import Path
from datetime import datetime
from spectramind.models.multi_scale_decoder import MultiScaleDecoder
from spectramind.models.flow_uncertainty_head import FlowUncertaintyHead
from spectramind.models.fgs1_mamba import FGS1MambaEncoder
from spectramind.models.airs_gnn import AIRSSpectralGNN
from spectramind.symbolic.symbolic_loss import symbolic_loss
from spectramind.diagnostics.coherence_curve_plot import compute_coherence_curve, plot_and_save
from spectramind.diagnostics.generate_diagnostic_summary import generate as generate_diag
from spectramind.diagnostics.generate_html_report import write_report
from spectramind.diagnostics.symbolic_violation_html_summary import write_html_from_log
from spectramind.calibration.temperature_scaling import apply_temperature_to_sigmas, save_temperature
from spectramind.calibration.spectral_conformal import conformalize_sigmas, save_corel_meta
from spectramind.utils.ripple_logger import update_run_hash_summary

app = typer.Typer(add_completion=False)

def _log(msg: str):
    Path('v50_debug_log.md').write_text(Path('v50_debug_log.md').read_text() + msg + "\n" if Path('v50_debug_log.md').exists() else msg + "\n")

def _load_cfg(path: str):
    return yaml.safe_load(Path(path).read_text())

def _append_violation(rule: str, value: float):
    p = Path('constraint_violation_log.json')
    obj = json.loads(p.read_text()) if p.exists() else {"violations": []}
    obj["violations"].append({"rule": rule, "value": float(value), "ts": datetime.utcnow().isoformat()})
    p.write_text(json.dumps(obj, indent=2))

@app.command()
def version():
    print("SpectraMind V50 CLI v0.4.0")

@app.command()
def selftest(mode: str = typer.Option("fast", help="fast|deep")):
    update_run_hash_summary(f"selftest:{mode}")
    _log(f"selftest:{mode}:{datetime.utcnow().isoformat()}Z")
    print("✅ selftest ok (stub)")

@app.command()
def train(config: str = typer.Option("configs/config_v50.yaml"), dry_run: bool = False, confirm: bool = False, log: bool = True):
    cfg = _load_cfg(config); out_bins = cfg["model"]["out_bins"]
    update_run_hash_summary("train")
    if dry_run and not confirm:
        print("DRY-RUN: would initialize model and perform a quick step."); return
    torch.manual_seed(cfg["training"]["seed"])
    # Fake inputs for encoders
    fgs_in = torch.randn(cfg["training"]["batch_size"], 64)
    airs_in = torch.randn(cfg["training"]["batch_size"], 64)
    fgs = FGS1MambaEncoder(64, cfg["model"]["latent_dim"])(fgs_in)
    air = AIRSSpectralGNN(64, cfg["model"]["latent_dim"])(airs_in)
    z = torch.cat([fgs, air], dim=-1).mean(dim=-1, keepdim=True).repeat(1, cfg["model"]["latent_dim"])
    dec = MultiScaleDecoder(cfg["model"]["latent_dim"], out_bins)
    uq = FlowUncertaintyHead(cfg["model"]["latent_dim"], out_bins)
    mu = dec(z); sigma = uq(z)
    sym = symbolic_loss(mu, cfg["model"]["smooth_lambda"], cfg["model"]["nonneg_lambda"])
    if log: _log(f"symbolic_loss={sym.item():.6f}")
    _append_violation("smoothness+nonneg", float(sym.item()))
    Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
    torch.save({"decoder": dec.state_dict(), "uq": uq.state_dict()}, Path(cfg["paths"]["outputs"])/"model.pt")
    print("✅ train: wrote outputs/model.pt")

@app.command()
def calibrate(config: str = typer.Option("configs/config_v50.yaml"), temperature: float = None):
    cfg = _load_cfg(config)
    update_run_hash_summary("calibrate")
    t = float(temperature if temperature is not None else cfg["calibration"]["temperature"])
    save_temperature(t)
    _log(f"calibration.temperature={t}")
    print(f"✅ calibrated with temperature={t}")

@app.command()
def predict(config: str = typer.Option("configs/config_v50.yaml"), confirm: bool = False, conformal_q: float = 0.0):
    cfg = _load_cfg(config); out_bins = cfg["model"]["out_bins"]
    update_run_hash_summary("predict")
    Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
    # Fake inputs for encoders
    fgs_in = torch.randn(2, 64)
    airs_in = torch.randn(2, 64)
    fgs = FGS1MambaEncoder(64, cfg["model"]["latent_dim"])(fgs_in)
    air = AIRSSpectralGNN(64, cfg["model"]["latent_dim"])(airs_in)
    z = torch.cat([fgs, air], dim=-1).mean(dim=-1, keepdim=True).repeat(1, cfg["model"]["latent_dim"])
    mu = MultiScaleDecoder(cfg["model"]["latent_dim"], out_bins)(z).detach().numpy()
    sg = FlowUncertaintyHead(cfg["model"]["latent_dim"], out_bins)(z).detach().numpy()
    # Temperature scaling
    t_json = Path("outputs/temperature.json")
    t = float(json.loads(t_json.read_text())["temperature"]) if t_json.exists() else float(cfg["calibration"]["temperature"])
    sg = apply_temperature_to_sigmas(sg, t)
    # Optional COREL-style conformal inflation
    if conformal_q and conformal_q > 0.0:
        sg = conformalize_sigmas(sg, quantile=conformal_q)
        save_corel_meta(conformal_q)
    import pandas as pd
    rows = []
    for i, pid in enumerate(["PL_0001","PL_0002"]):
        row = {"planet_id": pid}
        row.update({f"mu_{k}": float(mu[i, k]) for k in range(out_bins)})
        row.update({f"sigma_{k}": float(sg[i, k]) for k in range(out_bins)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(Path(cfg["paths"]["outputs"])/"submission.csv", index=False)
    # Diagnostics: coherence on μ
    curve = compute_coherence_curve(mu)
    plot_and_save(curve, outdir=cfg["paths"]["diagnostics"])
    # HTML
    write_html_from_log()
    write_report()
    print("✅ predict: wrote outputs/submission.csv and diagnostics")

@app.command()
def submit(config: str = typer.Option("configs/config_v50.yaml")):
    cfg = _load_cfg(config)
    update_run_hash_summary("submit")
    sub = Path(cfg["paths"]["outputs"])/"submission.csv"; bundle = Path(cfg["paths"]["outputs"])/"submission_bundle.zip"
    import zipfile
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        if sub.exists(): z.write(sub, arcname=sub.name)
        model = Path(cfg["paths"]["outputs"])/"model.pt"
        if model.exists(): z.write(model, arcname=model.name)
    print(f"📦 bundle: {bundle}")

@app.command()
def diagnose(config: str = typer.Option("configs/config_v50.yaml")):
    update_run_hash_summary("diagnose")
    write_html_from_log()
    html = write_report()
    print(f"🔬 diagnostics: {html}")

if __name__ == "__main__":
    app()
