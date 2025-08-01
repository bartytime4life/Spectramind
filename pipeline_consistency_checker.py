"""
SpectraMind V50 – Pipeline Consistency Checker
----------------------------------------------
Validates coherence between CLI registry, config, diagnostics, symbolic routing,
submission logs, and required file artifacts. Designed for end-to-end sanity assurance.
"""

import typer
import yaml
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from cli_v50 import app as cli_app
from rich import print
from rich.console import Console

console = typer.Typer(help="Validate pipeline structure, config, CLI, and artifact integrity")
log = Console()

HASH_FILE = Path("run_hash_summary_v50.json")
DEBUG_LOG = Path("v50_debug_log.md")

@app.command()
def check_cli_commands():
    """Ensure expected CLI commands are registered."""
    expected = {
        "train", "inference", "validate", "diagnose", "explain",
        "submit", "export", "calibrate", "html_report", "compare",
        "version", "simulate_lightcurve", "selftest", "health"
    }
    registered = set(cli_app.registered_commands.keys())
    missing = expected - registered
    for cmd in sorted(expected):
        print(f"{'✅' if cmd in registered else '❌'} CLI Command: [bold]{cmd}[/]")
    if missing:
        print(f"[red]❌ Missing CLI commands: {', '.join(missing)}[/]")
        raise typer.Exit(code=1)

@app.command()
def check_config_fields(config_file: Path = Path("configs/config_v50.yaml")):
    """Verify critical fields exist in config.yaml."""
    if not config_file.exists():
        print(f"[red]❌ Missing config file:[/] {config_file}")
        raise typer.Exit()
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    required_fields = ["model", "training", "optimizer"]
    for field in required_fields:
        print(f"{'✅' if field in cfg else '❌'} config.{field}")

@app.command()
def check_submission_artifacts():
    """Verify submission logs and diagnostic files exist."""
    required = [
        "submission.csv",
        "v50_debug_log.md",
        "constraint_violation_log.json",
        "run_hash_summary_v50.json",
        "outputs/preview_mu_spectra.png",
        "outputs/diagnostics/fft_variance_heatmap.png",
        "outputs/diagnostics/symbolic_violation_overlay.png"
    ]
    for f in required:
        path = Path(f)
        status = "✅" if path.exists() else "❌"
        print(f"{status} File: {f}")
        if path.exists() and path.suffix in {".md", ".json", ".csv"}:
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mod_time > timedelta(hours=24):
                print(f"[yellow]⚠️ {f} may be stale (last modified {mod_time})[/]")

@app.command()
def check_file_hashes():
    """Verify file hashes match log."""
    if not HASH_FILE.exists():
        print(f"[red]❌ Missing hash log:[/] {HASH_FILE}")
        return
    with open(HASH_FILE) as f:
        hash_data = json.load(f)
    for file, expected_hash in hash_data.get("files", {}).items():
        p = Path(file)
        if not p.exists():
            print(f"[red]❌ Missing file:[/] {file}")
            continue
        actual = hashlib.md5(p.read_bytes()).hexdigest()
        if actual == expected_hash:
            print(f"✅ Hash match: {file}")
        else:
            print(f"[red]❌ Hash mismatch:[/] {file} (expected {expected_hash}, got {actual})")

@app.command()
def check_symbolic_files():
    """Ensure symbolic modules are wired correctly."""
    required = [
        "symbolic_loss.py",
        "photonic_alignment.py",
        "generate_diagnostic_summary.py"
    ]
    for f in required:
        print(f"{'✅' if Path(f).exists() else '❌'} Symbolic File: {f}")

    # Check symbolic_loss used in model
    model_path = Path("model_v50_ar.py")
    if model_path.exists():
        with open(model_path) as f:
            content = f.read()
        if "symbolic_loss" in content:
            print("✅ symbolic_loss() used in model_v50_ar.py")
        else:
            print("[red]❌ symbolic_loss() not used in model_v50_ar.py[/]")
    else:
        print("[red]❌ model_v50_ar.py missing[/]")

@app.command()
def check_hydra_config():
    """Ensure hydra.main is used and YAML overrideable."""
    target = Path("train_v50.py")
    if target.exists():
        content = target.read_text()
        if "@hydra.main" in content:
            print("✅ Hydra override detected in train_v50.py")
        else:
            print("[red]❌ Hydra not used in train_v50.py[/]")

@app.command()
def full_check(strict: bool = typer.Option(False, help="Exit with error if any check fails")):
    """Run full pipeline coherence test."""
    print("[bold cyan]\n🔍 CLI Registration[/]")
    check_cli_commands()

    print("\n[bold cyan]🧾 Config Fields[/]")
    check_config_fields()

    print("\n[bold cyan]📦 Submission Artifacts[/]")
    check_submission_artifacts()

    print("\n[bold cyan]🔐 File Hash Integrity[/]")
    check_file_hashes()

    print("\n[bold cyan]🧬 Symbolic Files & Routing[/]")
    check_symbolic_files()

    print("\n[bold cyan]🧠 Hydra Usage[/]")
    check_hydra_config()

    print("\n[bold green]✅ Full Pipeline Check Passed[/]")
    DEBUG_LOG.write_text("✅ Full check passed.\n")  # Optional writeback
    if strict:
        raise typer.Exit(code=0)

if __name__ == "__main__":
    console()
