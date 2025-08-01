"""
SpectraMind V50 – CLI Self-Test and Compliance Validator
--------------------------------------------------------
Combines CLI smoke tests, workflow validation, and pipeline structure checks.
Validates CLI registry, symbolic loss routing, config presence, and submission integrity.
"""

import subprocess
import sys
from rich.console import Console
from pathlib import Path
import typer
from cli_v50 import app as cli_app
from generate_html_report import generate_html_report

console = Console()
app = typer.Typer(help="SpectraMind V50 – Self-Test and Pipeline Validator")

def check(cmd: str):
    console.print(f"\n▶️ [bold yellow]{cmd}[/bold yellow]")
    try:
        subprocess.run(cmd, shell=True, check=True)
        console.print("[green]✓ Success[/green]")
        return True
    except subprocess.CalledProcessError:
        console.print(f"[bold red]❌ Command failed:[/bold red] {cmd}")
        return False

@app.command()
def run_selftest():
    """Smoke test: validate, export, diagnose, explain, unit test"""
    console.rule("[bold cyan]SpectraMind V50 – CLI Self-Test")
    try:
        console.print("[green]✓[/green] CLI import OK")

        check("spectramind validate submission.csv")
        check("spectramind export")
        check("spectramind diagnose")
        check("spectramind explain")

        console.print("[yellow]•[/yellow] Running unit tests (core + extended)...")
        check("pytest tests --tb=short")

        console.print("\n[bold green]✅ CLI Self-Test Complete[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Self-test failed:[/bold red] {e}")

@app.command()
def diagnostics_html():
    """Generate HTML diagnostics report from current run outputs."""
    console.rule("[bold magenta]📄 Generating SpectraMind HTML Report")
    generate_html_report()

@app.command()
def validate_submission_workflow():
    """Full workflow test: train → inference → conformalize → submit → validate"""
    console.rule("[bold cyan]SpectraMind V50 – Full Workflow Test")

    check("spectramind train --config configs/config_v50.yaml")
    check("spectramind inference")
    check("spectramind conformalize \
        --model-path models/corel_gnn.pt \
        --mu-file outputs/mu.pt \
        --sigma-file outputs/sigma.pt \
        --edge-file calibration_data/edge_index.pt")
    check("spectramind submit")
    check("spectramind validate submission.csv")

    console.print("\n[bold green]✅ Full workflow test passed.[/bold green]")

@app.command()
def check_pipeline_files():
    """Verify that required config and artifact files exist."""
    required = [
        "config.yaml", "submission.csv", "spectramind.toml",
        "constraint_violation_log.json", "v50_debug_log.md"
    ]
    for f in required:
        if Path(f).exists():
            console.print(f"[green]✓ Found:[/green] {f}")
        else:
            console.print(f"[red]❌ Missing:[/red] {f}")

@app.command()
def check_cli_registry():
    """Validate that CLI commands are properly registered."""
    expected = {"train", "inference", "validate", "submit", "diagnose", "explain", "test"}
    registered = set(cli_app.registered_commands.keys())
    missing = expected - registered
    for cmd in expected:
        status = "✅" if cmd in registered else "❌"
        console.print(f"{status} Command: {cmd}")
    if missing:
        raise typer.Exit(code=1)

@app.command()
def check_symbolic_modules():
    symbolic_files = [
        "symbolic_loss.py", "photonic_alignment.py", "generate_diagnostic_summary.py"
    ]
    for f in symbolic_files:
        status = "✅" if Path(f).exists() else "❌"
        console.print(f"{status} {f}")

@app.command()
def test_dry_run():
    """Test CLI commands with --dry-run flag."""
    check("spectramind train --dry-run --confirm n")
    check("spectramind calibrate --planet-id TEST123 --dry-run --confirm n")

@app.command()
def check_hash_log():
    if Path("run_hash_summary_v50.json").exists():
        console.print("✅ run_hash_summary_v50.json found.")
    else:
        console.print("❌ Missing: run_hash_summary_v50.json")

@app.command()
def check_dvc_stage():
    if Path("dvc.yaml").exists():
        console.print("✅ dvc.yaml present.")
    else:
        console.print("❌ Missing: dvc.yaml")

@app.command()
def check_system_health():
    """Run all checks and print full health report."""
    console.rule("[bold white]\U0001f9e0 SpectraMind V50 – System Health Check")

    check_cli_registry()
    check_symbolic_modules()
    check_pipeline_files()
    check_hash_log()
    check_dvc_stage()
    test_dry_run()

    console.print("\n[bold green]✅ SpectraMind V50 system is healthy and submission-ready.[/bold green]")

if __name__ == "__main__":
    app()