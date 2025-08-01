"""
SpectraMind V50 – CLI Command Wrappers for Dashboard and Dispatcher
-------------------------------------------------------------------
Supports both subprocess-based CLI dispatch (used by Rich dashboard)
and optional internal module invocation via Typer CLI.
"""

import subprocess
import shlex
import datetime
from pathlib import Path
from rich.console import Console
import typer

console = Console()
app = typer.Typer()
LOG_PATH = Path("v50_debug_log.md")

def _log_cli_dispatch(label: str, cmd: list[str]):
    timestamp = datetime.datetime.now().isoformat()
    line = f"[{label}] [{timestamp}] DISPATCH: {' '.join(map(shlex.quote, cmd))}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)

def _run(cmd: list[str], label: str):
    _log_cli_dispatch(label, cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print_exception()
        raise typer.Exit(code=e.returncode)

# --- Subprocess wrappers (used by cli_dashboard_mini.py) ---

def run_train(dry_run=False, confirm=False):
    console.rule("[bold cyan]Training Model")
    cmd = ["python", "cli_v50.py", "train"]
    if dry_run: cmd.append("--dry-run")
    if confirm: cmd.append("--confirm")
    _run(cmd, "TRAIN")

def run_predict(dry_run=False, confirm=False):
    console.rule("[bold green]Running Inference")
    cmd = ["python", "cli_v50.py", "inference"]
    if dry_run: cmd.append("--dry-run")
    if confirm: cmd.append("--confirm")
    _run(cmd, "INFER")

def run_validate():
    console.rule("[bold yellow]Validating Submission")
    _run(["python", "cli_v50.py", "validate"], "VALIDATE")

def run_finalize():
    console.rule("[bold magenta]Generating Diagnostics + HTML")
    _run(["python", "cli_v50.py", "diagnose"], "DIAGNOSE")
    _run(["python", "cli_v50.py", "html_report"], "HTML")

def run_package():
    console.rule("[bold blue]Packaging Submission")
    _run(["python", "cli_v50.py", "submit", "--html"], "SUBMIT")

def run_all_pipeline(confirm=False):
    console.rule("[bold white]🚀 Full SpectraMind V50 Pipeline")
    run_train(confirm=confirm)
    run_predict(confirm=confirm)
    run_validate()
    run_finalize()
    run_package()

# --- Typer CLI command registration ---

@app.command("train")
def cli_train(dry_run: bool = False, confirm: bool = True):
    """Train the SpectraMind model."""
    run_train(dry_run=dry_run, confirm=confirm)

@app.command("predict")
def cli_predict(dry_run: bool = False, confirm: bool = True):
    """Generate submission.csv via inference."""
    run_predict(dry_run=dry_run, confirm=confirm)

@app.command("validate")
def cli_validate():
    """Validate submission format and contents."""
    run_validate()

@app.command("finalize")
def cli_finalize():
    """Generate diagnostics and HTML report."""
    run_finalize()

@app.command("package")
def cli_package():
    """Zip submission + logs + hashes."""
    run_package()

@app.command("run-all")
def cli_run_all(confirm: bool = True):
    """Full end-to-end run: train → infer → validate → finalize → package"""
    run_all_pipeline(confirm=confirm)

@app.command("register")
def cli_register():
    """Register all CLI command wrappers for SpectraMind V50."""
    console.rule("[bold white]Registering CLI commands")
    console.print("[green]✅ All CLI command wrappers registered into dispatcher.")

if __name__ == "__main__":
    app()
