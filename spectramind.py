"""
SpectraMind V50 – Unified CLI Interface
---------------------------------------
Main command-line interface for:
- Training and inference
- Diagnostics and dashboard generation
- Submission orchestration
- System health self-testing

Includes versioning, shell completion, and config hash display.
"""

import typer
from pathlib import Path
from datetime import datetime
import json

from cli_core_v50 import app as core_app
from cli_diagnose import app as diagnose_app
from cli_submit import app as submit_app
from selftest import app as test_app

app = typer.Typer(help="SpectraMind V50 – Unified CLI for scientific modeling and submission")

# Version and hash constants
__VERSION__ = "v50.1.0"
__BUILD_TIME__ = datetime.utcnow().isoformat()
__HASH_FILE__ = Path("run_hash_summary_v50.json")

def get_latest_config_hash():
    if __HASH_FILE__.exists():
        with open(__HASH_FILE__) as f:
            data = json.load(f)
            if data:
                last_tag = list(data.keys())[-1]
                return data[last_tag].get("hash", "unknown")
    return "unknown"

@app.callback()
def main(
    version: bool = typer.Option(None, "--version", help="Show CLI version and config hash")
):
    if version:
        typer.echo(f"SpectraMind CLI {__VERSION__}")
        typer.echo(f"Build Time: {__BUILD_TIME__}")
        typer.echo(f"Config Hash: {get_latest_config_hash()}")
        raise typer.Exit()

# Subcommands
app.add_typer(core_app, name="core", help="Core model: train, predict, package")
app.add_typer(diagnose_app, name="diagnose", help="Diagnostics: SHAP, symbolic, UMAP, t-SNE")
app.add_typer(submit_app, name="submit", help="Train → inference → zip submission")
app.add_typer(test_app, name="test", help="CLI and pipeline system validator")

@app.command("completion")
def completion(shell: str = typer.Option("bash", help="Shell type: bash, zsh, fish")):
    """Generate shell completion script"""
    from typer.main import get_command
    command = get_command(app)
    typer.echo(f"# Add this to your shell config (e.g. ~/.bashrc)")
    typer.echo(f"source <({command.name} --show-completion {shell})")

if __name__ == "__main__":
    app()