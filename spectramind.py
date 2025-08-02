"""
SpectraMind V50 – Unified CLI Interface
---------------------------------------
Main command-line interface for:
- Training and inference
- Diagnostics and dashboard generation
- Submission orchestration
- System health self-testing

Logs version + config hash on every CLI call.
"""

import typer
from pathlib import Path
from datetime import datetime
import json
import sys

from cli_core_v50 import app as core_app
from cli_diagnose import app as diagnose_app
from cli_submit import app as submit_app
from selftest import app as test_app

app = typer.Typer(help="SpectraMind V50 – Unified CLI for scientific modeling and submission")

# Constants
__VERSION__ = "v50.1.0"
__BUILD_TIME__ = datetime.utcnow().isoformat()
__HASH_FILE__ = Path("run_hash_summary_v50.json")
__LOG_FILE__ = Path("v50_debug_log.md")

def get_latest_config_hash():
    if __HASH_FILE__.exists():
        with open(__HASH_FILE__) as f:
            data = json.load(f)
            if data:
                last_tag = list(data.keys())[-1]
                return data[last_tag].get("hash", "unknown")
    return "unknown"

def log_cli_call():
    cmd = " ".join(sys.argv)
    hash_val = get_latest_config_hash()
    now = datetime.utcnow().isoformat()
    entry = f"\n### CLI Call @ {now}\n- Command: `{cmd}`\n- Version: {__VERSION__}\n- Config Hash: {hash_val}\n"
    if __LOG_FILE__.exists():
        __LOG_FILE__.write_text(__LOG_FILE__.read_text() + entry)
    else:
        __LOG_FILE__.write_text(entry)

@app.callback()
def main(
    version: bool = typer.Option(None, "--version", help="Show CLI version and config hash")
):
    if version:
        typer.echo(f"SpectraMind CLI {__VERSION__}")
        typer.echo(f"Build Time: {__BUILD_TIME__}")
        typer.echo(f"Config Hash: {get_latest_config_hash()}")
        raise typer.Exit()
    log_cli_call()

# Register subcommands
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

@app.command("analyze-log")
def analyze_log(limit: int = typer.Option(10, help="Max entries to show from debug log")):
    """Parse recent CLI calls from v50_debug_log.md and display as Markdown table"""
    if not __LOG_FILE__.exists():
        typer.secho("❌ No log file found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    entries = []
    with open(__LOG_FILE__) as f:
        lines = f.readlines()

    current = {}
    for line in lines:
        if line.startswith("### CLI Call @ "):
            if current:
                entries.append(current)
                current = {}
            current["time"] = line.strip().split("@ ")[1]
        elif line.startswith("- Command: "):
            current["command"] = line.strip().split("`")[1]
        elif line.startswith("- Version:"):
            current["version"] = line.strip().split(": ")[1]
        elif line.startswith("- Config Hash:"):
            current["hash"] = line.strip().split(": ")[1]

    if current:
        entries.append(current)

    entries = sorted(entries, key=lambda x: x["time"], reverse=True)[:limit]

    typer.echo("| Time (UTC)           | Command                                | Version  | Config Hash       |")
    typer.echo("|----------------------|-----------------------------------------|----------|-------------------|")
    for e in entries:
        typer.echo(f"| {e['time'][:19]} | {e['command'][:41]:<41} | {e['version']} | {e['hash'][:15]}... |")

if __name__ == "__main__":
    app()