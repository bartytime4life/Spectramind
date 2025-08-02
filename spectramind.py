"""
SpectraMind V50 – Unified CLI Interface
---------------------------------------
Main CLI for:
• Training and inference
• Diagnostics and dashboards
• Submission orchestration
• System self-test + logging
• CLI call history analysis with grouped exports
• Dummy test data generation
"""

import typer
from pathlib import Path
from datetime import datetime
import json
import sys
import csv
from collections import defaultdict

from cli_core_v50 import app as core_app
from cli_diagnose import app as diagnose_app
from cli_submit import app as submit_app
from selftest import app as test_app

app = typer.Typer(help="SpectraMind V50 – Unified CLI for training, submission, diagnostics")

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
def analyze_log(
    limit: int = typer.Option(10, help="Max entries to show in terminal"),
    out_csv: Path = typer.Option("diagnostics/cli_log.csv"),
    out_md: Path = typer.Option("diagnostics/log_table.md")
):
    """Parse CLI log into Markdown and CSV, grouped by config hash"""
    if not __LOG_FILE__.exists():
        typer.secho("❌ No v50_debug_log.md found.", fg=typer.colors.RED)
        raise typer.Exit()

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

    # Sort by time descending
    entries = sorted(entries, key=lambda x: x["time"], reverse=True)

    # Terminal Markdown table preview
    typer.echo("| Time (UTC)           | Command                                | Version  | Config Hash       |")
    typer.echo("|----------------------|-----------------------------------------|----------|-------------------|")
    for e in entries[:limit]:
        typer.echo(f"| {e['time'][:19]} | {e['command'][:41]:<41} | {e['version']} | {e['hash'][:15]}... |")

    # --- Export grouped Markdown ---
    grouped = defaultdict(list)
    for e in entries:
        grouped[e["hash"]].append(e)

    lines = []
    for h, group in grouped.items():
        lines.append(f"\n### Config Hash: `{h}`\n")
        lines.append("| Time (UTC)           | Command                                | Version  |")
        lines.append("|----------------------|-----------------------------------------|----------|")
        for e in group:
            lines.append(f"| {e['time'][:19]} | {e['command'][:41]:<41} | {e['version']} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    typer.echo(f"📄 Markdown log written to: {out_md}")

    # --- CSV Export ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "command", "version", "config_hash"])
        for e in entries:
            writer.writerow([e["time"], e["command"], e["version"], e["hash"]])
    typer.echo(f"📊 CSV log written to: {out_csv}")

@app.command("generate-dummy-data")
def generate_dummy_data():
    """Generate dummy μ, σ, y, COREL model, edge_index, and symbolic overlays."""
    try:
        import generate_dummy_v50_test_data
        typer.secho("✅ Dummy V50 test data successfully generated.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"❌ Failed to generate dummy data: {e}", fg=typer.colors.RED)

if __name__ == "__main__":
    app()