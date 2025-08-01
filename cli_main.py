"""
SpectraMind V50 – Unified CLI Entry Point
-----------------------------------------
Top-level CLI for training, inference, diagnostics, submission packaging, and validation.
"""

import click
from rich.console import Console

# Subcommand groups
from cli_core import core           # train, predict, tune, score
from cli_validate import validate   # dataset + submission validation
from cli_diag import diag           # SHAP, FFT, latent/fusion diagnostics
from cli_bundle import bundle       # full-run orchestrator + zip
# from cli_utils import utils       # optional tools: version, test-pipeline, hash-check

console = Console()

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("1.0.0", prog_name="SpectraMind V50 CLI")
def cli():
    """
    🌌 SpectraMind CLI – NeurIPS 2025 Ariel Data Challenge

    Modular command-line interface for the full AI modeling and scientific diagnostics pipeline.
    Use --help on subgroups (e.g. cli core --help) to explore available commands.
    """
    console.print("[bold cyan]Welcome to the SpectraMind V50 CLI[/] 🌠")
    console.print("Use [yellow]--help[/] on any command group to see options.\n")


# Register active command groups
cli.add_command(core)
cli.add_command(validate)
cli.add_command(diag)
cli.add_command(bundle)
# cli.add_command(utils)  # optional tools

if __name__ == "__main__":
    cli()
