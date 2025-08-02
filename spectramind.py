import typer
import subprocess
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from rich.console import Console
from cli_explain_util import validate_files_exist, export_to_markdown, export_to_json

app = typer.Typer()

__LOG_FILE__ = Path("v50_debug_log.md")

@app.command("corel-train")
def corel_train(
    input_dir: Path = typer.Option("calibration_data/"),
    output_path: Path = typer.Option("models/corel_gnn.pt"),
    epochs: int = typer.Option(50)
):
    typer.secho(f"\U0001f9e0 Training SpectralCOREL from {input_dir} for {epochs} epochs...", fg=typer.colors.CYAN)
    cmd = f"python train_corel.py --input_dir {input_dir} --output_path {output_path} --epochs {epochs}"
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        typer.secho(f"✅ Training complete. Model saved to {output_path}", fg=typer.colors.GREEN)
    else:
        typer.secho("❌ COREL training failed. Check logs or inputs.", fg=typer.colors.RED)

@app.command("check-cli-map")
def check_cli_command_map():
    console = Console()
    console.rule("[bold yellow]