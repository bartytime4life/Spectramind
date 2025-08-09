import typer
from pathlib import Path

app = typer.Typer(help="SpectraMind V50 – Unified CLI")

@app.command()
def version():
    typer.echo("SpectraMind V50 0.1.0")

@app.command()
def check_pipeline():
    # Lightweight check
    need = ["configs/config_v50.yaml"]
    missing = [p for p in need if not Path(p).exists()]
    if missing:
        typer.echo(f"Missing: {missing}")
        raise typer.Exit(1)
    typer.echo("Pipeline check passed.")

if __name__ == "__main__":
    app()
