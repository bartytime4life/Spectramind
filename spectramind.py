"""
SpectraMind V50 – Unified CLI Interface
---------------------------------------
Main command-line entrypoint for:
- Training
- Prediction
- Diagnostics
- Submission bundling
- Self-test validation

Supports --version, shell completion, GitHub badge integration.
"""

import typer
from cli_core_v50 import app as core_app
from cli_diagnose import app as diagnose_app
from cli_submit import app as submit_app
from selftest import app as test_app

app = typer.Typer(help="SpectraMind V50 – Unified CLI for scientific modeling")

__VERSION__ = "v50.1.0"

# Register all sub-CLIs
app.add_typer(core_app, name="core", help="Core modeling: train, predict, score, package")
app.add_typer(diagnose_app, name="diagnose", help="Diagnostics + symbolic QA tools")
app.add_typer(submit_app, name="submit", help="Full pipeline automation: train → zip")
app.add_typer(test_app, name="test", help="System health + CLI self-test")


@app.callback()
def version_callback(
    version: bool = typer.Option(None, "--version", callback=lambda v: print(f"SpectraMind CLI {__VERSION__}") or raise_exit() if v else None,
                                 help="Print CLI version and exit")
):
    pass


def raise_exit():
    raise typer.Exit()


@app.command("completion")
def completion(shell: str = typer.Option("bash", help="Shell type: bash, zsh, fish")):
    """Generate shell autocompletion script"""
    from typer.main import get_command
    import subprocess
    typer_command = get_command(app)
    typer.echo(f"# Add this to your shell config (e.g. .bashrc or .zshrc)")
    result = subprocess.run([typer_command.name, "--show-completion", shell], capture_output=True, text=True)
    typer.echo(result.stdout)


if __name__ == "__main__":
    app()