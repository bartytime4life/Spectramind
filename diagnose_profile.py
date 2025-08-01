"""
SpectraMind V50 – CLI Diagnostics Dashboard
-------------------------------------------
Entry point for full diagnostics run: symbolic, SHAP, FFT, DAG view, etc.
"""

import typer
import subprocess
import os
from pathlib import Path
from cli_guardrails import guardrails_wrapper, _log_cli_call

app = typer.Typer(help="Run diagnostic profile suite for SpectraMind V50")

@app.command()
def diagnose_profile(
    submission: Path = typer.Option("submission.csv", help="Path to μ, σ submission"),
    outdir: Path = typer.Option("outputs/diagnostics", help="Diagnostics output directory"),
    dry_run: bool = False,
    confirm: bool = True,
    skip_fft: bool = False,
    skip_symbolic: bool = False,
    skip_shap: bool = False,
    skip_dag: bool = False
):
    """
    Full profiling: FFT, symbolic overlays, DAG graph, uncertainty entropy, SHAP, etc.
    """
    args = {
        "submission": str(submission),
        "outdir": str(outdir),
        "dry_run": dry_run,
        "confirm": confirm,
        "skip_fft": skip_fft,
        "skip_symbolic": skip_symbolic,
        "skip_shap": skip_shap,
        "skip_dag": skip_dag
    }
    guardrails_wrapper("diagnose_profile", args, dry_run=dry_run, confirm=confirm)

    os.makedirs(outdir, exist_ok=True)

    def run_step(label: str, script: str, skip: bool = False, extra_args: list = None):
        if skip:
            typer.echo(f"⏭️  Skipping {label}")
            return
        cmd = ["python", script, "--submission", str(submission), "--outdir", str(outdir)]
        if extra_args:
            cmd += extra_args
        typer.echo(f"🔍 {label}...")
        _log_cli_call(script, {"submission": str(submission), "outdir": str(outdir)})
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Error in {script}: {e}")
            raise typer.Exit(code=e.returncode)

    run_step("FFT variance diagnostic", "fft_variance_heatmap.py", skip_fft)
    run_step("Symbolic violation overlay", "violation_heatmap.py", skip_symbolic)
    run_step("SHAP + symbolic fusion", "shap_symbolic_overlay.py", skip_shap)
    run_step("Pipeline DAG graph", "generate_pipeline_graph.py", skip_dag)

    svg_path = outdir / "pipeline_graph.svg"
    if not skip_dag and svg_path.exists():
        typer.echo(f"🧭 DAG visual saved to: {svg_path}")

    typer.echo(f"✅ All diagnostics complete. Output dir: [bold green]{outdir}[/]")

if __name__ == "__main__":
    app()
