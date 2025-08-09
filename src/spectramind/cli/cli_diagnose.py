from __future__ import annotations
import typer
from pathlib import Path

app = typer.Typer(help="SpectraMind V50 – Diagnostics CLI")

@app.command()
def dashboard(html: str = "outputs/diagnostics/diagnostic_report_v50.html"):
    from ..diagnostics.generate_html_report import generate
    generate(output_html=html)

@app.command()
def fft_variance(in_csv: str = "outputs/submission.csv", out_png: str = "outputs/diagnostics/fft_variance.png"):
    # Placeholder; integrate your actual fft_variance_heatmap here.
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_png).write_bytes(b"")  # stub
