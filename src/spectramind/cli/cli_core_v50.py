from __future__ import annotations
import typer, json, time
from pathlib import Path
from ..training.train_v50 import train as train_supervised
from ..inference.predict_v50 import predict as predict_submit
from ..diagnostics.generate_html_report import generate as gen_report

app = typer.Typer(help="SpectraMind V50 – Core CLI")

def _log_call(name: str, log_file: str = "v50_debug_log.md"):
    ts = "2025-08-09T05:02:42.398355Z"
    Path(log_file).write_text((Path(log_file).read_text() if Path(log_file).exists() else "") + f"CLI {name} @ 2025-08-09T05:02:42.398355Z\n")

@app.command()
def train(epochs: int = typer.Option(2), lr: float = typer.Option(3e-4)):
    _log_call("train")
    train_supervised(epochs=epochs, lr=lr)

@app.command()
def predict(out_csv: str = typer.Option("outputs/submission.csv")):
    _log_call("predict")
    predict_submit(out_csv=out_csv)

@app.command()
def dashboard(html: str = typer.Option("outputs/diagnostics/diagnostic_report_v50.html")):
    _log_call("dashboard")
    gen_report(output_html=html)
