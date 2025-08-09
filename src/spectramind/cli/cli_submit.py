from __future__ import annotations
import typer, json
from pathlib import Path

app = typer.Typer(help="SpectraMind V50 – Submission CLI")

@app.command()
def bundle(submission: str = "outputs/submission.csv", out_zip: str = "outputs/submission_bundle.zip"):
    Path(out_zip).parent.mkdir(parents=True, exist_ok=True)
    import zipfile
    with zipfile.ZipFile(out_zip, "w") as z:
        z.write(submission, arcname="submission.csv")
        for extra in ["run_hash_summary_v50.json", "v50_debug_log.md"]:
            if Path(extra).exists():
                z.write(extra, arcname=extra)
    print(f"Bundled -> {out_zip}")
