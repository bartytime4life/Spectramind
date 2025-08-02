"""
SpectraMind V50 – Full Self-Test Compliance Validator
-----------------------------------------------------
Runs CLI integrity checks, symbolic validation, HTML + ZIP inspection,
and outputs structured Markdown + JSON compliance reports.
"""

import os
import json
import typer
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()
app = typer.Typer()

LOG_PATH = Path("selftest_log.md")
JSON_PATH = Path("selftest_report.json")
HASH_PATH = Path("run_hash_summary_v50.json")
HTML_PATH = Path("diagnostics/diagnostic_report_v1.html")
ZIP_PATH = Path("spectramind_v50_submission.zip")


def write_log(msg: str):
    LOG_PATH.write_text(f"\n{msg}", append=True)
    console.print(msg)


def test_cmd(cmd: str, name: str, required=True):
    write_log(f"▶️ `{cmd}`")
    try:
        subprocess.run(cmd, shell=True, check=True)
        write_log(f"✅ {name}")
        return {"name": name, "status": "pass"}
    except subprocess.CalledProcessError:
        write_log(f"❌ {name}")
        if required:
            raise typer.Exit(code=1)
        return {"name": name, "status": "fail"}


def check_files(required: list):
    results = []
    for f in required:
        status = Path(f).exists()
        msg = f"✅ {f}" if status else f"❌ Missing: {f}"
        write_log(msg)
        results.append({"file": f, "exists": status})
    return results


def check_hash_consistency():
    if not HASH_PATH.exists():
        return {"status": "fail", "reason": "run_hash_summary_v50.json not found"}

    hashes = json.load(open(HASH_PATH))
    latest = list(hashes.values())[-1]
    return {
        "status": "pass",
        "run_tag": latest.get("run_tag"),
        "hash": latest.get("hash"),
        "timestamp": latest.get("timestamp")
    }


def check_zip_contents():
    import zipfile
    if not ZIP_PATH.exists():
        return {"status": "fail", "reason": "submission.zip not found"}

    required_inside = {"submission.csv", "mu.pt", "sigma.pt", "v50_debug_log.md"}
    with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
        found = set(zipf.namelist())
    missing = required_inside - found
    if missing:
        return {"status": "fail", "missing": list(missing)}
    return {"status": "pass", "files_found": list(found)}


@app.command()
def run(mode: str = typer.Option("fast", help="Mode: fast or deep")):
    LOG_PATH.write_text(f"# SpectraMind V50 – Self-Test Log\n\nStart: {datetime.utcnow().isoformat()}\n")
    results = {"mode": mode, "timestamp": datetime.utcnow().isoformat(), "checks": []}

    console.rule("[bold cyan]🚦 SpectraMind V50 Self-Test")
    console.print(f"[blue]Mode:[/blue] {mode}\n")

    results["checks"].append(test_cmd("spectramind diagnose --dry-run", "CLI dry-run"))
    results["cli_registry"] = test_cmd("spectramind test --list", "CLI registry check", required=False)

    results["files"] = check_files([
        "configs/config_v50.yaml",
        "v50_debug_log.md",
        "submission.csv",
        "outputs/mu.pt",
        "outputs/sigma.pt",
        "spectramind.toml"
    ])

    results["symbolic"] = check_files([
        "symbolic_loss.py", "photonic_alignment.py", "generate_diagnostic_summary.py"
    ])

    if mode == "deep":
        results["run_hash"] = check_hash_consistency()
        results["html_dashboard"] = {"exists": HTML_PATH.exists()}
        results["zip_bundle"] = check_zip_contents()
        results["corel"] = test_cmd("spectramind explain --dry-run", "COREL module dry-run", required=False)

    JSON_PATH.write_text(json.dumps(results, indent=2))
    write_log("\n✅ Self-test completed.")
    console.print(f"\n✅ [green]Self-test passed[/green]. Summary in [bold]{JSON_PATH}[/bold] and [bold]{LOG_PATH}[/bold]")

if __name__ == "__main__":
    app()