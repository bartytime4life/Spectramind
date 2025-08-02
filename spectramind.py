"""
SpectraMind V50 – Unified CLI Entry Point
------------------------------------------
Central Typer interface combining:
- core     → training, prediction, scoring, packaging
- diagnose → overlays, t-SNE/UMAP, dashboard
- submit   → full pipeline orchestration
"""

import typer
from cli_core_v50 import app as core_app
from cli_diagnose import app as diagnose_app
from cli_submit import app as submit_app

app = typer.Typer(help="SpectraMind V50 – Unified CLI")

# Register subcommands
app.add_typer(core_app, name="core", help="Core modeling commands")
app.add_typer(diagnose_app, name="diagnose", help="Diagnostics + dashboard tools")
app.add_typer(submit_app, name="submit", help="Submission pipeline runner")

if __name__ == "__main__":
    app()