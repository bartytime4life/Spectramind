"""
SpectraMind V50 – Unified CLI Interface
---------------------------------------
Combines core model commands, diagnostics, submission, and system health checks.

Commands:
  • core      – Train, predict, package
  • diagnose  – SHAP, symbolic, UMAP, t-SNE, dashboard
  • submit    – Train → inference → package orchestration
  • test      – CLI and pipeline self-test validator
"""

import typer
from cli_core_v50 import app as core_app
from cli_diagnose import app as diagnose_app
from cli_submit import app as submit_app
from selftest import app as test_app

app = typer.Typer(help="SpectraMind V50 – Unified CLI for training, diagnostics, submission, testing")

# Register all sub-CLIs
app.add_typer(core_app, name="core", help="Model training, inference, scoring")
app.add_typer(diagnose_app, name="diagnose", help="Diagnostics + HTML overlay tools")
app.add_typer(submit_app, name="submit", help="Full pipeline orchestration + zip")
app.add_typer(test_app, name="test", help="System self-test + validation CLI")

if __name__ == "__main__":
    app()