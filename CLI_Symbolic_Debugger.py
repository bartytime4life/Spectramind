"""
SpectraMind V50 – CLI Symbolic Debugger (Ultimate Version)
----------------------------------------------------------
CLI tool to inspect symbolic rule violations per planet.
Features:
- Colored violation table
- Summary statistics (max, mean, entropy)
- Hooks into planet memory, HTML diagnostics, symbolic overlay tools
- Part of `spectramind diagnose symbolic` CLI group
"""

import json
import typer
from rich import print
from rich.table import Table
from rich.panel import Panel
import numpy as np
from pathlib import Path

app = typer.Typer()

@app.command()
def debug(
    planet_id: str,
    memory_path: str = "outputs/memory/planet_memory.json",
    show_summary: bool = True
):
    """
    View symbolic rule violations for a given planet.
    """
    if not planet_id:
        print("[red]❌ No planet_id provided.")
        raise typer.Exit(1)

    memory_path = Path(memory_path)
    if not memory_path.exists():
        print(f"[red]❌ Memory file not found: {memory_path}")
        raise typer.Exit(1)

    with open(memory_path, 'r') as f:
        memory = json.load(f)
    entry = memory.get(planet_id)
    if not entry:
        print(f"[yellow]⚠️ Planet {planet_id} not found in memory.")
        raise typer.Exit(0)

    violations = entry.get("violations", {})
    if not violations:
        print("[green]✅ No symbolic violations recorded.")
        raise typer.Exit(0)

    # Table
    table = Table(title=f"Symbolic Debug – {planet_id}")
    table.add_column("Rule", style="cyan")
    table.add_column("Violation Score", style="magenta")
    vals = []
    for k, v in violations.items():
        vals.append(v)
        level = "green" if v < 0.05 else "yellow" if v < 0.15 else "red"
        table.add_row(k, f"[{level}]{v:.4f}[/{level}]")
    print(table)

    # Summary block
    if show_summary:
        arr = np.array(vals)
        summary = {
            "max_violation": float(arr.max()),
            "mean_violation": float(arr.mean()),
            "entropy": float(-(arr * np.log(arr + 1e-8)).sum())
        }
        stats = "\n".join([f"[bold]{k}:[/bold] {v:.4f}" for k, v in summary.items()])
        print(Panel(stats, title="Violation Summary", subtitle="Per-planet Symbolic Metrics"))


if __name__ == "__main__":
    app()