import json, typer
from rich import print
from rich.table import Table
from pathlib import Path
from datetime import datetime
app = typer.Typer()
LOG_PATH = Path("v50_debug_log.md")
def log_cli_call(planet_id: str, memory_path: Path):
    ts = datetime.utcnow().isoformat()
    entry = f"\n### CLI: symbolic-debug\n- Timestamp: {ts}\n- Planet ID: `{planet_id}`\n- Memory File: `{memory_path}`\n"
    LOG_PATH.write_text(LOG_PATH.read_text()+entry if LOG_PATH.exists() else entry)
@app.command()
def debug(planet_id: str, memory_path: Path = Path("outputs/memory/planet_memory.json")):
    if not memory_path.exists():
        print(f"[red]❌ Memory file not found: {memory_path}"); raise typer.Exit(1)
    entry = json.loads(memory_path.read_text()).get(planet_id)
    if not entry:
        print(f"[yellow]⚠️ Planet {planet_id} not found."); raise typer.Exit(0)
    violations = entry.get("violations", {})
    table = Table(title=f"🔬 Symbolic Rule Violations – [bold]{planet_id}[/bold]")
    table.add_column("Rule", style="cyan"); table.add_column("Violation Score", style="magenta", justify="right")
    for rule, score in sorted(violations.items(), key=lambda x: x[1], reverse=True)[:5]:
        level = "green" if score < 0.05 else "yellow" if score < 0.15 else "red"
        table.add_row(rule, f"[{level}]{score:.4f}[/{level}]")
    print(table); log_cli_call(planet_id, memory_path)
if __name__ == "__main__": app()