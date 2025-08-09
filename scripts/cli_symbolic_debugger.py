import json, typer
from rich import print
from rich.table import Table
from pathlib import Path
from datetime import datetime

app = typer.Typer()

LOG_PATH = Path("v50_debug_log.md")

def log_cli_call(planet_id: str, memory_path: Path):
    timestamp = datetime.utcnow().isoformat()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"CLI: symbolic-debug | {timestamp} | planet={planet_id} | memory={memory_path}\n")

@app.command()
def debug(planet_id: str, memory_path: Path = Path("outputs/memory/planet_memory.json")):
    if not memory_path.exists():
        print(f"[red]No memory at {memory_path}")
        raise typer.Exit(1)
    memory = json.loads(memory_path.read_text())
    entry = memory.get(planet_id)
    if not entry:
        print(f"[yellow]Planet {planet_id} not found")
        raise typer.Exit(0)
    violations = entry.get("violations", {})
    table = Table(title=f"Symbolic Rule Violations – {planet_id}")
    table.add_column("Rule", style="cyan"); table.add_column("Score", justify="right")
    for rule, score in sorted(violations.items(), key=lambda x: x[1], reverse=True)[:5]:
        table.add_row(rule, f"{score:.4f}")
    print(table); log_cli_call(planet_id, memory_path)

if __name__ == "__main__":
    app()
