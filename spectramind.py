import typer
import subprocess
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from rich.console import Console
from cli_explain_util import validate_files_exist, export_to_markdown, export_to_json
import matplotlib.pyplot as plt
import pandas as pd

app = typer.Typer()

__LOG_FILE__ = Path("v50_debug_log.md")

# --- Helper Functions ---
def parse_cli_log():
    entries, raw_blocks, current_lines, seen_keys = [], [], [], set()
    lines = __LOG_FILE__.read_text().splitlines()

    for line in lines:
        if line.startswith("### CLI Call @ "):
            if current_lines:
                raw_blocks.append(current_lines)
                current_lines = []
        current_lines.append(line)
    if current_lines:
        raw_blocks.append(current_lines)

    for block in raw_blocks:
        entry = {}
        for line in block:
            if line.startswith("### CLI Call @ "):
                entry["time"] = line.strip().split("@ ")[1]
            elif "- Command:" in line:
                entry["command"] = line.split("`")[1]
            elif "- Version:" in line:
                entry["version"] = line.split(": ")[1]
            elif "- Config Hash:" in line:
                entry["hash"] = line.split(": ")[1]
        key = (entry.get("command"), entry.get("hash"), entry.get("time")[:16])
        if entry and key not in seen_keys:
            entries.append(entry)
            seen_keys.add(key)
    return sorted(entries, key=lambda x: x["time"], reverse=True)

def export_markdown(entries, out_md):
    grouped = defaultdict(list)
    for e in entries:
        grouped[e["hash"]].append(e)

    lines = []
    for h, group in grouped.items():
        lines.append(f"\n### Config Hash: `{h}`\n")
        lines.append("| Time (UTC)           | Command                                | Version  |")
        lines.append("|----------------------|-----------------------------------------|----------|")
        for e in group:
            lines.append(f"| {e['time'][:19]} | {e['command'][:41]:<41} | {e['version']} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def export_csv(entries, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "command", "version", "config_hash"])
        for e in entries:
            writer.writerow([e["time"], e["command"], e["version"], e["hash"]])


def generate_plots(entries):
    times = [datetime.fromisoformat(e["time"]) for e in entries]
    hours = [t.hour for t in times]
    hashes = [e["hash"] for e in entries]

    # Hourly volume
    plt.figure(figsize=(10, 4))
    plt.hist(hours, bins=24, range=(0, 24), color="skyblue")
    plt.title("CLI Call Volume by Hour (UTC)")
    plt.xlabel("Hour")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("diagnostics/cli_heatmap.png")
    plt.close()

    # Config hash trend
    plt.figure(figsize=(10, 4))
    hash_counts = Counter(hashes)
    items = sorted(hash_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    labels, counts = zip(*items)
    plt.barh(labels, counts, color="orange")
    plt.title("Top Config Hashes")
    plt.xlabel("Uses")
    plt.tight_layout()
    plt.savefig("diagnostics/cli_hash_trend.png")
    plt.close()

@app.command("analyze-log")
def analyze_log(
    limit: int = typer.Option(10),
    out_md: Path = typer.Option("diagnostics/log_table.md"),
    out_csv: Path = typer.Option("diagnostics/cli_log.csv"),
    clean: bool = typer.Option(False)
):
    if not __LOG_FILE__.exists():
        typer.secho("❌ v50_debug_log.md not found.", fg=typer.colors.RED)
        raise typer.Exit()

    entries = parse_cli_log()

    if clean:
        typer.echo("🧹 Deduplicating CLI log...")
        clean_lines = []
        for e in entries:
            clean_lines.append(f"### CLI Call @ {e['time']}")
            clean_lines.append(f"- Command: `{e['command']}`")
            clean_lines.append(f"- Version: {e['version']}")
            clean_lines.append(f"- Config Hash: {e['hash']}")
        __LOG_FILE__.write_text("\n".join(clean_lines))
        typer.secho("✅ Log cleaned and saved.", fg=typer.colors.GREEN)

    # Preview
    typer.echo("| Time (UTC)           | Command                                | Version  | Config Hash       |")
    typer.echo("|----------------------|-----------------------------------------|----------|-------------------|")
    for e in entries[:limit]:
        typer.echo(f"| {e['time'][:19]} | {e['command'][:41]:<41} | {e['version']} | {e['hash'][:15]}... |")

    export_markdown(entries, out_md)
    typer.echo(f"📄 Markdown log written to: {out_md}")

    export_csv(entries, out_csv)
    typer.echo(f"📊 CSV log written to: {out_csv}")

    try:
        generate_plots(entries)
        typer.echo("🕒 Heatmap saved to diagnostics/cli_heatmap.png")
        typer.echo("📈 Config hash trend saved to diagnostics/cli_hash_trend.png")
    except Exception as e:
        typer.secho(f"⚠️ Plotting failed: {e}", fg=typer.colors.YELLOW)

if __name__ == "__main__":
    app()