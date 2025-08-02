# ... [imports and constants remain unchanged above]

@app.command("corel-train")
def train_corel():
    """Train SpectralCOREL GNN using validation μ, σ, y, and edge_index"""
    import subprocess
    cmd = [
        "python", "train_corel.py",
        "--input_dir", "calibration_data/",
        "--output_path", "models/corel_gnn.pt",
        "--epochs", "50"
    ]
    typer.echo("🧠 Launching COREL GNN training ...")
    subprocess.run(cmd)

@app.command("analyze-log")
def analyze_log(
    limit: int = typer.Option(10, help="Max entries to show in terminal"),
    out_csv: Path = typer.Option("diagnostics/cli_log.csv"),
    out_md: Path = typer.Option("diagnostics/log_table.md"),
    clean: bool = typer.Option(False, help="Clean duplicate log entries")
):
    """Parse CLI log into Markdown and CSV, grouped by config hash"""
    if not __LOG_FILE__.exists():
        typer.secho("❌ No v50_debug_log.md found.", fg=typer.colors.RED)
        raise typer.Exit()

    entries = []
    raw_blocks = []
    seen = set()
    current_lines = []

    with open(__LOG_FILE__) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("### CLI Call @ "):
            if current_lines:
                raw_blocks.append(current_lines)
                current_lines = []
        current_lines.append(line)
    if current_lines:
        raw_blocks.append(current_lines)

    parsed = []
    for block in raw_blocks:
        entry = {}
        for line in block:
            if line.startswith("### CLI Call @ "):
                entry["time"] = line.strip().split("@ ")[1]
            elif line.startswith("- Command: "):
                entry["command"] = line.strip().split("`")[1]
            elif line.startswith("- Version:"):
                entry["version"] = line.strip().split(": ")[1]
            elif line.startswith("- Config Hash:"):
                entry["hash"] = line.strip().split(": ")[1]
        key = (entry.get("command"), entry.get("hash"), entry.get("time")[:16])
        if entry and key not in seen:
            parsed.append(entry)
            seen.add(key)

    if clean:
        typer.echo("🧹 Cleaning duplicates from v50_debug_log.md ...")
        dedup_lines = []
        for e in parsed:
            dedup_lines.append(f"### CLI Call @ {e['time']}\n")
            dedup_lines.append(f"- Command: `{e['command']}`\n")
            dedup_lines.append(f"- Version: {e['version']}\n")
            dedup_lines.append(f"- Config Hash: {e['hash']}\n")
        __LOG_FILE__.write_text("".join(dedup_lines))
        typer.secho("✅ Log deduplicated and saved.", fg=typer.colors.GREEN)

    entries = sorted(parsed, key=lambda x: x["time"], reverse=True)

    # Terminal preview
    typer.echo("| Time (UTC)           | Command                                | Version  | Config Hash       |")
    typer.echo("|----------------------|-----------------------------------------|----------|-------------------|")
    for e in entries[:limit]:
        typer.echo(f"| {e['time'][:19]} | {e['command'][:41]:<41} | {e['version']} | {e['hash'][:15]}... |")

    # Grouped markdown
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
    typer.echo(f"📄 Markdown log written to: {out_md}")

    # CSV output
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "command", "version", "config_hash"])
        for e in entries:
            writer.writerow([e["time"], e["command"], e["version"], e["hash"]])
    typer.echo(f"📊 CSV log written to: {out_csv}")