"""
SpectraMind V50 – Failure Log Analyzer
--------------------------------------
Summarizes symbolic rule violations and SHAP spikes from self_diagnosis_log.json.
Outputs top failing planets, rule frequency, SHAP anomaly stats, and optional CSV export.
"""

import json
from collections import Counter
from pathlib import Path
import typer
import pandas as pd
from datetime import datetime

app = typer.Typer(help="Summarize symbolic and SHAP failure diagnostics")

@app.command()
def summarize(
    log_path: Path = typer.Option("outputs/diagnostics/self_diagnosis_log.json", help="Path to self-diagnosis log"),
    top_k: int = typer.Option(5, help="How many top items to display"),
    save_csv: bool = typer.Option(True, help="Save summary as CSV"),
    out_dir: Path = typer.Option("outputs/diagnostics", help="Directory to save outputs")
):
    if not log_path.exists():
        print("❌ Diagnosis log not found.")
        raise typer.Exit(1)

    with open(log_path) as f:
        entries = json.load(f)

    rule_counter = Counter()
    shap_bins = Counter()
    worst_planets = []
    detailed_rows = []

    for entry in entries:
        pid = entry.get("planet_id")
        rules = entry.get("rule_violations", [])
        shap = entry.get("shap_anomaly_bins", [])
        shift = entry.get("prediction_shift", {})
        note = entry.get("notes", "")
        timestamp = entry.get("timestamp")

        rule_counter.update(rules)
        shap_bins.update(map(int, shap))
        worst_planets.append((pid, len(rules), len(shap)))

        detailed_rows.append({
            "planet_id": pid,
            "timestamp": timestamp,
            "rule_count": len(rules),
            "shap_spike_count": len(shap),
            "mu_shift": shift.get("mu_shift"),
            "sigma_spike": shift.get("sigma_spike"),
            "notes": note
        })

    print("\n📉 Top Symbolic Rule Failures:")
    for rule, count in rule_counter.most_common(top_k):
        print(f"- {rule}: {count} occurrences")

    print("\n🔥 Most Frequent SHAP Anomaly Bins:")
    for b, count in shap_bins.most_common(top_k):
        print(f"- Bin {b}: {count} flagged")

    print("\n🚨 Top Offender Planets:")
    worst_planets.sort(key=lambda x: (x[1] + x[2]), reverse=True)
    for pid, r, s in worst_planets[:top_k]:
        print(f"- {pid}: {r} rule fails, {s} SHAP spikes")

    if save_csv:
        df = pd.DataFrame(detailed_rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"symbolic_failure_summary_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.csv"
        df.to_csv(out_path, index=False)
        print(f"📄 Full failure log exported to {out_path}")

if __name__ == "__main__":
    app()
