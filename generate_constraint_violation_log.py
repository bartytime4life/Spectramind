"""
SpectraMind V50 – Symbolic Constraint Violation Logger
-------------------------------------------------------
Evaluates symbolic and physics-inspired constraints on model outputs μ, σ.
Logs rule-wise and bin-level violations per sample, with diagnostics compatibility.

✅ Smoothness, non-negativity, uncertainty upper-bound
✅ Modular, extensible structure for future rule sets
✅ CLI-integrated, JSON-formatted output
✅ Compatible with symbolic_score_overlay, dashboard, and corel
"""

import torch
import json
import numpy as np
from pathlib import Path
import typer

app = typer.Typer(help="SpectraMind – Constraint Violation Logger")

def check_smoothness(mu: torch.Tensor, threshold: float = 0.05):
    """
    Flags bins where the slope (Δμ) exceeds a smoothness threshold.
    Returns: dict[rule_id -> list of violated bin indices]
    """
    diff = torch.diff(mu, dim=1)
    violations = (diff.abs() > threshold).nonzero(as_tuple=True)
    result = {}
    for sample_idx, bin_idx in zip(*violations):
        rule = f"smoothness::sample{sample_idx}"
        result.setdefault(rule, []).append(int(bin_idx))
    return result

def check_nonnegativity(mu: torch.Tensor):
    violations = (mu < 0).nonzero(as_tuple=True)
    result = {}
    for sample_idx, bin_idx in zip(*violations):
        rule = f"nonnegativity::sample{sample_idx}"
        result.setdefault(rule, []).append(int(bin_idx))
    return result

def check_uncertainty_bound(sigma: torch.Tensor, max_sigma: float = 200.0):
    violations = (sigma > max_sigma).nonzero(as_tuple=True)
    result = {}
    for sample_idx, bin_idx in zip(*violations):
        rule = f"uncertainty_bound::sample{sample_idx}"
        result.setdefault(rule, []).append(int(bin_idx))
    return result

def merge_violations(*violation_dicts):
    log = {}
    for vdict in violation_dicts:
        for k, bins in vdict.items():
            log.setdefault(k, []).extend(bins)
    return log

def compute_violation_stats(log: dict):
    """
    Generates summary stats for each rule.
    """
    stats = {}
    for rule_id, bins in log.items():
        base = rule_id.split("::")[0]
        stats.setdefault(base, 0)
        stats[base] += len(bins)
    return stats

def generate_violation_log(
    mu_path: str = "outputs/mu.pt",
    sigma_path: str = "outputs/sigma.pt",
    output_path: str = "constraint_violation_log.json",
    smoothness_thresh: float = 0.05,
    uncertainty_limit: float = 200.0,
):
    mu = torch.load(mu_path)
    sigma = torch.load(sigma_path) if Path(sigma_path).exists() else torch.ones_like(mu)

    log = merge_violations(
        check_smoothness(mu, smoothness_thresh),
        check_nonnegativity(mu),
        check_uncertainty_bound(sigma, uncertainty_limit),
    )

    stats = compute_violation_stats(log)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "violations": log,
            "summary": stats
        }, f, indent=2)

    print(f"✅ constraint_violation_log.json written to {output_path}")
    print(f"🔍 Rule violation counts: {json.dumps(stats, indent=2)}")

# CLI Entrypoint
@app.command()
def run(
    mu_path: str = typer.Option("outputs/mu.pt", help="Path to μ tensor (.pt)"),
    sigma_path: str = typer.Option("outputs/sigma.pt", help="Path to σ tensor (.pt)"),
    output_path: str = typer.Option("constraint_violation_log.json", help="Output JSON file"),
    smoothness_thresh: float = typer.Option(0.05, help="Δμ threshold for smoothness violation"),
    uncertainty_limit: float = typer.Option(200.0, help="Upper limit for σ"),
):
    generate_violation_log(mu_path, sigma_path, output_path, smoothness_thresh, uncertainty_limit)

if __name__ == "__main__":
    app()