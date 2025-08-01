"""
SpectraMind V50 – Self-Diagnosis Tracker
----------------------------------------
Logs symbolic rule violations, SHAP anomalies, and key failure metadata per planet.
Tracks violations over time and appends to persistent failure log.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

LOG_FILE = Path("outputs/diagnostics/self_diagnosis_log.json")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def record_failure(
    planet_id: str,
    rule_failures: List[str],
    shap_spikes: List[int],
    prediction_shift: Dict[str, float] = None,
    notes: str = ""
):
    """
    Records symbolic + SHAP-related anomalies for posthoc diagnostics.

    Args:
        planet_id (str): Target planet ID
        rule_failures (List[str]): Names of violated symbolic rules
        shap_spikes (List[int]): Spectral bins with anomalously high SHAP
        prediction_shift (Dict[str, float], optional): e.g. {'mu_shift': ..., 'sigma_spike': ...}
        notes (str, optional): Human annotation or explanation
    """
    entry = {
        "planet_id": planet_id,
        "timestamp": datetime.utcnow().isoformat(),
        "rule_violations": rule_failures,
        "shap_anomaly_bins": shap_spikes,
        "prediction_shift": prediction_shift or {},
        "notes": notes
    }

    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

    print(f"🧠 Diagnosis logged for planet {planet_id}: {len(rule_failures)} rules, {len(shap_spikes)} SHAP spikes")

if __name__ == "__main__":
    record_failure(
        planet_id="WASP-77b",
        rule_failures=["smoothness", "thermal_consistency"],
        shap_spikes=[42, 97, 101],
        prediction_shift={"mu_shift": 8.3, "sigma_spike": 1.7},
        notes="Sudden edge SHAP spike near water band"
    )
