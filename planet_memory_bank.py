"""
SpectraMind V50 – Planet Memory Bank
------------------------------------
Stores per-planet statistical summaries for adaptive reasoning or symbolic tracking.
Now supports drift tracking via planet_drift_tracker.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from planet_drift_tracker import PlanetDriftTracker


class PlanetMemoryBank:
    def __init__(self, filepath: str = "outputs/memory/planet_memory.json"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.memory = self._load()
        self.drift = PlanetDriftTracker()

    def _load(self) -> Dict:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("⚠️ Memory bank file is corrupted. Reinitializing.")
                    return {}
        return {}

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.memory, f, indent=2)

    def update(
        self,
        planet_id: str,
        z: Optional[List[float]] = None,
        mu: Optional[List[float]] = None,
        violations: Optional[Dict[str, float]] = None
    ):
        entry = self.memory.get(planet_id, {
            "planet_id": planet_id,
            "num_updates": 0,
        })

        if mu:
            mu_mean = sum(mu) / len(mu)
            mu_std = (sum((m - mu_mean) ** 2 for m in mu) / len(mu)) ** 0.5 if len(mu) > 1 else 0.0
            entry["mu_mean"] = float(mu_mean)
            entry["mu_std"] = float(mu_std)

        if z:
            z_mean = sum(z) / len(z)
            z_std = (sum((zi - z_mean) ** 2 for zi in z) / len(z)) ** 0.5 if len(z) > 1 else 0.0
            entry["z_mean"] = float(z_mean)
            entry["z_std"] = float(z_std)
            entry["z"] = z

        if violations:
            entry.setdefault("violations", {}).update(violations)
            entry["max_violation"] = max(violations.values())

        entry["last_updated"] = datetime.utcnow().isoformat()
        entry["num_updates"] += 1

        self.memory[planet_id] = entry
        self.save()

        # Track drift if applicable
        if z and violations:
            self.drift.log_state(planet_id, z, violations)

    def get(self, planet_id: str) -> Dict:
        return self.memory.get(planet_id, {})

    def get_all(self) -> Dict[str, Dict]:
        return self.memory

    def summary(self, planet_id: str) -> str:
        entry = self.get(planet_id)
        if not entry:
            return f"No entry for planet '{planet_id}'."
        return json.dumps(entry, indent=2)

    def clear(self):
        self.memory = {}
        self.save()