"""
SpectraMind V50 – Planet Drift Tracker (Ultimate Version)
----------------------------------------------------------
Tracks symbolic violation scores and latent z-vector drift per planet across inference runs.
Integrates with:
- PlanetMemoryBank (auto-logging)
- generate_html_report.py (dashboard summary)
- plot_umap_v50.py / plot_tsne_interactive.py (drift overlays)
- anomaly_feedback_trainer.py (drift-guided retraining)
"""

import json
import os
import numpy as np
from typing import Dict, List, Union
from pathlib import Path


class PlanetDriftTracker:
    def __init__(self, history_dir: str = "outputs/memory/drift_log"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def log_state(
        self,
        planet_id: str,
        z: Union[List[float], np.ndarray],
        violations: Dict[str, float],
        tag: str = None,
    ):
        """
        Appends a new drift record for a planet.
        Args:
            planet_id: planet name
            z: latent vector
            violations: symbolic rule score dict
            tag: version/run ID
        """
        path = self.history_dir / f"{planet_id}.json"
        history = []
        if path.exists():
            with open(path, "r") as f:
                history = json.load(f)

        entry = {
            "z": list(z),
            "violations": violations,
            "tag": tag
        }
        history.append(entry)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def compute_violation_drift(self, planet_id: str) -> Dict[str, float]:
        """
        Returns per-rule stddev across episodes.
        """
        path = self.history_dir / f"{planet_id}.json"
        if not path.exists():
            return {}

        with open(path, "r") as f:
            episodes = json.load(f)

        keys = episodes[0]["violations"].keys()
        arrays = {k: [ep["violations"].get(k, 0.0) for ep in episodes] for k in keys}
        return {k: float(np.std(v)) for k, v in arrays.items()}

    def compute_latent_drift(self, planet_id: str) -> float:
        """
        Returns L2 norm of std deviation of z across episodes.
        """
        path = self.history_dir / f"{planet_id}.json"
        if not path.exists():
            return 0.0

        with open(path, "r") as f:
            episodes = json.load(f)

        z_stack = np.array([ep["z"] for ep in episodes])
        return float(np.linalg.norm(np.std(z_stack, axis=0)))

    def summarize_drift(self, planet_id: str) -> Dict[str, float]:
        """
        Returns a summary: latent drift + per-rule symbolic drift.
        """
        return {
            "latent_drift": self.compute_latent_drift(planet_id),
            **self.compute_violation_drift(planet_id)
        }

    def list_tracked_planets(self) -> List[str]:
        return [p.stem for p in self.history_dir.glob("*.json")]


if __name__ == "__main__":
    tracker = PlanetDriftTracker()
    tracker.log_state("planet-001", z=np.ones(16), violations={"smooth": 0.02, "nonneg": 0.01}, tag="v50r1")
    tracker.log_state("planet-001", z=np.random.randn(16), violations={"smooth": 0.15, "nonneg": 0.05}, tag="v50r2")

    drift = tracker.summarize_drift("planet-001")
    print("📈 Drift Summary:", drift)