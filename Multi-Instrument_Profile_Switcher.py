"""
SpectraMind V50 – Multi-Instrument Profile Switcher (Ultimate+ Version)
------------------------------------------------------------------------
Dynamically selects symbolic logic profiles based on instrument availability.
Integrates with:
- symbolic_logic_engine.py (profile routing)
- symbolic_violation_predictor.py (profile-aware scoring)
- PlanetMemoryBank (profile audit log)
- cli_diagnose.py / diagnostics dashboard
- anomaly_feedback_trainer.py (retraining by profile class)
"""

from typing import Dict, Optional
import json
import os
from pathlib import Path


class MultiInstrumentProfileSwitcher:
    def __init__(self, profiles: Dict[str, dict]):
        """
        Args:
            profiles: dict of profile_name → symbolic config dict
        """
        self.profiles = profiles

    def choose_profile(self, instrument_flags: Dict[str, bool], fallback: Optional[str] = "safe") -> dict:
        """
        Selects symbolic config profile based on instrument usage.

        Args:
            instrument_flags: {"FGS1": True, "AIRS": False}
            fallback: profile to use if nothing matches

        Returns:
            symbolic config dict
        """
        fgs = instrument_flags.get("FGS1", False)
        airs = instrument_flags.get("AIRS", False)

        if fgs and airs:
            profile = "full"
        elif airs:
            profile = "airs_only"
        elif fgs:
            profile = "fgs_only"
        else:
            profile = fallback or "safe"

        return self.profiles.get(profile, self.profiles.get("safe", {}))

    def profile_name(self, instrument_flags: Dict[str, bool]) -> str:
        fgs = instrument_flags.get("FGS1", False)
        airs = instrument_flags.get("AIRS", False)
        if fgs and airs:
            return "full"
        elif airs:
            return "airs_only"
        elif fgs:
            return "fgs_only"
        return "safe"

    def export_profile_manifest(self, out_path: str = "outputs/profile_manifest.json"):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self.profiles, f, indent=2)
        print(f"🧾 Exported symbolic profiles to {out_path}")


if __name__ == "__main__":
    dummy_profiles = {
        "full": {"smoothness": True, "nonnegativity": True},
        "airs_only": {"enable_photonic": True},
        "fgs_only": {"monotonicity": "decreasing"},
        "safe": {"nonnegativity": True}
    }

    switcher = MultiInstrumentProfileSwitcher(dummy_profiles)

    test_cases = [
        ({"AIRS": True, "FGS1": True}, "full"),
        ({"AIRS": True, "FGS1": False}, "airs_only"),
        ({"AIRS": False, "FGS1": True}, "fgs_only"),
        ({"AIRS": False, "FGS1": False}, "safe")
    ]

    for flags, expected in test_cases:
        profile = switcher.choose_profile(flags)
        label = switcher.profile_name(flags)
        print(f"🔧 Flags: {flags} → Profile: {label} → Config: {profile}")

    switcher.export_profile_manifest()