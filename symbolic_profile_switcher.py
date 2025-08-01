"""
SpectraMind V50 – Symbolic Profile Switcher
-------------------------------------------
Selects appropriate symbolic constraint config based on metadata or training phase.
"""

import os
import yaml
from typing import Dict


class SymbolicProfileSwitcher:
    def __init__(self, profile_dir: str = "configs/symbolic"):
        """
        Args:
            profile_dir: path to directory containing symbolic YAML configs
        """
        self.profile_dir = profile_dir
        self.profiles = {}
        for fname in os.listdir(profile_dir):
            if fname.endswith(".yaml"):
                name = fname.replace(".yaml", "")
                with open(os.path.join(profile_dir, fname), 'r') as f:
                    self.profiles[name] = yaml.safe_load(f)

    def get_profile(self, metadata: Dict, phase: str = "train") -> Dict:
        """
        Select symbolic profile based on stellar metadata and training phase.

        Default rules:
            - train, Ts > 5500 → full.yaml
            - train, Ts ≤ 5500 → minimal.yaml
            - inference         → safe.yaml
            - fallback          → default.yaml

        Args:
            metadata: dict with keys like Ts (temperature), Rs, etc.
            phase: 'train', 'val', or 'inference'

        Returns:
            symbolic config dict
        """
        Ts = metadata.get("Ts", 5000)
        Rs = metadata.get("Rs", 1.0)

        if phase == "inference":
            profile_name = "safe"
        elif phase == "train":
            if Ts > 5500:
                profile_name = "full"
            else:
                profile_name = "minimal"
        else:
            profile_name = "default"

        profile = self.profiles.get(profile_name)
        if not profile:
            raise ValueError(f"⚠️ Symbolic profile '{profile_name}' not found in {self.profile_dir}")

        print(f"🔧 Using symbolic profile: '{profile_name}' for Ts={Ts}, phase={phase}")
        return profile

    def list_profiles(self):
        return list(self.profiles.keys())