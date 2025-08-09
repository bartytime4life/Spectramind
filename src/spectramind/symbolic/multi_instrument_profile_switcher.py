from typing import Dict, Optional
class MultiInstrumentProfileSwitcher:
    def __init__(self, profiles: Dict[str, dict]): self.profiles = profiles
    def choose_profile(self, instrument_flags: Dict[str, bool], fallback: Optional[str] = "safe") -> dict:
        fgs = instrument_flags.get("FGS1", False); airs = instrument_flags.get("AIRS", False)
        profile = "full" if fgs and airs else ("airs_only" if airs else ("fgs_only" if fgs else fallback))
        return self.profiles.get(profile, self.profiles.get("safe", {}))
    def profile_name(self, instrument_flags: Dict[str, bool]) -> str:
        fgs = instrument_flags.get("FGS1", False); airs = instrument_flags.get("AIRS", False)
        return "full" if (fgs and airs) else ("airs_only" if airs else ("fgs_only" if fgs else "safe"))