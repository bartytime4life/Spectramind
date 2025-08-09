import json
from pathlib import Path

def apply_temperature_to_sigmas(sigmas, temperature: float):
    # Basic temperature scaling: sigma_T = sigma * T
    return sigmas * float(temperature)

def save_temperature(t: float, out_path: str = "outputs/temperature.json"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"temperature": float(t)}, indent=2))
