# COREL-style conformalization stub: scales sigmas using a quantile factor.
# Replace with real residual-based COREL when labels/val folds are available.
import numpy as np, json
from pathlib import Path

def conformalize_sigmas(sigmas: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    # Simple monotone inflation to emulate coverage targeting
    factor = 1.0 + max(0.0, float(quantile) - 0.5)  # 1.0..1.5 for q in [0.5, 1.0]
    return sigmas * factor

def save_corel_meta(quantile: float, out_path: str = "outputs/corel_meta.json"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"quantile": float(quantile)}, indent=2))
