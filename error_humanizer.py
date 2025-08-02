"""
SpectraMind V50 – Error Humanizer (Ultimate Version)
-----------------------------------------------------
Translates raw Python errors and stack traces into human-readable guidance.
Integrates with:
- cli_core_v50.py (top-level try/except wrappers)
- train_v50.py / predict_v50.py (failure catching)
- run_integration.py (wrap all pipeline stages)
- diagnostics logging system (log explanations)
- generate_html_report.py (optional error banner)
- CLI decorators for @handle_errors (automatic parsing)
"""

import re
from rich.console import Console
from typing import Optional

console = Console()

COMMON_ERRORS = {
    "FileNotFoundError": "🔍 File not found. Did you run calibration or generate input data first?",
    "CUDA out of memory": "💥 GPU out of memory. Try smaller batch size or reduce model size.",
    "KeyError": "🧩 Missing dictionary key. Check your config or metadata.",
    "ModuleNotFoundError": "📦 Missing module. Did you run `poetry install` or activate the correct environment?",
    "shape mismatch": "📐 Tensor shape mismatch. Validate your μ, σ, latent, or decoder output shapes.",
    "invalid index": "📛 Invalid array index. Check for off-by-one errors or malformed inputs.",
    "RuntimeError": "⚠️ Runtime error. Check for NaNs, gradients, or hardware instability.",
    "Permission denied": "🔒 Permission error. Check file access rights or running inside containers."
}


def humanize_exception(err_msg: str, origin: Optional[str] = None):
    """
    Parses an error message and prints a human-readable explanation.

    Args:
        err_msg: raw error or traceback string
        origin: optional label for where the error occurred (e.g. 'train')
    """
    matched = False
    header = f"[bold red]Pipeline Error" + (f" in {origin}" if origin else "") + ":[/bold red]"
    console.print(header)

    for key, explanation in COMMON_ERRORS.items():
        if key.lower() in err_msg.lower():
            console.print(f"[yellow]Hint:[/yellow] {explanation}")
            matched = True
            break

    if not matched:
        console.print(f"[dim]{err_msg.strip()}[/dim]")


def handle_errors(func):
    """Decorator to wrap CLI functions with error humanization."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            humanize_exception(tb, origin=func.__name__)
            raise e
    return wrapper


if __name__ == "__main__":
    test_errors = [
        "FileNotFoundError: 'fgs1_tensor.npy' not found",
        "RuntimeError: CUDA out of memory",
        "KeyError: 'Ts'",
        "ModuleNotFoundError: No module named 'torch'",
        "IndexError: index 284 is out of bounds for axis 0 with size 283"
    ]
    for err in test_errors:
        humanize_exception(err)