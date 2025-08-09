from __future__ import annotations
def run(shap_json: str = "outputs/shap_overlay.json", out_png: str = "outputs/diagnostics/shap_overlay.png"):
    open(out_png, "wb").write(b"")
