from __future__ import annotations
def run(attn_json: str = "outputs/attn_overlay.json", out_png: str = "outputs/diagnostics/shap_attention.png"):
    open(out_png, "wb").write(b"")
