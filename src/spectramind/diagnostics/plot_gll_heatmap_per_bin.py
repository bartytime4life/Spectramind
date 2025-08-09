from __future__ import annotations
def run(in_json: str = "outputs/diagnostics/diagnostic_summary.json", out_png: str = "outputs/diagnostics/gll_heatmap.png"):
    open(out_png, "wb").write(b"")
