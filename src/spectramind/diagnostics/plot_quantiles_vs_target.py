from __future__ import annotations
def run(in_csv: str = "outputs/submission.csv", out_png: str = "outputs/diagnostics/quantile_violations.png"):
    open(out_png, "wb").write(b"")
