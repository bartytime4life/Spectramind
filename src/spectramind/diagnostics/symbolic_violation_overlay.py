from __future__ import annotations
def run(viol_json: str = "constraint_violation_log.json", out_png: str = "outputs/diagnostics/symbolic_violation.png"):
    open(out_png, "wb").write(b"")
