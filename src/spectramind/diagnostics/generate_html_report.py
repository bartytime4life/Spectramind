from pathlib import Path

HTML_TEMPLATE = """
<html><head><title>diagnostic_report_v50</title></head>
<body>
<h1>SpectraMind V50 Diagnostics</h1>
<ul>
  <li><a href="../outputs/submission.csv">submission.csv</a></li>
  <li><a href="coherence_curve.png">coherence_curve.png</a></li>
  <li><a href="../constraint_violation_log.json">constraint_violation_log.json</a></li>
  <li><a href="../run_hash_summary_v50.json">run_hash_summary_v50.json</a></li>
  <li><a href="symbolic_violation_html_summary.html">symbolic violations</a></li>
</ul>
</body></html>
"""

def write_report(outdir: str = "diagnostics"):
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    (p / "diagnostic_report_v50.html").write_text(HTML_TEMPLATE, encoding="utf-8")
    return str(p / "diagnostic_report_v50.html")

if __name__ == "__main__":
    print(write_report())
