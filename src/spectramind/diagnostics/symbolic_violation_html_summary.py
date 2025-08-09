import json
from pathlib import Path

def write_html_from_log(log_path: str = "constraint_violation_log.json",
                        out_path: str = "diagnostics/symbolic_violation_html_summary.html"):
    data = json.loads(Path(log_path).read_text()) if Path(log_path).exists() else {"violations": []}
    rows = "".join([f"<tr><td>{v['ts']}</td><td>{v['rule']}</td><td>{v['value']:.6f}</td></tr>" for v in data.get("violations", [])])
    html = f"""<html><body><h2>Symbolic Violations</h2>
    <table border="1" cellpadding="6"><tr><th>Time</th><th>Rule</th><th>Value</th></tr>{rows}</table>
    </body></html>"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path

if __name__ == "__main__":
    print(write_html_from_log())
