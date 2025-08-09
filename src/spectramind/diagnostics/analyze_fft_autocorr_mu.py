import numpy as np, pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
import plotly.graph_objects as go
from pathlib import Path
def analyze_fft(submission_csv: str, output_base: str = "outputs/diagnostics/mu_fft_autocorr_v50"):
    Path("outputs/diagnostics").mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(submission_csv).set_index("planet_id")
    mu_cols = [c for c in sub.columns if c.startswith("mu_")]
    spectra = sub[mu_cols]
    fig = go.Figure()
    for pid, row in spectra.iterrows():
        mu = row.values.astype(np.float32)
        mu = (mu - mu.mean())/(mu.std()+1e-8)
        fft_s = gaussian_filter1d(np.abs(rfft(mu)), sigma=2.0)
        freq = rfftfreq(len(mu), d=1)
        fig.add_trace(go.Scatter(x=freq, y=fft_s, mode="lines", name=pid, opacity=0.5))
    fig.update_layout(title="μ-Spectrum FFTs", xaxis_title="FFT Frequency", yaxis_title="Smoothed Magnitude", template="plotly_white")
    html_out = f"{output_base}_fft_clusters.html"; fig.write_html(html_out); return html_out