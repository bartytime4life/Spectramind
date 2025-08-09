import numpy as np, matplotlib.pyplot as plt, plotly.graph_objects as go, os, json
from pathlib import Path
import typer
from datetime import datetime

app = typer.Typer(help="Simulate and diagnose noisy exoplanet transit stacks")

def generate_single_transit(T=1000, depth=0.001, duration=0.1, sigma=0.0005, baseline=1.0, random_phase=True):
    time = np.linspace(0, 1, T)
    flux = np.ones(T) * baseline
    center = np.random.uniform(0.4, 0.6) if random_phase else 0.5
    in_transit = np.abs(time - center) < (duration / 2)
    flux[in_transit] -= depth
    return flux + np.random.normal(0, sigma, size=T)

def stack_transits(n=10, T=1000, depth=0.001, duration=0.1, sigma=0.0005, baseline=1.0, random_phase=True, return_all=False):
    stack = np.stack([generate_single_transit(T, depth, duration, sigma, baseline, random_phase) for _ in range(n)])
    return stack if return_all else stack.mean(axis=0)

@app.command()
def simulate_transit(n: int = 12, T: int = 1200, depth: float = 0.002, duration: float = 0.1, sigma: float = 0.0005,
                     baseline: float = 1.0, random_phase: bool = True, outdir: str = "outputs/diagnostics", tag: str = "default"):
    stack = stack_transits(n=n, T=T, depth=depth, duration=duration, sigma=sigma, baseline=baseline, random_phase=random_phase, return_all=True)
    mean_stack = stack.mean(axis=0)
    out_base = f"{outdir}/stacked_transit"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    np.save(f"{out_base}.npy", mean_stack)
    json.dump({"stack_shape": stack.shape, "mean_flux": mean_stack.tolist()}, open(f"{out_base}.json", "w"), indent=2)
    plt.figure(); [plt.plot(s, alpha=0.15, color="gray") for s in stack]; plt.plot(mean_stack, lw=2, color="black")
    plt.title("Stacked Transit Light Curve"); plt.tight_layout(); plt.savefig(f"{out_base}.png"); plt.close()
    fft = np.abs(np.fft.rfft(mean_stack)); freqs = np.fft.rfftfreq(len(mean_stack))
    fig = go.Figure(); fig.add_trace(go.Scatter(x=freqs, y=fft, mode="lines")); fig.update_layout(template="plotly_white")
    fig.write_html(f"{out_base}_fft.html")
    with open("v50_debug_log.md", "a") as f:
        f.write(f"Transit Stack Simulation {datetime.utcnow().isoformat()}Z | T={T}, n={n}, depth={depth}, sigma={sigma}\n")
    print("Simulation complete.")
if __name__ == "__main__":
    app()
