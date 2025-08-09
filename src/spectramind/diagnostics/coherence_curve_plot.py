import numpy as np, json, os, matplotlib.pyplot as plt

def compute_coherence_curve(mu: np.ndarray):
    d2 = mu[:, :-2] - 2*mu[:, 1:-1] + mu[:, 2:]
    curve = (d2**2).mean(axis=0)
    return np.pad(curve, (1,1), mode='edge')

def plot_and_save(curve, outdir='diagnostics', name='coherence_curve'):
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(10,3))
    plt.plot(curve, lw=2); plt.title('Coherence Curve'); plt.tight_layout()
    png = os.path.join(outdir, f'{name}.png'); fig.savefig(png); plt.close(fig)
    with open(os.path.join(outdir, f'{name}.json'), 'w') as f: json.dump({'curve': curve.tolist()}, f, indent=2)
    return png
