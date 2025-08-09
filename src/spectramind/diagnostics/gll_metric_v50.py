import numpy as np

TOTAL_BINS = 567
FGS_WEIGHT = 0.4
AIRS_WEIGHT = 0.6 / (TOTAL_BINS - 1)
SPECTRAL_WEIGHTS = np.array([FGS_WEIGHT] + [AIRS_WEIGHT]*(TOTAL_BINS-1), dtype=np.float32)

def compute_gll(y_true, mu_pred, sigma_pred):
    sigma_pred = np.clip(sigma_pred, 1e-8, 1e6)
    squared = ((y_true - mu_pred) / sigma_pred) ** 2
    logterm = 2*np.log(sigma_pred) + np.log(2*np.pi)
    gll = (squared + logterm) * SPECTRAL_WEIGHTS
    return float(np.mean(np.sum(gll, axis=1)))
