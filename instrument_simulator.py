"""
SpectraMind V50 – Instrument Simulator (Symbolic Violation Enabled)
-------------------------------------------------------------------
Simulates AIRS/FGS1 detector output and injects symbolic violations for testing.
"""

import numpy as np

def inject_symbolic_violations(mu: np.ndarray, violations: list[str], strength: float = 0.5) -> np.ndarray:
    """
    Intentionally corrupts spectrum to violate symbolic constraints.

    Args:
        mu: (283,) original spectrum
        violations: list of symbolic constraints to break ("smoothness", "nonnegativity", "symmetry")
        strength: magnitude of violation

    Returns:
        corrupted μ: (283,) spectrum with injected violations
    """
    mu = mu.copy()

    if "smoothness" in violations:
        idx = np.random.randint(20, 263)  # safe margin
        mu[idx - 2:idx + 2] += np.array([20, -40, 30, -10]) * strength

    if "nonnegativity" in violations:
        neg_bins = np.random.choice(len(mu), size=3, replace=False)
        mu[neg_bins] -= np.abs(mu[neg_bins]) + strength * 50

    if "symmetry" in violations:
        midpoint = len(mu) // 2
        asym_idx = np.random.randint(midpoint)
        mu[midpoint + asym_idx] += strength * 100

    return mu


def simulate_airs_signal(mu: np.ndarray, n_timesteps: int = 11250, noise_level: float = 10.0,
                         variable_noise: bool = False, symbolic_violation=None) -> np.ndarray:
    """
    Simulates time-series AIRS output, optionally injecting symbolic errors.

    Args:
        mu: (283,) spectrum
        n_timesteps: number of time steps
        noise_level: base Gaussian noise
        variable_noise: scale σ by bin magnitude
        symbolic_violation: list of constraints to violate

    Returns:
        (n_timesteps, 283) array
    """
    if symbolic_violation:
        mu = inject_symbolic_violations(mu, symbolic_violation)

    mu_expanded = np.tile(mu, (n_timesteps, 1)).astype(np.float32)
    if variable_noise:
        noise_std = noise_level * (1 + mu / (np.mean(mu) + 1e-5))
    else:
        noise_std = np.full_like(mu, noise_level)

    noise = np.random.normal(0, noise_std, size=mu_expanded.shape).astype(np.float32)
    return mu_expanded + noise


def simulate_fgs1_lightcurve(depth: float, n_timesteps: int = 135000, duration_frac: float = 0.05,
                             noise_level: float = 5.0, jitter_amplitude: float = 2.0,
                             transit_shape: str = "flat", jitter_freq=3) -> np.ndarray:
    """
    Simulates a broadband light curve with optional low-frequency jitter.

    Returns:
        (n_timesteps,) array
    """
    flux = np.ones(n_timesteps) * 1000.0
    transit_len = int(n_timesteps * duration_frac)
    center = n_timesteps // 2
    start = center - transit_len // 2

    if transit_shape == "flat":
        flux[start:start + transit_len] -= depth
    elif transit_shape == "smooth":
        t = np.linspace(-1, 1, transit_len)
        transit = 1 - depth * (1 - t**2)
        flux[start:start + transit_len] *= transit

    noise = np.random.normal(0, noise_level, n_timesteps)
    jitter = np.sin(np.linspace(0, 2 * np.pi * jitter_freq, n_timesteps)) * jitter_amplitude
    return (flux + noise + jitter).astype(np.float32)