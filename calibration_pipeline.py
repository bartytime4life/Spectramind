"""
SpectraMind V50 – Calibration Pipeline (Final Version)
------------------------------------------------------
Converts raw detector data (FGS1, AIRS-CH0) into calibrated, science-ready light curves
for input to modeling, diagnostics, and inference.

Implements:
- ADC conversion
- Nonlinearity correction
- Dead/hot pixel masking
- Dark current subtraction
- Flat fielding
- CDS differencing
- Aperture photometry (FGS1)
- Spectral trace extraction (AIRS)
- Common-mode normalization
- Time/phase alignment
"""

import numpy as np
import pyarrow.parquet as pq
from typing import Tuple, Dict
import os


def load_and_reshape_frames(path: str, shape: Tuple[int, int, int]) -> np.ndarray:
    print(f"📥 Loading frames from {path}")
    raw = pq.read_table(path).to_pandas().values.astype(np.uint16)
    return raw.reshape(shape)


def apply_adc_conversion(data: np.ndarray, gain: float, offset: float) -> np.ndarray:
    print("⚙️  ADC conversion")
    return data.astype(np.float32) * gain + offset


def mask_bad_pixels(stack: np.ndarray, dead_mask: np.ndarray) -> np.ndarray:
    print("🧼 Masking bad pixels")
    return np.where(dead_mask, np.nan, stack)


def correct_nonlinearity(stack: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    print("🔄 Applying nonlinearity correction")
    corrected = np.zeros_like(stack, dtype=np.float32)
    for deg, c in enumerate(coeffs.transpose(2, 0, 1)):
        corrected += c * np.power(stack, deg)
    return corrected


def subtract_dark_current(stack: np.ndarray, dark_map: np.ndarray, dt: float) -> np.ndarray:
    print("🌑 Subtracting dark current")
    return stack - dark_map * dt


def flat_field_correct(stack: np.ndarray, flat_map: np.ndarray) -> np.ndarray:
    print("📸 Flat field correction")
    return stack / flat_map


def apply_cds(frames: np.ndarray) -> np.ndarray:
    print("➖ Applying CDS (frame differencing)")
    return frames[1::2] - frames[::2]


def extract_lightcurve_aperture(stack: np.ndarray, aperture_mask: np.ndarray) -> np.ndarray:
    print("🔍 Extracting FGS1 light curve via aperture")
    return np.nansum(stack * aperture_mask, axis=(1, 2))


def extract_spectrum_trace(stack: np.ndarray, axis: int = 1) -> np.ndarray:
    print("📈 Extracting AIRS spectral trace")
    return np.nansum(stack, axis=axis)


def normalize_by_fgs1(airs_lc: np.ndarray, fgs_lc: np.ndarray) -> np.ndarray:
    print("🔗 Normalizing AIRS by FGS1")
    return airs_lc / fgs_lc[:, None]


def phase_fold(time_array: np.ndarray, t0: float, period: float) -> np.ndarray:
    phase = ((time_array - t0 + 0.5 * period) % period) / period - 0.5
    return phase


def run_calibration(planet_id: str, metadata: dict, paths: dict) -> dict:
    """
    Calibrates a single planet using raw frame paths + metadata.

    Args:
        planet_id: ID string
        metadata: dict with gains, t0, period, dt, etc.
        paths: dict of input frame + calibration component file paths

    Returns:
        dict with:
            - 'planet_id'
            - 'fgs1_lc': (N,)
            - 'airs_lc': (N, 356)
            - 'time': (N,)
            - 'phase': (N,)
    """
    print(f"\n🚀 Starting calibration for {planet_id}")

    # --- FGS1 ---
    fgs_frames = load_and_reshape_frames(paths['fgs_signal'], (135000, 32, 32))
    fgs_frames = apply_adc_conversion(fgs_frames, metadata['fgs_gain'], metadata['fgs_offset'])
    fgs_frames = mask_bad_pixels(fgs_frames, np.load(paths['fgs_dead']))
    fgs_frames = correct_nonlinearity(fgs_frames, np.load(paths['fgs_linear']))
    fgs_frames = subtract_dark_current(fgs_frames, np.load(paths['fgs_dark']), metadata['dt_fgs'])
    fgs_frames = flat_field_correct(fgs_frames, np.load(paths['fgs_flat']))
    fgs_imgs = apply_cds(fgs_frames)
    fgs_lc = extract_lightcurve_aperture(fgs_imgs, np.load(paths['fgs_aperture']))

    # --- AIRS ---
    airs_frames = load_and_reshape_frames(paths['airs_signal'], (11250, 32, 356))
    airs_frames = apply_adc_conversion(airs_frames, metadata['airs_gain'], metadata['airs_offset'])
    airs_frames = mask_bad_pixels(airs_frames, np.load(paths['airs_dead']))
    airs_frames = correct_nonlinearity(airs_frames, np.load(paths['airs_linear']))
    airs_frames = subtract_dark_current(airs_frames, np.load(paths['airs_dark']), metadata['dt_airs'])
    airs_frames = flat_field_correct(airs_frames, np.load(paths['airs_flat']))
    airs_imgs = apply_cds(airs_frames)
    airs_lc = extract_spectrum_trace(airs_imgs, axis=1)  # collapse rows

    # --- Normalize & Time Alignment ---
    norm_lc = normalize_by_fgs1(airs_lc, fgs_lc)
    time = np.arange(len(fgs_lc)) * metadata['dt_fgs']
    phase = phase_fold(time, metadata['t0'], metadata['period'])

    print(f"✅ Calibration complete for {planet_id}")
    return {
        'planet_id': planet_id,
        'fgs1_lc': fgs_lc.astype(np.float32),
        'airs_lc': norm_lc.astype(np.float32),
        'time': time.astype(np.float32),
        'phase': phase.astype(np.float32)
    }