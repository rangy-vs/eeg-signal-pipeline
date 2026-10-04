from __future__ import annotations

import numpy as np
from scipy import signal

from .synth import FS


def bandpass(x: np.ndarray, lo: float = 1.0, hi: float = 40.0, fs: int = FS, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis (second-order sections = numerically stable)."""
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, hz: float = 60.0, q: float = 30.0, fs: int = FS) -> np.ndarray:
    b, a = signal.iirnotch(hz, q, fs=fs)
    return signal.filtfilt(b, a, x, axis=-1)


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """Subtract the across-channel mean at each sample (channels on axis -2)."""
    return x - x.mean(axis=-2, keepdims=True)


def peak_to_peak(x: np.ndarray) -> np.ndarray:
    return x.max(axis=-1) - x.min(axis=-1)


def reject_artifacts(epochs: np.ndarray, ptp_uv: float = 100.0):
    """Drop epochs where ANY channel's peak-to-peak exceeds the threshold (blinks, motion).
    Returns (clean_epochs, boolean keep-mask) so labels can be filtered the same way."""
    keep = (peak_to_peak(epochs).max(axis=-1) < ptp_uv)
    return epochs[keep], keep


def filter_epochs(epochs: np.ndarray, mains_hz: float = 60.0) -> np.ndarray:
    """Notch + band-pass only. Artifact rejection happens on this output, BEFORE re-referencing:
    drift/mains inflate raw peak-to-peak range, and CAR would spread a blink onto every channel."""
    return bandpass(notch(epochs, mains_hz))


def preprocess(epochs: np.ndarray, mains_hz: float = 60.0) -> np.ndarray:
    return common_average_reference(filter_epochs(epochs, mains_hz))
