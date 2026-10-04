from __future__ import annotations

import numpy as np
from scipy import signal

from . import BANDS
from .synth import FS


def band_powers(epochs: np.ndarray, fs: int = FS, relative: bool = True) -> np.ndarray:
    """Welch PSD -> power per band per channel. epochs: (n, ch, t) -> (n, ch, n_bands)."""
    f, pxx = signal.welch(epochs, fs=fs, nperseg=min(epochs.shape[-1], 2 * fs), axis=-1)
    df = f[1] - f[0]
    out = np.stack([pxx[..., (f >= lo) & (f < hi)].sum(-1) * df for lo, hi in BANDS.values()], axis=-1)
    if relative:
        out = out / out.sum(-1, keepdims=True)
    return out


def feature_matrix(epochs: np.ndarray) -> np.ndarray:
    """Flatten log relative band power across channels into one feature vector per epoch."""
    bp = band_powers(epochs, relative=True)
    return np.log(bp + 1e-12).reshape(len(epochs), -1)
