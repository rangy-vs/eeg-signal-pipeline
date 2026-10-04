"""Synthetic multichannel EEG so the whole pipeline is testable with known ground truth.

Model per channel: 1/f background + 50/60 Hz mains + slow drift + occasional eye blinks, plus an
alpha (10 Hz) rhythm whose amplitude is higher in the 'closed' state than the 'open' state
(a well-known real effect, used here only to give the classifier something verifiable)."""
from __future__ import annotations

import numpy as np

FS = 256  # Hz


def _pink(n: int, rng) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def make_epoch(state: str, seconds: float = 4.0, n_ch: int = 4, blink_prob: float = 0.1,
               mains_hz: float = 60.0, rng=None) -> np.ndarray:
    """Return array (n_ch, n_samples) in microvolts. state: 'open' | 'closed'."""
    rng = rng or np.random.default_rng()
    n = int(seconds * FS)
    t = np.arange(n) / FS
    alpha_amp = 6.0 if state == "closed" else 2.0
    alpha_hz = rng.uniform(9.5, 10.5)
    weights = np.linspace(0.4, 1.0, n_ch)          # alpha is stronger over posterior channels
    x = np.empty((n_ch, n))
    for c in range(n_ch):
        x[c] = 8.0 * _pink(n, rng)                                     # background
        # independent phase per channel: a perfectly shared rhythm would be cancelled by CAR
        x[c] += alpha_amp * weights[c] * np.sin(2 * np.pi * alpha_hz * t + rng.uniform(0, 2 * np.pi))
        x[c] += 15.0 * np.sin(2 * np.pi * mains_hz * t + rng.uniform(0, 6.28))   # mains hum
        x[c] += rng.uniform(-30, 30) * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28))  # drift
    if rng.random() < blink_prob:                                      # frontal blink artifact
        center = rng.integers(FS, n - FS)
        blink = 180.0 * np.exp(-0.5 * ((np.arange(n) - center) / (0.08 * FS)) ** 2)
        x[0] += blink; x[1] += 0.6 * blink
    return x


def make_dataset(n_per_class: int = 100, seed: int = 0, **kw):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, state in enumerate(["open", "closed"]):
        for _ in range(n_per_class):
            X.append(make_epoch(state, rng=rng, **kw)); y.append(label)
    idx = rng.permutation(len(y))
    return np.array(X)[idx], np.array(y)[idx]
