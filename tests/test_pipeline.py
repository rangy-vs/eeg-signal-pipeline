import numpy as np
import pytest
from scipy import signal

from eegpipe import BANDS
from eegpipe.classify import run_experiment
from eegpipe.features import band_powers
from eegpipe.preprocess import bandpass, common_average_reference, filter_epochs, notch, reject_artifacts
from eegpipe.synth import FS, make_epoch

T = np.arange(0, 8, 1 / FS)


def rms(x): return np.sqrt(np.mean(x ** 2))


def gain_db(fn, hz):
    x = np.sin(2 * np.pi * hz * T)
    y = fn(x)[FS:-FS]  # trim filter edge effects
    return 20 * np.log10(rms(y) / rms(x[FS:-FS]))


def test_bandpass_passes_alpha_and_kills_extremes():
    f = lambda x: bandpass(x, 1, 40)
    assert abs(gain_db(f, 10)) < 0.5            # passband flat
    assert gain_db(f, 0.1) < -30                # drift removed
    assert gain_db(f, 100) < -30                # high-freq removed


def test_notch_removes_mains_but_spares_neighbours():
    f = lambda x: notch(x, 60)
    assert gain_db(f, 60) < -30
    assert abs(gain_db(f, 40)) < 1.0


def test_filters_are_zero_phase():
    x = np.sin(2 * np.pi * 10 * T)
    y = bandpass(x)
    lag = np.argmax(signal.correlate(y[FS:-FS], x[FS:-FS], "full")) - (len(x[FS:-FS]) - 1)
    assert lag == 0                             # no phase shift: matters for event-locked analysis


def test_car_makes_channels_sum_to_zero():
    x = np.random.default_rng(0).standard_normal((4, 100))
    assert np.allclose(common_average_reference(x).sum(axis=0), 0)


def test_band_power_peaks_in_correct_band():
    for hz, band in [(2, "delta"), (6, "theta"), (10, "alpha"), (20, "beta"), (35, "gamma")]:
        x = np.sin(2 * np.pi * hz * T)[None, None, : 4 * FS]
        bp = band_powers(x)[0, 0]
        assert list(BANDS)[int(np.argmax(bp))] == band, hz


def test_relative_power_sums_to_one():
    bp = band_powers(make_epoch("open", rng=np.random.default_rng(0))[None])
    assert np.allclose(bp.sum(-1), 1)


def test_artifact_rejection_flags_blinks_only():
    rng = np.random.default_rng(3)
    clean = [make_epoch("open", blink_prob=0, rng=rng) for _ in range(20)]
    blinks = [make_epoch("open", blink_prob=1, rng=rng) for _ in range(20)]
    _, keep = reject_artifacts(filter_epochs(np.array(clean + blinks)), ptp_uv=100)
    assert keep[:20].mean() >= 0.9 and keep[20:].mean() <= 0.1


def test_classifier_beats_chance_by_a_lot_and_is_reproducible():
    a, b = run_experiment(seed=0), run_experiment(seed=0)
    assert a == b
    assert a["accuracy"] > 0.9
    assert a["epochs_rejected"] > 0


def test_filtering_prevents_over_rejection():
    """Drift + mains hum inflate raw peak-to-peak range. Without filtering, the amplitude-based
    rejector throws away most good epochs; with filtering it removes roughly just the blinks (~10%)."""
    with_filter = run_experiment(seed=0)
    without = run_experiment(seed=0, use_filter=False)
    assert with_filter["epochs_rejected"] / with_filter["epochs_total"] < 0.25
    assert without["epochs_rejected"] / without["epochs_total"] > 0.40


def test_car_cancels_perfectly_shared_signal():
    """Documents a real property (and a bug I hit in the synthetic data): CAR subtracts anything
    identical across channels, so a rhythm with the same phase everywhere would vanish."""
    shared = np.tile(np.sin(2 * np.pi * 10 * T[: 2 * FS]), (4, 1))
    assert np.abs(common_average_reference(shared)).max() < 1e-9


def test_accuracy_stable_across_seeds():
    accs = [run_experiment(seed=s)["accuracy"] for s in range(3)]
    assert min(accs) > 0.9
