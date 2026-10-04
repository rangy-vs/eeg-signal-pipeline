"""python -m eegpipe demo   -> runs the experiment and writes docs/psd_before_after.png"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .classify import run_experiment
from .preprocess import preprocess
from .synth import FS, make_epoch


def plot_psd(path: Path) -> None:
    rng = np.random.default_rng(1)
    raw = make_epoch("closed", blink_prob=0, rng=rng)
    clean = preprocess(raw[None])[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    for sig, label in [(raw[3], "raw"), (clean[3], "notch + 1-40 Hz band-pass + CAR")]:
        f, p = signal.welch(sig, fs=FS, nperseg=2 * FS)
        ax.semilogy(f, p, label=label)
    ax.axvspan(8, 13, alpha=0.15, label="alpha band")
    ax.set(xlim=(0, 80), ylim=(1e-4, 1e3), xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)",
           title="Synthetic EEG (eyes closed), posterior channel")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(prog="eegpipe")
    ap.add_argument("cmd", choices=["demo"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    Path("docs").mkdir(exist_ok=True)
    plot_psd(Path("docs/psd_before_after.png"))
    res = run_experiment(seed=args.seed)
    print(json.dumps(res, indent=2))
    print("figure -> docs/psd_before_after.png")


if __name__ == "__main__":
    main()
