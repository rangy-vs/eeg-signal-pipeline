# EEG Signal Pipeline

![ci](https://github.com/rangy-vs/eeg-signal-pipeline/actions/workflows/ci.yml/badge.svg)

A tested EEG preprocessing → feature → classification pipeline, run on **synthetic** data so every stage has known ground truth. It does not use or claim any real recordings.

![PSD before/after](docs/psd_before_after.png)

```bash
pip install -r requirements.txt
python -m eegpipe demo     # runs the experiment, regenerates the figure
python -m pytest -q        # 11 tests
```

## Pipeline
1. **Notch (60 Hz) + 1–40 Hz Butterworth band-pass.** Zero-phase (`sosfiltfilt`) with second-order sections for numerical stability; a test verifies zero lag.
2. **Artifact rejection** (peak-to-peak threshold) *after* filtering and *before* re-referencing.
3. **Common average reference.**
4. **Features:** Welch PSD → relative power in delta/theta/alpha/beta/gamma per channel, log-transformed.
5. **Classifier:** standardized logistic regression, eyes-open vs eyes-closed (alpha amplitude differs between the synthetic states).

## Results (synthetic data, 200 epochs, mean ± std over 5 seeds)
| Variant | Accuracy | Epochs rejected (seed 0) |
|---|---|---|
| Full pipeline | 0.956 ± 0.014 | 26 / 200 |
| No CAR | 1.000 ± 0.000 | 26 / 200 |
| No artifact rejection | 0.957 ± 0.008 | 0 / 200 |
| No filtering | 0.979 ± 0.017 | **105 / 200** |
| Nothing (raw) | 1.000 ± 0.000 | 0 / 200 |

## What I learned (including what didn't work)
- **Preprocessing does not raise accuracy on this task.** The synthetic problem is easy, and log relative band power over 1–40 Hz is already insensitive to mains hum and drift. I expected preprocessing to help; the ablation says otherwise, so the claims above are limited to what the tests actually verify.
- **What filtering does buy:** without it, drift and mains hum inflate peak-to-peak amplitude and the rejector discards **52%** of epochs instead of ~13%. `test_filtering_prevents_over_rejection` locks that in.
- **CAR can erase real signal.** My first synthetic data used the same alpha phase on every channel and the classifier fell to chance, because CAR subtracts anything common to all channels. The data generator now uses per-channel phase, and `test_car_cancels_perfectly_shared_signal` documents the behavior.
- **Order matters:** rejecting on raw amplitude failed, and rejecting before CAR keeps blinks from being smeared onto every channel.

## Limits / next steps
Synthetic data validates the *code*, not neuroscience. Next: run the same pipeline on a public dataset (e.g. PhysioNet EEG Motor Movement/Imagery), report cross-validated results per subject, and replace threshold rejection with ICA-based blink removal.
