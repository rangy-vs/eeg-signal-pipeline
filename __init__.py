"""Small, tested EEG preprocessing + feature + classification pipeline (runs on synthetic data)."""
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 40)}
