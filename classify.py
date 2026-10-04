from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_matrix
from .preprocess import common_average_reference, filter_epochs, reject_artifacts
from .synth import make_dataset


def run_experiment(n_per_class: int = 100, seed: int = 0, ptp_uv: float = 100.0, reject: bool = True,
                   use_car: bool = True, use_filter: bool = True) -> dict:
    X, y = make_dataset(n_per_class, seed)
    n_raw = len(y)
    if use_filter:
        X = filter_epochs(X)
    if reject:
        X, keep = reject_artifacts(X, ptp_uv=ptp_uv)
        y = y[keep]
    if use_car:
        X = common_average_reference(X)
    F = feature_matrix(X)
    Ftr, Fte, ytr, yte = train_test_split(F, y, test_size=0.3, random_state=seed, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Ftr, ytr)
    pred = model.predict(Fte)
    return {"epochs_total": n_raw, "epochs_rejected": int(n_raw - len(y)),
            "accuracy": float(accuracy_score(yte, pred)), "confusion": confusion_matrix(yte, pred).tolist(),
            "n_test": int(len(yte))}
