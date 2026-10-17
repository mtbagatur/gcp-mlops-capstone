"""Input drift between the training data and recent prediction requests (requirement G8).

Population Stability Index (PSI) per feature:
  numeric     - 10 bins from training-data quantiles
  categorical - one bin per training category, plus "other" for unseen values
PSI < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant drift (common rule of thumb).
"""
import numpy as np
import pandas as pd

try:
    from training.features import CATEGORICAL, NUMERIC
except ImportError:
    from features import CATEGORICAL, NUMERIC

EPS = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(train: pd.Series, recent: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(train.astype(float), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(train.astype(float), edges)[0] / len(train)
    a = np.histogram(recent.astype(float), edges)[0] / len(recent)
    return _psi(e, a)


def psi_categorical(train: pd.Series, recent: pd.Series) -> float:
    cats = train.astype(str).value_counts(normalize=True)
    r = recent.astype(str)
    r = r.where(r.isin(cats.index), "__other__").value_counts(normalize=True)
    e = cats.reindex(list(cats.index) + ["__other__"], fill_value=0.0).to_numpy()
    a = r.reindex(list(cats.index) + ["__other__"], fill_value=0.0).to_numpy()
    return _psi(e, a)


def drift_report(train: pd.DataFrame, recent: pd.DataFrame, threshold: float = 0.2) -> dict:
    scores = {c: psi_numeric(train[c], recent[c]) for c in NUMERIC}
    scores |= {c: psi_categorical(train[c], recent[c]) for c in CATEGORICAL}
    scores = {k: round(v, 4) for k, v in scores.items()}
    drifted = sorted([k for k, v in scores.items() if v > threshold], key=lambda k: -scores[k])
    return {"n_recent": int(len(recent)), "threshold": threshold, "psi": scores, "drifted": drifted}
