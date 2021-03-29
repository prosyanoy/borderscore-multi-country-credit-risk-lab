"""Discrimination and calibration metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from .data import TARGET

_EPS = 1e-6


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-frequency bins (defaults are rare, so equal-width bins are uninformative)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, min(bins, len(p))):
        if len(chunk):
            ece += len(chunk) / len(p) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece)


def evaluate(y, p) -> dict[str, float]:
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both classes to compute AUC/KS")
    return {
        "auc": float(roc_auc_score(y, p)),
        "ks": ks_statistic(y, p),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


def evaluate_by_country(df: pd.DataFrame, p: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-country metrics plus a ``macro`` entry (unweighted mean over countries)."""
    p = np.asarray(p)
    out = {}
    for code, idx in df.groupby("country").indices.items():
        out[code] = evaluate(df[TARGET].to_numpy()[idx], p[idx])
    keys = next(iter(out.values())).keys()
    out["macro"] = {k: float(np.mean([m[k] for m in out.values()])) for k in keys}
    return out
