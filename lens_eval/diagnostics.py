"""Diagnostics computed on training data after fit.

Returns a structured dict consumed by the report layer. Dict shape is stable
across combiner types; combiner-specific items go in ``diagnostics['by_combiner']``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.stats import pearsonr


def compute_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    *,
    feature_names: List[str],
    task: str,
    target_type: str,
    ece_n_bins: int = 10,
) -> Dict[str, Any]:
    """Run all built-in diagnostics."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    out: Dict[str, Any] = {}

    # --- Residuals ------------------------------------------------------
    resid = y - y_pred
    out["residuals"] = {
        "mean":    float(np.mean(resid)),
        "std":     float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0,
        "min":     float(np.min(resid)),
        "max":     float(np.max(resid)),
        "abs_max": float(np.max(np.abs(resid))),
    }

    # --- Calibration ----------------------------------------------------
    # Binary/pairwise: reliability + ECE on probabilities.
    # Bounded regression: binary proxy (above-median) AND a regression curve.
    # Continuous regression: no calibration (would be misleading).
    if task == "pairwise" or target_type == "binary":
        out["calibration"] = {
            "kind": "binary",
            **_calibration_binary(np.clip(y_pred, 0.0, 1.0), y.astype(int), n_bins=ece_n_bins),
        }
    elif task == "regression" and target_type == "bounded":
        y_bin = (y > np.median(y)).astype(int)
        out["calibration"] = {
            "kind": "bounded_regression",
            **_calibration_binary(np.clip(y_pred, 0.0, 1.0), y_bin, n_bins=ece_n_bins),
            "regression": _calibration_regression(y_pred, y, n_bins=ece_n_bins),
        }
    else:
        out["calibration"] = None

    # --- Feature correlations ------------------------------------------
    # Pairwise Pearson; max_offdiag flags multicollinearity that would make
    # GLM coefficients hard to interpret.
    D = X.shape[1]
    corr = np.eye(D)
    for i in range(D):
        for j in range(i + 1, D):
            try:
                r, _ = pearsonr(X[:, i], X[:, j])
            except Exception:
                # Constant column → undefined; show NaN rather than crash.
                r = float("nan")
            corr[i, j] = corr[j, i] = r
    out["feature_correlations"] = {
        "feature_names": list(feature_names),
        "matrix":        corr.tolist(),
        "max_offdiag":   float(np.max(np.abs(corr - np.eye(D)))) if D > 1 else 0.0,
    }

    # --- Monotonicity ---------------------------------------------------
    # Bin predictions by each feature; classify the per-bin mean trend.
    # 1e-3 tolerance absorbs sampling noise on <100-row bins.
    mono: Dict[str, Dict[str, str]] = {}
    for j, name in enumerate(feature_names):
        order = np.argsort(X[:, j])
        binned = np.array_split(y_pred[order], min(10, len(order)))
        means = np.asarray([float(np.mean(b)) for b in binned if len(b)])
        if len(means) < 2:
            mono[name] = {"status": "indeterminate"}
            continue
        diffs = np.diff(means)
        if np.all(diffs >= -1e-3):
            mono[name] = {"status": "increasing"}
        elif np.all(diffs <= 1e-3):
            mono[name] = {"status": "decreasing"}
        else:
            mono[name] = {"status": "non-monotone"}
    out["monotonicity"] = mono

    # --- Outliers ------------------------------------------------------
    # Standardised residual > 3 (cap the index list at 50).
    std_resid = out["residuals"]["std"] if out["residuals"]["std"] > 1e-9 else 1.0
    z = resid / std_resid
    out["outliers"] = {
        "count":   int(np.sum(np.abs(z) > 3)),
        "indices": np.where(np.abs(z) > 3)[0].tolist()[:50],
    }

    out["warnings"] = []
    return out


def _equal_mass_bins(order: np.ndarray, n_bins: int) -> List[np.ndarray]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return [b for b in np.array_split(order, min(n_bins, max(len(order), 1))) if len(b)]


def _calibration_binary(p: np.ndarray, y: np.ndarray, *, n_bins: int) -> Dict[str, Any]:
    """Equal-mass reliability curve + ECE for binary probabilities."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=int).ravel()
    n = len(p)
    if n == 0:
        return {"ece": float("nan"), "brier": float("nan"), "reliability": None}

    bins = _equal_mass_bins(np.argsort(p), n_bins)
    bin_conf, bin_acc, weights = [], [], []
    ece = 0.0
    for idx in bins:
        conf = float(np.mean(p[idx]))
        acc  = float(np.mean(y[idx]))
        bin_conf.append(conf)
        bin_acc.append(acc)
        weights.append(int(len(idx)))
        ece += (len(idx) / n) * abs(conf - acc)
    return {
        "ece":   float(ece),
        "brier": float(np.mean((p - y) ** 2)),
        "reliability": {"confidence": bin_conf, "accuracy": bin_acc, "weights": weights},
    }


def _calibration_regression(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int,
) -> Dict[str, Any]:
    """Equal-mass regression calibration: mean(pred) vs mean(true) per bin."""
    yp = np.asarray(y_pred, dtype=float).ravel()
    yt = np.asarray(y_true, dtype=float).ravel()
    m = np.isfinite(yp) & np.isfinite(yt)
    yp, yt = yp[m], yt[m]
    if len(yp) == 0:
        return {"cal_mae": float("nan"), "curve": None}

    bins = _equal_mass_bins(np.argsort(yp), n_bins)
    mean_pred, mean_true, weights = [], [], []
    for idx in bins:
        mean_pred.append(float(np.mean(yp[idx])))
        mean_true.append(float(np.mean(yt[idx])))
        weights.append(int(len(idx)))

    cal_mae = float(np.average(np.abs(np.asarray(mean_pred) - np.asarray(mean_true)), weights=weights))
    return {
        "cal_mae": cal_mae,
        "curve":   {"mean_pred": mean_pred, "mean_true": mean_true, "weights": weights},
    }
