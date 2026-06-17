"""Auto-selection: capacity gating, cross-validated scoring, 1-SE rule.

The selection layer is the only place that knows about the combiner ladder.
Determinism: every CV split is seeded by ``random_state``.

Metrics are computed by exactly one function, :func:`compute_metrics`. During
CV we extract only the primary metric per fold (one number, used by the 1-SE
rule). The full panel is computed once for the winning model in
:func:`panel_for`, from its aggregated out-of-sample predictions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from .combiners import AVAILABLE, CAPACITY_ORDER, make as make_combiner
from .errors import CombinerBackendMissing


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    # Short-circuit degenerate inputs so spearmanr/kendalltau don't NaN-warn.
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    c = fn(x, y).correlation
    return float(c) if c is not None else float("nan")


# ---------------------------------------------------------------------------
# Capacity gate
# ---------------------------------------------------------------------------

def capacity_gate(n: int, mode: str = "auto") -> List[str]:
    """Candidate combiner list for sample size ``n``.

    ``"auto"`` → standard spec ladder, ``"fast"`` → glm only,
    ``"exhaustive"`` → all four tiers regardless of n.
    """
    if mode == "fast":
        return ["glm"]
    if mode == "exhaustive":
        return list(CAPACITY_ORDER)
    if n < 200:
        return ["glm"]
    if n < 1000:
        return ["glm", "glm_interactions"]
    if n < 5000:
        return ["glm", "glm_interactions", "ebm"]
    return list(CAPACITY_ORDER)


def filter_to_available(candidates: List[str]) -> Tuple[List[str], List[str]]:
    """Drop combiners whose backend isn't installed. Returns ``(kept, dropped)``."""
    kept, dropped = [], []
    for c in candidates:
        _, check = AVAILABLE[c]
        (kept if check() else dropped).append(c)
    return kept, dropped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# The primary metric used to pick a winner is pinned per task — no user-facing
# override; if you want a different ranking, compute it from `winner_panel`.
_PRIMARY = {
    "regression": "spearman",
    "pairwise":   "auc",
    "ranking":    "kendall",
}
METRICS_LOWER_BETTER = {"mae", "brier"}


def primary_metric_for(task: str) -> str:
    return _PRIMARY[task]


def compute_metrics(y_true, y_pred, task: str, target_type: str) -> Dict[str, float]:
    """The single metric function — returns a stable-shape panel.

    Used twice in the pipeline: per fold (we extract just the primary) and once
    on the winner's aggregated out-of-sample predictions (the full panel).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: Dict[str, float] = {
        "spearman": safe_corr(spearmanr,  y_true, y_pred),
        "kendall":  safe_corr(kendalltau, y_true, y_pred),
    }
    out["pearson"] = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")
    )
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    if target_type == "binary" or task == "pairwise":
        try:
            out["auc"] = float(roc_auc_score(y_true.astype(int), y_pred))
        except ValueError:
            out["auc"] = float("nan")
        out["brier"] = float(brier_score_loss(y_true.astype(int), np.clip(y_pred, 0.0, 1.0)))
    else:
        out["brier"] = float("nan")
        out["auc"]   = float("nan")
    return out


# ---------------------------------------------------------------------------
# Cross-validated scoring
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    combiner_type: str
    primary_metric: str
    primary_mean: float
    primary_std: float
    # Per-fold primary metric values (NaN-filtered). Used by `primary_se` and
    # serialized for audit; the full panel is NOT stored per fold.
    per_fold: List[float] = field(default_factory=list)
    # Per-fold out-of-sample predictions, kept so the winner can compute its
    # full metric panel later. Always None for failed CV runs.
    fold_preds: Optional[List[Tuple[np.ndarray, np.ndarray]]] = field(default=None, repr=False)
    error: Optional[str] = None

    @property
    def primary_se(self) -> float:
        # SE = std / sqrt(n_folds), used by the 1-SE rule.
        n = max(len(self.per_fold), 1)
        return self.primary_std / np.sqrt(n) if self.primary_std > 0 else 0.0


def _make_splitter(task, target_type, groups, n_splits, random_state):
    """GroupKFold > StratifiedKFold (discrete y) > KFold (continuous y)."""
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    if task == "pairwise" or target_type in {"binary", "ordinal"}:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_combiner(
    combiner_type: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    target_type: str,
    link: str,
    feature_names: Optional[List[str]],
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    random_state: int = 42,
    combiner_kwargs: Optional[Dict[str, Any]] = None,
) -> CVResult:
    """Outer K-fold CV with each tier's default hyperparameters.

    Per fold: fit on train, predict on test, compute the primary metric only.
    Predictions are kept on the result for post-selection panel computation.
    """
    combiner_kwargs = combiner_kwargs or {}
    primary_metric = primary_metric_for(task)
    X = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    splitter = _make_splitter(task, target_type, groups, n_splits, random_state)
    fold_preds: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_primary: List[float] = []

    # Whole-CV try/except: one combiner crash shouldn't kill the ladder.
    try:
        if isinstance(splitter, GroupKFold):
            split_iter = splitter.split(X, y_arr, groups=groups)
        elif isinstance(splitter, StratifiedKFold):
            split_iter = splitter.split(X, y_arr.astype(int))
        else:
            split_iter = splitter.split(X)

        for tr, te in split_iter:
            cmb = make_combiner(
                combiner_type,
                task=task, target_type=target_type, link=link,
                feature_names=feature_names,
                random_state=random_state,
                **combiner_kwargs,
            )
            fit_kw: Dict[str, Any] = {}
            if task == "ranking" and groups is not None:
                fit_kw["ranks"] = y_arr[tr]
                fit_kw["groups"] = groups[tr]
            cmb.fit(X[tr], y_arr[tr], **fit_kw)
            y_pred = cmb.predict(X[te])
            fold_preds.append((y_arr[te].copy(), np.asarray(y_pred, dtype=float)))
            fold_primary.append(
                compute_metrics(y_arr[te], y_pred,
                                task=task, target_type=target_type)[primary_metric]
            )
    except CombinerBackendMissing as exc:
        return CVResult(combiner_type=combiner_type, primary_metric=primary_metric,
                        primary_mean=float("nan"), primary_std=float("nan"),
                        error=str(exc))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"{combiner_type}: CV failed ({exc}); marking nan.", RuntimeWarning)
        return CVResult(combiner_type=combiner_type, primary_metric=primary_metric,
                        primary_mean=float("nan"), primary_std=float("nan"),
                        error=str(exc))

    valid = np.asarray([v for v in fold_primary if not np.isnan(v)], dtype=float)
    if len(valid) == 0:
        return CVResult(combiner_type=combiner_type, primary_metric=primary_metric,
                        primary_mean=float("nan"), primary_std=float("nan"),
                        per_fold=fold_primary, fold_preds=fold_preds,
                        error="all folds produced NaN primary metric")
    return CVResult(
        combiner_type=combiner_type,
        primary_metric=primary_metric,
        primary_mean=float(np.mean(valid)),
        primary_std=float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
        per_fold=fold_primary,
        fold_preds=fold_preds,
    )


def panel_for(result: CVResult, *, task: str, target_type: str) -> Dict[str, float]:
    """Full metric panel for a CV'd model, computed on aggregated out-of-sample
    predictions via :func:`compute_metrics`."""
    if not result.fold_preds:
        return {}
    y_true = np.concatenate([yt for yt, _ in result.fold_preds])
    y_pred = np.concatenate([yp for _, yp in result.fold_preds])
    return compute_metrics(y_true, y_pred, task=task, target_type=target_type)


# ---------------------------------------------------------------------------
# 1-SE rule with capacity ordering
# ---------------------------------------------------------------------------

def select_with_1se_rule(results: List[CVResult]) -> Tuple[str, str]:
    """Pick the lowest-capacity combiner within 1 SE of the best CV score.

    Returns ``(winner_name, reason_text)``. Skips combiners that errored.
    """
    valid = [r for r in results if not np.isnan(r.primary_mean)]
    if not valid:
        raise RuntimeError("no combiner produced a valid CV score")

    is_lower = valid[0].primary_metric in METRICS_LOWER_BETTER
    best = (min(valid, key=lambda r: r.primary_mean) if is_lower
            else max(valid, key=lambda r: r.primary_mean))

    se = best.primary_se
    if is_lower:
        within = [r for r in valid if r.primary_mean <= best.primary_mean + se]
    else:
        within = [r for r in valid if r.primary_mean >= best.primary_mean - se]

    # Bias toward lower capacity within the 1-SE band → better generalisation.
    cap_idx = {name: i for i, name in enumerate(CAPACITY_ORDER)}
    within.sort(key=lambda r: cap_idx.get(r.combiner_type, 999))
    winner = within[0]

    if winner.combiner_type == best.combiner_type:
        reason = (
            f"{winner.combiner_type} won outright (mean primary = "
            f"{winner.primary_mean:.4f}; nothing within 1 SE = {se:.4f})."
        )
    else:
        reason = (
            f"{best.combiner_type} had the best mean ({best.primary_mean:.4f}); "
            f"{winner.combiner_type} was within 1 SE = {se:.4f} and is "
            f"lower-capacity — selected {winner.combiner_type}. "
            f"(To force {best.combiner_type}, pass selection={best.combiner_type!r}.)"
        )
    return winner.combiner_type, reason
