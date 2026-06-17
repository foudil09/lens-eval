"""EBM combiner — wraps interpret-ml's Explainable Boosting Machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseCombiner, expand_pairs


@dataclass
class EBMCombiner(BaseCombiner):
    max_bins:     int = 256
    outer_bags:   int = 8
    interactions: Any = "auto"   # 0, int, or "auto"

    model_: Optional[Any] = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        pairs: Optional[np.ndarray] = None,
        ranks: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "EBMCombiner":
        # Deferred import so the module is importable without interpret-ml.
        from interpret.glassbox import (
            ExplainableBoostingClassifier,
            ExplainableBoostingRegressor,
        )

        X = np.asarray(X, dtype=float)

        # `interactions`: number of pairwise interactions EBM auto-discovers
        # via FAST. n_features - 1 is a sensible default at D=4 (3 pairs).
        inter = max(0, X.shape[1] - 1) if self.interactions == "auto" else self.interactions

        common = dict(
            max_bins=self.max_bins,
            outer_bags=self.outer_bags,
            interactions=inter,
            random_state=self.random_state,
        )
        # Push real dim names through to interpret-ml so explain_global /
        # explain_local don't return "feature_0000" — that breaks both
        # LENS.feature_importance() (name-based lookup) and the text report
        # (interactions render as "feature_0000 & feature_0001" otherwise).
        if self.feature_names:
            common["feature_names"] = list(self.feature_names)

        if self.task == "pairwise":
            if pairs is not None:
                X_use, y_use = expand_pairs(X, pairs)
            else:
                X_use = X
                y_use = np.asarray(y, dtype=int).ravel()
            self.model_ = ExplainableBoostingClassifier(**common)
            self.model_.fit(X_use, y_use)
        elif self.target_type == "binary":
            self.model_ = ExplainableBoostingClassifier(**common)
            self.model_.fit(X, np.asarray(y, dtype=int).ravel())
        else:
            self.model_ = ExplainableBoostingRegressor(**common)
            self.model_.fit(X, np.asarray(y, dtype=float).ravel())

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        # Classifier → positive-class probability, matching GLM.predict shape.
        if self.task == "pairwise" or self.target_type == "binary":
            return self.model_.predict_proba(X)[:, 1]
        return self.model_.predict(X)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature local contributions via interpret-ml's ``explain_local``.

        Interaction-term contributions are split 50/50 between their parents so
        the result stays (N, D_base). Falls back to finite differences when
        ``explain_local`` raises (the API shifts across interpret-ml versions).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        D = X.shape[1]
        try:
            explanation = self.model_.explain_local(X)
            contribs = np.zeros((X.shape[0], D), dtype=float)
            base_names = (list(self.model_.feature_names_in_)
                          if hasattr(self.model_, "feature_names_in_")
                          else [f"feature_{i:04d}" for i in range(D)])
            name_to_idx = {n: i for i, n in enumerate(base_names)}
            for row_idx in range(X.shape[0]):
                row = explanation.data(row_idx)
                if row is None:
                    continue
                for n, s in zip(row["names"], row["scores"]):
                    n = str(n)
                    if " & " in n:
                        a, b = n.split(" & ", 1)
                        if a in name_to_idx and b in name_to_idx:
                            contribs[row_idx, name_to_idx[a]] += 0.5 * float(s)
                            contribs[row_idx, name_to_idx[b]] += 0.5 * float(s)
                    elif n in name_to_idx:
                        contribs[row_idx, name_to_idx[n]] += float(s)
            return contribs
        except Exception:
            return _finite_diff_contribs(self.predict, X)

    def coefficients(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.model_ is None:
            return out
        try:
            g = self.model_.explain_global().data()
            out["feature_names"] = list(g.get("names", []))
            out["feature_importances"] = list(g.get("scores", []))
        except Exception:
            pass
        out["hyperparameters"] = {
            "max_bins":     self.max_bins,
            "outer_bags":   self.outer_bags,
            "interactions": self.interactions,
        }
        return out


def _finite_diff_contribs(predict_fn, X: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Centred-difference fallback: ∂f/∂x_j × x_j (Taylor expansion around 0)."""
    D = X.shape[1]
    out = np.zeros_like(X, dtype=float)
    for j in range(D):
        Xp = X.copy(); Xp[:, j] += eps
        Xm = X.copy(); Xm[:, j] -= eps
        out[:, j] = (predict_fn(Xp) - predict_fn(Xm)) / (2 * eps) * X[:, j]
    return out
