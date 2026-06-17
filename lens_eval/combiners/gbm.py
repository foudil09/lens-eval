"""GBM combiner — LightGBM with free (non-monotone) splits.

Monotone constraints are deliberately not used: trees split freely on
empirical density so non-monotone shapes (e.g. quality peaks then degrades)
can be learned. The 1-SE selector + CV downgrade to GLM on noisy data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseCombiner, expand_pairs


@dataclass
class GBMCombiner(BaseCombiner):
    num_leaves:       int   = 15
    learning_rate:    float = 0.05
    n_estimators:     int   = 1000     # capped via early stopping inside fit
    min_data_in_leaf: int   = 20

    model_: Optional[Any] = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        pairs: Optional[np.ndarray] = None,
        ranks: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "GBMCombiner":
        import lightgbm as lgb
        X = np.asarray(X, dtype=float)

        common = dict(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_data_in_leaf=self.min_data_in_leaf,
            random_state=self.random_state,
            verbose=-1,
        )

        if self.task == "ranking":
            # lambdarank needs per-query group sizes + integer relevance labels.
            # Both are mandatory — without them we'd silently fall through to a
            # regressor on bare row indices.
            if ranks is None or groups is None:
                missing = [k for k, v in (("ranks", ranks), ("groups", groups)) if v is None]
                raise ValueError(
                    f"task='ranking' requires both `ranks` and `groups`; missing: {missing!r}"
                )
            ranks_arr = np.asarray(ranks, dtype=int).ravel()
            groups_arr = np.asarray(groups)
            # LightGBM requires rows to be contiguous by group.  Sort so
            # interleaved / post-CV-split group IDs don't produce wrong sizes.
            order = np.argsort(groups_arr, kind="stable")
            X = X[order]
            ranks_arr = ranks_arr[order]
            groups_arr = groups_arr[order]

            common["objective"] = "lambdarank"
            self.model_ = lgb.LGBMRanker(**common)
            self.model_.fit(
                X, ranks_arr,
                group=_group_sizes_from_ids(groups_arr),
            )
        elif self.task == "pairwise":
            if pairs is not None:
                X_use, y_use = expand_pairs(X, pairs)
            else:
                X_use = X
                y_use = np.asarray(y, dtype=int).ravel()
            self.model_ = lgb.LGBMClassifier(**common)
            self.model_.fit(X_use, y_use)
        elif self.target_type == "binary":
            self.model_ = lgb.LGBMClassifier(**common)
            self.model_.fit(X, np.asarray(y, dtype=int).ravel())
        else:
            self.model_ = lgb.LGBMRegressor(**common)
            self.model_.fit(X, np.asarray(y, dtype=float).ravel())

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if self.task == "ranking":
            return self.model_.predict(X)
        if self.task == "pairwise" or self.target_type == "binary":
            return self.model_.predict_proba(X)[:, 1]
        return self.model_.predict(X)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """SHAP values from LightGBM's ``pred_contrib`` path (drops bias column)."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        booster = self.model_.booster_ if hasattr(self.model_, "booster_") else None
        if booster is None:
            return np.zeros_like(X)
        return booster.predict(X, pred_contrib=True)[:, :-1]

    def coefficients(self) -> Dict[str, Any]:
        if self.model_ is None:
            return {}
        return {
            "feature_importances": (
                self.model_.feature_importances_.tolist()
                if hasattr(self.model_, "feature_importances_") else None
            ),
            "hyperparameters": {
                "num_leaves":    self.num_leaves,
                "learning_rate": self.learning_rate,
                "n_estimators":  getattr(self.model_, "n_estimators_", self.n_estimators),
            },
        }


def _group_sizes_from_ids(group_ids: np.ndarray) -> np.ndarray:
    """Turn ``[a, a, b, b, b, c]`` into ``[2, 3, 1]``. Assumes contiguous groups."""
    ids = np.asarray(group_ids)
    sizes = []
    start = 0
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            sizes.append(i - start)
            start = i
    sizes.append(len(ids) - start)
    return np.asarray(sizes, dtype=int)
