"""BaseCombiner: the protocol every combiner tier implements.

Invariants:

1. ``predict(X)`` returns scores in target space (logit/latent/raw — the
   combiner decides). LENS applies any inverse link before returning to the
   user; ranking only needs ordering, so we don't enforce calibration here.
2. ``contributions(X)`` returns an (N, D) attribution matrix. Linear combiners
   return ``X * coef``; EBM/GBM return per-feature SHAP-style contributions.
3. ``coefficients()`` returns a dict of interpretable parameters used by the
   report layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BaseCombiner(ABC):
    """Common state every combiner carries."""

    task: str = "regression"            # 'regression' | 'pairwise' | 'ranking'
    target_type: str = "continuous"     # 'bounded' | 'ordinal' | 'binary' | 'continuous'
    link: str = "identity"              # 'identity' | 'logit' | 'cumulative_logit'
    random_state: int = 42
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: Optional[list] = None

    # Subclasses MUST set this to True at the end of fit().
    is_fitted_: bool = field(default=False, init=False)

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        pairs: Optional[np.ndarray] = None,
        ranks: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "BaseCombiner":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score in target space; higher = better quality."""

    @abstractmethod
    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-sample per-feature contribution matrix, shape (N, D)."""

    def coefficients(self) -> Dict[str, Any]:
        """Interpretable parameters. Override to expose more."""
        return {}

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call .fit() before prediction."
            )

    @property
    def type_name(self) -> str:
        return _TYPE_NAME.get(type(self).__name__, type(self).__name__.lower())


_TYPE_NAME = {
    "GLMCombiner":              "glm",
    "GLMInteractionsCombiner":  "glm_interactions",
    "EBMCombiner":              "ebm",
    "GBMCombiner":              "gbm",
}


def expand_pairs(X: np.ndarray, pairs: np.ndarray):
    """Bradley-Terry antisymmetric expansion.

    Each pair ``(a, b)`` (a beat b) becomes two training rows: ``(x_a - x_b, 1)``
    and ``(x_b - x_a, 0)``. Returns ``(X_diff, y_bin)`` of length ``2 * len(pairs)``.
    """
    pairs = np.asarray(pairs, dtype=int)
    diff = X[pairs[:, 0]] - X[pairs[:, 1]]
    X_use = np.vstack([diff, -diff])
    y_use = np.concatenate([
        np.ones(len(pairs), dtype=int),
        np.zeros(len(pairs), dtype=int),
    ])
    return X_use, y_use
