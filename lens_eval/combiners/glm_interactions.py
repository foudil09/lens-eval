"""GLM + interactions.

Same engine as :class:`GLMCombiner` but the design matrix is augmented with
pairwise interaction terms. ``hypothesized_interactions`` (if set) selects
exact pairs; otherwise all D*(D-1)/2 pairs are added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .glm import GLMCombiner


@dataclass
class GLMInteractionsCombiner(GLMCombiner):
    """GLM with explicit pairwise interaction terms.

    ``hypothesized_interactions`` accepts (i, j) integer pairs or (name_i, name_j)
    string pairs (resolved against ``feature_names``). Unknown / self pairs are
    silently dropped so user typos don't break the fit.
    """

    hypothesized_interactions: Optional[List[Tuple[Any, Any]]] = None
    _interaction_idx: List[Tuple[int, int]] = field(default_factory=list, init=False, repr=False)

    def _resolve_interactions(self, D: int) -> List[Tuple[int, int]]:
        if self.hypothesized_interactions is None:
            return list(combinations(range(D), 2))

        seen, out = set(), []
        for a, b in self.hypothesized_interactions:
            ia = self._name_to_idx(a, D)
            ib = self._name_to_idx(b, D)
            if ia is None or ib is None or ia == ib:
                continue
            pair = tuple(sorted((ia, ib)))   # canonical order; collapse (a,b) and (b,a)
            if pair in seen:
                continue
            seen.add(pair)
            out.append(pair)
        return out

    def _name_to_idx(self, name, D: int) -> Optional[int]:
        if isinstance(name, (int, np.integer)):
            return int(name) if 0 <= int(name) < D else None
        if self.feature_names is None:
            return None
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        # Layout: [base D cols | interaction cols in _interaction_idx order].
        if not self._interaction_idx:
            return X
        cols = [X[:, i] * X[:, j] for i, j in self._interaction_idx]
        return np.column_stack([X] + cols)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        pairs: Optional[np.ndarray] = None,
        ranks: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "GLMInteractionsCombiner":
        X = np.asarray(X, dtype=float)
        self._interaction_idx = self._resolve_interactions(X.shape[1])
        return super().fit(self._augment(X), y, pairs=pairs, ranks=ranks, groups=groups)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(self._augment(np.asarray(X, dtype=float)))

    def predict_latent(self, X: np.ndarray) -> np.ndarray:
        return super().predict_latent(self._augment(np.asarray(X, dtype=float)))

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-base-dim contribution; interaction contributions split 50/50.

        Output shape stays (N, D) so plotting code doesn't have to deal with
        O(D²) extra columns. Row sum equals ``X_aug · coef`` (the latent score).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        D = X.shape[1]
        full = self._augment(X) * self.coef_     # (N, D + n_inter)
        base = full[:, :D].copy()
        for k, (i, j) in enumerate(self._interaction_idx):
            half = 0.5 * full[:, D + k]
            base[:, i] += half
            base[:, j] += half
        return base

    def coefficients(self) -> Dict[str, Any]:
        out = super().coefficients()
        out["interactions"] = list(self._interaction_idx)
        if self.feature_names:
            out["interaction_names"] = [
                f"{self.feature_names[i]} × {self.feature_names[j]}"
                for i, j in self._interaction_idx
            ]
        return out
