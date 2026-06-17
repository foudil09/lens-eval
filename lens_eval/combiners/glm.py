"""GLM combiner.

Defaults to sklearn's Ridge / LogisticRegression so the combiner ladder runs
without statsmodels. If statsmodels is importable, OLS-shaped fits (continuous
/ bounded) get re-fit through it for std-errs, CIs, and deviance — surfaced
via :meth:`coefficients`. Pairwise data trains on ``x_a - x_b`` with a logistic
link (Bradley-Terry).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseCombiner, expand_pairs


def _has_statsmodels() -> bool:
    try:
        import statsmodels.api  # noqa: F401
        return True
    except ImportError:
        return False


def _safe_logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    # Clip the 0/1 boundary so log(0) doesn't blow up.
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip ±30 to cover float64 dynamic range without overflow warnings.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class GLMCombiner(BaseCombiner):
    """Linear-additive combiner with link function matched to target type."""

    alpha: float = 0.1  # Ridge L2 strength; C = 1/alpha for logistic.
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)
    stats_: Dict[str, Any] = field(default_factory=dict, init=False)

    def _y_for_glm(self, y: np.ndarray) -> np.ndarray:
        """Transform y into the latent space OLS targets."""
        if self.task == "pairwise":
            return y  # already binary; logistic handles it.
        if self.target_type == "bounded":
            # Rescale to [0, 1] then logit-transform → OLS in latent space
            # behaves like a Beta GLM in raw space. predict() inverts.
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            if ymax > 1.5:
                y01 = (y - ymin) / (ymax - ymin + 1e-12)
                self.stats_["y01_range"] = (float(ymin), float(ymax))
            else:
                y01 = y
                self.stats_["y01_range"] = (0.0, 1.0)
            return _safe_logit(y01)
        return np.asarray(y, dtype=float)

    def _predict_inverse_link(self, z: np.ndarray) -> np.ndarray:
        if self.task == "pairwise" or self.target_type == "binary":
            return _sigmoid(z)
        if self.target_type == "bounded":
            p = _sigmoid(z)
            r = self.stats_.get("y01_range", (0.0, 1.0))
            return r[0] + p * (r[1] - r[0])
        return z

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        pairs: Optional[np.ndarray] = None,
        ranks: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "GLMCombiner":
        X = np.asarray(X, dtype=float)

        if self.task == "pairwise":
            # Raw pairs → antisymmetric expansion; pre-diffed X → use as-is.
            if pairs is not None:
                X_diff, y_bin = expand_pairs(X, pairs)
            else:
                X_diff = X
                y_bin = np.asarray(y, dtype=int).ravel()
            self._fit_logistic(X_diff, y_bin)
        elif self.target_type == "binary" or (self.task == "regression" and _is_binary(y)):
            # Auto-upgrade when the caller said "regression" but y is {0, 1}.
            self.target_type = "binary"
            self.link = "logit"
            self._fit_logistic(X, np.asarray(y, dtype=int).ravel())
        else:
            y_t = self._y_for_glm(np.asarray(y, dtype=float).ravel())
            self._fit_linear(X, y_t)

        self.is_fitted_ = True
        return self

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        # max(alpha, 1e-8) keeps Ridge from collapsing into raw OLS on the
        # often-collinear LENS dimensions.
        m = Ridge(alpha=max(self.alpha, 1e-8), random_state=self.random_state)
        m.fit(X, y)
        self.coef_ = m.coef_.astype(float)
        self.intercept_ = float(m.intercept_)
        self.stats_["sklearn_model"] = "Ridge"
        if _has_statsmodels() and self.target_type in {"bounded", "continuous"}:
            self._enrich_with_statsmodels(X, y)

    def _fit_logistic(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        # sklearn's C is inverse regularisation → C = 1/alpha keeps one knob
        # consistent across the linear/logistic paths.
        C = 1.0 / max(self.alpha, 1e-8)
        m = LogisticRegression(C=C, fit_intercept=True, max_iter=2000,
                               random_state=self.random_state)
        m.fit(X, y)
        self.coef_ = m.coef_.ravel().astype(float)
        self.intercept_ = float(m.intercept_[0])
        self.link = "logit"
        self.stats_["sklearn_model"] = "LogisticRegression"

    def _enrich_with_statsmodels(self, X: np.ndarray, y: np.ndarray) -> None:
        """Refit via statsmodels OLS for std-errs and CIs (best-effort)."""
        try:
            import statsmodels.api as sm
            Xc = sm.add_constant(X, has_constant="add")
            res = sm.OLS(y, Xc).fit()
            self.stats_["statsmodels_summary"] = {
                "params":   np.asarray(res.params).tolist(),
                "bse":      np.asarray(res.bse).tolist(),
                "tvalues":  np.asarray(res.tvalues).tolist(),
                "pvalues":  np.asarray(res.pvalues).tolist(),
                "rsquared": float(res.rsquared),
                "aic":      float(res.aic),
                "bic":      float(res.bic),
                "df_resid": int(res.df_resid),
            }
            # With α≈0 prefer statsmodels' un-regularised coefficients.
            if self.alpha < 1e-6:
                params = np.asarray(res.params)
                self.intercept_ = float(params[0])
                self.coef_ = params[1:].astype(float)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"statsmodels enrichment failed: {exc}", RuntimeWarning)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return self._predict_inverse_link(X @ self.coef_ + self.intercept_)

    def predict_latent(self, X: np.ndarray) -> np.ndarray:
        """Score in latent (link) space — preserves order for ranking."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

    def contributions(self, X: np.ndarray) -> np.ndarray:
        # Per-dim contribution = x_i * β_i. Sum + intercept reproduces latent score.
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return X * self.coef_

    def coefficients(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "intercept":     float(self.intercept_),
            "coef":          None if self.coef_ is None else self.coef_.tolist(),
            "feature_names": list(self.feature_names) if self.feature_names else None,
        }
        if self.stats_:
            out["stats"] = dict(self.stats_)
        return out


def _is_binary(y) -> bool:
    if y is None:
        return False
    u = np.unique(np.asarray(y))
    return len(u) <= 2 and set(u).issubset({0, 1, 0.0, 1.0})
