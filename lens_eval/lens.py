"""LENS: the public-facing object.

End-to-end usage:

    lens = LENS()
    lens.fit(texts, references=refs, scores=y)   # or pass features=X
    s = lens.score(texts, references=refs)
    d = lens.compare(a_texts, b_texts, references=refs)
    order = lens.rank(candidates, references=refs)

``fit()`` runs the full pipeline: validate inputs, infer target/link, featurise
(via :mod:`lens_eval.encoders` unless ``features=`` is supplied), apply the
capacity gate, CV each candidate combiner, pick the winner with the 1-SE rule,
refit on all data, compute diagnostics, and populate
``selection_report_`` / ``diagnostics_`` / ``combiner_``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import combiners as cmb
from . import encoders as enc
from ._validate import validate_task_channels
from .diagnostics import compute_diagnostics
from .errors import (
    CombinerBackendMissing,
    DegenerateTargetError,
    InsufficientDataError,
    LensEvalError,
    ReferenceModeError,
)
from .report import format_html, format_text, summarise_diagnostics
from .selection import (
    capacity_gate,
    compute_metrics,
    cross_validate_combiner,
    filter_to_available,
    panel_for,
    primary_metric_for,
    select_with_1se_rule,
)

DIMENSIONS = enc.DIMENSIONS


# ---------------------------------------------------------------------------
# Post-fit feature analyses
# ---------------------------------------------------------------------------

def _feature_ablation(
    combiner,
    X: np.ndarray,
    y: np.ndarray,
    *,
    primary_metric: str,
    feature_names: Sequence[str],
    task: str,
    target_type: str,
) -> Dict[str, Any]:
    """Drop-column importance: substitute each feature's training mean, re-predict.

    Returns ``{"baseline": float, "by_feature": [{"name", "score", "delta"}, ...]}``
    where ``delta = baseline - masked_score`` (positive ⇒ feature was important
    for the primary metric).
    """
    means = X.mean(axis=0)
    baseline = compute_metrics(
        y, combiner.predict(X), task=task, target_type=target_type,
    )[primary_metric]
    rows = []
    for j, name in enumerate(feature_names):
        Xm = X.copy()
        Xm[:, j] = means[j]
        masked = compute_metrics(
            y, combiner.predict(Xm), task=task, target_type=target_type,
        )[primary_metric]
        rows.append({
            "name":  name,
            "score": float(masked),
            "delta": float(baseline - masked),
        })
    return {"baseline": float(baseline), "by_feature": rows}


def _marginal_impact_bins(
    combiner,
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_range: int,
) -> Dict[str, List[Dict[str, float]]]:
    """For each feature, mean per-dim contribution within ``n_range`` quantile bins.

    Returns ``{feature: [{"x_lo", "x_hi", "x_mean", "contribution_mean", "n"}, ...]}``.
    Duplicate quantile edges (heavy-tailed columns) collapse, yielding fewer
    than ``n_range`` bins for that feature.
    """
    contribs = combiner.contributions(X)
    out: Dict[str, List[Dict[str, float]]] = {}
    for j, name in enumerate(feature_names):
        x = X[:, j]
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_range + 1)))
        n_bins = max(len(edges) - 1, 0)
        if n_bins < 1:
            out[name] = []
            continue
        bin_idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, n_bins - 1)
        bins: List[Dict[str, float]] = []
        for k in range(n_bins):
            mask = bin_idx == k
            n = int(mask.sum())
            bins.append({
                "x_lo":              float(edges[k]),
                "x_hi":              float(edges[k + 1]),
                "x_mean":            float(x[mask].mean()) if n else float("nan"),
                "contribution_mean": float(contribs[mask, j].mean()) if n else float("nan"),
                "n":                 n,
            })
        out[name] = bins
    return out


# ---------------------------------------------------------------------------
# Target-type / link auto-detection
# ---------------------------------------------------------------------------

def _infer_target_type(y: np.ndarray) -> str:
    """Integer / [0, 1] / [0, 100] targets → bounded; {0,1} → binary;
    free-floating floats → continuous. Users can override with ``target_type=``."""
    y = np.asarray(y).ravel()
    u = np.unique(y[~np.isnan(y.astype(float, copy=False))])
    if set(u).issubset({0, 1, 0.0, 1.0}) and len(u) <= 2:
        return "binary"

    # essentially, if the values are all integers (not floats)
    is_integer = np.allclose(y, np.round(y), atol=1e-9)
    fmin, fmax = float(np.nanmin(y)), float(np.nanmax(y))
    if is_integer and fmax > fmin:  # fmax > fmin is to catch degenerate-constant-target cases
        return "bounded"
    if (fmin >= 0.0 and fmax <= 1.0 + 1e-9) or (fmin >= 0.0 and fmax <= 100.0 + 1e-9 and fmax > 1.5):
        return "bounded"
    return "continuous"


def _infer_link(target_type: str, task: str) -> str:
    if task == "pairwise" or target_type == "binary":
        return "logit"
    if target_type == "ordinal":
        return "cumulative_logit"
    if target_type == "bounded":
        return "logit"
    return "identity"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class LENS:
    random_state: int = 42

    # sklearn-style trailing-underscore fitted attrs.
    combiner_:         Any = field(default=None, init=False, repr=False)
    combiner_type_:    Optional[str] = field(default=None, init=False)
    task_:             Optional[str] = field(default=None, init=False)
    target_type_:      Optional[str] = field(default=None, init=False)
    link_function_:    Optional[str] = field(default=None, init=False)
    dimensions_used_:  Tuple[str, ...] = field(default=DIMENSIONS, init=False)
    cv_scores_:        Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    selection_report_: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    diagnostics_:      Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # Cached (min, max) for bounded targets so score() can rescale predictions
    # back to the user's original range. None when no rescale was applied.
    target_range_: Optional[Tuple[float, float]] = field(default=None, init=False)

    _fitted_:                bool = field(default=False, init=False, repr=False)
    _fitted_with_references: bool = field(default=True, init=False, repr=False)
    _n_train:                int  = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: Optional[Sequence[str]] = None,
        references: Optional[Sequence[str]] = None,
        features: Optional[np.ndarray] = None,
        scores: Optional[Sequence[float]] = None,
        pairs:  Optional[Sequence[Tuple[int, int]]] = None,
        ranks:  Optional[Sequence[int]]  = None,
        groups: Optional[Sequence[Any]]  = None,
        primary_metric: str = "auto",
        task: str = "auto",
        target_type: str = "auto",
        selection: str = "auto",
        dimensions: Optional[Sequence[str]] = None,
        hypothesized_interactions: Optional[List[Tuple[Any, Any]]] = None,
        cv_splits: int = 5,
        n_range: Optional[int] = None,
        verbose: bool = False,
    ) -> "LENS":
        """Featurise + fit a combiner.

        Pass ``features`` to skip encoding when scores are already on disk.
        Otherwise supply ``texts`` (and ``references`` for reference-mode dims).
        """
        if texts is None and features is None:
            raise ValueError("must pass either `texts` or `features`")

        _, inferred_task = validate_task_channels(
            scores=scores, pairs=pairs, ranks=ranks, groups=groups,
        )
        if task == "auto":
            task = inferred_task

        y, target_type = self._prep_y(scores, target_type, task)
        link = _infer_link(target_type, task)

        # Bounded target → cache (min, max) and rescale to [0, 1] for fit.
        self.target_range_ = None
        if task == "regression" and target_type == "bounded" and y is not None:
            y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
            self.target_range_ = (y_min, y_max)
            y = (y - y_min) / max(y_max - y_min, 1e-12)

        features, dimensions = self._coerce_feature_table(features, dimensions)
        dims_used = tuple(dimensions) if dimensions else DIMENSIONS
        X, dims_used = self._featurize_fit(texts, references, features, dims_used)

        n = X.shape[0]
        if n < 50:
            raise InsufficientDataError(
                f"need at least 50 rows to fit; got {n}."
            )
        if n < 200 and selection == "auto":
            warnings.warn(
                f"n={n} < 200 — restricting to GLM-only. "
                f"Pass selection='exhaustive' to override.",
                RuntimeWarning,
            )

        X, y_for_cv, groups_arr = self._task_prep(X, y, task, pairs, ranks, groups)
        candidates, dropped = self._candidate_set(selection, len(X))

        if primary_metric == "auto":
            primary = primary_metric_for(task)
        else:
            primary = primary_metric

        cv_results = []
        for ct in candidates:
            kw = {}
            if ct == "glm_interactions" and hypothesized_interactions is not None:
                kw["hypothesized_interactions"] = list(hypothesized_interactions)
            if verbose:
                print(f"  [CV] {ct} ...")
            cv_results.append(cross_validate_combiner(
                ct, X, y_for_cv,
                task=task, target_type=target_type, link=link,
                feature_names=list(dims_used),
                groups=groups_arr,
                n_splits=cv_splits,
                random_state=self.random_state,
                combiner_kwargs=kw,
            ))

        if not any(not np.isnan(r.primary_mean) for r in cv_results):
            raise LensEvalError(
                "all candidate combiners failed CV. See per-tier .error fields: "
                + "; ".join(f"{r.combiner_type}: {r.error}" for r in cv_results)
            )

        winner_type, reason = select_with_1se_rule(cv_results)
        winner_cv = next(r for r in cv_results if r.combiner_type == winner_type)
        winner_panel = panel_for(winner_cv, task=task, target_type=target_type)

        # Refit winner on all training data — CV was for selection only.
        winner_kwargs: Dict[str, Any] = {}
        if winner_type == "glm_interactions" and hypothesized_interactions is not None:
            winner_kwargs["hypothesized_interactions"] = list(hypothesized_interactions)
        combiner = cmb.make(
            winner_type,
            task=task, target_type=target_type, link=link,
            random_state=self.random_state, feature_names=list(dims_used),
            **winner_kwargs,
        )
        fit_kw: Dict[str, Any] = {}
        if task == "ranking" and groups_arr is not None:
            fit_kw["ranks"] = y_for_cv
            fit_kw["groups"] = groups_arr
        combiner.fit(X, y_for_cv, **fit_kw)

        ablation = _feature_ablation(
            combiner, X, y_for_cv,
            primary_metric=primary, feature_names=dims_used,
            task=task, target_type=target_type,
        )
        marginal = (_marginal_impact_bins(combiner, X, dims_used, n_range=n_range)
                    if n_range is not None else None)

        diag = self._safe_diagnostics(X, y_for_cv, combiner, dims_used, task, target_type)
        report = self._build_report(
            task, target_type, link, X, dims_used,
            candidates, dropped, primary, cv_splits, self.random_state,
            cv_results, winner_type, reason, winner_panel,
            ablation, marginal, combiner, diag,
        )

        self.combiner_         = combiner
        self.combiner_type_    = winner_type
        self.task_             = task
        self.target_type_      = target_type
        self.link_function_    = link
        self.dimensions_used_  = dims_used
        self.cv_scores_        = {
            r.combiner_type: {
                "mean":     r.primary_mean,
                "std":      r.primary_std,
                "per_fold": list(r.per_fold),
            }
            for r in cv_results
        }
        self.selection_report_ = report
        self.diagnostics_      = diag
        self._fitted_with_references = references is not None
        self._n_train          = int(len(X))
        self._fitted_          = True

        if verbose:
            self.report()
        return self

    # -- fit helpers --

    @staticmethod
    def _prep_y(scores, target_type, task):
        if scores is None:
            if target_type != "auto":
                return None, target_type
            # Pairwise targets are binary (0/1 from expand_pairs).
            # Ranking targets are integer relevance labels — not binary.
            return None, ("binary" if task == "pairwise" else "continuous")
        y = np.asarray(scores, dtype=float).ravel()
        if len(np.unique(y[~np.isnan(y)])) <= 1:
            raise DegenerateTargetError("target has zero variance; nothing to learn.")
        if target_type == "auto":
            target_type = _infer_target_type(y)
        return y, target_type

    @staticmethod
    def _coerce_feature_table(features, dimensions):
        """DataFrame ``features`` → ndarray, using the column headers as the
        dimension names.

        If an explicit ``dimensions=`` is also given it must match the headers
        exactly (same names, same order) — we assert rather than silently
        reorder, so a mismatch is surfaced instead of producing a model whose
        feature names disagree with its columns. Non-DataFrame ``features`` (or
        ``None``) pass through untouched.
        """
        if features is None or not hasattr(features, "columns"):
            return features, dimensions
        cols = [str(c) for c in features.columns]
        if dimensions is not None and list(dimensions) != cols:
            raise ValueError(
                f"`dimensions` {list(dimensions)} does not match the feature-table "
                f"columns {cols}. Omit `dimensions` to use the headers, or pass "
                f"them in column order."
            )
        return features.to_numpy(dtype=float), cols

    @staticmethod
    def _featurize_fit(texts, references, features, dims_used):
        if features is not None:
            X = np.asarray(features, dtype=float)
            if X.shape[1] != len(dims_used):
                raise ValueError(
                    f"`features` has {X.shape[1]} columns but {len(dims_used)} "
                    f"dimensions are requested ({dims_used})."
                )
        else:
            X = enc.featurize(list(texts), references=references, dimensions=dims_used)
        # Drop all-NaN columns — a ref-mode dim with no refs (encoding path) or an
        # empty column in a supplied feature table both carry no signal to fit.
        bad = np.all(np.isnan(X), axis=0)
        if bad.any():
            kept = ~bad
            dropped = [d for d, k in zip(dims_used, kept) if not k]
            if dropped:
                warnings.warn(
                    f"Dropping dimensions with no signal under this fit mode: {dropped}",
                    RuntimeWarning,
                )
            X = X[:, kept]
            dims_used = tuple(d for d, k in zip(dims_used, kept) if k)
        return X, dims_used

    @staticmethod
    def _task_prep(X, y, task, pairs, ranks, groups):
        groups_arr = None if groups is None else np.asarray(groups)
        if task == "pairwise":
            X, y_for_cv = cmb.expand_pairs(X, pairs)
        elif task == "ranking":
            y_for_cv = np.asarray(ranks, dtype=int).astype(float)
        else:
            y_for_cv = y
        return X, y_for_cv, groups_arr

    @staticmethod
    def _candidate_set(selection, n):
        if selection in cmb.AVAILABLE:
            candidates = [selection]
        else:
            candidates = capacity_gate(n, mode=selection)
        kept, dropped = filter_to_available(candidates)
        if not kept:
            raise CombinerBackendMissing(
                f"none of {candidates!r} are installable in this environment. "
                f"Install with: pip install 'lens-eval[all]'"
            )
        return kept, dropped

    @staticmethod
    def _safe_diagnostics(X, y, combiner, dims_used, task, target_type):
        try:
            y_pred = combiner.predict(X)
            return compute_diagnostics(
                X, y, y_pred,
                feature_names=list(dims_used),
                task=task, target_type=target_type,
            )
        except Exception as exc:
            warnings.warn(f"diagnostics failed: {exc}", RuntimeWarning)
            return {"warnings": [f"diagnostics failed: {exc}"]}

    @staticmethod
    def _build_report(task, target_type, link, X, dims_used, candidates, dropped,
                      primary, cv_splits, random_state,
                      cv_results, winner_type, reason, winner_panel,
                      ablation, marginal, combiner, diag):
        cv_score_rows = [{
            "combiner_type": r.combiner_type,
            "primary_mean":  r.primary_mean,
            "primary_std":   r.primary_std,
            "primary_se":    r.primary_se,
            "per_fold":      list(r.per_fold),
            "error":         r.error,
        } for r in cv_results]
        return {
            "task":              task,
            "target_type":       target_type,
            "link":               link,
            "n_samples":         int(len(X)),
            "dimensions_used":   list(dims_used),
            "candidates":        candidates,
            "dropped_backends":  dropped,
            "primary_metric":    primary,
            "cv_splits":         cv_splits,
            "random_state":      int(random_state),
            "cv_scores":         cv_score_rows,
            "winner":            winner_type,
            "reason":            reason,
            # Full panel for the selected model, computed once from its
            # aggregated out-of-sample predictions.
            "winner_panel":      winner_panel,
            # Drop-column ablation on training data; baseline + per-feature delta.
            "feature_ablation":  ablation,
            # Optional: per-feature mean contribution within n_range quantile bins.
            "marginal_impact":   marginal,
            "fitted_coefficients": combiner.coefficients(),
            "diagnostics_summary": summarise_diagnostics(diag),
            "diagnostics":       diag,
            "warnings":          diag.get("warnings", []),
        }

    # ------------------------------------------------------------------
    # score / compare / rank
    # ------------------------------------------------------------------

    def _coerce_feature_table_score(self, features):
        """DataFrame ``features`` → ndarray at score-time, selecting the fitted
        dimensions *by name* in ``dimensions_used_`` order.

        Selecting by header means the caller's column order doesn't matter and
        extra columns are ignored; a missing fitted dimension is an error. A
        plain ndarray (or ``None``) passes through and is matched positionally.
        """
        if features is None or not hasattr(features, "columns"):
            return features
        want = list(self.dimensions_used_)
        cols = [str(c) for c in features.columns]
        missing = [d for d in want if d not in cols]
        if missing:
            raise ValueError(
                f"feature table is missing fitted dimension(s) {missing}; "
                f"it has columns {cols}, needs {want}."
            )
        return features.loc[:, want].to_numpy(dtype=float)

    def _featurize(self, texts, references, features=None):
        self._require_fit()
        features = self._coerce_feature_table_score(features)
        if features is not None:
            X = np.asarray(features, dtype=float)
            if X.shape[1] != len(self.dimensions_used_):
                raise ValueError(
                    f"`features` has {X.shape[1]} columns but the combiner was "
                    f"trained on {len(self.dimensions_used_)} dimensions."
                )
            return X
        # Short-circuit BEFORE encoding — otherwise a user without encoder
        # weights would see a cryptic encoder error instead of the actionable one.
        if self._fitted_with_references and references is None:
            raise ReferenceModeError(
                "this LENS was fitted with references but `references=None` was "
                "passed at score-time. Provide references or refit reference-free."
            )
        return enc.featurize(list(texts), references=references,
                             dimensions=self.dimensions_used_)

    def score(
        self,
        texts: Optional[Sequence[str]] = None,
        references: Optional[Sequence[str]] = None,
        *,
        features: Optional[np.ndarray] = None,
        discretize: bool = False,
    ) -> np.ndarray:
        """Per-row quality score in the user's target range.

        Bounded targets are always rescaled from the combiner's [0, 1] latent
        space back to the cached ``target_range_``. Pass ``discretize=True`` to
        also round + clip to integers (useful for Likert / DA scales).
        """
        X = self._featurize(texts, references, features=features)
        pred = self._rescale(self.combiner_.predict(X))
        if not discretize:
            return pred
        if self.target_range_ is not None:
            lo, hi = self.target_range_
            return np.clip(np.rint(pred), lo, hi).astype(int)
        return np.rint(pred).astype(int)

    def _rescale(self, y_pred: np.ndarray) -> np.ndarray:
        if self.target_range_ is None:
            return np.asarray(y_pred, dtype=float)
        y_min, y_max = self.target_range_
        return np.asarray(y_pred, dtype=float) * (y_max - y_min) + y_min

    def compare(
        self,
        texts_a: Optional[Sequence[str]] = None,
        texts_b: Optional[Sequence[str]] = None,
        references: Optional[Sequence[str]] = None,
        *,
        features_a: Optional[np.ndarray] = None,
        features_b: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """score(a) - score(b). Positive ⇒ a is better than b."""
        return (self.score(texts_a, references, features=features_a)
                - self.score(texts_b, references, features=features_b))

    def rank(
        self,
        candidates: Optional[Sequence[str]] = None,
        references: Optional[Sequence[str]] = None,
        *,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """argsort by score (descending). Best-first."""
        return np.argsort(-self.score(candidates, references, features=features))

    def contributions(
        self,
        texts: Optional[Sequence[str]] = None,
        references: Optional[Sequence[str]] = None,
        *,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.combiner_.contributions(self._featurize(texts, references, features=features))

    def feature_importance(self) -> Tuple[List[str], List[float], str]:
        """Uniform per-base-dim importance: ``(names, values, kind)``.

        ``kind`` is ``"coef"`` (signed, GLM family) or ``"importance"``
        (non-negative, EBM / GBM). GLM+interactions returns leading-D main-effect
        coefficients; EBM strips interaction-term rows.
        """
        self._require_fit()
        names = list(self.dimensions_used_)
        c = self.combiner_.coefficients() or {}
        coef = c.get("coef")
        if coef is not None:
            coef = [float(v) for v in coef]
            if len(coef) >= len(names):
                return names, coef[: len(names)], "coef"
        imps = [float(v) for v in (c.get("feature_importances") or [])]
        feat_names = c.get("feature_names")
        if feat_names:
            lookup = dict(zip(feat_names, imps))
            return names, [float(lookup.get(n, 0.0)) for n in names], "importance"
        if len(imps) == len(names):
            return names, imps, "importance"
        return names, [0.0] * len(names), "importance"

    def report(self) -> None:
        self._require_fit()
        print(format_text(self.selection_report_))

    def report_html(self, path: str | Path) -> None:
        self._require_fit()
        Path(path).write_text(format_html(self.selection_report_, self.diagnostics_))

    def _require_fit(self) -> None:
        if not self._fitted_:
            raise RuntimeError("call .fit(...) before using this method")

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        from .persistence import save_lens
        save_lens(self, path)

    @classmethod
    def load(cls, path: str | Path, **hub_kwargs) -> "LENS":
        """Load from a local directory or a Hugging Face ``owner/repo`` id.

        An existing local path is read directly; otherwise a repo id is fetched
        from the Hub. ``hub_kwargs`` (e.g. ``revision``, ``token``, ``cache_dir``)
        are forwarded to ``snapshot_download``.
        """
        from .persistence import _resolve_model_dir, load_lens
        return load_lens(_resolve_model_dir(path, **hub_kwargs))

    def push_to_hub(self, repo_id: str, *, private: bool = False,
                    commit_message: str | None = None, **upload_kwargs) -> str:
        """Upload this fitted combiner to ``repo_id`` on the Hugging Face Hub.

        Returns the commit URL. ``upload_kwargs`` (e.g. ``token``, ``revision``)
        are forwarded to ``upload_folder``.
        """
        self._require_fit()
        from .persistence import push_lens_to_hub
        return push_lens_to_hub(self, repo_id, private=private,
                                commit_message=commit_message, **upload_kwargs)
