from __future__ import annotations

import numpy as np
import pytest

from lens_eval import LENS
from lens_eval import encoders as enc
from lens_eval.errors import (
    AmbiguousTaskError,
    DegenerateTargetError,
    InsufficientDataError,
    ReferenceModeError,
)
from .conftest import needs_interpret, needs_lightgbm


@pytest.fixture(autouse=True)
def _reset_encoders_state():
    """Tests configure module-level encoder state — restore between runs."""
    saved = {k: v for k, v in enc._CONFIG.items()}
    autoload = enc._CENTROID_AUTOLOADED
    yield
    enc._CONFIG.update(saved)
    enc._CACHE.clear()
    enc._CENTROID_AUTOLOADED = autoload


def _bare(mode="reference", random_state=0):
    enc.configure(naturalness_mode=mode)
    return LENS(random_state=random_state)


# ---------------------------------------------------------------------------
# Fit (regression)
# ---------------------------------------------------------------------------

def test_fit_regression_populates_selection_report(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y, verbose=False)
    rpt = lens.selection_report_
    assert rpt["task"] == "regression"
    assert rpt["target_type"] == "bounded"
    assert rpt["link"] == "logit"
    assert rpt["winner"] in {"glm", "glm_interactions"}
    assert rpt["n_samples"] == len(X)
    assert rpt["candidates"] == ["glm", "glm_interactions"]
    assert "reason" in rpt and rpt["reason"]


def test_fit_regression_score_correlates_with_target(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X), scores=y)
    pred = lens.score(references=[""] * len(X), features=X)
    assert np.corrcoef(pred, y)[0, 1] > 0.85


def test_fit_score_compare_rank_basic(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X), scores=y)
    s = lens.score(references=[""] * 10, features=X[:10])
    d = lens.compare(references=[""] * 10,
                     features_a=X[:10], features_b=X[10:20])
    r = lens.rank(references=[""] * 10, features=X[:10])
    assert s.shape == (10,)
    assert d.shape == (10,)
    assert r.shape == (10,)
    assert set(r) == set(range(10))


# ---------------------------------------------------------------------------
# Continuity-enriched regression (Phase 1)
# ---------------------------------------------------------------------------

def test_likert_1to5_caches_range_and_predicts_in_range(rng):
    X = rng.uniform(0, 1, size=(400, 4)).astype(np.float64)
    raw = X @ np.array([0.45, 0.30, 0.15, 0.10])
    qs = np.quantile(raw, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(raw, qs) + 1  # 1-5

    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y.astype(float))
    assert lens.target_range_ == (1.0, 5.0)
    assert lens.target_type_ == "bounded"

    pred = lens.score(references=[""] * len(X), features=X)
    assert pred.min() >= 0.5
    assert pred.max() <= 5.5
    assert np.corrcoef(pred, y)[0, 1] > 0.80


def test_score_discretize_returns_integers_within_range(rng):
    """discretize=True rounds to int and clips to the cached target range."""
    X = rng.uniform(0, 1, size=(400, 4)).astype(np.float64)
    raw = X @ np.array([0.45, 0.30, 0.15, 0.10])
    qs = np.quantile(raw, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(raw, qs) + 1

    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y.astype(float))
    cont = lens.score(references=[""] * 30, features=X[:30])
    label = lens.score(references=[""] * 30, features=X[:30], discretize=True)

    assert label.shape == cont.shape
    assert label.dtype.kind in {"i", "u"}
    assert int(label.min()) >= 1 and int(label.max()) <= 5
    np.testing.assert_array_equal(label, np.clip(np.rint(cont), 1, 5).astype(int))


def test_score_discretize_works_on_arbitrary_integer_range(rng):
    """0-100 DA scale: discretize=True must clip to the observed range."""
    X = rng.uniform(0, 1, size=(400, 4)).astype(np.float64)
    raw = X @ np.array([0.4, 0.3, 0.2, 0.1])
    y = np.round(raw * 100).astype(int)
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y.astype(float))
    assert lens.target_range_[0] >= 0 and lens.target_range_[1] <= 100
    label = lens.score(references=[""] * 10, features=X[:10], discretize=True)
    assert (label >= 0).all() and (label <= 100).all()


def test_target_range_round_trips_through_save_load(tmp_path, rng):
    from lens_eval.persistence import load_lens, save_lens
    X = rng.uniform(0, 1, size=(400, 4)).astype(np.float64)
    raw = X @ np.array([0.45, 0.30, 0.15, 0.10])
    qs = np.quantile(raw, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(raw, qs) + 1

    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y.astype(float))
    save_lens(lens, tmp_path / "lens")
    lens2 = LENS.load(tmp_path / "lens")
    assert lens2.target_range_ == lens.target_range_
    np.testing.assert_allclose(
        lens2.score(references=[""] * 20, features=X[:20]),
        lens.score(references=[""] * 20, features=X[:20]),
        atol=1e-9,
    )


def test_contributions_shape_matches_features(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X), scores=y)
    c = lens.contributions(references=[""] * 10, features=X[:10])
    assert c.shape == (10, X.shape[1])


def test_fit_with_explicit_dimensions_subset(synth_reg):
    X, y = synth_reg
    X3 = X[:, :3]
    lens = _bare().fit(features=X3, references=[""] * len(X3),
                       scores=y, dimensions=["semantic", "nli", "naturalness"])
    assert lens.dimensions_used_ == ("semantic", "nli", "naturalness")
    s = lens.score(references=[""] * 10, features=X3[:10])
    assert s.shape == (10,)


def test_fit_features_dim_mismatch_raises(synth_reg):
    X, y = synth_reg
    bad = X[:, :3]
    with pytest.raises(ValueError, match="columns"):
        _bare().fit(features=bad, references=[""] * len(bad), scores=y)


# ---------------------------------------------------------------------------
# Auto-selection behaviour
# ---------------------------------------------------------------------------

def test_selection_force_glm_skips_other_tiers(synth_reg_large):
    X, y = synth_reg_large
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y, selection="glm")
    assert lens.combiner_type_ == "glm"
    assert lens.selection_report_["candidates"] == ["glm"]


def test_selection_fast_mode(synth_reg_large):
    X, y = synth_reg_large
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y, selection="fast")
    assert lens.selection_report_["candidates"] == ["glm"]


def test_warn_when_n_below_200(rng):
    X = rng.uniform(0, 1, size=(150, 4))
    y = (X @ np.array([0.4, 0.3, 0.15, 0.15])).clip(0, 1)
    with pytest.warns(RuntimeWarning, match="n=.*200"):
        _bare().fit(features=X, references=[""] * len(X), scores=y)


# ---------------------------------------------------------------------------
# Auto-selection end-to-end coverage: the n>=1000 (EBM) and n>=5000 (GBM)
# branches of the capacity gate were exercised only by capacity_gate() unit
# tests, never via a real fit. These tests close that gap so a subtle bug in
# the CV loop, the 1-SE rule, or a combiner's fit path on a freshly-fit auto
# winner would surface here instead of silently shipping.
# ---------------------------------------------------------------------------

@needs_interpret
def test_auto_selection_offers_ebm_at_n_geq_1000(synth_reg_large):
    """n=1500 hits the n>=1000 branch — auto must include 'ebm', all 3 candidates
    must produce a valid CV score, and the winner must be one of them."""
    X, y = synth_reg_large
    lens = _bare().fit(features=X, references=[""] * len(X), scores=y)
    candidates = lens.selection_report_["candidates"]
    assert candidates == ["glm", "glm_interactions", "ebm"]
    cv_scores = lens.selection_report_["cv_scores"]
    assert {r["combiner_type"] for r in cv_scores} == set(candidates)
    assert all(not np.isnan(r["primary_mean"]) for r in cv_scores)
    assert lens.combiner_type_ in candidates


@needs_interpret
@needs_lightgbm
def test_auto_selection_offers_all_four_tiers_at_n_geq_5000(rng):
    """n=5000 hits the top branch — auto must include all four tiers. We use
    cv_splits=2 to keep the GBM CV runtime modest; the test is about plumbing,
    not statistical rigor."""
    n = 5000
    X = rng.uniform(0, 1, size=(n, 4)).astype(np.float64)
    w = np.array([0.45, 0.30, 0.15, 0.10])
    y_lat = X @ w + 0.05 * X[:, 0] * X[:, 1] + rng.normal(0, 0.02, n)
    y = (y_lat - y_lat.min()) / (y_lat.max() - y_lat.min() + 1e-12)

    lens = _bare().fit(
        features=X, references=[""] * len(X), scores=y, cv_splits=2,
    )
    candidates = lens.selection_report_["candidates"]
    assert candidates == ["glm", "glm_interactions", "ebm", "gbm"]
    cv_scores = lens.selection_report_["cv_scores"]
    assert {r["combiner_type"] for r in cv_scores} == set(candidates)
    assert all(not np.isnan(r["primary_mean"]) for r in cv_scores)
    assert lens.combiner_type_ in candidates


@needs_interpret
@needs_lightgbm
def test_exhaustive_selection_runs_all_tiers_at_small_n(synth_reg):
    """selection='exhaustive' bypasses the capacity gate. Use synth_reg (n=400)
    — normally only glm+glm_interactions — and force all four tiers."""
    X, y = synth_reg
    lens = _bare().fit(
        features=X, references=[""] * len(X), scores=y,
        selection="exhaustive", cv_splits=2,
    )
    candidates = lens.selection_report_["candidates"]
    assert candidates == ["glm", "glm_interactions", "ebm", "gbm"]
    cv_scores = lens.selection_report_["cv_scores"]
    assert all(not np.isnan(r["primary_mean"]) for r in cv_scores)


def test_fit_with_hypothesized_interactions(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(
        features=X, references=[""] * len(X), scores=y,
        selection="glm_interactions",
        hypothesized_interactions=[("semantic", "nli")],
    )
    interactions = lens.combiner_.coefficients()["interactions"]
    assert interactions == [(0, 1)]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_fit_zero_variance_target_raises(rng):
    X = rng.uniform(0, 1, size=(100, 4))
    y = np.zeros(100)
    with pytest.raises(DegenerateTargetError):
        _bare().fit(features=X, references=[""] * len(X), scores=y)


def test_fit_ambiguous_task_raises(rng):
    X = rng.uniform(0, 1, size=(100, 4))
    y = np.linspace(0, 1, 100)
    pairs = np.array([[0, 1], [1, 2]])
    with pytest.raises(AmbiguousTaskError):
        _bare().fit(features=X, references=[""] * len(X),
                    scores=y, pairs=pairs)


def test_fit_no_task_signal_raises(rng):
    X = rng.uniform(0, 1, size=(100, 4))
    with pytest.raises(AmbiguousTaskError):
        _bare().fit(features=X, references=[""] * len(X))


def test_fit_too_few_samples_raises(rng):
    X = rng.uniform(0, 1, size=(40, 4))
    y = np.linspace(0, 1, 40)
    with pytest.raises(InsufficientDataError):
        _bare().fit(features=X, references=[""] * len(X), scores=y)


def test_score_before_fit_raises(synth_reg):
    X, _ = synth_reg
    lens = LENS(random_state=0)
    with pytest.raises(RuntimeError, match="fit"):
        lens.score(references=[""] * 5, features=X[:5])


def test_score_without_references_after_with_ref_fit_raises(synth_reg, monkeypatch):
    """ReferenceModeError must fire BEFORE any encoder work — otherwise users
    without local model weights would see a cryptic encoder error. Verify by
    monkeypatching the module-level featurize into a tripwire."""
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X), scores=y)

    def _tripwire(*args, **kwargs):
        raise AssertionError(
            "encoders.featurize was reached — the ReferenceModeError guard should "
            "have short-circuited before any encoding work."
        )
    monkeypatch.setattr(enc, "featurize", _tripwire)

    with pytest.raises(ReferenceModeError, match="references"):
        lens.score(["a", "b"], references=None)


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------

def test_pairwise_fit_and_in_sample_accuracy(synth_pairs):
    X, idx = synth_pairs
    lens = _bare().fit(features=X, references=[""] * len(X),
                       pairs=idx, task="pairwise")
    assert lens.task_ == "pairwise"
    s_a = lens.score(references=[""] * len(idx), features=X[idx[:, 0]])
    s_b = lens.score(references=[""] * len(idx), features=X[idx[:, 1]])
    assert float(np.mean(s_a > s_b)) > 0.85


def test_pairwise_compare_sign_is_correct(synth_pairs):
    X, idx = synth_pairs
    lens = _bare().fit(features=X, references=[""] * len(X),
                       pairs=idx, task="pairwise")
    diffs = lens.compare(
        references=[""] * len(idx),
        features_a=X[idx[:, 0]], features_b=X[idx[:, 1]],
    )
    assert float(np.mean(diffs > 0)) > 0.85


# ---------------------------------------------------------------------------
# feature_importance: uniform across tiers
# ---------------------------------------------------------------------------

def test_feature_importance_glm_returns_coef_per_dim(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y, selection="glm")
    names, vals, kind = lens.feature_importance()
    assert kind == "coef"
    assert names == list(lens.dimensions_used_)
    assert len(vals) == X.shape[1]


def test_feature_importance_glm_interactions_returns_only_main_effects(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y, selection="glm_interactions")
    names, vals, kind = lens.feature_importance()
    assert kind == "coef"
    assert len(vals) == X.shape[1]


def test_feature_importance_uses_dimensions_used_(synth_reg):
    X, y = synth_reg
    X3 = X[:, :3]
    lens = _bare().fit(features=X3, references=[""] * len(X3),
                       scores=y, dimensions=["semantic", "nli", "naturalness"])
    names, vals, _ = lens.feature_importance()
    assert names == ["semantic", "nli", "naturalness"]
    assert len(vals) == 3


def test_feature_importance_ebm_uses_dim_names_not_internal(synth_reg_large):
    """Regression: EBM previously kept interpret-ml's auto-names
    (feature_0000, …). LENS.feature_importance() did name-based lookup against
    dimensions_used_, so every key missed and returned all-zero importances."""
    try:
        from lens_eval.combiners import make as _make
        _make("ebm", task="regression", target_type="continuous")
    except Exception:
        pytest.skip("interpret-ml not available")

    X, y = synth_reg_large
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y, selection="ebm")
    names, vals, kind = lens.feature_importance()
    assert kind == "importance"
    assert names == list(lens.dimensions_used_)
    assert any(v != 0.0 for v in vals), "EBM importances must not all be zero"
    # And the dim with the largest true weight should rank first.
    assert int(np.argmax(vals)) == 0


def test_ebm_coefficients_expose_real_dim_names_including_interactions(synth_reg_large):
    """Regression: with the fix, interpret-ml's explain_global() reports real
    names for both main effects and `a & b` interaction rows."""
    try:
        from lens_eval.combiners import make as _make
        _make("ebm", task="regression", target_type="continuous")
    except Exception:
        pytest.skip("interpret-ml not available")

    X, y = synth_reg_large
    lens = _bare().fit(features=X, references=[""] * len(X),
                       scores=y, selection="ebm")
    coef = lens.combiner_.coefficients()
    fnames = coef.get("feature_names") or []
    # No "feature_0000" leakage anywhere.
    assert not any(n.startswith("feature_0") for n in fnames)
    # Main effects use real dim names.
    for d in lens.dimensions_used_:
        assert d in fnames


def test_feature_ablation_runs_unconditionally_and_ranks_dominant_dim(synth_reg):
    """Ablation is always populated; dim with largest true weight should have the
    largest delta (most impact on the primary metric when masked)."""
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X), scores=y)
    abl = lens.selection_report_["feature_ablation"]
    assert "baseline" in abl
    assert len(abl["by_feature"]) == X.shape[1]
    # Fixture weights are w=[0.45, 0.30, 0.15, 0.10] → dim 0 dominates.
    deltas = [row["delta"] for row in abl["by_feature"]]
    assert int(np.argmax(deltas)) == 0


def test_marginal_impact_only_present_when_n_range_set(synth_reg):
    """marginal_impact is None unless n_range is passed."""
    X, y = synth_reg
    lens_a = _bare().fit(features=X, references=[""] * len(X), scores=y)
    assert lens_a.selection_report_["marginal_impact"] is None

    lens_b = _bare().fit(features=X, references=[""] * len(X), scores=y, n_range=10)
    marg = lens_b.selection_report_["marginal_impact"]
    assert marg is not None
    assert set(marg) == set(lens_b.dimensions_used_)
    for name, bins in marg.items():
        # Up to 10 bins per feature (fewer when quantile edges collapse).
        assert 1 <= len(bins) <= 10
        for b in bins:
            assert {"x_lo", "x_hi", "x_mean", "contribution_mean", "n"}.issubset(b)


def test_marginal_impact_bins_sum_of_n_equals_dataset_size(synth_reg):
    X, y = synth_reg
    lens = _bare().fit(features=X, references=[""] * len(X), scores=y, n_range=5)
    for name, bins in lens.selection_report_["marginal_impact"].items():
        assert sum(b["n"] for b in bins) == len(X)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

@needs_lightgbm
def test_ranking_fit_end_to_end(synth_ranking):
    """End-to-end ranking: fit with ranks+groups via GBM, score, rank."""
    X, ranks, groups = synth_ranking
    lens = _bare().fit(
        features=X, references=[""] * len(X),
        ranks=ranks, groups=groups,
        selection="gbm", cv_splits=3,
    )
    assert lens.task_ == "ranking"
    assert lens.combiner_type_ == "gbm"
    assert lens._fitted_

    s = lens.score(references=[""] * 20, features=X[:20])
    assert s.shape == (20,)

    r = lens.rank(references=[""] * 20, features=X[:20])
    assert r.shape == (20,)
    assert set(r) == set(range(20))


def test_score_is_deterministic_across_runs(synth_reg):
    X, y = synth_reg
    lens1 = _bare(random_state=7).fit(features=X, references=[""] * len(X), scores=y)
    lens2 = _bare(random_state=7).fit(features=X, references=[""] * len(X), scores=y)
    s1 = lens1.score(references=[""] * 50, features=X[:50])
    s2 = lens2.score(references=[""] * 50, features=X[:50])
    np.testing.assert_allclose(s1, s2, atol=0)


# ---------------------------------------------------------------------------
# DataFrame feature tables (headers = dimension names)
# ---------------------------------------------------------------------------

def _df(X, names):
    import pandas as pd
    return pd.DataFrame(X, columns=list(names))


def test_fit_with_feature_table_uses_headers_as_dims(synth_reg):
    X, y = synth_reg
    names = ["bleu", "comet", "chrf", "len_ratio"]
    lens = _bare().fit(features=_df(X, names), references=[""] * len(X), scores=y)
    assert list(lens.dimensions_used_) == names
    assert lens.selection_report_["dimensions_used"] == names


def test_feature_table_matches_ndarray_fit(synth_reg):
    X, y = synth_reg
    names = ["a", "b", "c", "d"]
    arr_pred = _bare().fit(features=X, references=[""] * len(X), scores=y) \
        .score(features=X, references=[""] * len(X))
    df_pred = _bare().fit(features=_df(X, names), references=[""] * len(X), scores=y) \
        .score(features=_df(X, names), references=[""] * len(X))
    np.testing.assert_allclose(df_pred, arr_pred, rtol=1e-9, atol=1e-9)


def test_score_table_column_order_is_irrelevant(synth_reg):
    X, y = synth_reg
    names = ["a", "b", "c", "d"]
    lens = _bare().fit(features=_df(X, names), references=[""] * len(X), scores=y)
    ordered = lens.score(features=_df(X[:50], names), references=[""] * 50)
    flipped_names = names[::-1]
    flipped = lens.score(
        features=_df(X[:50, ::-1], flipped_names), references=[""] * 50,
    )
    np.testing.assert_allclose(flipped, ordered, rtol=1e-12, atol=1e-12)


def test_score_table_ignores_extra_columns(synth_reg):
    X, y = synth_reg
    names = ["a", "b", "c", "d"]
    lens = _bare().fit(features=_df(X, names), references=[""] * len(X), scores=y)
    base = lens.score(features=_df(X[:30], names), references=[""] * 30)
    aug = _df(X[:30], names)
    aug["junk"] = 1.23
    np.testing.assert_allclose(
        lens.score(features=aug, references=[""] * 30), base, rtol=1e-12, atol=1e-12,
    )


def test_score_table_missing_dim_raises(synth_reg):
    X, y = synth_reg
    names = ["a", "b", "c", "d"]
    lens = _bare().fit(features=_df(X, names), references=[""] * len(X), scores=y)
    bad = _df(X[:30], names).drop(columns=["b"])
    with pytest.raises(ValueError, match="missing fitted dimension"):
        lens.score(features=bad, references=[""] * 30)


def test_explicit_dimensions_must_match_headers(synth_reg):
    X, y = synth_reg
    names = ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="does not match the feature-table columns"):
        _bare().fit(
            features=_df(X, names), references=[""] * len(X), scores=y,
            dimensions=["a", "b", "c", "WRONG"],
        )


def test_feature_table_all_nan_column_dropped(synth_reg):
    X, y = synth_reg
    names = ["a", "b", "c", "d"]
    Xn = X.copy()
    Xn[:, 2] = np.nan
    with pytest.warns(RuntimeWarning, match="no signal"):
        lens = _bare().fit(features=_df(Xn, names), references=[""] * len(X), scores=y)
    assert list(lens.dimensions_used_) == ["a", "b", "d"]
