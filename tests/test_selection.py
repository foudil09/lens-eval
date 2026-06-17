from __future__ import annotations

import numpy as np
import pytest

from lens_eval.selection import (
    capacity_gate,
    compute_metrics,
    cross_validate_combiner,
    filter_to_available,
    panel_for,
    primary_metric_for,
    select_with_1se_rule,
    CVResult,
)
from .conftest import needs_lightgbm


# ---------------------------------------------------------------------------
# Capacity gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (50,    ["glm"]),
    (199,   ["glm"]),
    (200,   ["glm", "glm_interactions"]),
    (999,   ["glm", "glm_interactions"]),
    (1000,  ["glm", "glm_interactions", "ebm"]),
    (4999,  ["glm", "glm_interactions", "ebm"]),
    (5000,  ["glm", "glm_interactions", "ebm", "gbm"]),
    (10_000, ["glm", "glm_interactions", "ebm", "gbm"]),
])
def test_capacity_gate_auto(n, expected):
    assert capacity_gate(n, "auto") == expected


def test_capacity_gate_fast_is_glm_only():
    assert capacity_gate(10_000, "fast") == ["glm"]


def test_capacity_gate_exhaustive_returns_all():
    assert capacity_gate(50, "exhaustive") == ["glm", "glm_interactions", "ebm", "gbm"]


def test_filter_to_available_drops_missing_backends():
    """Stub out a backend check, confirm it's dropped."""
    from lens_eval import combiners as cmb
    factory, check = cmb.AVAILABLE["gbm"]
    cmb.AVAILABLE["gbm"] = (factory, lambda: False)
    try:
        kept, dropped = filter_to_available(["glm", "ebm", "gbm"])
        assert "gbm" in dropped
        assert "gbm" not in kept
    finally:
        cmb.AVAILABLE["gbm"] = (factory, check)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_continuous():
    y = np.linspace(0, 1, 100)
    p = y + 0.01 * np.random.default_rng(0).normal(size=100)
    m = compute_metrics(y, p, task="regression", target_type="continuous")
    assert m["spearman"] > 0.95
    assert m["pearson"] > 0.95
    assert m["mae"] < 0.05
    assert np.isnan(m["brier"])


def test_compute_metrics_binary_has_brier_and_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = 0.6 * y + 0.2 * rng.uniform(0, 1, 200)
    m = compute_metrics(y, p, task="regression", target_type="binary")
    assert 0.0 <= m["brier"] <= 1.0
    assert not np.isnan(m["auc"])


def test_primary_metric_for():
    assert primary_metric_for("regression") == "spearman"
    assert primary_metric_for("pairwise")   == "auc"
    assert primary_metric_for("ranking")    == "kendall"


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------

def test_cv_glm_produces_per_fold_scores(synth_reg):
    X, y = synth_reg
    r = cross_validate_combiner(
        "glm", X, y,
        task="regression", target_type="continuous", link="identity",
        feature_names=["a", "b", "c", "d"],
        random_state=0, n_splits=5,
    )
    assert len(r.per_fold) == 5
    # per_fold is now a list of primary-metric values (floats), not dicts.
    assert all(isinstance(v, float) for v in r.per_fold)
    assert not np.isnan(r.primary_mean)
    assert r.primary_se >= 0.0
    # fold_preds are kept so the winner panel can be computed later.
    assert r.fold_preds is not None and len(r.fold_preds) == 5


def test_panel_for_returns_full_metric_panel(synth_reg):
    X, y = synth_reg
    r = cross_validate_combiner(
        "glm", X, y,
        task="regression", target_type="continuous", link="identity",
        feature_names=["a", "b", "c", "d"],
        random_state=0, n_splits=5,
    )
    panel = panel_for(r, task="regression", target_type="continuous")
    assert set(panel) == {"spearman", "kendall", "pearson", "mae", "auc", "brier"}
    assert not np.isnan(panel["spearman"])


def test_cv_handles_combiner_failure_gracefully(synth_reg):
    """If a combiner raises, the CV result holds the error string and NaN mean."""
    from lens_eval import combiners as cmb
    factory, check = cmb.AVAILABLE["glm"]

    def explode(**kw):
        raise RuntimeError("boom")

    cmb.AVAILABLE["glm"] = (explode, lambda: True)
    try:
        with pytest.warns():
            r = cross_validate_combiner(
                "glm", *synth_reg,
                task="regression", target_type="continuous", link="identity",
                feature_names=["a", "b", "c", "d"],
            )
        assert np.isnan(r.primary_mean)
        assert r.error and "boom" in r.error
    finally:
        cmb.AVAILABLE["glm"] = (factory, check)


# ---------------------------------------------------------------------------
# 1-SE rule
# ---------------------------------------------------------------------------

def _cvresult(name, mean, std, n_folds=5):
    return CVResult(
        combiner_type=name,
        primary_metric="spearman",
        primary_mean=mean,
        primary_std=std,
        per_fold=[mean] * n_folds,
    )


def test_1se_simpler_wins_within_threshold():
    """ebm beats glm_interactions by 0.005, SE is 0.02 → 1-SE picks glm_interactions."""
    results = [
        _cvresult("glm",              0.80, 0.05),
        _cvresult("glm_interactions", 0.84, 0.04),
        _cvresult("ebm",              0.845, 0.05),
    ]
    winner, reason = select_with_1se_rule(results)
    assert winner == "glm_interactions"
    assert "lower-capacity" in reason or "lower capacity" in reason or "within 1 SE" in reason


def test_1se_outright_winner_when_gap_large():
    """ebm beats everything by 0.3 → outright winner."""
    results = [
        _cvresult("glm",              0.50, 0.01),
        _cvresult("glm_interactions", 0.55, 0.01),
        _cvresult("ebm",              0.85, 0.01),
    ]
    winner, _ = select_with_1se_rule(results)
    assert winner == "ebm"


def test_1se_picks_glm_when_all_tied():
    results = [
        _cvresult("glm",              0.80, 0.05),
        _cvresult("glm_interactions", 0.80, 0.05),
        _cvresult("ebm",              0.80, 0.05),
        _cvresult("gbm",              0.80, 0.05),
    ]
    winner, _ = select_with_1se_rule(results)
    assert winner == "glm"


def test_1se_skips_failed_combiners():
    """Combiners with NaN mean are excluded; the rule still produces a winner."""
    results = [
        _cvresult("glm",              0.80, 0.05),
        _cvresult("glm_interactions", float("nan"), float("nan")),
        _cvresult("ebm",              0.81, 0.05),
    ]
    winner, _ = select_with_1se_rule(results)
    assert winner == "glm"  # within 1 SE of ebm, lower-capacity


def test_1se_all_failed_raises():
    results = [
        _cvresult("glm", float("nan"), float("nan")),
        _cvresult("ebm", float("nan"), float("nan")),
    ]
    with pytest.raises(RuntimeError, match="no combiner"):
        select_with_1se_rule(results)


# ---------------------------------------------------------------------------
# CV: ranking with groups
# ---------------------------------------------------------------------------

@needs_lightgbm
def test_cv_ranking_with_groups(synth_ranking):
    """CV on ranking task must forward ranks+groups to the combiner and
    produce valid per-fold Kendall tau scores."""
    X, ranks, groups = synth_ranking
    r = cross_validate_combiner(
        "gbm", X, ranks.astype(float),
        task="ranking", target_type="continuous", link="identity",
        feature_names=["a", "b", "c", "d"],
        groups=groups,
        n_splits=3,
        random_state=42,
    )
    assert r.error is None, f"CV failed: {r.error}"
    assert not np.isnan(r.primary_mean)
    assert r.primary_metric == "kendall"
    assert len(r.per_fold) == 3
    assert all(not np.isnan(v) for v in r.per_fold)
