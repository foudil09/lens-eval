from __future__ import annotations

import numpy as np
import pytest

from lens_eval.diagnostics import compute_diagnostics
from lens_eval.report import format_html, format_text, summarise_diagnostics
from .conftest import needs_jinja2


# ---------------------------------------------------------------------------
# Diagnostics dict
# ---------------------------------------------------------------------------

def test_diagnostics_dict_has_expected_keys(synth_reg):
    X, y = synth_reg
    pred = y + 0.01 * np.random.default_rng(0).normal(size=len(y))
    d = compute_diagnostics(
        X, y, pred,
        feature_names=["a", "b", "c", "d"],
        task="regression",
        target_type="continuous",
    )
    assert {"residuals", "calibration", "feature_correlations",
            "monotonicity", "outliers", "warnings"}.issubset(d)


def test_residuals_zero_mean_when_predict_equals_target(synth_reg):
    X, y = synth_reg
    d = compute_diagnostics(X, y, y, feature_names=["a", "b", "c", "d"],
                            task="regression", target_type="continuous")
    assert abs(d["residuals"]["mean"]) < 1e-9
    assert d["residuals"]["abs_max"] < 1e-9


def test_calibration_returned_for_bounded(synth_reg):
    X, y = synth_reg
    pred = np.clip(y + 0.02 * np.random.default_rng(0).normal(size=len(y)), 0, 1)
    d = compute_diagnostics(X, y, pred, feature_names=["a", "b", "c", "d"],
                            task="regression", target_type="bounded")
    assert d["calibration"] is not None
    assert "ece" in d["calibration"]
    assert "brier" in d["calibration"]


def test_calibration_skipped_for_continuous(synth_reg):
    X, y = synth_reg
    d = compute_diagnostics(X, y, y, feature_names=["a", "b", "c", "d"],
                            task="regression", target_type="continuous")
    assert d["calibration"] is None


def test_monotonicity_detects_increasing_relationship():
    """y = x[:,0] strictly → predictor monotone-increasing in dim 0."""
    X = np.linspace(0, 1, 200).reshape(-1, 1)
    X = np.column_stack([X] * 4)
    y = X[:, 0].copy()
    d = compute_diagnostics(X, y, y, feature_names=list("abcd"),
                            task="regression", target_type="bounded")
    # All four features are identical so all should be increasing.
    # Phase 2: monotonicity is now a dict per dim with at least a "status".
    for entry in d["monotonicity"].values():
        assert entry["status"] == "increasing"


def test_monotonicity_detects_non_monotone_shape():
    """Concave-down y vs x → bin means rise then fall → flagged non-monotone."""
    n = 400
    x0 = np.linspace(0, 1, n)
    y = -(x0 - 0.5) ** 2 + 0.25
    X = np.column_stack([x0, np.zeros(n), np.zeros(n), np.zeros(n)])
    d = compute_diagnostics(X, y, y, feature_names=list("abcd"),
                            task="regression", target_type="continuous")
    assert d["monotonicity"]["a"]["status"] == "non-monotone"


def test_feature_correlations_symmetric(synth_reg):
    X, y = synth_reg
    d = compute_diagnostics(X, y, y, feature_names=list("abcd"),
                            task="regression", target_type="continuous")
    M = np.asarray(d["feature_correlations"]["matrix"])
    assert M.shape == (4, 4)
    np.testing.assert_allclose(M, M.T, atol=1e-9)
    np.testing.assert_allclose(np.diag(M), [1, 1, 1, 1], atol=1e-9)


def test_outliers_flagged_when_huge_residual(synth_reg):
    X, y = synth_reg
    pred = y.copy()
    pred[:5] += 10.0  # massive synthetic outliers
    d = compute_diagnostics(X, y, pred, feature_names=list("abcd"),
                            task="regression", target_type="continuous")
    assert d["outliers"]["count"] >= 5


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

@pytest.fixture
def report_dict():
    return {
        "task": "regression", "target_type": "bounded", "link": "logit",
        "n_samples": 1500, "dimensions_used": ["semantic", "nli", "naturalness", "emotion"],
        "candidates": ["glm", "glm_interactions"], "dropped_backends": ["ebm"],
        "primary_metric": "spearman", "cv_splits": 5, "random_state": 0,
        "cv_scores": [
            {"combiner_type": "glm",              "primary_mean": 0.99,  "primary_std": 0.005},
            {"combiner_type": "glm_interactions", "primary_mean": 0.991, "primary_std": 0.005},
        ],
        "winner": "glm",
        "reason": "glm won outright.",
        "winner_panel": {
            "spearman": 0.99, "kendall": 0.93, "pearson": 0.99,
            "mae": 0.02, "auc": float("nan"), "brier": float("nan"),
        },
        "fitted_coefficients": {"intercept": -2.8, "coef": [2.8, 1.7, 0.9, 0.3],
                                "feature_names": ["semantic", "nli", "naturalness", "emotion"]},
        "diagnostics_summary": {"Residuals": "mean=0.0 std=0.03"},
        "warnings": [],
    }


def test_format_text_includes_winner_and_candidates(report_dict):
    out = format_text(report_dict)
    assert "Winner: glm" in out
    assert "glm_interactions" in out
    assert "ebm" in out and "dropped" in out


def test_format_text_dropped_backend_includes_install_hint(report_dict):
    """When a tier is dropped, the user must see what's missing AND how to install it."""
    out = format_text(report_dict)
    # Backend name, description, and install command must all appear.
    assert "ebm" in out
    assert "Explainable Boosting" in out
    assert "interpret-ml" in out
    assert "pip install 'lens-eval[ebm]'" in out
    # The catch-all hint mentions the [all] extras group.
    assert "lens-eval[all]" in out


def test_format_text_dropped_backend_includes_gbm_install_hint():
    rpt = {
        "task": "regression", "target_type": "continuous", "link": "identity",
        "n_samples": 10_000, "dimensions_used": ["semantic", "nli", "naturalness", "emotion"],
        "candidates": ["glm", "glm_interactions"], "dropped_backends": ["ebm", "gbm"],
        "primary_metric": "spearman", "cv_splits": 5, "random_state": 0,
        "cv_scores": [], "winner": "glm", "reason": "ok",
    }
    out = format_text(rpt)
    assert "gbm" in out
    assert "lightgbm" in out
    assert "pip install 'lens-eval[gbm]'" in out


def test_format_text_no_dropped_section_when_nothing_dropped(report_dict):
    rpt = dict(report_dict)
    rpt["dropped_backends"] = []
    out = format_text(rpt)
    assert "dropped (backend not installed)" not in out


def test_format_text_unknown_backend_falls_back_to_name():
    """If a future combiner name has no install hint registered, render its name only."""
    rpt = {
        "task": "regression", "target_type": "continuous", "link": "identity",
        "n_samples": 1000, "dimensions_used": ["a"],
        "candidates": ["glm"], "dropped_backends": ["mystery"],
        "primary_metric": "spearman", "cv_splits": 5, "random_state": 0,
        "cv_scores": [], "winner": "glm", "reason": "ok",
    }
    out = format_text(rpt)
    assert "mystery" in out
    # No crash, no install command (we don't know what to recommend).
    assert "pip install 'lens-eval[mystery]'" not in out


def test_format_text_renders_winner_panel(report_dict):
    """The full metric panel for the selected model lands in a dedicated section."""
    out = format_text(report_dict)
    assert "Winner panel" in out
    for k in ("spearman", "kendall", "pearson", "mae"):
        assert k in out


def test_format_text_renders_coefficients(report_dict):
    out = format_text(report_dict)
    assert "semantic" in out
    assert "+2.8" in out or "2.8163" in out or "2.8" in out


def test_summarise_diagnostics_returns_dict_of_strings():
    """Monotonicity entries are dicts; the summariser names non-monotone dims."""
    diag = {
        "residuals": {"mean": 0.0, "std": 0.03, "abs_max": 0.1},
        "calibration": {"ece": 0.01, "brier": 0.05, "reliability": {}},
        "feature_correlations": {"max_offdiag": 0.4},
        "monotonicity": {
            "a": {"status": "increasing"},
            "b": {"status": "non-monotone"},
        },
        "outliers": {"count": 3, "indices": [0, 1, 2]},
    }
    s = summarise_diagnostics(diag)
    assert all(isinstance(v, str) for v in s.values())
    assert "Monotonicity" in s
    assert "non-monotone" in s["Monotonicity"]
    assert "b" in s["Monotonicity"]


def test_format_html_returns_html(report_dict):
    html = format_html(report_dict, diagnostics=None)
    assert html.lower().startswith("<!doctype html>")
    assert "LENS combiner selection report" in html
