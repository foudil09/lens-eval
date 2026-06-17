from __future__ import annotations

import numpy as np
import pytest

from lens_eval.combiners import (
    AVAILABLE,
    CAPACITY_ORDER,
    GLMCombiner,
    GLMInteractionsCombiner,
    make,
)
from lens_eval.errors import CombinerBackendMissing
from .conftest import needs_interpret, needs_lightgbm


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_has_all_four():
    assert set(AVAILABLE) == {"glm", "glm_interactions", "ebm", "gbm"}


def test_capacity_order_simplest_first():
    assert CAPACITY_ORDER == ("glm", "glm_interactions", "ebm", "gbm")


def test_make_unknown_raises():
    with pytest.raises(ValueError, match="unknown combiner"):
        make("not_a_combiner")


def test_make_missing_backend_raises():
    # We can't uninstall sklearn, so ask for one that needs an optional dep
    # and stub its check function temporarily.
    from lens_eval import combiners as cmb

    factory, check = cmb.AVAILABLE["ebm"]
    cmb.AVAILABLE["ebm"] = (factory, lambda: False)
    try:
        with pytest.raises(CombinerBackendMissing):
            make("ebm")
    finally:
        cmb.AVAILABLE["ebm"] = (factory, check)


# ---------------------------------------------------------------------------
# GLM
# ---------------------------------------------------------------------------

def test_glm_fits_continuous(synth_reg):
    X, y = synth_reg
    m = GLMCombiner(target_type="continuous", task="regression")
    m.fit(X, y)
    assert m.is_fitted_
    assert m.coef_.shape == (X.shape[1],)
    p = m.predict(X)
    assert p.shape == (X.shape[0],)
    # Strong signal data → high R²-like behaviour.
    assert np.corrcoef(p, y)[0, 1] > 0.85


def test_glm_fits_bounded_with_logit(synth_reg):
    X, y = synth_reg
    m = GLMCombiner(target_type="bounded", task="regression")
    m.fit(X, y)
    p = m.predict(X)
    assert (p >= 0.0).all() and (p <= 1.0).all()
    assert np.corrcoef(p, y)[0, 1] > 0.85


def test_glm_fits_binary(synth_binary):
    X, y = synth_binary
    m = GLMCombiner(target_type="binary", task="regression")
    m.fit(X, y)
    p = m.predict(X)
    assert (p >= 0.0).all() and (p <= 1.0).all()
    assert m.link == "logit"


def test_glm_contributions_shape(synth_reg):
    X, y = synth_reg
    m = GLMCombiner(target_type="continuous").fit(X, y)
    c = m.contributions(X)
    assert c.shape == X.shape
    # Linear → contributions sum + intercept ≈ prediction.
    p = m.predict(X)
    if m.target_type == "continuous":
        np.testing.assert_allclose(c.sum(axis=1) + m.intercept_, p, atol=1e-9)


def test_glm_coefficients_dict_has_intercept_and_coef(synth_reg):
    X, y = synth_reg
    m = GLMCombiner(target_type="continuous", feature_names=["a", "b", "c", "d"]).fit(X, y)
    out = m.coefficients()
    assert "intercept" in out and "coef" in out
    assert out["feature_names"] == ["a", "b", "c", "d"]
    assert len(out["coef"]) == 4


# ---------------------------------------------------------------------------
# GLM + interactions
# ---------------------------------------------------------------------------

def test_glm_interactions_all_pairs_count(synth_reg):
    X, y = synth_reg
    m = GLMInteractionsCombiner(target_type="continuous").fit(X, y)
    assert len(m._interaction_idx) == 6   # C(4, 2)
    # Base coef + 6 interactions = 10 features.
    assert m.coef_.shape == (10,)


def test_glm_interactions_resolve_named_pairs(synth_reg):
    X, y = synth_reg
    names = ["semantic", "nli", "naturalness", "emotion"]
    m = GLMInteractionsCombiner(
        target_type="continuous",
        feature_names=names,
        hypothesized_interactions=[("semantic", "nli"), ("semantic", "emotion")],
    ).fit(X, y)
    assert m._interaction_idx == [(0, 1), (0, 3)]


def test_glm_interactions_resolve_index_pairs(synth_reg):
    X, y = synth_reg
    m = GLMInteractionsCombiner(
        target_type="continuous",
        hypothesized_interactions=[(0, 2), (1, 3)],
    ).fit(X, y)
    assert m._interaction_idx == [(0, 2), (1, 3)]


def test_glm_interactions_drops_unknown_names(synth_reg):
    X, y = synth_reg
    m = GLMInteractionsCombiner(
        target_type="continuous",
        feature_names=["a", "b", "c", "d"],
        hypothesized_interactions=[("a", "nope"), ("b", "c")],
    ).fit(X, y)
    assert m._interaction_idx == [(1, 2)]


def test_glm_interactions_contributions_shape_stays_D(synth_reg):
    """Interaction contributions should be redistributed back onto the D base dims."""
    X, y = synth_reg
    m = GLMInteractionsCombiner(target_type="continuous").fit(X, y)
    c = m.contributions(X)
    assert c.shape == X.shape


def test_glm_interactions_contributions_sum_matches_prediction(synth_reg):
    """contribs sum + intercept ≈ prediction in identity-link mode."""
    X, y = synth_reg
    m = GLMInteractionsCombiner(target_type="continuous").fit(X, y)
    c = m.contributions(X)
    p = m.predict(X)
    np.testing.assert_allclose(c.sum(axis=1) + m.intercept_, p, atol=1e-9)


def test_glm_interactions_predict_matches_augmented_X(synth_reg):
    X, y = synth_reg
    m = GLMInteractionsCombiner(target_type="continuous").fit(X, y)
    p1 = m.predict(X)
    # Manually build the augmented matrix and verify.
    cols = [X[:, i] * X[:, j] for i, j in m._interaction_idx]
    X_aug = np.column_stack([X] + cols)
    p2 = X_aug @ m.coef_ + m.intercept_
    np.testing.assert_allclose(p1, p2, atol=1e-9)


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------

def test_glm_pairwise_with_explicit_pairs(synth_pairs):
    X, idx = synth_pairs
    m = GLMCombiner(task="pairwise")
    m.fit(X, pairs=idx)
    # Convention: 'a' is the winner → predicted prob > 0.5 on average.
    diff = X[idx[:, 0]] - X[idx[:, 1]]
    probs = m._predict_inverse_link(diff @ m.coef_ + m.intercept_)
    assert probs.mean() > 0.5


def test_glm_pairwise_predict_latent_ranks_items(synth_pairs):
    X, idx = synth_pairs
    # Train via pre-diffed X (the path LENS.fit uses internally).
    diff = X[idx[:, 0]] - X[idx[:, 1]]
    Xtrain = np.vstack([diff, -diff])
    ytrain = np.concatenate([np.ones(len(idx), dtype=int), np.zeros(len(idx), dtype=int)])
    m = GLMCombiner(task="pairwise")
    m.fit(Xtrain, ytrain)
    latent = m.predict_latent(X)
    # Correlate latent with the true quality.
    w = np.array([0.6, 0.3, 0.05, 0.05])
    q = X @ w
    assert np.corrcoef(latent, q)[0, 1] > 0.95


# ---------------------------------------------------------------------------
# EBM / GBM (gated by backend)
# ---------------------------------------------------------------------------

@needs_interpret
def test_ebm_fits_and_predicts(synth_reg_large):
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("ebm", task="regression", target_type="continuous")
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (X.shape[0],)
    assert np.corrcoef(p, y)[0, 1] > 0.85


@needs_interpret
def test_ebm_contributions_shape(synth_reg_large):
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("ebm", task="regression", target_type="continuous")
    m.fit(X, y)
    c = m.contributions(X[:20])
    assert c.shape == (20, X.shape[1])


@needs_interpret
def test_ebm_contributions_track_known_weight_order(synth_reg_large):
    """The dim with the largest true weight should also have the largest
    mean |contribution|. The synth_reg fixture has w=[0.45, 0.30, 0.15, 0.10]."""
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("ebm", task="regression", target_type="continuous")
    m.fit(X, y)
    c = m.contributions(X)
    mag = np.abs(c).mean(axis=0)
    # dim 0 is the dominant driver in the fixture — should rank first.
    assert int(np.argmax(mag)) == 0


@needs_interpret
def test_ebm_contributions_sum_correlates_with_prediction(synth_reg_large):
    """Per-row contribution sum should track the prediction (up to a roughly
    constant intercept). We test correlation rather than exact equality
    because interpret-ml's local explanations live in latent space while
    predict() applies the link / mean adjustment."""
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("ebm", task="regression", target_type="continuous")
    m.fit(X, y)
    c = m.contributions(X[:200])
    p = m.predict(X[:200])
    rho = float(np.corrcoef(c.sum(axis=1), p)[0, 1])
    assert rho > 0.95


@needs_lightgbm
def test_gbm_fits_with_monotone_constraints(synth_reg_large):
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("gbm", task="regression", target_type="continuous")
    m.fit(X, y)
    p = m.predict(X)
    assert np.corrcoef(p, y)[0, 1] > 0.85


@needs_lightgbm
def test_gbm_still_picks_up_signal_after_constraints_stripped(synth_reg_large):
    """Phase 2 removed monotone constraints, so we no longer test that
    bumping a feature always raises the prediction. We DO still expect the
    fit to correlate strongly with the target — Phase 2's promise is that
    we lose the constraint, not the signal. (Feature-ordering via SHAP
    contributions is covered by test_gbm_contributions_track_known_weight_order.)"""
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("gbm", task="regression", target_type="continuous")
    m.fit(X, y)
    p = m.predict(X)
    assert np.corrcoef(p, y)[0, 1] > 0.85


@needs_lightgbm
def test_gbm_can_learn_non_monotone_shape(rng):
    """With monotone constraints stripped, the GBM is now free to fit a
    concave-down ('quality peaks then degrades') pattern in a feature.
    Verify by synthesising a one-feature concave-down target and checking
    the predictions also peak in the interior of the feature range."""
    n = 2000
    X = rng.uniform(0, 1, size=(n, 4))
    # y peaks when x0 ≈ 0.5, falls off symmetrically — strictly non-monotone.
    y = -(X[:, 0] - 0.5) ** 2 + 0.25 + 0.01 * rng.normal(size=n)
    from lens_eval.combiners import make
    m = make("gbm", task="regression", target_type="continuous")
    m.fit(X, y)
    # Grid over x0; other features at their mean — see if the prediction
    # peaks somewhere in the interior rather than monotonically rising.
    xg = np.linspace(0.05, 0.95, 19)
    X_grid = np.tile(X.mean(axis=0), (19, 1))
    X_grid[:, 0] = xg
    preds = m.predict(X_grid)
    peak_idx = int(np.argmax(preds))
    # Interior peak — not at either endpoint. Monotone constraints would
    # force the argmax to the upper edge.
    assert 2 <= peak_idx <= 16, (
        f"GBM still behaves monotonically — peak at index {peak_idx} "
        f"(expected interior). Constraints may not be fully stripped."
    )


@needs_lightgbm
def test_gbm_contributions_shape(synth_reg_large):
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("gbm", task="regression", target_type="continuous")
    m.fit(X, y)
    c = m.contributions(X[:20])
    assert c.shape == (20, X.shape[1])


@needs_lightgbm
def test_gbm_contributions_sum_plus_bias_recovers_prediction(synth_reg_large):
    """LightGBM's pred_contrib output (SHAP) sums to the raw prediction once
    you add the dropped bias column. Our combiner returns shap[:, :-1], so
    contributions.sum + constant_bias should equal predict(X) row-wise."""
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("gbm", task="regression", target_type="continuous")
    m.fit(X, y)
    c = m.contributions(X[:200])
    p = m.predict(X[:200])
    # The bias is a constant across rows — its variance across the residual
    # is what we expect, not its magnitude.
    residual = p - c.sum(axis=1)
    assert float(np.std(residual)) < 1e-6
    # And the residual should equal the booster's reported base value.
    bias_col = m.model_.booster_.predict(X[:200], pred_contrib=True)[:, -1]
    np.testing.assert_allclose(residual, bias_col, atol=1e-9)


@needs_lightgbm
def test_gbm_contributions_track_known_weight_order(synth_reg_large):
    """Most-weighted dim in the fixture (w[0]=0.45) should dominate SHAP magnitude."""
    X, y = synth_reg_large
    from lens_eval.combiners import make
    m = make("gbm", task="regression", target_type="continuous")
    m.fit(X, y)
    c = m.contributions(X)
    mag = np.abs(c).mean(axis=0)
    assert int(np.argmax(mag)) == 0


@needs_lightgbm
def test_gbm_ranking_requires_ranks_and_groups():
    """task='ranking' without ranks or groups must raise — never silently
    fall through to a regressor on bare row indices."""
    X = np.random.default_rng(0).uniform(0, 1, size=(60, 4))
    from lens_eval.combiners import make
    m = make("gbm", task="ranking", target_type="continuous")
    with pytest.raises(ValueError, match="ranks"):
        m.fit(X, y=None)
    with pytest.raises(ValueError, match="groups"):
        m.fit(X, y=None, ranks=np.arange(60))


@needs_lightgbm
def test_gbm_ranking_fit_and_predict(synth_ranking):
    """GBM ranking with valid ranks + groups fits and produces predictions
    that correlate with relevance labels."""
    X, ranks, groups = synth_ranking
    from lens_eval.combiners import make
    m = make("gbm", task="ranking", target_type="continuous")
    m.fit(X, y=None, ranks=ranks, groups=groups)
    p = m.predict(X)
    assert p.shape == (X.shape[0],)
    assert np.corrcoef(p, ranks)[0, 1] > 0.3


@needs_lightgbm
def test_gbm_ranking_non_contiguous_groups(synth_ranking, rng):
    """Non-contiguous (interleaved) group IDs must not crash or silently
    mis-specify the LightGBM group structure."""
    X, ranks, groups = synth_ranking
    shuffle = rng.permutation(len(X))
    X_s, ranks_s, groups_s = X[shuffle], ranks[shuffle], groups[shuffle]

    from lens_eval.combiners import make
    m = make("gbm", task="ranking", target_type="continuous", random_state=42)
    m.fit(X_s, y=None, ranks=ranks_s, groups=groups_s)
    p = m.predict(X_s)
    assert p.shape == (len(X),)
    assert np.corrcoef(p, ranks_s)[0, 1] > 0.3
