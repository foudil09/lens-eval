from __future__ import annotations

import numpy as np

from lens_eval.lens import _infer_link, _infer_target_type


def test_binary_y_inferred():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert _infer_target_type(y) == "binary"


def test_integer_likert_inferred_as_bounded():
    """Continuity-enriched: 1-5 Likert routes through the bounded scaling layer."""
    y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 3, 3])
    assert _infer_target_type(y) == "bounded"


def test_integer_count_inferred_as_bounded():
    """Integer y with many unique levels still goes bounded — no Likert cap."""
    y = np.array(list(range(10)) * 5)
    assert _infer_target_type(y) == "bounded"


def test_explicit_ordinal_still_allowed_as_override():
    """Auto-detect doesn't pick ordinal; the user override path still works."""
    assert _infer_link("ordinal", "regression") == "cumulative_logit"


def test_bounded_unit_interval_inferred():
    y = np.random.default_rng(0).uniform(0, 1, 200)
    assert _infer_target_type(y) == "bounded"


def test_bounded_zero_to_hundred_inferred():
    y = np.random.default_rng(0).uniform(0, 100, 200)
    assert _infer_target_type(y) == "bounded"


def test_continuous_negative_values():
    y = np.random.default_rng(0).normal(0, 1, 200)
    assert _infer_target_type(y) == "continuous"


# ---------------------------------------------------------------------------
# Link function inference
# ---------------------------------------------------------------------------

def test_link_pairwise_is_logit():
    assert _infer_link("continuous", "pairwise") == "logit"


def test_link_binary_is_logit():
    assert _infer_link("binary", "regression") == "logit"


def test_link_ordinal_is_cumulative_logit():
    assert _infer_link("ordinal", "regression") == "cumulative_logit"


def test_link_bounded_is_logit():
    assert _infer_link("bounded", "regression") == "logit"


def test_link_continuous_is_identity():
    assert _infer_link("continuous", "regression") == "identity"
