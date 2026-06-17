"""Tests for the centralized input-validation gateway.

Both `LENS.fit` and the CLI route through `lens_eval._validate.validate_task_channels`,
so the same wrong input produces the same exception type and message regardless
of entry point. These tests exercise the gateway directly — the CLI tests in
test_cli.py and the LENS tests in test_lens.py cover the routing.
"""

from __future__ import annotations

import numpy as np
import pytest

from lens_eval._validate import validate_task_channels
from lens_eval.errors import AmbiguousTaskError


# ---------------------------------------------------------------------------
# Exactly-one-channel rule
# ---------------------------------------------------------------------------

def test_no_channel_raises_ambiguous():
    with pytest.raises(AmbiguousTaskError, match="exactly one"):
        validate_task_channels()


def test_two_channels_raises_ambiguous():
    with pytest.raises(AmbiguousTaskError, match="exactly one"):
        validate_task_channels(scores=np.zeros(5), pairs=np.zeros((3, 2), dtype=int))


def test_three_channels_raises_ambiguous():
    with pytest.raises(AmbiguousTaskError, match="exactly one"):
        validate_task_channels(
            scores=np.zeros(5),
            pairs=np.zeros((3, 2), dtype=int),
            ranks=np.zeros(5, dtype=int),
            groups=np.zeros(5, dtype=int),
        )


# ---------------------------------------------------------------------------
# Channel → task mapping
# ---------------------------------------------------------------------------

def test_scores_maps_to_regression():
    channel, task = validate_task_channels(scores=np.linspace(0, 1, 10))
    assert channel == "scores"
    assert task == "regression"


def test_pairs_maps_to_pairwise():
    channel, task = validate_task_channels(pairs=np.array([[0, 1], [2, 3]]))
    assert channel == "pairs"
    assert task == "pairwise"


def test_ranks_with_groups_maps_to_ranking():
    channel, task = validate_task_channels(
        ranks=np.array([3, 1, 2, 1, 2]),
        groups=np.array([0, 0, 0, 1, 1]),
    )
    assert channel == "ranks"
    assert task == "ranking"


# ---------------------------------------------------------------------------
# Channel-specific shape validation
# ---------------------------------------------------------------------------

def test_pairs_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        validate_task_channels(pairs=np.array([1, 2, 3]))   # 1D


def test_pairs_three_columns_raises():
    with pytest.raises(ValueError, match="shape"):
        validate_task_channels(pairs=np.zeros((5, 3), dtype=int))


def test_pairs_fractional_indices_raise():
    with pytest.raises(ValueError, match="integer"):
        validate_task_channels(pairs=np.array([[0.5, 1.5], [2.0, 3.0]]))


def test_pairs_float_but_integer_valued_accepted():
    """Float arrays whose values are integer-valued (e.g. read from CSV) are OK."""
    channel, _ = validate_task_channels(pairs=np.array([[0.0, 1.0], [2.0, 3.0]]))
    assert channel == "pairs"


# ---------------------------------------------------------------------------
# Ranking-specific structure
# ---------------------------------------------------------------------------

def test_ranking_without_groups_raises():
    with pytest.raises(ValueError, match="groups"):
        validate_task_channels(ranks=np.array([1, 2, 3, 1, 2]))


def test_ranking_length_mismatch_raises():
    with pytest.raises(ValueError, match="align"):
        validate_task_channels(
            ranks=np.array([1, 2, 3]),
            groups=np.array([0, 0, 0, 1, 1]),
        )
