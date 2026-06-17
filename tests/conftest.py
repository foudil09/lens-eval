"""Shared pytest fixtures.

Encoder-dependent tests are gated by `--run-encoders` (off by default) so the
suite stays offline-fast. Combiner backend tests are auto-skipped if the
backend isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-encoders",
        action="store_true",
        default=False,
        help="Run tests that load real sentence-transformer / HF encoders.",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-encoders"):
        skip = pytest.mark.skip(reason="needs --run-encoders to load real models")
        for item in items:
            if "encoders" in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _synthetic_regression(rng, n=400, d=4):
    """4 'dimension' features uniform in [0, 1], y is a bounded mix of them."""
    X = rng.uniform(0, 1, size=(n, d)).astype(np.float64)
    w = np.array([0.45, 0.30, 0.15, 0.10])
    y_lat = X @ w + 0.05 * X[:, 0] * X[:, 1] + rng.normal(0, 0.02, n)
    y = (y_lat - y_lat.min()) / (y_lat.max() - y_lat.min() + 1e-12)
    return X, y


@pytest.fixture
def synth_reg(rng):
    return _synthetic_regression(rng, n=400)


@pytest.fixture
def synth_reg_small(rng):
    return _synthetic_regression(rng, n=120)


@pytest.fixture
def synth_reg_large(rng):
    return _synthetic_regression(rng, n=1500)


@pytest.fixture
def synth_pairs(rng):
    """N=200 items, M=600 pairs with a known latent ordering."""
    n, m = 200, 600
    X = rng.uniform(0, 1, size=(n, 4)).astype(np.float64)
    w = np.array([0.6, 0.3, 0.05, 0.05])
    q = X @ w
    idx = rng.integers(0, n, size=(m, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    swap = q[idx[:, 0]] < q[idx[:, 1]]
    idx[swap] = idx[swap][:, ::-1]
    return X, idx


@pytest.fixture
def synth_ranking(rng):
    """10 groups of 30 items, relevance (0-3) derived from features."""
    n_groups, per_group = 10, 30
    n = n_groups * per_group
    X = rng.uniform(0, 1, size=(n, 4)).astype(np.float64)
    w = np.array([0.45, 0.30, 0.15, 0.10])
    quality = X @ w
    groups = np.repeat(np.arange(n_groups), per_group)
    ranks = np.zeros(n, dtype=int)
    for g in range(n_groups):
        mask = groups == g
        edges = np.quantile(quality[mask], [0.25, 0.5, 0.75])
        ranks[mask] = np.digitize(quality[mask], edges)
    return X, ranks, groups


@pytest.fixture
def synth_binary(rng):
    X = rng.uniform(0, 1, size=(300, 4)).astype(np.float64)
    w = np.array([1.5, 1.0, 0.5, 0.2])
    z = X @ w - 1.2
    y = (rng.uniform(0, 1, 300) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Backend availability markers
# ---------------------------------------------------------------------------

def _has(name):
    try:
        __import__(name)
        return True
    except Exception:  # ImportError or OSError (e.g. lightgbm without libomp)
        return False


needs_statsmodels = pytest.mark.skipif(not _has("statsmodels"), reason="statsmodels not installed")
needs_interpret   = pytest.mark.skipif(not _has("interpret"),   reason="interpret-ml not installed")
needs_lightgbm    = pytest.mark.skipif(not _has("lightgbm"),    reason="lightgbm not installed")
needs_jinja2      = pytest.mark.skipif(not _has("jinja2"),      reason="jinja2 not installed")
