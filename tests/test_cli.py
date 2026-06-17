"""CLI subprocess smoke tests.

These shell out to the installed `lens-eval` console script. If the package is
checked out but not installed, the tests fall back to `python -m lens_eval.cli`.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _cli_cmd():
    """Return the argv prefix for invoking the CLI."""
    if shutil.which("lens-eval"):
        return ["lens-eval"]
    return [sys.executable, "-m", "lens_eval.cli"]


def _write_features_csv(path: Path, X: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in X:
            w.writerow([float(x) for x in row])


def _write_scores_csv(path: Path, y: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for v in y:
            w.writerow([float(v)])


def _write_texts(path: Path, n: int) -> None:
    path.write_text("\n".join(["x"] * n))


@pytest.fixture
def small_dataset(tmp_path, synth_reg):
    X, y = synth_reg
    paths = {
        "features": tmp_path / "features.csv",
        "scores":   tmp_path / "scores.csv",
        "texts":    tmp_path / "texts.txt",
        "refs":     tmp_path / "refs.txt",
        "model":    tmp_path / "lens-cli",
        "preds":    tmp_path / "preds.csv",
        "html":     tmp_path / "report.html",
    }
    _write_features_csv(paths["features"], X)
    _write_scores_csv(paths["scores"], y)
    _write_texts(paths["texts"], len(X))
    _write_texts(paths["refs"], len(X))
    return paths


def test_cli_help_exits_zero():
    res = subprocess.run(_cli_cmd() + ["--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "fit" in res.stdout and "score" in res.stdout and "report" in res.stdout


def test_cli_fit_then_score_then_report(small_dataset):
    p = small_dataset

    fit = subprocess.run(
        _cli_cmd() + [
            "fit",
            "--texts",    str(p["texts"]),
            "--refs",     str(p["refs"]),
            "--features", str(p["features"]),
            "--scores",   str(p["scores"]),
            "--output",   str(p["model"]),
            "--naturalness-mode", "reference",
        ],
        capture_output=True, text=True,
    )
    assert fit.returncode == 0, fit.stderr
    assert (p["model"] / "manifest.json").exists()
    assert (p["model"] / "combiner.pkl").exists()

    score = subprocess.run(
        _cli_cmd() + [
            "score",
            "--model",    str(p["model"]),
            "--features", str(p["features"]),
            "--texts",    str(p["texts"]),
            "--refs",     str(p["refs"]),
            "--output",   str(p["preds"]),
        ],
        capture_output=True, text=True,
    )
    assert score.returncode == 0, score.stderr
    assert p["preds"].exists()

    # Reload predictions, sanity-check shape.
    with open(p["preds"]) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["score"]
    assert len(rows) - 1 == 400   # synth_reg fixture is n=400

    rpt = subprocess.run(
        _cli_cmd() + ["report", str(p["model"]), "--html", str(p["html"])],
        capture_output=True, text=True,
    )
    assert rpt.returncode == 0, rpt.stderr
    assert p["html"].exists()
    assert "Winner" in rpt.stdout


def test_cli_fit_without_any_task_channel_errors(small_dataset):
    """No --scores, --pairs, or --ranks → AmbiguousTaskError via the shared gateway."""
    p = small_dataset
    res = subprocess.run(
        _cli_cmd() + [
            "fit",
            "--features", str(p["features"]),
            "--output",   str(p["model"]),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    # Same exception type & verbiage as the Python API — the centralized
    # gateway guarantees that.
    assert "scores" in res.stderr and "pairs" in res.stderr and "ranks" in res.stderr


def test_cli_fit_with_two_task_channels_errors(small_dataset, tmp_path):
    """Both --scores and --ranks → AmbiguousTaskError."""
    p = small_dataset
    ranks_path = tmp_path / "ranks.csv"
    with open(ranks_path, "w", newline="") as f:
        w = csv.writer(f)
        for v in range(400):
            w.writerow([v])
    res = subprocess.run(
        _cli_cmd() + [
            "fit",
            "--features", str(p["features"]),
            "--scores",   str(p["scores"]),
            "--ranks",    str(ranks_path),
            "--output",   str(p["model"]),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "exactly one" in res.stderr


def test_cli_ranks_requires_groups(small_dataset, tmp_path):
    """--ranks without --groups must fail with the gateway's standard message."""
    p = small_dataset
    ranks_path = tmp_path / "ranks.csv"
    with open(ranks_path, "w", newline="") as f:
        w = csv.writer(f)
        for v in range(400):
            w.writerow([v])
    res = subprocess.run(
        _cli_cmd() + [
            "fit",
            "--features", str(p["features"]),
            "--ranks",    str(ranks_path),
            "--output",   str(p["model"]),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "groups" in res.stderr


def test_cli_score_without_features_or_texts_errors(small_dataset):
    p = small_dataset
    # First fit something.
    subprocess.run(
        _cli_cmd() + [
            "fit",
            "--features", str(p["features"]),
            "--scores",   str(p["scores"]),
            "--output",   str(p["model"]),
            "--naturalness-mode", "reference",
        ],
        check=True, capture_output=True,
    )
    res = subprocess.run(
        _cli_cmd() + [
            "score",
            "--model", str(p["model"]),
            "--output", str(p["preds"]),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
