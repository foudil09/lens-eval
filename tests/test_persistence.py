from __future__ import annotations

import json

import numpy as np
import pytest

from lens_eval import LENS
from lens_eval import encoders as enc


@pytest.fixture(autouse=True)
def _reset_encoders_state():
    saved = {k: v for k, v in enc._CONFIG.items()}
    autoload = enc._CENTROID_AUTOLOADED
    yield
    enc._CONFIG.update(saved)
    enc._CACHE.clear()
    enc._CENTROID_AUTOLOADED = autoload


def _fit_small(features, scores, mode="reference"):
    enc.configure(naturalness_mode=mode)
    lens = LENS(random_state=0)
    lens.fit(features=features, references=[""] * len(features),
             scores=scores, verbose=False)
    return lens


def test_save_writes_expected_files(tmp_path, synth_reg):
    X, y = synth_reg
    lens = _fit_small(X, y)
    out = tmp_path / "lens"
    lens.save(out)
    files = {p.name for p in out.iterdir()}
    assert files == {"manifest.json", "combiner.pkl", "selection_report.json"}


def test_load_round_trip_predicts_identically(tmp_path, synth_reg):
    X, y = synth_reg
    lens = _fit_small(X, y)
    out = tmp_path / "lens"
    lens.save(out)

    lens2 = LENS.load(out)
    s1 = lens.score(references=[""] * 30, features=X[:30])
    s2 = lens2.score(references=[""] * 30, features=X[:30])
    np.testing.assert_allclose(s1, s2, atol=0)


def test_load_restores_naturalness_mode(tmp_path, synth_reg):
    X, y = synth_reg
    lens = _fit_small(X, y, mode="reference")
    out = tmp_path / "lens"
    lens.save(out)

    # Mutate global state, then loading must reset it.
    enc.configure(naturalness_mode="centroid")
    LENS.load(out)
    assert enc._CONFIG["naturalness_mode"] == "reference"


def test_load_restores_combiner_metadata(tmp_path, synth_reg):
    X, y = synth_reg
    lens = _fit_small(X, y)
    out = tmp_path / "lens"
    lens.save(out)
    lens2 = LENS.load(out)
    assert lens2.combiner_type_   == lens.combiner_type_
    assert lens2.task_            == lens.task_
    assert lens2.target_type_     == lens.target_type_
    assert lens2.link_function_   == lens.link_function_
    assert lens2.dimensions_used_ == lens.dimensions_used_


def test_load_round_trips_naturalness_centroid(tmp_path, synth_reg):
    X, y = synth_reg
    centroid = np.linspace(-1, 1, 64).astype(np.float32)
    enc.configure(naturalness_mode="centroid", naturalness_centroid=centroid)
    lens = LENS(random_state=0)
    lens.fit(features=X, references=[""] * len(X), scores=y, verbose=False)
    out = tmp_path / "lens"
    lens.save(out)

    # Wipe centroid; load must put it back.
    enc.configure(naturalness_mode="reference")
    enc._CONFIG["naturalness_centroid"] = None
    LENS.load(out)
    np.testing.assert_allclose(enc._CONFIG["naturalness_centroid"], centroid, atol=0)


def test_load_ignores_hash_mismatch_when_runtime_has_no_local_checkpoint(tmp_path, synth_reg):
    """Public-checkpoint paths have no on-disk fingerprint, so a stale saved
    hash can't trigger a mismatch — the loader must not raise."""
    X, y = synth_reg
    lens = _fit_small(X, y)
    out = tmp_path / "lens"
    lens.save(out)

    manifest = json.loads((out / "manifest.json").read_text())
    manifest["encoder_manifest"]["hashes"]["semantic"] = "deadbeefdeadbeef"
    (out / "manifest.json").write_text(json.dumps(manifest))

    lens2 = LENS.load(out)
    assert lens2 is not None


def test_selection_report_has_per_fold_scores(tmp_path, synth_reg):
    """Per-fold per-candidate scores live inside selection_report.json; no
    separate CSV is written any more — derive a DataFrame from the JSON when
    needed."""
    X, y = synth_reg
    lens = _fit_small(X, y)
    out = tmp_path / "lens"
    lens.save(out)

    report = json.loads((out / "selection_report.json").read_text())
    for r in report["cv_scores"]:
        assert len(r["per_fold"]) == 5
