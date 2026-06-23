from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from lens_eval import encoders as enc
from lens_eval.encoders import DIMENSIONS, NATURALNESS_MODES

# Torch is an optional dep; the mocked-encoder tests below need it but
# everything else in this file stays offline-importable.
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _reset_encoders_state():
    """Each test runs against a clean module-level config + cache."""
    saved = {
        "paths": dict(enc._CONFIG["paths"]),
        "device": enc._CONFIG["device"],
        "batch_size": enc._CONFIG["batch_size"],
        "naturalness_mode": enc._CONFIG["naturalness_mode"],
        "naturalness_centroid": enc._CONFIG["naturalness_centroid"],
    }
    autoload = enc._CENTROID_AUTOLOADED
    cache = dict(enc._CACHE)
    resolved = enc._RESOLVED_DEVICE
    enc._CACHE.clear()
    yield
    enc._CONFIG.update(saved)
    enc._CACHE.clear()
    enc._CACHE.update(cache)
    enc._CENTROID_AUTOLOADED = autoload
    enc._RESOLVED_DEVICE = resolved


def test_default_paths_have_all_four_dims():
    assert set(enc._DEFAULT_PATHS) == set(DIMENSIONS)


def test_naturalness_modes_are_centroid_or_reference():
    assert set(NATURALNESS_MODES) == {"centroid", "reference"}


def test_configure_rejects_invalid_mode():
    with pytest.raises(ValueError, match="naturalness_mode"):
        enc.configure(naturalness_mode="wrong")


def test_configure_rejects_unknown_dimension_paths():
    with pytest.raises(ValueError, match="unknown dimension"):
        enc.configure(paths={"not_a_dim": "/tmp/anything"})


def test_manifest_reports_mode_and_centroid_dim():
    enc.configure(naturalness_mode="centroid",
                  naturalness_centroid=np.ones(128, dtype=np.float32))
    m = enc.manifest()
    assert m["naturalness_mode"] == "centroid"
    assert m["has_naturalness_centroid"] is True
    assert m["naturalness_centroid_dim"] == 128


def test_manifest_reports_reference_mode_no_centroid():
    enc.configure(naturalness_mode="reference")
    enc._CONFIG["naturalness_centroid"] = None  # explicit reset
    m = enc.manifest()
    assert m["naturalness_mode"] == "reference"
    assert m["has_naturalness_centroid"] is False


def test_centroid_mode_falls_back_when_no_paired_encoder(monkeypatch):
    """If no paired NatureEncoder is on disk, naturalness_score auto-falls back
    to reference mode with a warning rather than producing nonsense numbers."""
    monkeypatch.setattr(enc, "_paired_encoder_path", lambda meta: None)

    # Make sure we're in centroid mode but with no centroid loaded — that
    # triggers the auto-load path.
    enc.configure(naturalness_mode="centroid")
    enc._CONFIG["naturalness_centroid"] = None
    enc._CENTROID_AUTOLOADED = False

    monkeypatch.setattr(enc, "_encode",
                        lambda dim, texts: np.zeros((len(texts), 128), dtype=np.float32))

    with pytest.warns(RuntimeWarning, match="bundled naturalness centroid"):
        out = enc.naturalness_score(["a"])
    assert enc._CONFIG["naturalness_mode"] == "reference"
    # In reference mode with no refs the column is NaN.
    assert np.isnan(out).all()


def test_user_supplied_centroid_skips_bundled_lookup(monkeypatch):
    """A user-supplied centroid suppresses the bundled auto-load."""
    monkeypatch.setattr(enc, "_load_bundled_centroid",
                        lambda: pytest.fail("should not be called"))
    arr = np.ones(64, dtype=np.float32)
    enc.configure(naturalness_mode="centroid", naturalness_centroid=arr)
    np.testing.assert_array_equal(enc._CONFIG["naturalness_centroid"], arr)


def test_naturalness_centroid_mode_without_centroid_raises(monkeypatch):
    """If centroid auto-load fails to populate, score-time gives a useful error."""
    enc.configure(naturalness_mode="centroid")
    enc._CONFIG["naturalness_centroid"] = None
    enc._CENTROID_AUTOLOADED = True   # pretend we tried and have nothing
    with pytest.raises(ValueError, match="no centroid is loaded"):
        enc.naturalness_score(["hi"])


def test_reference_dim_with_no_refs_returns_nan_column(monkeypatch):
    monkeypatch.setattr(enc, "_encode",
                        lambda dim, texts: np.zeros((len(texts), 4), dtype=np.float32))
    out = enc.semantic_score(["a", "b"], None)  # type: ignore[arg-type]
    assert out.shape == (2,)
    assert np.isnan(out).all()


def test_featurize_returns_one_column_per_dimension(monkeypatch):
    monkeypatch.setattr(enc, "_encode",
                        lambda dim, texts: np.ones((len(texts), 4), dtype=np.float32))

    class _FakeCE:
        def entail(self, premises, hypotheses):
            return np.full(len(premises), 0.5, dtype=np.float32)

    monkeypatch.setattr(enc, "_get_cross_encoder", lambda: _FakeCE())
    enc.configure(naturalness_mode="centroid",
                  naturalness_centroid=np.array([1, 0, 0, 0], dtype=np.float32))
    X = enc.featurize(["a", "b"], references=["r1", "r2"])
    assert X.shape == (2, 4)


def test_nli_score_passes_candidate_as_premise_reference_as_hypothesis(monkeypatch):
    seen = {}

    class _FakeCE:
        def entail(self, premises, hypotheses):
            seen["premises"], seen["hypotheses"] = list(premises), list(hypotheses)
            return np.arange(len(premises), dtype=np.float32)

    monkeypatch.setattr(enc, "_get_cross_encoder", lambda: _FakeCE())
    out = enc.nli_score(["cand1", "cand2"], ["ref1", "ref2"])
    assert seen["premises"] == ["cand1", "cand2"]      # candidate is the premise
    assert seen["hypotheses"] == ["ref1", "ref2"]      # reference is the hypothesis
    assert out.tolist() == [0.0, 1.0]


def test_nli_score_without_refs_returns_nan_column():
    out = enc.nli_score(["a", "b"], None)  # type: ignore[arg-type]
    assert out.shape == (2,) and np.isnan(out).all()


# ---------------------------------------------------------------------------
# Mocked transformer / tokenizer — pin the pooling + L2-norm math.
# ---------------------------------------------------------------------------

class _FakeBatchEncoding(dict):
    def to(self, _):
        return self


def _stub_hf_encoder(*, last_hidden_state, attention_mask):
    enc_obj = enc._HFBackboneEncoder.__new__(enc._HFBackboneEncoder)
    enc_obj.path = "stub"
    enc_obj.device = "cpu"
    enc_obj.batch_size = 64
    enc_obj.max_length = 256
    enc_obj._torch = torch

    def fake_tokenizer(_batch, **_kw):
        return _FakeBatchEncoding(
            input_ids=torch.zeros_like(attention_mask),
            attention_mask=attention_mask,
        )
    enc_obj.tokenizer = fake_tokenizer

    fake_model = MagicMock()
    fake_model.return_value = MagicMock(last_hidden_state=last_hidden_state)
    enc_obj.model = fake_model
    return enc_obj


def test_hf_encoder_mean_pool_ignores_padding_tokens():
    hidden = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0]],
    ])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    enc_obj = _stub_hf_encoder(last_hidden_state=hidden, attention_mask=mask)
    out = enc_obj.encode(["abc", "ab"])
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out[0], np.array([1, 1, 1, 0]) / np.sqrt(3), atol=1e-6)
    np.testing.assert_allclose(out[1], np.array([1, 1, 0, 0]) / np.sqrt(2), atol=1e-6)


def test_hf_encoder_outputs_are_l2_unit_normalised():
    rng = np.random.default_rng(0)
    hidden = torch.tensor(rng.normal(size=(3, 5, 8)), dtype=torch.float32)
    mask = torch.ones(3, 5, dtype=torch.long)
    enc_obj = _stub_hf_encoder(last_hidden_state=hidden, attention_mask=mask)
    out = enc_obj.encode(["x", "y", "z"])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.ones(3), atol=1e-6)


def test_hf_encoder_handles_all_padding_row_without_divbyzero():
    hidden = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    mask = torch.tensor([[0, 0]])
    enc_obj = _stub_hf_encoder(last_hidden_state=hidden, attention_mask=mask)
    out = enc_obj.encode(["zilch"])
    assert out.shape == (1, 2)
    assert np.all(np.isfinite(out))


def test_hf_encoder_empty_input_returns_correct_width():
    hidden = torch.zeros(1, 1, 7)
    mask = torch.ones(1, 1, dtype=torch.long)
    enc_obj = _stub_hf_encoder(last_hidden_state=hidden, attention_mask=mask)
    enc_obj.model.config = MagicMock()
    enc_obj.model.config.hidden_size = 7
    out = enc_obj.encode([])
    assert out.shape == (0, 7)


# ---------------------------------------------------------------------------
# Mocked dimension-score path — pin cosine + clip + centroid math.
# ---------------------------------------------------------------------------

def test_semantic_score_is_row_wise_cosine(monkeypatch):
    """cos(a, b) for two pairs of unit vectors at known angles."""
    a = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    calls = {"n": 0}

    def fake_encode(dim, texts):
        calls["n"] += 1
        return a if calls["n"] == 1 else b
    monkeypatch.setattr(enc, "_encode", fake_encode)

    out = enc.semantic_score(["t1", "t2"], ["r1", "r2"])
    np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-6)


def test_semantic_score_clips_into_minus1_to_1(monkeypatch):
    """Float overshoot from the dot product must be clipped to the cosine range."""
    over = np.array([[1.000001, 0.0]], dtype=np.float32)
    monkeypatch.setattr(enc, "_encode", lambda dim, texts: over)
    out = enc.semantic_score(["t"], ["r"])
    assert -1.0 <= out[0] <= 1.0


def test_naturalness_centroid_mode_uses_dot_with_centroid(monkeypatch):
    centroid = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    emb = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
    ], dtype=np.float32)
    enc.configure(naturalness_mode="centroid", naturalness_centroid=centroid)
    monkeypatch.setattr(enc, "_encode", lambda dim, texts: emb)
    out = enc.naturalness_score(["a", "b", "c"])
    np.testing.assert_allclose(out, [1.0, 0.0, -1.0], atol=1e-6)


def test_naturalness_centroid_mode_renormalises_non_unit_centroid(monkeypatch):
    centroid = np.array([0.0, 3.0, 0.0, 0.0], dtype=np.float32)
    emb = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    enc.configure(naturalness_mode="centroid", naturalness_centroid=centroid)
    monkeypatch.setattr(enc, "_encode", lambda dim, texts: emb)
    out = enc.naturalness_score(["x"])
    np.testing.assert_allclose(out, [1.0], atol=1e-6)


def test_naturalness_centroid_mode_rejects_wrong_dim_centroid(monkeypatch):
    centroid = np.array([0.0, 1.0], dtype=np.float32)        # 2D
    emb = np.zeros((1, 4), dtype=np.float32)                  # encoder is 4D
    enc.configure(naturalness_mode="centroid", naturalness_centroid=centroid)
    monkeypatch.setattr(enc, "_encode", lambda dim, texts: emb)
    with pytest.raises(ValueError, match="centroid dim"):
        enc.naturalness_score(["x"])


def test_semantic_score_length_mismatch_raises(monkeypatch):
    monkeypatch.setattr(enc, "_encode",
                        lambda dim, texts: np.zeros((len(texts), 4), dtype=np.float32))
    with pytest.raises(ValueError, match="same length"):
        enc.semantic_score(["a", "b"], ["only one"])
