"""Encoders for the four LENS dimensions.

Public API — call the per-axis functions directly or via ``featurize``:

    from lens_eval import semantic_score, naturalness_score, configure
    configure(device="cuda")
    s = semantic_score(texts, refs)        # cosine via mpnet
    n = naturalness_score(texts)           # cosine vs centroid (default)
    X = featurize(texts, refs)             # (N, 4) feature matrix

State lives in module-level config + a lazy encoder cache. ``configure()`` is
idempotent and clears the cache when paths/device/batch_size change.

The semantic, naturalness, and emotion axes produce L2-normalised embeddings
and score by cosine in [-1, 1]. The nli axis is different: it runs the NLI
cross-encoder on the (candidate, reference) pair and returns P(entailment) in
[0, 1] — directional, since entailment is not symmetric.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


DIMENSIONS = ("semantic", "nli", "naturalness", "emotion")
NATURALNESS_MODES = ("centroid", "reference")

_DEFAULT_PATHS = {
    "semantic":    "sentence-transformers/all-mpnet-base-v2",
    "nli":         "cross-encoder/nli-deberta-v3-base",
    "naturalness": "foudil/lens-naturalness-encoder",
    "emotion":     "foudil/lens-emotion-encoder",
}

_CONFIG: dict = {
    "paths": dict(_DEFAULT_PATHS),
    "device": None,
    "batch_size": 64,
    "naturalness_mode": "centroid",
    "naturalness_centroid": None,
}

_CACHE: dict = {}
_RESOLVED_DEVICE: Optional[str] = None
_CENTROID_AUTOLOADED = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def configure(
    *,
    paths: Optional[dict] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    naturalness_mode: Optional[str] = None,
    naturalness_centroid: Optional[np.ndarray] = None,
) -> None:
    """Update the per-process encoder configuration.

    Pass a partial dict — only the keys you supply are touched. Encoder cache
    is invalidated when ``paths`` / ``device`` / ``batch_size`` change.
    """
    global _RESOLVED_DEVICE, _CENTROID_AUTOLOADED
    invalidate = False

    if paths is not None:
        unknown = set(paths) - set(DIMENSIONS)
        if unknown:
            raise ValueError(
                f"unknown dimension(s) {sorted(unknown)}; must be subset of {DIMENSIONS}"
            )
        merged = dict(_CONFIG["paths"])
        merged.update(paths)
        _CONFIG["paths"] = merged
        invalidate = True

    if device is not None:
        _CONFIG["device"] = device
        _RESOLVED_DEVICE = None
        invalidate = True

    if batch_size is not None:
        _CONFIG["batch_size"] = int(batch_size)
        invalidate = True

    if naturalness_mode is not None:
        if naturalness_mode not in NATURALNESS_MODES:
            raise ValueError(
                f"naturalness_mode must be one of {NATURALNESS_MODES}; got {naturalness_mode!r}"
            )
        _CONFIG["naturalness_mode"] = naturalness_mode
        _CENTROID_AUTOLOADED = False

    if naturalness_centroid is not None:
        _CONFIG["naturalness_centroid"] = np.asarray(naturalness_centroid, dtype=np.float32)
        # User-supplied centroid → suppress bundled auto-load.
        _CENTROID_AUTOLOADED = True

    if invalidate:
        _CACHE.clear()


def free() -> None:
    """Drop cached encoders (frees GPU memory)."""
    _CACHE.clear()


def manifest() -> dict:
    """Config snapshot for persistence."""
    c = _CONFIG["naturalness_centroid"]
    return {
        "paths": dict(_CONFIG["paths"]),
        "hashes": {d: _file_hash(_CONFIG["paths"][d]) for d in DIMENSIONS},
        "naturalness_mode": _CONFIG["naturalness_mode"],
        "has_naturalness_centroid": c is not None,
        "naturalness_centroid_dim": None if c is None else int(np.asarray(c).size),
    }


# ---------------------------------------------------------------------------
# Public per-axis score functions
# ---------------------------------------------------------------------------

def semantic_score(texts: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """cos(emb(text), emb(reference)) via the semantic encoder."""
    return _cosine_score("semantic", texts, references)


def nli_score(texts: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """P(entailment) for premise=text, hypothesis=reference via the NLI
    cross-encoder. Asymmetric by design: it scores whether the candidate
    entails the reference, which a symmetric cosine cannot capture."""
    if references is None:
        return np.full(len(texts), np.nan, dtype=np.float32)
    if len(texts) != len(references):
        raise ValueError(
            f"texts ({len(texts)}) and references ({len(references)}) must have the same length"
        )
    return _get_cross_encoder().entail(texts, references)


def naturalness_score(
    texts: Sequence[str],
    references: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Naturalness score.

    ``naturalness_mode='centroid'`` (default): cos(emb(text), centroid), a
    single-text quality signal that ignores ``references``. ``'reference'``:
    cos(emb(text), emb(reference)) like the other dims.
    """
    if _CONFIG["naturalness_mode"] == "centroid":
        _ensure_centroid_loaded()
        # _ensure_centroid_loaded may have downgraded mode to "reference" when
        # no paired encoder is available — re-check before assuming centroid.
        if _CONFIG["naturalness_mode"] != "centroid":
            return _cosine_score("naturalness", texts, references)
        c = _CONFIG["naturalness_centroid"]
        if c is None:
            raise ValueError(
                "naturalness_mode='centroid' but no centroid is loaded. "
                "Pass `naturalness_centroid=...` to configure() or set mode='reference'."
            )
        emb = _encode("naturalness", texts)
        c = np.asarray(c, dtype=np.float32).ravel()
        if c.size != emb.shape[1]:
            raise ValueError(
                f"naturalness centroid dim ({c.size}) does not match the encoder "
                f"embedding dim ({emb.shape[1]}). The bundled centroid is for "
                f"NatureEncoder (dim=128); point `paths['naturalness']` at the "
                f"matching encoder, supply your own centroid, or use mode='reference'."
            )
        cn = c / max(float(np.linalg.norm(c)), 1e-9)
        return np.clip(emb @ cn, -1.0, 1.0).astype(np.float32)
    return _cosine_score("naturalness", texts, references)


def emotion_score(texts: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """cos(emb(text), emb(reference)) via the emotion encoder backbone."""
    return _cosine_score("emotion", texts, references)


def featurize(
    texts: Sequence[str],
    references: Optional[Sequence[str]] = None,
    dimensions: Iterable[str] = DIMENSIONS,
) -> np.ndarray:
    """(N, len(dimensions)) feature matrix in the supplied dimension order."""
    return np.column_stack([_dim_score(d, texts, references) for d in dimensions])


def _dim_score(dim: str, texts, refs):
    if dim == "naturalness":
        return naturalness_score(texts, refs)
    if dim == "nli":
        return nli_score(texts, refs)
    if refs is None:
        # Reference-dependent dim with no refs → NaN column; fit() drops it.
        return np.full(len(texts), np.nan, dtype=np.float32)
    return _cosine_score(dim, texts, refs)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _cosine_score(dim: str, texts: Sequence[str], references: Optional[Sequence[str]]) -> np.ndarray:
    if references is None:
        return np.full(len(texts), np.nan, dtype=np.float32)
    if len(texts) != len(references):
        raise ValueError(
            f"texts ({len(texts)}) and references ({len(references)}) must have the same length"
        )
    a = _encode(dim, texts)
    b = _encode(dim, references)
    sims = (a * b).sum(axis=1)
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


def _encode(dim: str, texts: Sequence[str]) -> np.ndarray:
    return _get_encoder(dim).encode(texts)


def _get_encoder(dim: str):
    if dim not in DIMENSIONS:
        raise ValueError(f"unknown dimension {dim!r}")
    if dim in _CACHE:
        return _CACHE[dim]
    path = _CONFIG["paths"][dim]
    device = _resolved_device()
    batch_size = _CONFIG["batch_size"]
    cls = _STEncoder if dim in {"semantic", "naturalness"} else _HFBackboneEncoder
    _CACHE[dim] = cls(path, device=device, batch_size=batch_size)
    return _CACHE[dim]


def _get_cross_encoder():
    if "nli_ce" in _CACHE:
        return _CACHE["nli_ce"]
    _CACHE["nli_ce"] = _CrossEncoderScorer(
        _CONFIG["paths"]["nli"], _resolved_device(), _CONFIG["batch_size"]
    )
    return _CACHE["nli_ce"]


def _resolved_device() -> str:
    global _RESOLVED_DEVICE
    if _RESOLVED_DEVICE is not None:
        return _RESOLVED_DEVICE
    pref = _CONFIG["device"]
    if pref is not None:
        _RESOLVED_DEVICE = pref
        return pref
    try:
        import torch
    except ImportError:
        _RESOLVED_DEVICE = "cpu"
        return "cpu"
    if torch.cuda.is_available():
        _RESOLVED_DEVICE = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _RESOLVED_DEVICE = "mps"
    else:
        _RESOLVED_DEVICE = "cpu"
    return _RESOLVED_DEVICE


def _ensure_centroid_loaded() -> None:
    """Lazy auto-load of the bundled NatureEncoder centroid in centroid mode."""
    global _CENTROID_AUTOLOADED
    if _CENTROID_AUTOLOADED or _CONFIG["naturalness_centroid"] is not None:
        return
    meta = _load_bundled_centroid()
    explicit_nat_path = _CONFIG["paths"]["naturalness"] != _DEFAULT_PATHS["naturalness"]
    paired = _paired_encoder_path(meta)
    if explicit_nat_path:
        # User pinned their own encoder; trust the centroid's vector space matches.
        _CONFIG["naturalness_centroid"] = meta["centroid"]
    elif paired is not None:
        _CONFIG["paths"]["naturalness"] = paired
        _CACHE.pop("naturalness", None)
        _CONFIG["naturalness_centroid"] = meta["centroid"]
    else:
        # Bundled centroid lives in a vector space we can't reach → fall back
        # to reference mode rather than emit nonsense similarities.
        warnings.warn(
            "bundled naturalness centroid is paired with a local NatureEncoder "
            "checkpoint that wasn't found in this environment. Falling back to "
            "naturalness_mode='reference'. Pass `naturalness_centroid=...` or "
            "`paths={'naturalness': ...}` to configure() to keep centroid mode.",
            RuntimeWarning,
        )
        _CONFIG["naturalness_mode"] = "reference"
    _CENTROID_AUTOLOADED = True


def _load_bundled_centroid() -> dict:
    payload = json.loads(
        (Path(__file__).parent / "configs" / "naturalness_centroid_natureencoder.json").read_text()
    )
    payload["centroid"] = np.asarray(payload["centroid"], dtype=np.float32)
    return payload


def _paired_encoder_path(meta: dict) -> Optional[str]:
    raw = str(meta.get("encoder_path", ""))
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return raw if p.exists() else None
    # HF id or relative path — trust it; the caller will get a clean load error if wrong.
    return raw


# ---------------------------------------------------------------------------
# Encoder backbones
# ---------------------------------------------------------------------------

class _STEncoder:
    """sentence-transformers backbone with built-in normalisation."""

    def __init__(self, path: str, device: str, batch_size: int):
        from sentence_transformers import SentenceTransformer
        self.path = path
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(path, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        with torch.no_grad():
            emb = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return emb.float().cpu().numpy()


class _HFBackboneEncoder:
    """AutoModel + masked mean-pool + L2-normalise."""

    def __init__(self, path: str, device: str, batch_size: int, max_length: int = 256):
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.path = path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device)
        self.model.eval()
        self._torch = torch

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        texts = list(texts)
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                chunks.append(pooled.float().cpu().numpy())
        if chunks:
            return np.vstack(chunks)
        # Empty input → preserve (0, hidden_size) so downstream vstack/dot works.
        dim = int(self.model.config.hidden_size)
        return np.zeros((0, dim), dtype=np.float32)


class _CrossEncoderScorer:
    """NLI cross-encoder: P(entailment) for (premise, hypothesis) pairs.

    The pair is encoded jointly through the classification head, so the score
    is directional. Identical pairs are scored once and reused.
    """

    def __init__(self, path: str, device: str, batch_size: int, max_length: int = 256):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        self.model.eval()
        self._torch = torch
        labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.ent_idx = next((i for i, lab in labels.items() if "entail" in lab), None)
        if self.ent_idx is None:
            raise ValueError(f"no 'entailment' class in {path} id2label={labels}")

    def entail(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        torch = self._torch
        pairs = list(zip(map(str, premises), map(str, hypotheses)))
        uniq = list(dict.fromkeys(pairs))
        probs = np.empty(len(uniq), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(uniq), self.batch_size):
                chunk = uniq[i: i + self.batch_size]
                enc = self.tokenizer(
                    [p for p, _ in chunk], [h for _, h in chunk],
                    padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=1)[:, self.ent_idx]
                probs[i: i + len(chunk)] = p.float().cpu().numpy()
        lookup = dict(zip(uniq, probs))
        return np.array([lookup[pr] for pr in pairs], dtype=np.float32)


def _file_hash(path) -> Optional[str]:
    """Fingerprint of a local checkpoint directory (names + sizes, skips weights)."""
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        if f.is_file() and f.stat().st_size < 64 * 1024 * 1024:
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]
