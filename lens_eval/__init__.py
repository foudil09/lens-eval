"""lens-eval: interpretable multi-dimension text quality scoring.

Public entrypoints:

    from lens_eval import LENS
    from lens_eval import semantic_score, nli_score, naturalness_score, emotion_score
    from lens_eval import configure   # device, paths, naturalness mode/centroid

See README.md for usage.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .encoders import (
    DIMENSIONS,
    configure,
    emotion_score,
    featurize,
    free,
    naturalness_score,
    nli_score,
    semantic_score,
)
from .errors import (
    AmbiguousTaskError,
    CombinerBackendMissing,
    DegenerateTargetError,
    EncoderVersionMismatchError,
    InsufficientDataError,
    LensEvalError,
    ReferenceModeError,
)
from .lens import LENS

__all__ = [
    "LENS",
    "DIMENSIONS",
    "configure",
    "featurize",
    "free",
    "semantic_score",
    "nli_score",
    "naturalness_score",
    "emotion_score",
    "LensEvalError",
    "InsufficientDataError",
    "DegenerateTargetError",
    "AmbiguousTaskError",
    "ReferenceModeError",
    "EncoderVersionMismatchError",
    "CombinerBackendMissing",
    "__version__",
]
