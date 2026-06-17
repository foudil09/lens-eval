"""Custom exceptions for lens-eval. All errors carry actionable messages."""

from __future__ import annotations


# Single base so user code can do `except LensEvalError` to catch any
# library-raised error without also swallowing built-in exceptions.
class LensEvalError(Exception):
    """Base class for all lens-eval errors."""


class InsufficientDataError(LensEvalError):
    """Raised when the supplied dataset is too small to fit any combiner safely."""


class DegenerateTargetError(LensEvalError):
    """Raised when the target has zero variance (single value)."""


class AmbiguousTaskError(LensEvalError):
    """Raised when more than one of (scores, pairs, ranks) is supplied to fit()."""


class ReferenceModeError(LensEvalError):
    """Raised when score/predict-time reference mode disagrees with training-time."""


class EncoderVersionMismatchError(LensEvalError):
    """Raised by LENS.load() when the saved encoder hash doesn't match the runtime encoders."""


class CombinerBackendMissing(LensEvalError):
    """Raised when a combiner is requested that needs an optional dependency not installed."""
