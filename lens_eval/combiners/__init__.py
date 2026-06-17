"""Combiner registry.

Each combiner implements the BaseCombiner protocol. The `AVAILABLE` dict maps
combiner type names → (factory, is_backend_available) so the selection layer can
filter to what's actually installable without having to import each module
eagerly.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from .base import BaseCombiner, expand_pairs
from .glm import GLMCombiner
from .glm_interactions import GLMInteractionsCombiner


def _ebm_available() -> bool:
    # ImportError → package missing. OSError → native lib missing (e.g. some
    # interpret-ml builds without their native deps). Either way: not usable.
    try:
        import interpret.glassbox  # noqa: F401
    except Exception:
        return False
    return True


def _gbm_available() -> bool:
    # lightgbm's ctypes load can OSError on macOS without libomp; treat as
    # "not installed" rather than crashing the candidate filter.
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return False
    return True


def _ebm_factory(**kwargs) -> BaseCombiner:
    # Factory wraps the import so AVAILABLE can be built without importing
    # interpret-ml — the module is only touched when the user actually
    # picks this combiner.
    from .ebm import EBMCombiner
    return EBMCombiner(**kwargs)


def _gbm_factory(**kwargs) -> BaseCombiner:
    # Same deferred-import pattern as the EBM factory.
    from .gbm import GBMCombiner
    return GBMCombiner(**kwargs)


# Registry: name → (factory, availability_check). The factory is called only
# when this combiner wins selection; the check runs during candidate-filter
# without any heavy imports.
AVAILABLE: Dict[str, Tuple[Callable[..., BaseCombiner], Callable[[], bool]]] = {
    "glm":              (lambda **kw: GLMCombiner(**kw),             lambda: True),
    "glm_interactions": (lambda **kw: GLMInteractionsCombiner(**kw), lambda: True),
    "ebm":              (_ebm_factory,                                _ebm_available),
    "gbm":              (_gbm_factory,                                _gbm_available),
}

# Capacity ordering: simpler first. Used by the 1-SE selection rule to break
# ties toward lower-capacity (more interpretable, less prone to overfit).
CAPACITY_ORDER: Tuple[str, ...] = ("glm", "glm_interactions", "ebm", "gbm")


def make(combiner_type: str, **kwargs) -> BaseCombiner:
    """Build a combiner by name."""
    if combiner_type not in AVAILABLE:
        raise ValueError(
            f"unknown combiner {combiner_type!r}; choose from {list(AVAILABLE)}"
        )
    factory, check = AVAILABLE[combiner_type]
    # Double-check at construction time — selection layer should have
    # filtered already, but guard against a direct `make()` call too.
    if not check():
        from ..errors import CombinerBackendMissing
        raise CombinerBackendMissing(
            f"combiner {combiner_type!r} requires an optional dependency. "
            f"Install with: pip install 'lens-eval[{combiner_type}]'"
        )
    return factory(**kwargs)


__all__ = [
    "BaseCombiner",
    "GLMCombiner",
    "GLMInteractionsCombiner",
    "AVAILABLE",
    "CAPACITY_ORDER",
    "expand_pairs",
    "make",
]
