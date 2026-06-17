"""Centralized structural validation for fit-time inputs.

Both ``LENS.fit`` and the CLI route their inputs through this module so a
missing ``groups`` array, an ambiguous task channel, or a malformed pairs
matrix surfaces the *same* typed exception with the *same* error message
regardless of where the call originated. The gateway is intentionally
narrow — it does shape and channel checks only, no semantic validation.
That belongs in the combiner layer.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .errors import AmbiguousTaskError


TASK_CHANNELS = ("scores", "pairs", "ranks")
TASK_TO_NAME = {"scores": "regression", "pairs": "pairwise", "ranks": "ranking"}


def validate_task_channels(
    *,
    scores: Optional[Any] = None,
    pairs:  Optional[Any] = None,
    ranks:  Optional[Any] = None,
    groups: Optional[Any] = None,
) -> Tuple[str, str]:
    """Resolve the task channel and validate channel-specific structure.

    Returns ``(channel_name, task)`` where:
      * ``channel_name`` is one of ``"scores" | "pairs" | "ranks"``.
      * ``task``         is the inferred ``LENS`` task —
                         ``"regression" | "pairwise" | "ranking"``.

    Raises ``AmbiguousTaskError`` if zero or multiple channels are supplied.
    Raises ``ValueError`` for shape / dtype / coverage issues with the chosen
    channel (e.g. pairs must be ``(M, 2)``, ranking requires ``groups``).

    Used by both ``LENS.fit`` and the CLI so the user-visible failure modes
    are identical across entry points.
    """
    provided = [k for k, v in (("scores", scores), ("pairs", pairs), ("ranks", ranks))
                if v is not None]
    if len(provided) != 1:
        # Zero ⇒ nothing to fit on. Two+ ⇒ which to optimise? Both are bad.
        raise AmbiguousTaskError(
            f"fit() / CLI needs exactly one of (scores, pairs, ranks); got {provided!r}"
        )
    channel = provided[0]
    task = TASK_TO_NAME[channel]

    if channel == "pairs":
        arr = np.asarray(pairs)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"`pairs` must have shape (M, 2) of (winner_idx, loser_idx); got {arr.shape}"
            )
        if arr.size and not np.issubdtype(arr.dtype, np.integer):
            # Allow float-typed indices that round to int, but reject strings /
            # floats with fractional parts — those are user-error.
            if not np.allclose(arr, np.round(arr)):
                raise ValueError("`pairs` indices must be integer-valued.")
    elif channel == "ranks":
        # Ranking is the only channel that needs an auxiliary grouping array —
        # lambdarank needs to know which rows belong to the same query. Without
        # groups every row is treated as its own group, which silently
        # collapses to "fit a regressor on rank labels" rather than learning
        # within-query orderings. Reject upfront.
        if groups is None:
            raise ValueError(
                "task='ranking' requires `groups` so lambdarank can isolate "
                "within-query orderings — pass a per-row group-id array."
            )
        ranks_arr = np.asarray(ranks)
        groups_arr = np.asarray(groups)
        if ranks_arr.shape[0] != groups_arr.shape[0]:
            raise ValueError(
                f"`ranks` ({ranks_arr.shape[0]}) and `groups` ({groups_arr.shape[0]}) "
                f"must align row-wise."
            )

    return channel, task
