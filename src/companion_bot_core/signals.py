"""Shared regex signal scoring used by behavior detection and policy guardrails."""

from __future__ import annotations

import re
from typing import NamedTuple


class Signal(NamedTuple):
    """A compiled regex pattern paired with its confidence weight."""

    pattern: re.Pattern[str]
    weight: float


def compile_signals(
    patterns: list[tuple[str, float]],
    *,
    dotall: bool = False,
) -> list[Signal]:
    """Compile *patterns* into :class:`Signal` objects.

    Args:
        patterns: List of ``(regex_string, weight)`` tuples.  Each pattern
            is compiled with ``IGNORECASE`` and, when *dotall* is ``True``,
            ``DOTALL`` so ``.`` also matches newlines.
        dotall: When ``True`` add ``re.DOTALL`` to the flags.  Defaults to
            ``False`` to avoid cross-line matches in security-critical patterns
            where ``.{0,N}`` between keywords could produce false positives on
            innocent multiline messages.

    Returns:
        List of compiled :class:`Signal` objects.
    """
    flags = re.IGNORECASE | (re.DOTALL if dotall else 0)
    return [Signal(re.compile(p, flags), w) for p, w in patterns]


def score_signals(text: str, signals: list[Signal]) -> float:
    """Return the sum of weights for all matching signals, capped at 1.0."""
    return min(
        sum(sig.weight for sig in signals if sig.pattern.search(text)),
        1.0,
    )
