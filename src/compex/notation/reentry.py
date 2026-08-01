"""Read an emitted formula back to the three inputs that produced it.

Composition is deterministic, so ``SEED``, ``RUNTIME`` and ``MOOD`` are the
whole story — hand those back to the composer and the same piece comes out.
That makes the emitted formula genuinely re-entrant rather than decorative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from compex.generate.mood import AXES, Mood
from compex.notation import latex

_SEED = re.compile(r"\bSEED\s*=\s*(\d+)")
_RUNTIME = re.compile(r"\bRUNTIME\s*=\s*([\d.]+)")
_MOOD_BLOCK = re.compile(r"\bMOOD\s*=\s*\(([^)]*)\)")
_AXIS = re.compile(r"\b([vetdg])\s*([01]?\.?\d+)")

#: First letter of each axis, as written by the emitter.
_INITIALS = {axis[0]: axis for axis in AXES}


class ReentryError(ValueError):
    """Raised when a formula lacks the header needed to reproduce its track."""


@dataclass(frozen=True)
class Formula:
    seed: int
    runtime_s: float
    mood: Mood


def read_formula(text: str) -> Formula:
    """Extract seed, runtime and mood from emitted notation."""
    if not text or not text.strip():
        raise ReentryError("formula is empty")

    plain = latex.normalise(text)

    seed_match = _SEED.search(plain)
    if not seed_match:
        raise ReentryError("no SEED line — this formula cannot reproduce its track")
    runtime_match = _RUNTIME.search(plain)
    if not runtime_match:
        raise ReentryError("no RUNTIME line")

    mood_match = _MOOD_BLOCK.search(plain)
    if not mood_match:
        raise ReentryError("no MOOD line")

    axes: dict[str, float] = {}
    for initial, value in _AXIS.findall(mood_match.group(1)):
        axis = _INITIALS.get(initial)
        if axis:
            axes[axis] = float(value)

    missing = set(AXES) - set(axes)
    if missing:
        raise ReentryError(f"MOOD is missing {', '.join(sorted(missing))}")

    runtime = float(runtime_match.group(1))
    if runtime <= 0:
        raise ReentryError(f"RUNTIME must be positive, got {runtime}")

    return Formula(seed=int(seed_match.group(1)), runtime_s=runtime, mood=Mood.parse(axes))
