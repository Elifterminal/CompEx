"""The written material itself: movements, notes, strokes.

Kept apart from the composer so the critic can read what was written without
importing the thing that writes it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Movement:
    """One span of the invented form."""

    name: str
    beats: float
    energy: float
    tension: float


@dataclass(frozen=True)
class Note:
    start: float       # beats from the top
    duration: float    # beats
    pitch: float       # MIDI note number
    velocity: float
    voice: str
    #: Which state of that voice played it. Instruments drift as the piece
    #: goes, so "which voice" is no longer enough to know what it sounded like.
    timbre: int = 0


@dataclass(frozen=True)
class Stroke:
    """A percussion hit."""

    start: float
    velocity: float
    voice: str
