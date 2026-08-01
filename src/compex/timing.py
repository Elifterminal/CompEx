"""Rhythmic scaffolding shared by the composer and the arranger."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Comb:
    """A Dirac comb — ``sum_n delta(t - (period*n + offset))`` — measured in beats.

    The composer invents one per drum; it is the cleanest description of a
    repeating pulse, and it writes back out to notation unchanged.
    """

    name: str
    period_beats: float
    offset_beats: float

    def hits(self, start_beat: float, end_beat: float) -> tuple[float, ...]:
        """Every impulse falling in ``[start_beat, end_beat)``."""
        if self.period_beats <= 0:
            raise ValueError(f"comb {self.name!r} has non-positive period {self.period_beats}")
        n = ceil((start_beat - self.offset_beats) / self.period_beats)
        out: list[float] = []
        while True:
            beat = self.offset_beats + n * self.period_beats
            if beat >= end_beat:
                return tuple(out)
            if beat >= start_beat:
                out.append(beat)
            n += 1
