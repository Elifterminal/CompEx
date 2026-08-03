"""More than one clock, because nothing here has to keep time with a hand.

Every voice in this engine has always been locked to one tempo grid, which is
a habit inherited from people: an ensemble shares a pulse because humans
cannot reliably hold two at once. A machine can hold as many as it likes, and
scheduling them costs nothing when time is addressed positionally rather than
sequentially — `f(seed, stream, index)` does not care which grid the index is
counted on.

So voices can run at rational multiples of the base tempo: three against two,
five against four, seven against four. Rational rather than arbitrary, for one
reason worth stating — **irrational ratios never meet again**. A 3:2 voice
realigns with the pulse every two bars, and that returning coincidence is the
thing the ear latches onto. Without it you have not written polytempo, you
have written drift.

One anchor stays at 1. You can only hear three-against-two if something is
being the two, and the kick and the bass are the parts a listener uses as the
floor. That is not a concession to human hearing so much as a fact about what
makes a ratio audible at all: a ratio needs both of its terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from compex import rng
from compex.generate.mood import Mood
from compex.generate.palette import ROLE_BASS, ROLE_PERC, VoiceSpec

#: The ratios on offer, as (numerator, denominator) against the base pulse.
#: Kept small-integer so that the voices meet again inside a piece rather than
#: in some theoretical future.
RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 1), (1, 1),          # most voices stay on the pulse
    (3, 2), (2, 3), (4, 3), (3, 4),
    (5, 4), (4, 5), (5, 3), (7, 4),
)

ANCHORS = ("kick", "boom")   # drums that hold the floor along with the bass
ENABLED = True               # the whole mechanism, off


@dataclass(frozen=True)
class Clock:
    """One voice's own tempo, as a ratio against the pulse."""

    voice: str
    numerator: int = 1
    denominator: int = 1

    @property
    def ratio(self) -> float:
        return self.numerator / self.denominator

    @property
    def anchored(self) -> bool:
        return self.numerator == self.denominator

    def meets_every(self, beats_per_bar: float) -> float:
        """How often this voice and the pulse land together again, in beats.

        A cycle of the voice's grid is ``denominator/numerator`` of a beat, so
        the two coincide at the least common multiple — which for small
        integers is a handful of bars rather than never.
        """
        if self.anchored:
            return beats_per_bar
        return float(self.denominator * beats_per_bar / gcd(self.numerator, self.denominator))

    def describe(self) -> str:
        if self.anchored:
            return f"{self.voice} on the pulse"
        return (f"{self.voice} at {self.numerator}:{self.denominator} "
                f"({self.ratio:.2f}x the pulse)")


def assign(seed: int, voices: list[VoiceSpec], kit: tuple[VoiceSpec, ...],
           mood: Mood) -> dict[str, Clock]:
    """Give some voices their own tempo. How many depends on the mood.

    Tension and energy buy the right to leave the pulse; a serene piece keeps
    everything together because a serene piece that does not is simply
    confusing rather than serene.
    """
    appetite = 0.05 + mood.tension * 0.55 + mood.energy * 0.25
    clocks: dict[str, Clock] = {}

    for index, voice in enumerate(list(voices) + list(kit)):
        anchored = (voice.role == ROLE_BASS
                    or (voice.role == ROLE_PERC and voice.voice_id in ANCHORS))
        if not ENABLED or anchored or rng.uniform(seed, "clock-gate", index) > appetite:
            clocks[voice.voice_id] = Clock(voice.voice_id)
            continue
        numerator, denominator = rng.pick(seed, "clock-ratio", index, RATIOS)
        clocks[voice.voice_id] = Clock(voice.voice_id, int(numerator), int(denominator))

    return clocks


def loose(clocks: dict[str, Clock]) -> tuple[Clock, ...]:
    """The voices that are not on the pulse."""
    return tuple(clock for clock in clocks.values() if not clock.anchored)


def summary(clocks: dict[str, Clock], beats_per_bar: float) -> str:
    off = loose(clocks)
    if not off:
        return "one clock, everything on it"
    parts = ", ".join(f"{clock.voice} {clock.numerator}:{clock.denominator} "
                      f"(meets every {clock.meets_every(beats_per_bar):g} beats)"
                      for clock in off)
    return f"{len(off)} of {len(clocks)} voices on their own clock — {parts}"
