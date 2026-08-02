"""Lossy memory of the germ motif.

The composer does not keep the theme. It keeps a **sketch** of it — the
direction each step moved, roughly how long each note was, and how far the
widest leap reached. Everything else is thrown away.

So when the motif comes back, it cannot be retrieved. It has to be
*reconstructed*, and the gaps get filled using whatever the composer's drives
have become by then. A theme returning at minute four is the theme as
remembered by something that has changed since it first played it.

That is the point. Literal recapitulation is a copy; this is a memory.
"""

from __future__ import annotations

from dataclasses import dataclass

from compex import rng
from compex.generate.evolve import Drives
from compex.generate.theory import Motif

#: Coarse duration classes. Everything inside a class is indistinguishable once stored.
RHYTHM_CLASSES: tuple[tuple[float, ...], ...] = (
    (0.25, 0.25, 0.5),        # short
    (0.5, 0.75, 1.0),         # medium
    (1.5, 2.0, 3.0, 4.0),     # long
)


@dataclass(frozen=True)
class Sketch:
    """What survives of a motif in memory."""

    contour: tuple[int, ...]        # -1 / 0 / +1 per step
    rhythm_class: tuple[int, ...]   # index into RHYTHM_CLASSES
    reach: int                      # widest leap it remembers making
    original_length: int

    @property
    def bits(self) -> int:
        """Roughly how much was kept — two bits of direction, two of duration, per note."""
        return len(self.contour) * 4 + 6

    def fidelity_against(self, motif: Motif) -> float:
        """Share of the original's information the sketch still carries."""
        exact = len(motif.steps) * 12  # a rough count of what an exact copy would need
        return min(1.0, self.bits / max(exact, 1))


def compress(motif: Motif) -> Sketch:
    """Throw the motif away and keep only its shape."""
    steps = [b - a for a, b in zip(motif.steps, motif.steps[1:])]
    contour = tuple(1 if step > 0 else -1 if step < 0 else 0 for step in steps)
    reach = max((abs(step) for step in steps), default=1)
    return Sketch(
        contour=contour,
        rhythm_class=tuple(_class_of(value) for value in motif.rhythm),
        reach=max(1, int(reach)),
        original_length=len(motif.steps),
    )


def remember(sketch: Sketch, seed: int, index: int, drives: Drives) -> Motif:
    """Rebuild a motif from its sketch, filling the gaps with who the composer is now.

    Two drives do the filling. ``register_reach`` decides how far a remembered
    leap actually goes, and ``gap_fill`` decides whether a leap gets answered by
    a step the other way — so a composer that has learned to answer its leaps
    will remember the theme as better behaved than it was.
    """
    if not sketch.contour:
        return Motif(steps=(0,), rhythm=(1.0,))

    steps: list[int] = [0]
    previous = 0
    for position, direction in enumerate(sketch.contour):
        if direction == 0:
            steps.append(steps[-1])
            previous = 0
            continue

        span = 1 + drives.register_reach * (sketch.reach - 1)
        size = 1 + int(rng.uniform(seed, f"recall-size-{index}", position) * span)

        answering = abs(previous) > 2 and rng.uniform(
            seed, f"recall-answer-{index}", position) < drives.gap_fill
        if answering:
            direction = -1 if previous > 0 else 1
            size = 1

        move = direction * size
        steps.append(steps[-1] + move)
        previous = move

    rhythm = tuple(
        _value_in_class(sketch.rhythm_class[position], seed, index, position, drives)
        for position in range(len(sketch.rhythm_class))
    )
    # The sketch stores one duration per original note; the rebuilt contour may
    # be a different length, so trim or extend to match rather than crashing.
    rhythm = _fit(rhythm, len(steps), seed, index, drives)
    return Motif(steps=tuple(steps), rhythm=rhythm)


def _class_of(duration: float) -> int:
    if duration <= 0.5:
        return 0
    if duration <= 1.0:
        return 1
    return 2


def _value_in_class(index_of_class: int, seed: int, index: int, position: int,
                    drives: Drives) -> float:
    """Pick a duration inside the remembered class — denser drives pick shorter ones."""
    options = RHYTHM_CLASSES[min(index_of_class, len(RHYTHM_CLASSES) - 1)]
    pull = min(0.999, max(0.0, (drives.density_bias - 0.35) / 1.55))
    draw = rng.uniform(seed, f"recall-rhythm-{index}", position)
    # Bias the draw toward the short end of the class as density rises.
    biased = draw * (1.0 - pull * 0.7)
    return options[min(len(options) - 1, int(biased * len(options)))]


def _fit(rhythm: tuple[float, ...], wanted: int, seed: int, index: int,
         drives: Drives) -> tuple[float, ...]:
    if len(rhythm) == wanted:
        return rhythm
    if len(rhythm) > wanted:
        return rhythm[:wanted]
    out = list(rhythm)
    while len(out) < wanted:
        out.append(_value_in_class(1, seed, index, len(out), drives))
    return tuple(out)
