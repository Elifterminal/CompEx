"""What the engine carries from one piece to the next.

Until now every piece began from nothing. The composer learned inside a track —
its taste moved, its grids learned, its bounds gave way — and then the whole
thing was thrown away and the next piece started from the same mood-derived
defaults as the first. It has made hundreds and remembered none of them.

A machine has no excuse for that. It has perfect recall of everything it has
ever done, which is one of the few advantages it holds over a person outright,
and the obvious thing to do with it is the thing a person does badly: **get
bored of itself**. An engine that reaches for the same five instruments and the
same tuning every time is not being consistent, it is being stuck.

So a `Memory` carries three things forward: where taste ended up, what has been
used lately, and how many pieces there have been. It biases the next piece and
nothing more — it never overrides the mood, because a serene piece must still
be able to come out serene on the four hundredth attempt.

**Determinism survives because the memory is an explicit input.** Same seed,
same runtime, same mood, same memory digest gives the same audio, and that
digest is printed in the formula next to the seed. A hidden accumulator would
have quietly broken the one property this whole project rests on; a visible one
is just another argument.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

#: How strongly the past pulls on a new piece's starting taste. Well under a
#: half on purpose: this is a bias, not an inheritance, and a piece that cannot
#: escape its predecessors is not a new piece.
PULL = 0.35

#: How much having used something recently counts against using it again.
BOREDOM = 0.6

#: How fast the past fades. Each piece multiplies what came before by this, so
#: the last handful matter and the first hundred are a rumour.
DECAY = 0.72

#: Ignore anything that has decayed below this — otherwise the counters fill up
#: with ten thousand near-zero entries and the digest never stabilises.
FLOOR = 0.02

#: Decimal places every remembered weight is held at, in memory and on disk.
#: These must be the same number. They were not, briefly: values were kept at
#: six places and written at five, so saving a history and loading it back gave
#: a *different* memory that reported the *same* digest — the formula would
#: have sworn two different pieces were the same one. Rounding at the point of
#: creation is what makes the file and the object the same thing.
PRECISION = 5


@dataclass(frozen=True)
class Memory:
    """Everything the engine remembers about the pieces it has already made."""

    pieces: int = 0
    taste: tuple[tuple[str, float], ...] = ()      # where melodic taste tends to end up
    groove: tuple[tuple[str, float], ...] = ()     # and rhythmic taste
    engines: tuple[tuple[str, float], ...] = ()    # what it has been reaching for
    tunings: tuple[tuple[str, float], ...] = ()    # and which families of tuning
    themes: tuple[tuple[str, float], ...] = ()

    def is_blank(self) -> bool:
        return self.pieces == 0

    def weight(self, table: str, name: str) -> float:
        for key, value in getattr(self, table):
            if key == name:
                return value
        return 0.0

    def tired_of(self, table: str, name: str) -> float:
        """How stale something is, 0..1 — its share of everything remembered."""
        entries = getattr(self, table)
        total = sum(value for _, value in entries)
        if total <= 0:
            return 0.0
        return min(1.0, self.weight(table, name) / total * max(1, len(entries)) * 0.5)

    def digest(self) -> str:
        """A short hash of everything remembered.

        This goes in the formula. Two people with the same seed and different
        histories get different music, and this is how they can tell — without
        it, "same inputs, same output" would quietly become a lie.
        """
        payload = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=6).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "pieces": self.pieces,
            "taste": dict(self.taste),
            "groove": dict(self.groove),
            "engines": dict(self.engines),
            "tunings": dict(self.tunings),
            "themes": dict(self.themes),
        }

    def describe(self) -> str:
        if self.is_blank():
            return "nothing remembered — this is the first piece"
        recent = ", ".join(name for name, _ in sorted(
            self.engines, key=lambda pair: -pair[1])[:4])
        return (f"{self.pieces} pieces remembered ({self.digest()}), "
                f"lately reaching for {recent}")

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | str | None) -> "Memory":
        """Build a memory from untrusted input — a file, a browser, an API call."""
        if raw is None or raw == "":
            return cls()
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return cls()
        if not isinstance(raw, Mapping):
            return cls()

        def table(name: str) -> tuple[tuple[str, float], ...]:
            values = raw.get(name) or {}
            if not isinstance(values, Mapping):
                return ()
            out = []
            for key, value in values.items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if number == number and number > FLOOR:   # NaN never survives
                    out.append((str(key), round(number, PRECISION)))
            return tuple(sorted(out))

        try:
            pieces = max(0, int(raw.get("pieces", 0)))
        except (TypeError, ValueError):
            pieces = 0
        return cls(pieces=pieces, taste=table("taste"), groove=table("groove"),
                   engines=table("engines"), tunings=table("tunings"),
                   themes=table("themes"))


def learn(memory: Memory, composition) -> Memory:
    """Fold a finished piece into what is remembered.

    Everything decays first, so the last few pieces weigh more than the first
    few hundred. Without that the engine would become steadily more stuck the
    longer it ran, which is the opposite of the point.
    """
    taste = _fold(memory.taste, dict(composition.taste.weights()))
    groove = _fold(memory.groove, dict(composition.groove.weights()))
    engines = _fold(memory.engines,
                    {voice.engine: 1.0 for voice in composition.voices
                     if voice.role != "perc"})
    tunings = _fold(memory.tunings, {composition.tuning.family: 1.0,
                                     composition.tuning.name: 1.0})
    themes = _fold(memory.themes, {composition.mood.nearest_theme(): 1.0})

    return Memory(pieces=memory.pieces + 1, taste=taste, groove=groove,
                  engines=engines, tunings=tunings, themes=themes)


def _fold(existing: tuple[tuple[str, float], ...],
          arriving: dict[str, float]) -> tuple[tuple[str, float], ...]:
    merged = {name: value * DECAY for name, value in existing}
    for name, value in arriving.items():
        merged[name] = merged.get(name, 0.0) + value
    return tuple(sorted((name, round(value, PRECISION)) for name, value in merged.items()
                        if value > FLOOR))


def lean(memory: Memory, table: str, mood_value: float, name: str) -> float:
    """Pull a starting value toward where past pieces ended up.

    Only for the taste tables, and only part of the way. The mood still decides
    what kind of piece this is; the memory decides what kind of listener is
    writing it.
    """
    if memory.is_blank():
        return mood_value
    remembered = memory.weight(table, name)
    if remembered <= 0:
        return mood_value
    # Remembered weights are sums over decayed pieces, so normalise by what a
    # fully-decayed history of ones would total.
    scale = sum(DECAY ** step for step in range(min(memory.pieces, 24))) or 1.0
    average = remembered / scale
    return mood_value * (1.0 - PULL) + average * PULL


def bored_of(memory: Memory, table: str, name: str) -> float:
    """How much recent use should count against choosing this again, 0..1."""
    return BOREDOM * memory.tired_of(table, name)
