"""The music theory the composer works inside.

Nothing here decides anything on its own — it offers legal moves, and
``compose.py`` picks among them with the seeded RNG. That split is the whole
design: theory is the environment, the seed is a walk through it. Because
every move offered is already idiomatic, the composer never has to generate
garbage and filter it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from compex import rng

#: Semitone offsets from the tonic.
SCALES: dict[str, tuple[int, ...]] = {
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "hirajoshi": (0, 2, 3, 7, 8),
    "in_sen": (0, 1, 5, 7, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "phrygian_dominant": (0, 1, 4, 5, 7, 8, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "octatonic": (0, 2, 3, 5, 6, 8, 9, 11),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
}

#: Where each scale sits in (brightness, tension) — used to match a mood.
SCALE_COLOUR: dict[str, tuple[float, float]] = {
    "lydian": (0.96, 0.30),
    "ionian": (0.86, 0.14),
    "major_pentatonic": (0.80, 0.10),
    "mixolydian": (0.70, 0.26),
    "melodic_minor": (0.48, 0.50),
    "dorian": (0.50, 0.30),
    "minor_pentatonic": (0.36, 0.24),
    "aeolian": (0.30, 0.36),
    "hirajoshi": (0.28, 0.46),
    "in_sen": (0.22, 0.52),
    "harmonic_minor": (0.25, 0.60),
    "phrygian": (0.15, 0.62),
    "phrygian_dominant": (0.20, 0.70),
    "whole_tone": (0.55, 0.76),
    "octatonic": (0.34, 0.82),
    "locrian": (0.07, 0.90),
}

#: Root motion between chords, with weights. Fifths dominate, as they should.
ROOT_MOVES: tuple[tuple[int, float], ...] = (
    (3, 1.00),    # up a fourth / down a fifth
    (-3, 0.80),
    (4, 0.45),
    (-4, 0.45),
    (1, 0.55),
    (-1, 0.50),
    (2, 0.32),
    (-2, 0.32),
    (5, 0.18),
    (-5, 0.18),
)

#: Note values the composer builds rhythm from, with weights at low energy.
DURATIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

TRANSFORMS: tuple[str, ...] = (
    "transpose", "invert", "retrograde", "augment",
    "diminish", "rotate", "fragment", "extend",
)

A4_MIDI = 69
A4_HZ = 440.0


@dataclass(frozen=True)
class Chord:
    """A chord as a scale degree plus a stack height — mode-relative, not absolute."""

    degree: int
    size: int

    def semitones(self, scale: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(degree_semitone(scale, self.degree + 2 * step) for step in range(self.size))


@dataclass(frozen=True)
class Motif:
    """A germ idea: scale-degree steps and the rhythm they are spoken in."""

    steps: tuple[int, ...]
    rhythm: tuple[float, ...]

    @property
    def beats(self) -> float:
        return sum(self.rhythm)

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.rhythm):
            raise ValueError(
                f"motif has {len(self.steps)} steps but {len(self.rhythm)} durations"
            )
        if not self.steps:
            raise ValueError("motif cannot be empty")


def degree_semitone(scale: tuple[int, ...], degree: int) -> int:
    """Scale degree to semitones above the tonic, wrapping octaves in both directions."""
    if not scale:
        raise ValueError("scale cannot be empty")
    octave, index = divmod(degree, len(scale))
    return scale[index] + 12 * octave


def midi_to_hz(pitch: float) -> float:
    return A4_HZ * (2.0 ** ((pitch - A4_MIDI) / 12.0))


def pick_scale(seed: int, valence: float, tension: float, stream: str = "scale") -> str:
    """Choose a mode whose colour is near the mood, with room to surprise.

    The three nearest candidates go into the hat rather than just the winner,
    so two runs at the same mood do not always land on the same mode.
    """
    ranked = sorted(
        SCALE_COLOUR,
        key=lambda name: (SCALE_COLOUR[name][0] - valence) ** 2
        + (SCALE_COLOUR[name][1] - tension) ** 2,
    )
    return rng.pick(seed, stream, 0, ranked[:3])


def chord_sizes(tension: float) -> tuple[int, ...]:
    """Stack heights available at this tension — triads always, extensions as it rises."""
    sizes = [3, 3, 3]
    if tension > 0.35:
        sizes.append(4)
    if tension > 0.60:
        sizes.extend((4, 5))
    if tension > 0.82:
        sizes.append(5)
    return tuple(sizes)


def progression(seed: int, scale: tuple[int, ...], length: int, tension: float,
                stream: str = "harmony") -> tuple[Chord, ...]:
    """Walk the scale by weighted root motion. Returns at least one chord."""
    if length < 1:
        raise ValueError(f"progression length must be >= 1, got {length}")

    sizes = chord_sizes(tension)
    span = len(scale)
    chords: list[Chord] = [Chord(degree=0, size=rng.pick(seed, f"{stream}-size", 0, sizes))]

    degree = 0
    for step in range(1, length):
        degree = (degree + _weighted_move(seed, f"{stream}-move", step, tension)) % span
        chords.append(Chord(degree=degree, size=rng.pick(seed, f"{stream}-size", step, sizes)))

    # Land somewhere that feels like an ending unless the mood wants it left open.
    if tension < 0.62 and len(chords) > 2:
        chords[-1] = Chord(degree=0, size=chords[-1].size)
    return tuple(chords)


def _weighted_move(seed: int, stream: str, index: int, tension: float) -> int:
    """Pick a root motion. High tension flattens the weights toward the odd ones."""
    flatten = min(0.85, tension)
    weights = [(move, weight ** (1.0 - flatten)) for move, weight in ROOT_MOVES]
    total = sum(weight for _, weight in weights)
    target = rng.uniform(seed, stream, index) * total
    running = 0.0
    for move, weight in weights:
        running += weight
        if target <= running:
            return move
    return weights[-1][0]


def make_motif(seed: int, scale: tuple[int, ...], energy: float, tension: float,
               stream: str = "motif") -> Motif:
    """Invent a germ idea: a short contour with a rhythm that matches the energy."""
    length = 3 + int(rng.uniform(seed, f"{stream}-len", 0) * (3 + round(energy * 3)))
    reach = 2 + int(tension * 4)

    steps: list[int] = [0]
    for index in range(1, length):
        leap = int(rng.between(seed, f"{stream}-step", index, -reach, reach + 1))
        # Small motion is the norm; big leaps happen but stay rare at low tension.
        if abs(leap) > 2 and rng.uniform(seed, f"{stream}-tame", index) > 0.25 + tension * 0.5:
            leap = 1 if leap > 0 else -1
        steps.append(max(-len(scale), min(len(scale), steps[-1] + leap)))

    pool = _duration_pool(energy)
    rhythm = tuple(rng.pick(seed, f"{stream}-rhythm", index, pool) for index in range(length))
    return Motif(steps=tuple(steps), rhythm=rhythm)


def _duration_pool(energy: float) -> tuple[float, ...]:
    """Faster values dominate as energy climbs; slow values never vanish entirely."""
    pool: list[float] = []
    for duration in DURATIONS:
        weight = 1.0 - abs(_ideal_duration(energy) - duration) / 4.0
        for _ in range(max(1, int(round(weight * 5)))):
            pool.append(duration)
    return tuple(pool)


def _ideal_duration(energy: float) -> float:
    return 3.0 - 2.75 * max(0.0, min(1.0, energy))


def develop(motif: Motif, seed: int, index: int, tension: float,
            stream: str = "develop",
            fatigue: tuple[tuple[str, float], ...] = ()) -> tuple[Motif, str]:
    """Return a variation of ``motif`` — Schoenberg's developing variation, mechanised.

    Each movement transforms what came before rather than starting fresh, so
    the piece grows out of one idea instead of collecting unrelated ones.

    ``fatigue`` down-weights transforms used recently. Without it the same
    seed reaches for the same operation every time and the piece develops in
    one direction until it falls off the edge.
    """
    kind = _pick_transform(seed, f"{stream}-kind", index, fatigue)
    steps, rhythm = list(motif.steps), list(motif.rhythm)

    if kind == "transpose":
        shift = int(rng.between(seed, f"{stream}-amount", index, -3, 4)) or 2
        steps = [step + shift for step in steps]
    elif kind == "invert":
        pivot = steps[0]
        steps = [2 * pivot - step for step in steps]
    elif kind == "retrograde":
        steps, rhythm = steps[::-1], rhythm[::-1]
    elif kind == "augment":
        rhythm = [min(4.0, value * 2.0) for value in rhythm]
    elif kind == "diminish":
        rhythm = [max(0.25, value / 2.0) for value in rhythm]
    elif kind == "rotate":
        turn = 1 + int(rng.uniform(seed, f"{stream}-turn", index) * (len(steps) - 1))
        steps, rhythm = steps[turn:] + steps[:turn], rhythm[turn:] + rhythm[:turn]
    elif kind == "fragment" and len(steps) > 2:
        keep = 2 + int(rng.uniform(seed, f"{stream}-keep", index) * (len(steps) - 2))
        steps, rhythm = steps[:keep], rhythm[:keep]
    elif kind == "extend":
        tail = int(rng.between(seed, f"{stream}-tail", index, -2, 3))
        steps.append(steps[-1] + tail)
        rhythm.append(rng.pick(seed, f"{stream}-tailr", index, (0.5, 1.0, 1.5, 2.0)))

    if tension > 0.7 and rng.uniform(seed, f"{stream}-bite", index) > 0.6:
        steps = [step + (1 if step % 2 else -1) for step in steps]

    return Motif(steps=tuple(steps), rhythm=tuple(rhythm)), kind


def _pick_transform(seed: int, stream: str, index: int,
                    fatigue: tuple[tuple[str, float], ...]) -> str:
    """Weighted draw over the transforms, avoiding whatever was just used."""
    tired = dict(fatigue)
    weights = [(kind, 1.0 / (1.0 + tired.get(kind, 0.0) * 2.2)) for kind in TRANSFORMS]
    total = sum(weight for _, weight in weights)
    target = rng.uniform(seed, stream, index) * total
    running = 0.0
    for kind, weight in weights:
        running += weight
        if target <= running:
            return kind
    return TRANSFORMS[-1]
