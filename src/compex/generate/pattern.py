"""Rhythms the composer decides on, out of what it has already played.

The drums used to be Dirac combs: one period per voice, drawn from a table of
what a kick usually does. Every kick pattern in every piece was therefore the
same pattern at a different speed, and no amount of listening back could change
that, because there was nothing there to change.

A pattern here is a **grid of slots** — one cycle of the bar at some
subdivision, each slot either struck or not. Three things decide what gets
struck:

* **weights**, one per slot, starting from the metrical hierarchy the meter
  implies (downbeat strongest, then the halves, then the beats, then what is
  left) bent by the character of the voice — a shaker wants the gaps a kick
  wants the downbeat;
* **judgement** — several candidate patterns are drawn and mutated, then scored
  on whether they hold a pulse, syncopate as much as the mood wants, collide
  with what other voices already claimed, and read as a figure rather than a
  metronome;
* **its own output** — after each movement the weights move toward the pattern
  that actually won. Slots it keeps using get heavier, slots it keeps skipping
  fade.

That last one is why the weights are a starting point rather than a rule. They
are allowed past 1.0, past where any metrical hierarchy would put them, if
that is where the piece's own choices keep landing. What they cannot do is
stop meaning anything: there is a hard ceiling, and it is about the pattern
still being playable, not about what was expected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from compex import rng
from compex.generate.critic import Verdict
from compex.generate.evolve import Drives
from compex.generate.mood import Mood

CANDIDATES_MIN, CANDIDATES_MAX = 2, 6
MAX_SLOTS = 32
SUBDIVISIONS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

WEIGHT_FLOOR, WEIGHT_CEILING = 0.02, 1.75   # hard limits. The starting weights sit inside 0..1
LEARN_RATE = 0.30      # how fast a grid moves toward the pattern it just played. 0 freezes it
STRONG = 0.55          # a slot at least this heavy counts as a strong position
TASTE_BOUNDS = (0.05, 3.0)
TASTE_RATE = 0.20

@dataclass(frozen=True)
class Criterion:
    """One thing the composer can listen for in a candidate rhythm."""

    name: str
    why: str


#: Data rather than prose, for the same reason the melodic ones are: the page
#: introspects this, so it cannot describe a criterion nothing scores.
CRITERIA_DETAIL: tuple[Criterion, ...] = (
    Criterion(
        "pulse",
        "Something has to land where the meter says the bar starts, or there is no bar.",
    ),
    Criterion(
        "syncopation",
        "Hits off the strong positions, against how much of that the mood wants. A band, not "
        "a direction — everything off the beat is as unreadable as nothing off it.",
    ),
    Criterion(
        "fill",
        "Strikes per beat against what the movement's energy and the density drive ask for.",
    ),
    Criterion(
        "interlock",
        "How much it collides with what other voices already claimed. A little agreement is "
        "the pocket; total agreement is one thick voice pretending to be a kit.",
    ),
    Criterion(
        "figure",
        "How many different gaps there are between hits. One is a metronome, two or three is "
        "a figure — a clave, a tresillo, most things anyone can whistle.",
    ),
    Criterion(
        "continuity",
        "How close it stays to what this voice was doing before, against a target that loosens "
        "as the composer gets more unsettled.",
    ),
)

CRITERIA: tuple[str, ...] = tuple(criterion.name for criterion in CRITERIA_DETAIL)

#: Which groove criterion each aesthetic principle teaches. Fewer than the
#: melodic ones, because most of the principles are about a line, not a pulse.
TAUGHT_BY: dict[str, tuple[str, int]] = {
    "density": ("fill", +1),
    "repetition": ("continuity", +1),
    "novelty": ("continuity", -1),
}


@dataclass(frozen=True)
class Grid:
    """What one voice currently believes about where its hits belong."""

    voice: str
    subdivision: float          # beats per slot
    weights: tuple[float, ...]
    start: tuple[float, ...]    # the weights it began the piece with

    @property
    def cycle_beats(self) -> float:
        return len(self.weights) * self.subdivision

    def strayed(self) -> float:
        """Mean distance the weights have travelled from where they started."""
        if not self.start:
            return 0.0
        return sum(abs(now - then) for now, then in zip(self.weights, self.start)) / len(self.start)

    def past_start(self) -> int:
        """How many slots now sit outside the 0..1 range the hierarchy gave them."""
        return sum(1 for weight in self.weights if weight > 1.0 + 1e-6)


@dataclass(frozen=True)
class Pattern:
    """One cycle of one voice: a velocity per slot, zero for a rest."""

    voice: str
    slots: tuple[float, ...]
    subdivision: float
    origin: str
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError(f"pattern for {self.voice!r} has no slots")
        if self.subdivision <= 0:
            raise ValueError(f"pattern for {self.voice!r} has subdivision {self.subdivision}")

    @property
    def cycle_beats(self) -> float:
        return len(self.slots) * self.subdivision

    @property
    def hit_count(self) -> int:
        return sum(1 for velocity in self.slots if velocity > 0)

    def onsets(self) -> tuple[tuple[int, float], ...]:
        return tuple((index, velocity)
                     for index, velocity in enumerate(self.slots) if velocity > 0)

    def hits(self, start_beat: float, end_beat: float) -> tuple[tuple[float, float], ...]:
        """Every strike falling in ``[start_beat, end_beat)``, as (beat, velocity).

        The cycle is anchored to the top of the piece rather than to the span
        it is asked about, so a pattern does not restart at every movement
        boundary and the bar line stays where the listener put it.
        """
        cycle = self.cycle_beats
        out: list[tuple[float, float]] = []
        first = int(start_beat // cycle)
        cursor = first * cycle
        while cursor < end_beat:
            for index, velocity in self.onsets():
                beat = cursor + index * self.subdivision
                if start_beat <= beat < end_beat:
                    out.append((beat, velocity))
            cursor += cycle
        return tuple(sorted(out))

    def describe(self) -> str:
        return "".join("x" if velocity > 0.66 else "+" if velocity > 0 else "."
                       for velocity in self.slots)


@dataclass(frozen=True)
class Groove:
    """How much each groove criterion matters. Evolves, same as melodic taste."""

    pulse: float = 1.0
    syncopation: float = 0.8
    fill: float = 1.0
    interlock: float = 0.9
    figure: float = 0.9
    continuity: float = 0.8

    def weights(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in CRITERIA)

    def summary(self) -> str:
        return " · ".join(f"{name} {value:.2f}" for name, value in self.weights())


@dataclass(frozen=True)
class PatternChoice:
    """One pattern audition, kept so the piece can say why it grooves like that."""

    movement: int
    voice: str
    pattern: Pattern
    score: float
    scores: tuple[tuple[str, float], ...]
    rejected: tuple[tuple[str, float], ...]
    considered: int
    strayed: float

    @property
    def margin(self) -> float:
        if not self.rejected:
            return 0.0
        return self.score - max(score for _, score in self.rejected)

    def note(self) -> str:
        return (f"{self.voice}: |{self.pattern.describe()}| "
                f"({self.pattern.origin}, beat {self.pattern.subdivision:g}/slot, "
                f"{self.pattern.hit_count} hits, beat {self.margin:+.3f})")


def initial_groove(mood: Mood) -> Groove:
    """Where the composer's sense of groove starts. Derived, then left to move."""
    return Groove(
        pulse=0.60 + (1.0 - mood.tension) * 0.80,
        syncopation=0.25 + mood.tension * 1.15,
        fill=0.45 + mood.density * 1.00,
        interlock=0.50 + mood.density * 0.85,
        figure=0.45 + mood.energy * 0.80,
        continuity=0.35 + (1.0 - mood.energy) * 1.00,
    )


# ── where the weights come from ───────────────────────────────────────────

def metre_weights(beats_per_bar: int, subdivision: float, bars: int = 1) -> tuple[float, ...]:
    """The metrical hierarchy for this meter, derived rather than tabulated.

    A slot's weight is set by the coarsest level of the cycle that lands on it:
    the downbeat is heaviest, then the halves (or thirds, in a triple meter),
    then the beats, then the subdivisions. This is the standard hierarchy, and
    it falls out of the arithmetic for 7/4 the same way it does for 4/4 —
    which is exactly why it is derived and not typed out per meter.
    """
    slots = max(1, min(MAX_SLOTS, int(round(beats_per_bar * bars / subdivision))))
    per_bar = max(1, slots // max(1, bars))

    weights: list[float] = []
    for index in range(slots):
        within = index % per_bar
        beat, into_beat = divmod(within * subdivision, 1.0)
        if index == 0:
            weight = 1.0                      # the downbeat of the cycle
        elif within == 0:
            weight = 0.86                     # the downbeat of a later bar
        elif into_beat < 1e-9:
            level = _division_depth(int(beat), beats_per_bar)
            weight = 0.72 - 0.11 * level      # on a beat, deeper divisions weaker
        else:
            weight = 0.34 if abs(into_beat - 0.5) < 1e-9 else 0.18
        weights.append(round(max(0.05, weight), 4))
    return tuple(weights)


def _division_depth(beat: int, beats_per_bar: int) -> int:
    """How many halvings (or thirds) of the bar it takes to reach this beat.

    Beat 2 of 4/4 is one halving deep; beats 1 and 3 are two. That ordering is
    what makes a backbeat feel like a backbeat rather than an accident.
    """
    depth = 0
    span = beats_per_bar
    while span > 1:
        split = 2 if span % 2 == 0 else 3 if span % 3 == 0 else span
        step = span // split
        if step and beat % step == 0:
            return depth
        span = step or 1
        depth += 1
    return depth


def build_grid(voice: str, seed: int, index: int, beats_per_bar: int, mood: Mood,
               brightness: float, aggression: float) -> Grid:
    """Give a voice a subdivision and a starting set of weights that suit it.

    Bright voices get a finer grid and want the gaps; dark ones get a coarse
    grid and want the beats. Nothing here is per-instrument — a shaker behaves
    like a shaker because it is bright, not because it is named shaker.
    """
    subdivision = _subdivision(seed, voice, index, brightness, mood)
    bars = 2 if subdivision >= 1.0 and rng.uniform(seed, f"bars-{voice}", index) < 0.45 else 1
    base = metre_weights(beats_per_bar, subdivision, bars)

    # A bright voice trades some of the hierarchy for its inverse: what is weak
    # for the meter is where a hat lives. Aggression sharpens the contrast so a
    # violent kit hits harder in fewer places rather than everywhere at once.
    off = min(1.0, brightness * 0.85)
    contrast = 0.75 + aggression * 0.7
    weights = tuple(
        round(max(WEIGHT_FLOOR, min(1.0, ((1.0 - off) * weight + off * (1.0 - weight)) ** contrast
                                    * (0.55 + mood.density * 0.75))), 4)
        for weight in base
    )
    return Grid(voice=voice, subdivision=subdivision, weights=weights, start=weights)


def _subdivision(seed: int, voice: str, index: int, brightness: float, mood: Mood) -> float:
    """Coarse for low voices, fine for high ones, finer again in a dense mood."""
    ideal = (1.0 - brightness) * 1.6 - mood.density * 0.45 + mood.energy * -0.2
    ranked = sorted(SUBDIVISIONS, key=lambda value: abs(value - max(0.2, ideal)))
    return float(rng.pick(seed, f"subdiv-{voice}", index, ranked[:2]))


# ── inventing and choosing a pattern ──────────────────────────────────────

@dataclass(frozen=True)
class Bed:
    """What a candidate pattern is being judged inside."""

    energy: float
    tension: float
    drives: Drives
    bar_beats: float = 4.0
    claimed: frozenset[float] = frozenset()   # positions in the bar other voices took
    previous: Pattern | None = None


def invent(seed: int, movement: int, index: int, grid: Grid, groove: Groove,
           bed: Bed) -> PatternChoice:
    """Draw several patterns for this voice, judge them, keep the best."""
    candidates = _propose(seed, movement, index, grid, groove, bed)
    scored = [(pattern, criteria, _total(criteria, groove))
              for pattern, criteria in ((p, assess(p, grid, bed)) for p in candidates)]
    winner, criteria, score = max(scored, key=lambda row: row[2])

    return PatternChoice(
        movement=movement, voice=grid.voice, pattern=winner, score=score,
        scores=tuple(sorted(criteria.items())),
        rejected=tuple((pattern.origin, value)
                       for pattern, _, value in scored if pattern is not winner),
        considered=len(scored),
        strayed=grid.strayed(),
    )


def _propose(seed: int, movement: int, index: int, grid: Grid, groove: Groove,
             bed: Bed) -> tuple[Pattern, ...]:
    wanted = max(CANDIDATES_MIN, min(CANDIDATES_MAX,
                                     CANDIDATES_MIN + int(round(bed.drives.unrest * 4.0))))
    stream = f"{grid.voice}-{movement}-{index}"
    appetite = (0.45 + bed.energy * 0.9) * bed.drives.density_bias

    field: list[Pattern] = [
        _draw(seed, f"{stream}-a", grid, appetite),
        _draw(seed, f"{stream}-b", grid, appetite * 0.6),
    ]
    if bed.previous is not None:
        field.append(bed.previous)   # standing pat is always on the table
        for slot in range(wanted - 3):
            field.append(_mutate(bed.previous, seed, f"{stream}-m", slot, grid))
    while len(field) < wanted:
        field.append(_draw(seed, f"{stream}-c{len(field)}", grid, appetite * 1.3))
    return tuple(field[:wanted])


def _draw(seed: int, stream: str, grid: Grid, appetite: float) -> Pattern:
    """Sample the grid: heavier slots get struck more often, and struck harder."""
    slots: list[float] = []
    for index, weight in enumerate(grid.weights):
        if rng.uniform(seed, f"{stream}-hit", index) < min(0.98, weight * appetite):
            velocity = min(1.0, 0.42 + weight * 0.5
                           + rng.between(seed, f"{stream}-vel", index, -0.1, 0.1))
            slots.append(round(max(0.12, velocity), 3))
        else:
            slots.append(0.0)
    if not any(slots):
        # A silent voice is a legal pattern but a useless candidate; anchor it
        # on its own heaviest slot so the field always contains something.
        heaviest = max(range(len(grid.weights)), key=lambda i: grid.weights[i])
        slots[heaviest] = 0.7
    return Pattern(voice=grid.voice, slots=tuple(slots), subdivision=grid.subdivision,
                   origin="drawn", generation=0)


MUTATIONS: tuple[str, ...] = ("shift", "add", "drop", "accent", "double", "rotate", "ghost")


def _mutate(parent: Pattern, seed: int, stream: str, slot: int, grid: Grid) -> Pattern:
    """One change to a pattern that already works. Small, so a groove survives it."""
    kind = rng.pick(seed, f"{stream}-kind", slot, MUTATIONS)
    slots = list(parent.slots)
    onsets = [index for index, velocity in enumerate(slots) if velocity > 0]
    rests = [index for index, velocity in enumerate(slots) if velocity == 0]
    at = int(rng.uniform(seed, f"{stream}-at", slot) * len(slots))

    if kind == "shift" and onsets:
        source = onsets[at % len(onsets)]
        target = (source + (1 if rng.uniform(seed, f"{stream}-dir", slot) < 0.5 else -1)) % len(slots)
        slots[target], slots[source] = slots[source], 0.0
    elif kind == "add" and rests:
        # Add where the grid is heaviest among the free slots, not at random —
        # the point of the weights is that they say where a hit belongs.
        target = max(rests, key=lambda index: grid.weights[index])
        slots[target] = round(min(1.0, 0.4 + grid.weights[target] * 0.5), 3)
    elif kind == "drop" and len(onsets) > 1:
        target = min(onsets, key=lambda index: grid.weights[index])
        slots[target] = 0.0
    elif kind == "accent" and onsets:
        target = onsets[at % len(onsets)]
        slots[target] = round(min(1.0, slots[target] + 0.25), 3)
    elif kind == "double" and onsets:
        source = onsets[at % len(onsets)]
        target = (source + 1) % len(slots)
        if slots[target] == 0:
            slots[target] = round(max(0.12, slots[source] * 0.6), 3)
    elif kind == "rotate":
        turn = 1 + int(rng.uniform(seed, f"{stream}-turn", slot) * max(1, len(slots) - 1))
        slots = slots[turn:] + slots[:turn]
    elif kind == "ghost" and onsets:
        target = onsets[at % len(onsets)]
        slots[target] = round(max(0.12, slots[target] * 0.45), 3)

    if not any(slots):
        slots[0] = 0.6
    return Pattern(voice=parent.voice, slots=tuple(slots), subdivision=parent.subdivision,
                   origin=kind, generation=parent.generation + 1)


def _total(criteria: dict[str, float], groove: Groove) -> float:
    weights = dict(groove.weights())
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return sum(criteria[name] * weights[name] for name in CRITERIA) / total


# ── judging a pattern ─────────────────────────────────────────────────────

def assess(pattern: Pattern, grid: Grid, bed: Bed) -> dict[str, float]:
    """Score a candidate pattern on every criterion, 0..1, 1 being satisfied."""
    return {
        "pulse": _pulse(pattern, grid),
        "syncopation": _syncopation(pattern, grid, bed),
        "fill": _fill(pattern, bed),
        "interlock": _interlock(pattern, bed),
        "figure": _figure(pattern),
        "continuity": _continuity(pattern, bed),
    }


def _pulse(pattern: Pattern, grid: Grid) -> float:
    """Is it anchored? Something has to land where the meter says the bar starts."""
    strong = [index for index, weight in enumerate(grid.weights) if weight >= STRONG]
    if not strong:
        strong = [0]
    landed = sum(1 for index in strong if pattern.slots[index] > 0)
    return min(1.0, landed / max(1, min(len(strong), 2)))


def _syncopation(pattern: Pattern, grid: Grid, bed: Bed) -> float:
    """Hits off the strong positions, against how much of that the mood wants."""
    onsets = pattern.onsets()
    if not onsets:
        return 0.0
    weak = sum(1 for index, _ in onsets if grid.weights[index] < STRONG) / len(onsets)
    want = 0.15 + bed.tension * 0.45 + bed.energy * 0.15
    return max(0.0, 1.0 - abs(weak - want) / max(want, 1e-6))


def _fill(pattern: Pattern, bed: Bed) -> float:
    """Hits per beat, against what the energy and the density drive are asking."""
    rate = pattern.hit_count / max(pattern.cycle_beats, 1e-6)
    want = (0.25 + bed.energy * 1.6) * bed.drives.density_bias
    return max(0.0, 1.0 - abs(rate - want) / max(want, 1e-6))


def _interlock(pattern: Pattern, bed: Bed) -> float:
    """Does it leave room for what is already playing?

    Everything landing on the same slots is one thick voice rather than a kit.
    A little agreement is the pocket; total agreement is mud. Below a third of
    the hits colliding this stops caring.
    """
    onsets = pattern.onsets()
    if not onsets or not bed.claimed:
        return 0.8
    collisions = sum(1 for index, _ in onsets
                     if in_bar(index * pattern.subdivision, bed.bar_beats) in bed.claimed)
    share = collisions / len(onsets)
    return 1.0 if share <= 0.34 else max(0.0, 1.0 - (share - 0.34) / 0.66)


def in_bar(beat_offset: float, bar_beats: float) -> float:
    """Where a hit falls inside the bar, to a quarter of a beat.

    Rounded because two voices on grids of different fineness have to be able
    to agree that they are hitting the same place; exact floats never would.
    """
    if bar_beats <= 0:
        return round(beat_offset, 2)
    return round((beat_offset % bar_beats) * 4) / 4


def _figure(pattern: Pattern) -> float:
    """Does it read as a figure rather than a pulse or a spray?

    Measured by how many *different* gaps there are between hits. One gap is a
    metronome. Two or three is a figure — that is a clave, a tresillo, nearly
    every drum pattern anyone whistles. More than four and the ear stops
    hearing a pattern and starts hearing activity.
    """
    onsets = [index for index, _ in pattern.onsets()]
    if len(onsets) < 2:
        return 0.15
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]
    gaps.append(len(pattern.slots) - onsets[-1] + onsets[0])
    distinct = len(set(gaps))
    if distinct == 1:
        return 0.45
    return max(0.2, 1.0 - abs(distinct - 2.5) / 3.5)


def _continuity(pattern: Pattern, bed: Bed) -> float:
    """How close to what this voice was doing before, against how much it should be.

    The target moves with unrest: a settled piece should keep its groove, an
    unsettled one has earned the right to tear it up.
    """
    if bed.previous is None or len(bed.previous.slots) != len(pattern.slots):
        return 0.6
    same = sum(1 for now, then in zip(pattern.slots, bed.previous.slots)
               if (now > 0) == (then > 0)) / len(pattern.slots)
    want = max(0.35, 0.95 - bed.drives.unrest * 0.9)
    return max(0.0, 1.0 - abs(same - want) / max(want, 1e-6))


# ── learning from what it played ──────────────────────────────────────────

def learn(grid: Grid, pattern: Pattern, plasticity: float) -> Grid:
    """Move the weights toward the pattern that won.

    This is the channel that is not the critic. Whatever the composer keeps
    choosing to play becomes what it believes belongs there, whether or not the
    metrical hierarchy agrees — which is how a grid ends up with a weight above
    1.0 on a slot that started as an offbeat afterthought.
    """
    if LEARN_RATE <= 0.0 or len(pattern.slots) != len(grid.weights):
        return grid
    rate = min(0.6, LEARN_RATE * max(0.35, plasticity))
    moved = tuple(
        round(max(WEIGHT_FLOOR, min(WEIGHT_CEILING, weight + rate * (target - weight))), 4)
        for weight, target in (
            # A slot it struck aims above 1.0 on purpose. Keep choosing a slot
            # and it ends up more certain than the metrical hierarchy ever was
            # — which is the whole claim, and it has to be reachable to be true.
            (weight, 0.6 + velocity if velocity > 0 else weight * 0.55)
            for weight, velocity in zip(grid.weights, pattern.slots)
        )
    )
    return replace(grid, weights=moved)


@dataclass(frozen=True)
class Shift:
    """One groove weight moved, and what taught it."""

    criterion: str
    before: float
    after: float
    because: str

    def describe(self) -> str:
        arrow = "matters more" if self.after > self.before else "matters less"
        return (f"groove {self.criterion} {arrow} "
                f"({self.before:.2f}→{self.after:.2f}), taught by {self.because}")


def retune(groove: Groove, verdicts: tuple[Verdict, ...],
           drives: Drives) -> tuple[Groove, tuple[Shift, ...]]:
    """Let the critic teach the groove, the same way it teaches melodic taste."""
    values = {name: getattr(groove, name) for name in CRITERIA}
    shifts: list[Shift] = []
    low, high = TASTE_BOUNDS

    for verdict in verdicts:
        if verdict.satisfied:
            continue
        taught = TAUGHT_BY.get(verdict.principle.name)
        if taught is None:
            continue
        criterion, polarity = taught
        before = values[criterion]
        after = max(low, min(high, before + polarity * (-verdict.error)
                             * TASTE_RATE * drives.plasticity))
        if abs(after - before) < 1e-4:
            continue
        values[criterion] = after
        shifts.append(Shift(criterion, before, after, verdict.principle.attribution))

    return replace(groove, **values), tuple(shifts)


def trace(choices: tuple[PatternChoice, ...]) -> str:
    """A readable account of the patterns, for the formula."""
    if not choices:
        return "no patterns chosen"
    return "\n".join(f"[{choice.movement}] {choice.note()}" for choice in choices)
