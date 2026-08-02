"""Choosing a melody instead of reciting one.

Before this the lead had no choice to make: the germ motif came round, got
rotated a bit, and was spoken over whatever chord was underneath. It was a
recital. This makes it an audition.

For every phrase the composer needs, it **imagines several** — mutations of
the line it just played, a return to the germ, and the occasional line
invented from nothing — then judges them all against what it currently cares
about and plays the winner. The winner becomes the parent of the next
generation, so the melody has a lineage: what you hear in the last movement
descends from the first phrase by a chain of decisions, none of which were
written down in advance.

What it cares about is :class:`Taste` — a weight per criterion. The criteria
themselves are music theory (steps outnumber leaps, leaps get answered, a
phrase lands somewhere), but *how much each one matters* is not fixed. Taste
starts somewhere derived from the mood and then moves, taught by the critic:
principles that keep failing make their criterion matter more, and a criterion
that keeps deciding auditions while the piece is going well earns trust.

Nothing pins those weights to where they started. If the music the composer is
actually making argues for caring about something four times as much as it did
in bar one, it will, and the record says so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from compex import rng
from compex.generate import promise
from compex.generate.critic import ROUGHNESS, Verdict
from compex.generate.evolve import Drives
from compex.generate.mood import Mood
from compex.generate.promise import Ledger
from compex.generate.theory import Motif, degree_semitone

#: How many lines it will imagine before choosing one. Set both to 1 and the
#: audition collapses to a recital again — which is the switch that makes the
#: whole idea falsifiable.
CANDIDATES_MIN, CANDIDATES_MAX = 2, 7

LEAP_DEGREES = 2      # a move of more than this many scale degrees is a leap
STRAY_BAND = 0.25     # a weight this far from where it started counts as having strayed
TASTE_BOUNDS = (0.02, 3.20)   # hard limits on a weight — not where it starts, where it stops
LEARN_RATE = 0.22     # how fast a complaint moves the weight behind it
TRUST_RATE = 0.05     # how fast a criterion that keeps deciding auditions earns weight

#: Which criteria each aesthetic principle teaches, and which way.
#: +1 means "this principle reading below its band wants that criterion to
#: matter more". The critic is the teacher; this is the syllabus.
#:
#: Post-skip reversal teaches two, deliberately. Answering a leap inside the
#: phrase and answering one three phrases later are the same complaint at
#: different timescales, and the ledger is what makes the second possible.
#: Repetition teaches the promise weight *down*, and that pairing is the whole
#: reason the ledger does not run away with the piece: settling a debt means
#: returning to material the line has already been near, so a composer that
#: cares only about paying its debts becomes self-similar. One principle pushes
#: the weight up when leaps go unanswered, another pushes it down when the
#: answering has made the music repeat itself. The intent lives in the argument
#: between them.
TAUGHT_BY: dict[str, tuple[tuple[str, int], ...]] = {
    "post_skip_reversal": (("answer", +1), ("promise", +1)),
    "motif_presence": (("kinship", +1),),
    "novelty": (("freshness", +1),),
    "repetition": (("freshness", -1), ("promise", -1)),
    "register_spread": (("span", +1),),
    "dissonance": (("colour", -1),),
    "density": (("gait", +1),),
    "tessitura_drift": (("shape", +1),),
}

@dataclass(frozen=True)
class Criterion:
    """One thing the composer can listen for in a candidate line."""

    name: str
    attribution: str
    why: str


#: What the audition judges on, and where each criterion comes from. Kept as
#: data rather than as prose in a docstring so the public page can introspect
#: it and cannot describe a criterion the code does not score.
CRITERIA_DETAIL: tuple[Criterion, ...] = (
    Criterion(
        "line", "conjunct motion",
        "Melodies move by step far more than by leap, and how much more depends on how "
        "energetic the music is. This wants the ratio the energy asks for, not the maximum.",
    ),
    Criterion(
        "answer", "Huron's post-skip reversal",
        "A large leap is overwhelmingly followed by a step back the other way. Checking it "
        "before the phrase is played is cheaper than correcting it three movements later.",
    ),
    Criterion(
        "shape", "melodic arch",
        "A line that only climbs is a scale; one that turns on every note is a tremble. "
        "Around a third of the moves being direction changes reads as a shape.",
    ),
    Criterion(
        "kinship", "Schoenberg's developing variation",
        "How much of the germ's contour is in the candidate, forwards or inverted. This is "
        "what keeps a piece about one idea while the idea keeps changing.",
    ),
    Criterion(
        "freshness", "Berlyne's inverted-U",
        "New interval shapes against the ones already played, judged toward a target rather "
        "than maximised — too novel stops reading as music at all.",
    ),
    Criterion(
        "colour", "consonance and its uses",
        "Where the weight of the phrase lands: chord tones or the notes between them, weighted "
        "by duration, against the dissonance the composer will currently tolerate.",
    ),
    Criterion(
        "span", "tessitura",
        "The pitch range of the phrase against what the reach drive is asking for. One octave "
        "is monotonous, four stops sounding like a single instrument.",
    ),
    Criterion(
        "cadence", "phrase closure",
        "Whether it lands on a chord tone, and whether the landing note is held longer than "
        "what came before it. That is most of what makes a phrase sound finished.",
    ),
    Criterion(
        "gait", "arousal matching",
        "Whether it fills the slot it was given without running past the chord change, at a "
        "note rate the movement's energy actually wants.",
    ),
    Criterion(
        "promise", "Meyer's expectation, carried",
        "What it does about everything the piece has left hanging. Settling a ripe obligation "
        "scores highest, carrying one scores next, and settling one nobody is waiting for yet "
        "scores worst — which is what 'too obvious' means once it has a number.",
    ),
)

CRITERIA: tuple[str, ...] = tuple(criterion.name for criterion in CRITERIA_DETAIL)


@dataclass(frozen=True)
class Phrase:
    """A candidate line: scale-degree steps away from the chord, and their rhythm."""

    steps: tuple[int, ...]
    rhythm: tuple[float, ...]
    origin: str            # how it came to exist — "germ", "neighbour", "improvise"…
    generation: int = 0

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.rhythm):
            raise ValueError(
                f"phrase has {len(self.steps)} steps but {len(self.rhythm)} durations")
        if not self.steps:
            raise ValueError("phrase cannot be empty")

    @property
    def beats(self) -> float:
        return sum(self.rhythm)


@dataclass(frozen=True)
class Setting:
    """Where a phrase would land, and what has already been heard around it."""

    scale: tuple[int, ...]
    chord_degree: int
    chord_size: int
    span_beats: float
    energy: float
    tension: float
    drives: Drives
    germ: tuple[int, ...] = ()                      # the germ's contour, direction only
    heard: frozenset[tuple[int, int]] = frozenset()  # interval pairs already played
    approach: int | None = None                     # scale degree the last phrase ended on
    ledger: Ledger = Ledger()                       # what the piece owes so far
    now_beat: float = 0.0                           # where we are, for judging maturity

    def chord_tones(self) -> frozenset[int]:
        """The chord's own degrees, folded into one octave of the scale."""
        span = len(self.scale)
        return frozenset((2 * step) % span for step in range(max(1, self.chord_size)))


@dataclass(frozen=True)
class Taste:
    """How much each criterion matters right now. This is the thing that evolves."""

    line: float = 1.0
    answer: float = 1.0
    shape: float = 0.8
    kinship: float = 1.0
    freshness: float = 1.0
    colour: float = 1.0
    span: float = 0.7
    cadence: float = 0.8
    gait: float = 0.9
    promise: float = 1.0
    curiosity: float = 0.5   # not a criterion — how many lines it bothers to imagine

    def weights(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in CRITERIA)

    def summary(self) -> str:
        return " · ".join(f"{name} {value:.2f}" for name, value in self.weights())

    def strayed_from(self, start: "Taste") -> tuple[str, ...]:
        """Criteria that have left the band they started in."""
        return tuple(name for name in CRITERIA
                     if abs(getattr(self, name) - getattr(start, name)) > STRAY_BAND)


@dataclass(frozen=True)
class Choice:
    """One audition: what was imagined, what won, and by how much."""

    movement: int
    at_beat: float
    chosen: Phrase
    score: float
    scores: tuple[tuple[str, float], ...]      # the winner's criterion by criterion
    rejected: tuple[tuple[str, float], ...]    # origin and score of everything else
    considered: int

    @property
    def margin(self) -> float:
        """How much better the winner was than the best line it turned down."""
        if not self.rejected:
            return 0.0
        return self.score - max(score for _, score in self.rejected)

    def note(self) -> str:
        best = ", ".join(f"{name} {value:.2f}" for name, value in
                         sorted(self.scores, key=lambda pair: -pair[1])[:3])
        return (f"{self.chosen.origin} won {self.considered} ways in "
                f"by {self.margin:.3f} — strongest on {best}")


def initial_taste(mood: Mood) -> Taste:
    """Where the composer's ear starts, before it has heard itself.

    These are derived from the mood rather than picked: a serene piece should
    begin caring about line and consonance, a shattered one about freshness and
    gait. They are starting weights, and the whole point is that they move.
    """
    return Taste(
        line=0.55 + (1.0 - mood.energy) * 0.85,
        answer=0.60 + (1.0 - mood.tension) * 0.70,
        shape=0.45 + mood.valence * 0.65,
        kinship=0.50 + (1.0 - mood.tension) * 0.80,
        freshness=0.30 + mood.tension * 1.10,
        colour=0.35 + (1.0 - mood.tension) * 1.05,
        span=0.35 + mood.energy * 0.85,
        cadence=0.40 + (1.0 - mood.energy) * 0.75,
        gait=0.45 + mood.density * 0.85,
        # A weight of zero drops the criterion out of the audition entirely
        # rather than sitting in it as a constant, so switching the ledger off
        # gives back exactly the music that existed before there was one.
        promise=(0.35 + mood.tension * 0.60) if promise.ENABLED else 0.0,
        curiosity=0.30 + mood.tension * 0.45,
    )


# ── the audition ──────────────────────────────────────────────────────────

def choose(seed: int, movement: int, index: int, setting: Setting, taste: Taste,
           parent: Phrase | None, motif: Motif, at_beat: float = 0.0) -> Choice:
    """Imagine several lines for this slot, judge them all, keep the best one."""
    candidates = propose(seed, movement, index, setting, taste, parent, motif)

    # Judge the phrase as it will actually sound, not as it was imagined. The
    # register drives stretch and bound a line on its way to becoming pitches;
    # scoring the version before that happened would be scoring something
    # nobody hears.
    judged = [(phrase, assess(voiced(phrase, setting.drives), setting))
              for phrase in candidates]
    scored = [(phrase, criteria, _total(criteria, taste)) for phrase, criteria in judged]
    # max() keeps the first of equal scores, so a tie goes to the earlier
    # candidate — deterministic, and the earlier ones are the lineage.
    winner, criteria, score = max(scored, key=lambda row: row[2])

    return Choice(
        movement=movement,
        at_beat=at_beat,
        chosen=winner,
        score=score,
        scores=tuple(sorted(criteria.items())),
        rejected=tuple((phrase.origin, value)
                       for phrase, _, value in scored if phrase is not winner),
        considered=len(scored),
    )


def propose(seed: int, movement: int, index: int, setting: Setting, taste: Taste,
            parent: Phrase | None, motif: Motif) -> tuple[Phrase, ...]:
    """The field for one audition: descendants, a return to the germ, a stranger."""
    wanted = _how_many(taste, setting.drives)
    # A restless composer does not enter the germ from the same note every
    # chord. Without this the germ candidate is identical all through a
    # movement, it keeps winning, and the piece recites after all.
    turn = index if setting.drives.novelty_pressure > 0.5 else 0
    germ = Phrase(steps=_rotate(motif.steps, turn), rhythm=_rotate(motif.rhythm, turn),
                  origin="germ", generation=parent.generation if parent else 0)

    field: list[Phrase] = [germ]
    if parent is not None:
        if setting.approach is not None:
            field.append(_connect(parent, seed, f"con-{movement}-{index}", setting))
        for slot in range(wanted - len(field) - 1):
            field.append(_mutate(parent, seed, f"mut-{movement}-{index}", slot, setting))
    field.append(_improvise(seed, f"imp-{movement}-{index}", setting,
                            length=len(motif.steps)))
    while len(field) < wanted:
        field.append(_improvise(seed, f"imp2-{movement}-{index}", setting,
                                length=len(motif.steps) + len(field)))
    return tuple(field[:wanted])


def voiced(phrase: Phrase, drives: Drives) -> Phrase:
    """The phrase as the register drives will render it.

    ``register_reach`` scales the line as well as bounding it — clamping alone
    can only ever narrow a melody, so the drive could never answer a complaint
    that the range was too small. Applied at the boundary rather than folded
    into the lineage, so a phrase does not get stretched again every
    generation until it falls off the top of the scale.
    """
    stretch = 0.55 + drives.register_reach * 1.6
    reach = 1 + int(drives.register_reach * 6)
    return replace(phrase, steps=tuple(
        max(-reach, min(reach, int(round(step * stretch)))) for step in phrase.steps))


def _rotate(values: tuple, turn: int) -> tuple:
    if not values or turn % len(values) == 0:
        return values
    offset = turn % len(values)
    return values[offset:] + values[:offset]


def _how_many(taste: Taste, drives: Drives) -> int:
    """How hard to think about this phrase.

    Curiosity sets the floor; unrest raises it, because a composer that keeps
    failing its own principles should be considering more than two options.
    """
    wanted = CANDIDATES_MIN + int(round(taste.curiosity * 3.0 + drives.unrest * 3.0))
    return max(CANDIDATES_MIN, min(CANDIDATES_MAX, wanted))


def _total(criteria: dict[str, float], taste: Taste) -> float:
    """Weighted mean over the criteria that are switched on.

    A weight of zero drops its criterion out of the average rather than
    averaging in a constant, which is what makes an off switch mean *off*
    instead of *quietly shifting every score*.
    """
    weights = {name: value for name, value in taste.weights() if value > 1e-9}
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return sum(criteria[name] * weight for name, weight in weights.items()) / total


# ── making candidates ─────────────────────────────────────────────────────

MUTATIONS: tuple[str, ...] = (
    "neighbour", "passing", "answer", "displace", "turn",
    "trim", "sequence", "syncopate", "stretch", "crush",
)


def _mutate(parent: Phrase, seed: int, stream: str, slot: int, setting: Setting) -> Phrase:
    """One small change to a line that already exists.

    Small on purpose. A mutation that rewrites the phrase is just a new phrase,
    and then nothing accumulates across the piece.
    """
    kind = rng.pick(seed, f"{stream}-kind", slot, MUTATIONS)
    steps, rhythm = list(parent.steps), list(parent.rhythm)
    at = int(rng.uniform(seed, f"{stream}-at", slot) * len(steps))

    if kind == "neighbour" and len(steps) > 1:
        # Decorate a note with its upper or lower neighbour, borrowing time.
        direction = 1 if rng.uniform(seed, f"{stream}-dir", slot) < 0.5 else -1
        half = rhythm[at] / 2.0
        steps.insert(at + 1, steps[at] + direction)
        rhythm[at] = half
        rhythm.insert(at + 1, half)
    elif kind == "passing" and len(steps) > 1:
        # Fill a gap with the degree between its ends.
        nxt = (at + 1) % len(steps)
        gap = steps[nxt] - steps[at]
        if abs(gap) >= 2:
            half = rhythm[at] / 2.0
            steps.insert(at + 1, steps[at] + (1 if gap > 0 else -1))
            rhythm[at] = half
            rhythm.insert(at + 1, half)
    elif kind == "answer" and len(steps) > 2:
        # Huron's reversal, applied on purpose rather than hoped for.
        for position in range(1, len(steps) - 1):
            move = steps[position] - steps[position - 1]
            if abs(move) > LEAP_DEGREES:
                steps[position + 1] = steps[position] - (1 if move > 0 else -1)
                break
    elif kind == "displace":
        shift = int(rng.between(seed, f"{stream}-shift", slot, -2, 3)) or 1
        steps = [step + shift for step in steps]
    elif kind == "turn" and len(steps) > 2:
        steps[at], steps[(at + 1) % len(steps)] = steps[(at + 1) % len(steps)], steps[at]
    elif kind == "trim" and len(steps) > 2:
        steps.pop(at)
        rhythm.pop(at)
    elif kind == "sequence":
        # Say the shape again a degree up or down — the oldest trick there is.
        shift = 1 if rng.uniform(seed, f"{stream}-seq", slot) < 0.6 else -1
        keep = max(2, len(steps) // 2)
        steps = steps + [step + shift for step in steps[:keep]]
        rhythm = rhythm + rhythm[:keep]
    elif kind == "syncopate" and len(rhythm) > 1:
        nxt = (at + 1) % len(rhythm)
        borrow = min(rhythm[nxt] / 2.0, 0.5)
        rhythm[at] += borrow
        rhythm[nxt] -= borrow
        if rhythm[nxt] < 0.125:
            rhythm[nxt] = 0.125
    elif kind == "stretch":
        rhythm = [min(4.0, value * 1.5) for value in rhythm]
    elif kind == "crush":
        rhythm = [max(0.125, value / 1.5) for value in rhythm]

    return Phrase(steps=tuple(steps), rhythm=tuple(rhythm),
                  origin=kind, generation=parent.generation + 1)


def _connect(parent: Phrase, seed: int, stream: str, setting: Setting) -> Phrase:
    """The parent line, moved so it starts a step from where the last one stopped.

    Phrases were being judged in isolation and then played back to back, so
    every join was a jump nobody had looked at — and joins are exactly where a
    melody audibly lurches. This candidate is the obvious answer to that: same
    idea, entered from where the ear already is.

    The shift is computed in voiced space because that is what will be heard;
    a phrase transposed before the register drives stretch it does not land
    where you put it.
    """
    stretch = 0.55 + setting.drives.register_reach * 1.6
    direction = 1 if rng.uniform(seed, f"{stream}-dir", 0) < 0.5 else -1
    wanted = setting.approach - setting.chord_degree + direction
    raw = int(round(wanted / stretch)) if stretch else wanted
    shift = raw - parent.steps[0]
    return replace(parent, steps=tuple(step + shift for step in parent.steps),
                   origin="connect", generation=parent.generation + 1)


def _improvise(seed: int, stream: str, setting: Setting, length: int) -> Phrase:
    """A line with no ancestry, walked out under the current drives.

    It usually loses. It needs to exist anyway: without an outsider in the
    field, every phrase in the piece descends from one germ and the lineage can
    only ever narrow.
    """
    drives = setting.drives
    count = max(3, min(9, length + int(setting.energy * 3)))
    reach = 1 + int(drives.register_reach * 4)

    steps = [0]
    previous = 0
    for position in range(1, count):
        move = int(rng.between(seed, f"{stream}-step", position, -reach, reach + 1))
        if abs(previous) > LEAP_DEGREES and rng.uniform(
                seed, f"{stream}-ans", position) < drives.gap_fill:
            move = -1 if previous > 0 else 1
        steps.append(steps[-1] + move)
        previous = move

    ideal = 2.5 - 2.0 * setting.energy
    rhythm = tuple(
        max(0.125, round(rng.between(seed, f"{stream}-dur", position,
                                     ideal * 0.5, ideal * 1.5) * 4) / 4) or 0.25
        for position in range(count)
    )
    return Phrase(steps=tuple(steps), rhythm=rhythm, origin="improvise", generation=0)


# ── judging one candidate ─────────────────────────────────────────────────

def assess(phrase: Phrase, setting: Setting) -> dict[str, float]:
    """Score a candidate on every criterion. Each returns 0..1 — 1 is satisfied.

    Where a criterion has a target rather than a direction, the target comes
    from the drives, which are themselves moving. So the same phrase can win
    one audition and lose the next without a line of this being rewritten.
    """
    moves = [b - a for a, b in zip(phrase.steps, phrase.steps[1:])]
    if setting.approach is not None:
        # The step into the phrase counts. Judging phrases in isolation left
        # every join between them unexamined, and the joins are where a line
        # audibly lurches — the critic was measuring them all along.
        moves.insert(0, (setting.chord_degree + phrase.steps[0]) - setting.approach)
    tones = setting.chord_tones()
    span = len(setting.scale)

    return {
        "line": _line(moves, setting.energy),
        "answer": _answer(moves),
        "shape": _shape(moves),
        "kinship": _kinship(moves, setting.germ),
        "freshness": _freshness(moves, setting.heard, setting.drives.novelty_pressure),
        "colour": _colour(phrase, setting, tones),
        "span": _span(phrase, setting),
        "cadence": _cadence(phrase, tones, span),
        "gait": _gait(phrase, setting),
        "promise": _promise(phrase, setting),
    }


def _line(moves: list[int], energy: float) -> float:
    """Steps against leaps, judged toward what this energy should want."""
    if not moves:
        return 0.5
    small = sum(1 for move in moves if abs(move) <= LEAP_DEGREES) / len(moves)
    want = 0.88 - energy * 0.30
    return max(0.0, 1.0 - abs(small - want) / max(want, 1e-6))


def _answer(moves: list[int]) -> float:
    """Post-skip reversal inside the phrase itself, before anybody hears it."""
    leaps = answered = 0
    for move, nxt in zip(moves, moves[1:]):
        if abs(move) <= LEAP_DEGREES:
            continue
        leaps += 1
        if move * nxt < 0 and abs(nxt) <= LEAP_DEGREES:
            answered += 1
    if not leaps:
        return 0.75   # nothing to answer for is not the same as answering well
    return answered / leaps


def _shape(moves: list[int]) -> float:
    """Does the line turn around, and not too often?

    A phrase that only climbs is a scale; one that changes direction on every
    note is a tremble. Somewhere near a third of the moves being turns reads as
    a shape.
    """
    if len(moves) < 2:
        return 0.5
    turns = sum(1 for a, b in zip(moves, moves[1:]) if a * b < 0) / (len(moves) - 1)
    return max(0.0, 1.0 - abs(turns - 0.38) / 0.62)


def _kinship(moves: list[int], germ: tuple[int, ...]) -> float:
    """How much of the germ's contour is in here, forwards or inverted."""
    if not germ or not moves:
        return 0.5
    shape = [1 if step > 0 else -1 if step < 0 else 0 for step in germ]
    played = [1 if step > 0 else -1 if step < 0 else 0 for step in moves]
    if len(played) < len(shape):
        shape = shape[:len(played)]
    best = 0.0
    for start in range(len(played) - len(shape) + 1):
        window = played[start:start + len(shape)]
        forward = sum(1 for a, b in zip(window, shape) if a == b) / len(shape)
        mirrored = sum(1 for a, b in zip(window, shape) if a == -b) / len(shape)
        best = max(best, forward, mirrored)
    return best


def _freshness(moves: list[int], heard: frozenset[tuple[int, int]], want: float) -> float:
    """New shapes against the target the novelty drive is currently asking for."""
    pairs = {(a, b) for a, b in zip(moves, moves[1:])}
    if not pairs:
        return 0.5
    if not heard:
        return 0.6   # nothing to be fresh against yet; do not reward or punish
    fresh = len(pairs - heard) / len(pairs)
    return max(0.0, 1.0 - abs(fresh - want))


def _colour(phrase: Phrase, setting: Setting, tones: frozenset[int]) -> float:
    """Where the weight of the phrase falls: chord tones, or the notes between them.

    Weighted by duration, because a passing dissonance on a semiquaver is not
    the same event as sitting on a tritone for two beats. Judged against the
    dissonance the composer will currently tolerate, not against consonance.
    """
    span = len(setting.scale)
    total = sum(phrase.rhythm) or 1.0
    rough = 0.0
    for step, length in zip(phrase.steps, phrase.rhythm):
        if (step % span) in tones:
            continue
        semitone = degree_semitone(setting.scale, setting.chord_degree + step)
        rough += ROUGHNESS.get(semitone % 12, 0.5) * length
    measured = rough / total
    return max(0.0, 1.0 - abs(measured - setting.drives.dissonance_ceiling * 0.55))


def _span(phrase: Phrase, setting: Setting) -> float:
    """Pitch range against what the reach drive is asking for."""
    if len(phrase.steps) < 2:
        return 0.4
    scale = setting.scale
    pitches = [degree_semitone(scale, setting.chord_degree + step) for step in phrase.steps]
    measured = max(pitches) - min(pitches)
    want = 3.0 + setting.drives.register_reach * 20.0
    return max(0.0, 1.0 - abs(measured - want) / max(want, 1e-6))


def _cadence(phrase: Phrase, tones: frozenset[int], span: int) -> float:
    """Does it land somewhere, and is the landing note the longest one?

    Ending on a chord tone held longer than the notes around it is most of what
    makes a phrase sound finished rather than interrupted.
    """
    last_step, last_length = phrase.steps[-1], phrase.rhythm[-1]
    lands = 1.0 if (last_step % span) in tones else 0.25
    if len(phrase.rhythm) < 2:
        return lands * 0.8
    average = sum(phrase.rhythm[:-1]) / (len(phrase.rhythm) - 1)
    held = min(1.0, last_length / max(average, 1e-6) / 1.5)
    return lands * (0.55 + 0.45 * held)


def _gait(phrase: Phrase, setting: Setting) -> float:
    """Does it fill the slot it has been given, at a rate the energy wants?

    Two failures share this criterion: a phrase that runs past the chord change
    and gets cut off, and a phrase that leaves most of its bar silent.
    """
    if setting.span_beats <= 0:
        return 0.5
    fill = min(1.0, phrase.beats / setting.span_beats)
    overrun = max(0.0, phrase.beats - setting.span_beats) / setting.span_beats
    rate = len(phrase.steps) / max(phrase.beats, 1e-6)
    want_rate = 0.5 + setting.energy * 2.5 * setting.drives.density_bias
    fit = max(0.0, 1.0 - abs(rate - want_rate) / max(want_rate, 1e-6))
    return max(0.0, (0.45 * fill + 0.55 * fit) - overrun * 0.5)


def _promise(phrase: Phrase, setting: Setting) -> float:
    """What this candidate does about everything still hanging.

    The ordering that matters — settling early is worse than carrying, which
    is worse than settling something ripe — lives in
    :func:`compex.generate.promise.credit`, so the melodic and rhythmic sides
    cannot disagree about what "too soon" means.
    """
    live = setting.ledger.live(setting.now_beat, domain="pitch")
    if not live:
        return 0.5   # nothing owed. Not a virtue and not a failing

    best, ripening = 0.0, 0.0
    settling = False
    for owed in live:
        maturity = owed.maturity(setting.now_beat)
        ripening = max(ripening, maturity)
        if promise.would_settle(owed, phrase.steps, setting.chord_degree):
            settling = True
            best = max(best, maturity)
    return promise.credit(best, settling, ripening)


# ── taste moving ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Shift:
    """One weight moved, and what taught it."""

    criterion: str
    before: float
    after: float
    because: str
    past_start: bool

    def describe(self) -> str:
        arrow = "matters more" if self.after > self.before else "matters less"
        tail = " — past where it started" if self.past_start else ""
        return (f"{self.criterion} {arrow} ({self.before:.2f}→{self.after:.2f}), "
                f"taught by {self.because}{tail}")


def retune(taste: Taste, start: Taste, verdicts: tuple[Verdict, ...],
           choices: tuple[Choice, ...], drives: Drives) -> tuple[Taste, tuple[Shift, ...]]:
    """Move the weights: the critic teaches, and the auditions confirm.

    Two channels feed this and they are different in kind. The critic says what
    the music got wrong, which raises the weight of the criterion that would
    have caught it. The auditions say which criteria are actually *deciding*
    anything — a criterion that separated the winner from the field while the
    piece was going well earns a little trust on its own account.

    Nothing here clamps back toward the starting weights. That is deliberate.
    """
    values = {name: getattr(taste, name) for name in CRITERIA}
    shifts: list[Shift] = []
    low, high = TASTE_BOUNDS

    for verdict in verdicts:
        if verdict.satisfied:
            continue
        for criterion, polarity in TAUGHT_BY.get(verdict.principle.name, ()):
            before = values[criterion]
            if before <= 1e-9:
                continue   # a criterion switched off stays off; teaching cannot revive it
            delta = polarity * (-verdict.error) * LEARN_RATE * drives.plasticity
            after = max(low, min(high, before + delta))
            if abs(after - before) < 1e-4:
                continue
            values[criterion] = after
            shifts.append(Shift(criterion, before, after, verdict.principle.attribution,
                                _past(after, getattr(start, criterion))))

    # The trust channel only opens when the piece is going well. Reinforcing
    # whatever is deciding auditions during a bad stretch would teach the
    # composer to trust the taste that got it there.
    if choices and drives.unrest < 0.30:
        for criterion, weight in _deciding(choices).items():
            before = values[criterion]
            if before <= 1e-9:
                continue
            after = max(low, min(high, before + weight * TRUST_RATE))
            if abs(after - before) < 1e-4:
                continue
            values[criterion] = after
            shifts.append(Shift(criterion, before, after, "its own auditions",
                                _past(after, getattr(start, criterion))))

    updated = replace(taste, **values)
    updated = _reconsider_curiosity(updated, drives)
    return updated, tuple(shifts)


def _past(value: float, started_at: float) -> bool:
    return abs(value - started_at) > STRAY_BAND


def _deciding(choices: tuple[Choice, ...]) -> dict[str, float]:
    """Which criteria actually separated winners from the lines they beat.

    A criterion every candidate scores the same on decided nothing, however
    heavily it is weighted. This finds the ones doing the work.
    """
    strengths: dict[str, float] = {}
    for choice in choices:
        if not choice.rejected or choice.margin <= 0:
            continue
        for name, value in choice.scores:
            strengths[name] = strengths.get(name, 0.0) + value * choice.margin
    if not strengths:
        return {}
    top = max(strengths.values()) or 1.0
    return {name: value / top for name, value in strengths.items() if value / top > 0.6}


def _reconsider_curiosity(taste: Taste, drives: Drives) -> Taste:
    """Think harder when unsettled, less hard when it is working."""
    target = 0.2 + drives.unrest * 1.4
    moved = taste.curiosity + (target - taste.curiosity) * 0.35
    return replace(taste, curiosity=max(0.0, min(1.0, moved)))


def trace(choices: tuple[Choice, ...]) -> str:
    """A readable account of the auditions, for the formula."""
    if not choices:
        return "no melodies chosen"
    return "\n".join(
        f"[{choice.movement}] @ beat {choice.at_beat:g}: {choice.note()}"
        for choice in choices
    )
