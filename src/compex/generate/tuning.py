"""The tuning the piece is in, which the machine picks for itself.

Twelve notes to the octave is not a fact about music. It is a fact about
keyboards — a compromise adopted because a physical instrument cannot retune
between chords, and then inherited by everyone who learned on one. This engine
has no keyboard in it. It synthesises from frequencies, and every pitch that
leaves the composer is a float, so the twelve-tone grid was never anything but
a table I handed it.

So it chooses. Three families, all derived rather than looked up:

* **twelve** — the familiar grid, and the named modes on it. Still the right
  answer for a piece that wants to be legible, and still where most serene
  moods land.
* **equal** — some other equal division of the octave. Seventeen, nineteen,
  twenty-two. The steps are even, the intervals are unfamiliar, and it does
  not sound broken — it sounds like a system you have not heard before, which
  is exactly the difference between *differently tuned* and *out of tune*.
* **just** — degrees built from small whole-number frequency ratios, chosen by
  **Tenney height** (how simple the ratio is: log2 of numerator times
  denominator). Chords stop beating, because the partials line up exactly
  rather than nearly. No fixed-pitch instrument can do this and change key;
  this one changes tuning per piece and does not care.

Everything downstream already speaks in floating-point semitones, so nothing in
the synthesis had to change for any of it. Only the theory believed in twelve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from compex import rng
from compex.generate.mood import Mood

#: Prime limits available to the just family, by how strange the mood wants to be.
#: 5-limit is common practice; 7 brings in the blue seventh; 11 is where most
#: listeners stop having names for what they are hearing.
LIMITS: tuple[int, ...] = (5, 7, 11)

#: Divisions of the octave worth using. 12 is in the equal family too — it is
#: not special, it is just one of these that happened to win.
DIVISIONS: tuple[int, ...] = (5, 7, 10, 13, 15, 17, 19, 22, 24, 31)

OCTAVE = 12.0        # semitones. The octave itself stays a 2:1 — see the note below
MIN_STEP = 0.55      # semitones: closer than this and two degrees are one degree
FAMILIES: tuple[str, ...] = ("twelve", "equal", "just")


@dataclass(frozen=True)
class Tuning:
    """The pitches this piece is built from, as semitone offsets from the tonic.

    ``steps`` replaces what used to be a row from a table of scales. It is
    still ascending, still one octave wide, and now it is very often not
    integers.
    """

    name: str
    family: str
    steps: tuple[float, ...]
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a tuning needs at least one degree")
        if self.steps[0] != 0.0:
            raise ValueError(f"a tuning starts on its tonic, not {self.steps[0]}")
        if list(self.steps) != sorted(self.steps):
            raise ValueError("tuning degrees must ascend")

    def __len__(self) -> int:
        return len(self.steps)

    def cents(self) -> tuple[float, ...]:
        return tuple(step * 100.0 for step in self.steps)

    def describe(self) -> str:
        return f"{self.name} ({len(self.steps)} degrees)" + (f" — {self.detail}"
                                                             if self.detail else "")


def derive(seed: int, mood: Mood, stream: str = "tuning",
           bored: dict[str, float] | None = None) -> Tuning:
    """Choose a tuning for this piece.

    Leaving twelve is a decision about strangeness, so it is driven by tension
    and grit — a serene piece nearly always stays where the ear is comfortable,
    and a shattered one is free to go somewhere with no name.

    ``bored`` is how much it has been living in each family lately. An engine
    that has spent its last ten pieces in twelve should find leaving easier,
    which is a machine's version of wanting a change and is available to it
    precisely because it remembers every piece it has made.
    """
    stale = bored or {}
    strangeness = (0.55 * mood.tension + 0.3 * mood.grit + 0.15 * (1.0 - mood.valence)
                   + 0.35 * stale.get("twelve", 0.0)
                   - 0.2 * (stale.get("equal", 0.0) + stale.get("just", 0.0)))
    strangeness = max(0.0, min(1.0, strangeness))
    draw = rng.uniform(seed, f"{stream}-family", 0)

    if draw > strangeness:
        return twelve(seed, mood, stream)
    if rng.uniform(seed, f"{stream}-which", 0) < 0.55:
        return equal(seed, mood, stream)
    return just(seed, mood, stream)


# ── the three families ────────────────────────────────────────────────────

def twelve(seed: int, mood: Mood, stream: str = "tuning") -> Tuning:
    """The familiar grid — with a mode the machine works out rather than looks up.

    The named modes were the last table in the project that decided something
    musical. They are still here, but demoted to what they should always have
    been: **names for shapes**, checked against afterwards. The machine builds
    a set of degrees out of the grid by its own criteria, and then we tell the
    listener if it happens to have reinvented aeolian.
    """
    steps = invent(seed, mood, grid=tuple(index * 1.0 for index in range(12)),
                   stream=f"{stream}-mode")
    known = name_of(steps)
    return Tuning(
        name=known or "unnamed mode",
        family="twelve",
        steps=steps,
        detail="12-EDO, " + (f"which is {known}" if known else
                             "a set with no common name"),
    )


def invent(seed: int, mood: Mood, grid: tuple[float, ...],
           stream: str = "invent") -> tuple[float, ...]:
    """Build a scale out of an available grid, by what the machine can measure.

    Four things are scored, and none of them is "is this a mode somebody wrote
    down". A set wants a **strong interval** near a fifth to hold it together;
    **steps of more than one size**, because a set of equal steps has no
    positions in it and every note sounds like every other; **no hole too wide
    to step through**; and a **roughness against the tonic** that suits the
    mood, which is the one place the mood gets a vote.
    """
    size = max(4, min(len(grid), 5 + int(mood.density * 3.5)))
    best, chosen = -1.0, tuple(grid[: size])

    for attempt in range(48):
        offsets = sorted(rng.pick(seed, f"{stream}-{attempt}", index, grid[1:])
                         for index in range(size - 1))
        candidate = _thin((0.0,) + tuple(dict.fromkeys(offsets)))
        if len(candidate) < 4:
            continue
        score = _score(candidate, mood)
        if score > best:
            best, chosen = score, candidate
    return chosen


def _score(steps: tuple[float, ...], mood: Mood) -> float:
    """How good a scale this is, by criteria a machine can check."""
    gaps = [high - low for low, high in zip(steps, list(steps[1:]) + [OCTAVE])]
    if max(gaps) > MAX_GAP + MIN_STEP:
        return -1.0

    # Something near a 3:2 holds a set together — it is why every scale anyone
    # uses has one, and it is checkable rather than received.
    anchored = 1.0 if any(abs(step - 7.02) < 0.6 for step in steps) else 0.35

    # More than one size of step. All-equal steps (whole tone, chromatic) have
    # no landmarks in them, and a melody in one cannot be anywhere in particular.
    sizes = len({round(gap * 2) / 2 for gap in gaps})
    variety = min(1.0, (sizes - 1) / 3.0)

    rough = sum(roughness_at(step) for step in steps[1:]) / max(1, len(steps) - 1)
    wanted = 0.18 + mood.tension * 0.42
    fits = max(0.0, 1.0 - abs(rough - wanted) / max(wanted, 1e-6))

    return anchored * 0.9 + variety * 0.8 + fits * 1.1 + _focus(steps) * 1.0


def _focus(steps: tuple[float, ...]) -> float:
    """How much one interval recurs inside the set, 0..1.

    This is the property that actually makes the diatonic scale special, and it
    is checkable without knowing its name: count every interval between every
    pair of degrees and see whether one size dominates. The major scale has six
    fifths in it, which is why it holds together and why a chord built anywhere
    in it sounds related to a chord built anywhere else. A set with a flat
    interval profile has no such relationships — every pair is as unlike every
    other pair as possible, and there is nothing for harmony to be *about*.
    """
    if len(steps) < 3:
        return 0.0
    counts: dict[float, int] = {}
    for index, low in enumerate(steps):
        for high in steps[index + 1:]:
            interval = round(min(high - low, OCTAVE - (high - low)) * 2) / 2
            counts[interval] = counts.get(interval, 0) + 1
    pairs = sum(counts.values()) or 1
    return max(counts.values()) / pairs


def name_of(steps: tuple[float, ...]) -> str | None:
    """Did it happen to reinvent something with a name? Reported, never used."""
    from compex.generate.theory import SCALES

    rounded = tuple(round(step) for step in steps)
    for name, known in SCALES.items():
        if rounded == tuple(known):
            return name.replace("_", " ")
    return None


def equal(seed: int, mood: Mood, stream: str = "tuning") -> Tuning:
    """An equal division of the octave that is not twelve.

    The degrees are chosen by stacking the division's best fifth, which is how
    every equal temperament that anyone has ever found usable was built: it
    guarantees the set is connected by a strong interval rather than being an
    arbitrary handful of pitches.
    """
    divisions = int(rng.pick(seed, f"{stream}-edo", 0, DIVISIONS))
    step_size = OCTAVE / divisions

    # The step nearest a 3:2. In 12 this is 7; in 17 it is 10; in 19, 11.
    fifth = max(1, round(divisions * math.log2(1.5)))
    wanted = max(4, min(divisions, 4 + int(mood.density * 4) + 2))

    positions = {0}
    cursor = 0
    while len(positions) < wanted:
        cursor = (cursor + fifth) % divisions
        if cursor in positions:
            break
        positions.add(cursor)

    steps = tuple(sorted(position * step_size for position in positions))
    return Tuning(
        name=f"{divisions}-EDO",
        family="equal",
        steps=_thin(steps),
        detail=f"{len(positions)} of {divisions}, stacked on its best fifth "
               f"({fifth}/{divisions} of an octave)",
    )


def just(seed: int, mood: Mood, stream: str = "tuning") -> Tuning:
    """Degrees built from small whole-number ratios, ranked by Tenney height.

    Tenney height is log2(numerator x denominator): the plainest measure there
    is of how simple a ratio is, and the one that best predicts whether two
    tones lock together or beat against each other. Low tension takes the
    simplest ratios available; high tension is allowed the ones that are
    interesting precisely because they are not simple.
    """
    limit = int(rng.pick(seed, f"{stream}-limit", 0,
                         LIMITS[: 1 + int(mood.tension * (len(LIMITS) - 1) + 0.5)]))
    ratios = _candidates(limit)
    ranked = sorted(ratios, key=lambda pair: _tenney(*pair))

    wanted = max(4, min(len(ranked), 5 + int(mood.density * 4)))
    # A restless piece reaches past the obvious consonances into the ones with
    # no common name; a calm one takes them in order of simplicity.
    reach = 1.0 + mood.tension * 1.6
    chosen = {(1, 1)}
    for index in range(wanted - 1):
        pool = ranked[1:max(2, int(len(ranked) * min(1.0, 0.35 * reach)))]
        pick = rng.pick(seed, f"{stream}-ratio", index, tuple(pool))
        chosen.add(pick)

    chosen = _fill_gaps(chosen, ranked)
    ordered = sorted(chosen, key=lambda pair: _semitones(*pair))
    kept = _thin(tuple(_semitones(*pair) for pair in ordered))
    # Name the ratios that survived thinning, not the ones that were proposed —
    # a tuning that lists a degree it does not contain is a lie in a formula
    # that is supposed to regenerate the piece.
    names = ", ".join(f"{num}/{den}" for num, den in ordered
                      if any(abs(_semitones(num, den) - step) < 1e-9 for step in kept))
    return Tuning(name=f"just, {limit}-limit", family="just", steps=kept, detail=names)


# ── the arithmetic ────────────────────────────────────────────────────────

MAX_GAP = 3.6   # semitones: wider than a minor third and the scale stops being steppable


def _fill_gaps(chosen: set[tuple[int, int]],
               ranked: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Fill any hole too wide to step through, with the simplest ratio inside it.

    Ranking by simplicity alone clusters: the plainest ratios crowd around the
    fifth and the octave and leave a fourth of the octave empty. A scale with a
    hole that size is a chord, not a scale — a melody crossing it has to leap
    every time, which the audition then spends the whole piece answering.
    """
    for _ in range(8):
        steps = sorted(_semitones(*pair) for pair in chosen)
        widest, at = 0.0, None
        for low, high in zip(steps, list(steps[1:]) + [OCTAVE]):
            if high - low > widest:
                widest, at = high - low, (low, high)
        if at is None or widest <= MAX_GAP:
            break

        low, high = at
        inside = [pair for pair in ranked
                  if low + MIN_STEP <= _semitones(*pair) <= high - MIN_STEP]
        if not inside:
            break
        chosen.add(inside[0])   # ranked is already sorted by Tenney height
    return chosen


def _candidates(limit: int) -> tuple[tuple[int, int], ...]:
    """Every ratio inside one octave whose primes stop at ``limit``.

    Generated rather than listed, which matters: a table of "the good
    intervals" would be the same borrowed authority the twelve-tone grid was.
    """
    primes = [prime for prime in (2, 3, 5, 7, 11, 13) if prime <= limit]
    out: set[tuple[int, int]] = set()
    for numerator in range(1, 64):
        if not _smooth(numerator, primes):
            continue
        for denominator in range(1, 64):
            if not _smooth(denominator, primes):
                continue
            ratio = numerator / denominator
            if not 1.0 <= ratio < 2.0:
                continue
            if math.gcd(numerator, denominator) != 1:
                continue
            out.add((numerator, denominator))
    return tuple(sorted(out))


def _smooth(value: int, primes: list[int]) -> bool:
    """Does this number factor entirely into the given primes?"""
    for prime in primes:
        while value % prime == 0:
            value //= prime
    return value == 1


def _tenney(numerator: int, denominator: int) -> float:
    """log2(n x d) — how much information the ratio costs to state."""
    return math.log2(numerator * denominator)


def _semitones(numerator: int, denominator: int) -> float:
    return OCTAVE * math.log2(numerator / denominator)


def _thin(steps: tuple[float, ...]) -> tuple[float, ...]:
    """Drop degrees too close together to be heard as separate.

    Two pitches a fifth of a semitone apart are not two degrees of a scale;
    they are one degree and a tuning error, and a melody that "steps" between
    them reads as wobble rather than as motion.
    """
    kept: list[float] = []
    for step in steps:
        if not kept or step - kept[-1] >= MIN_STEP:
            kept.append(step)
    if kept and kept[0] != 0.0:
        kept.insert(0, 0.0)
    return tuple(kept) if len(kept) >= 3 else tuple(steps)


def roughness_at(semitones: float) -> float:
    """How rough an interval of this size sounds against the tonic, 0..1.

    The critic's roughness table is indexed by the twelve interval classes,
    which is fine until the piece is in nineteen. This interpolates between
    the nearest two classes, so a neutral third lands between a minor and a
    major one instead of being rounded into whichever it is closer to.
    """
    from compex.generate.critic import ROUGHNESS

    within = semitones % OCTAVE
    low = int(math.floor(within)) % 12
    high = (low + 1) % 12
    blend = within - math.floor(within)
    return ROUGHNESS.get(low, 0.5) * (1.0 - blend) + ROUGHNESS.get(high, 0.5) * blend
