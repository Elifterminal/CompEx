"""How much cheaper the piece's own past got.

The most interesting kind of resolution is not lowering tension. It is a late
event revealing that five strange, apparently unrelated things were one thing
rotated five ways — after which the earlier strangeness is *describable by a
simpler rule than it needed at the time*. The instability does not stop. It
becomes intelligible.

That is measurable here in a way it would not be almost anywhere else, because
this engine already writes a description of what it did and that description
regenerates the audio. So "the opening got cheaper to describe" is a number of
symbols, not a metaphor.

The measurement is minimum-description-length, done honestly:

* a phrase can be spelled out **literally** — a degree and a duration per note;
* or written **by reference** to something already in the vocabulary — "entry
  four, inverted", "the germ, fragmented" — which costs a handful of symbols
  whatever the phrase's length;
* the cost of a movement is the sum over its phrases of whichever is cheaper.

Describe the opening using only the vocabulary that existed while it was being
written, then describe it again using the vocabulary the finished piece ended
up with. The difference is the reveal. Because the late vocabulary contains
the early one, the number can never be negative — the question is only ever
*how much* hindsight bought.

The transforms recognised here are exactly the ones the composer can actually
perform (:mod:`compex.generate.theory`). Recognising transforms it cannot make
would credit the piece with structure nobody put there.

This sits at the top of the package rather than under ``notation`` on purpose:
the composer consults it *while choosing*, and ``notation`` imports the
finished composition. Putting it there would have made the audition depend on
the module that describes its own output.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The whole mechanism, off. With this False the criterion is dropped from the
#: audition entirely and the measurement is still reported — you can watch what
#: the piece reveals without letting it try to reveal anything.
ENABLED = True

#: Cost, in symbols, of each part of a description.
HEADER = 1.0       # naming a phrase at all
NOTE = 2.0         # spelling one note out: a degree and a duration
REFERENCE = 1.0    # pointing at something already in the vocabulary
PARAM = 1.0        # one transform parameter — how far, which way
REPAIR = 1.5       # one interval that does not match and has to be corrected

RHYTHM_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)  # diminution, as-is, augmentation
MIN_LENGTH = 3     # shorter than this and a "match" is a coincidence


@dataclass(frozen=True)
class Shape:
    """One piece of material, stripped to what a description would need."""

    label: str
    steps: tuple[int, ...]
    rhythm: tuple[float, ...]
    at_beat: float = 0.0

    @property
    def intervals(self) -> tuple[int, ...]:
        """The shape without its pitch level — transposition costs nothing here."""
        return tuple(b - a for a, b in zip(self.steps, self.steps[1:]))

    def literal(self) -> float:
        return HEADER + NOTE * len(self.steps)


@dataclass(frozen=True)
class Reveal:
    """What the piece's own past cost before and after the piece happened."""

    then: float          # symbols needed using only the vocabulary of the time
    now: float           # symbols needed using the vocabulary it ended with
    literal: float       # symbols with no vocabulary at all
    phrases: int
    vocabulary_then: int
    vocabulary_now: int

    @property
    def saved(self) -> float:
        return max(0.0, self.then - self.now)

    @property
    def share(self) -> float:
        """The reveal as a fraction of what spelling the opening out would cost.

        Normalised because the raw symbol count scales with how much music is
        in the movement, and a long opening would otherwise always look more
        revealing than a short one.
        """
        return self.saved / self.literal if self.literal > 0 else 0.0

    def describe(self) -> str:
        if self.saved <= 0:
            return (f"nothing was revealed — the opening costs {self.then:.0f} symbols "
                    f"either way")
        return (f"the opening cost {self.then:.0f} symbols as it was written and "
                f"{self.now:.0f} once the piece had finished explaining itself — "
                f"{self.share * 100:.0f}% cheaper in hindsight")


# ── describing one shape against a vocabulary ─────────────────────────────

def variants(intervals: tuple[int, ...]) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    """Every way an entry can be turned into something else, with what saying so costs.

    Note which is which: playing a melody backwards reverses its intervals *and*
    negates them, because the step that went up on the way out comes down on the
    way back. Reversal alone is the retrograde of the inversion. Getting those
    two labels the wrong way round would misname half the transforms in the
    formula while producing identical numbers, which is the kind of wrong that
    survives for months.
    """
    return (
        ("transposed", intervals, 1),
        ("inverted", tuple(-value for value in intervals), 1),
        ("retrograde", tuple(-value for value in intervals[::-1]), 1),
        ("retrograde inversion", intervals[::-1], 2),
    )


def match(shape: Shape, entry: Shape) -> tuple[str, float] | None:
    """The cheapest way to write ``shape`` as a transform of ``entry``, plus repairs.

    Exact matching was the first version of this and it was useless: two
    phrases in a real piece are almost never bit-identical under a transform,
    so nothing ever matched, the measurement read zero everywhere, and the
    criterion could not tell any candidate from any other.

    So a reference may be **approximate**. "Entry four, inverted, with the
    third interval a step wider" is a legitimate description and it costs what
    the repairs cost. If the repairs cost more than spelling the phrase out,
    :func:`cost` will spell it out — which is exactly the trade minimum
    description length is supposed to make.
    """
    if len(shape.steps) < MIN_LENGTH or len(entry.steps) < MIN_LENGTH:
        return None

    mine, theirs = shape.intervals, entry.intervals
    if not mine or not theirs:
        return None

    best: tuple[str, float] | None = None
    for name, turned, params in variants(theirs):
        for offset in range(max(1, len(turned) - len(mine) + 1)):
            window = turned[offset:offset + len(mine)]
            if not window:
                continue
            overlap = min(len(mine), len(window))
            repairs = sum(1 for a, b in zip(mine[:overlap], window[:overlap]) if a != b)
            missing = abs(len(mine) - len(window))
            priced = (PARAM * params
                      + PARAM * (1 if offset else 0)      # where in the entry it starts
                      + PARAM * (1 if missing else 0)     # how much of it is used
                      + REPAIR * repairs
                      + NOTE * missing)                   # anything left over is spelled out
            label = name if not offset and not missing else f"{name} fragment"
            if best is None or priced < best[1]:
                best = (label, priced)
    return best


def rhythm_cost(shape: Shape, entry: Shape) -> float:
    """What it costs to say how the borrowed shape is spoken.

    A uniform stretch or squeeze is one parameter. Anything else has to be
    spelled out, and spelling out durations is half the price of spelling out
    whole notes, because the pitches came free with the reference.
    """
    if len(shape.rhythm) == len(entry.rhythm):
        for scale in RHYTHM_SCALES:
            if all(abs(mine - theirs * scale) < 1e-9
                   for mine, theirs in zip(shape.rhythm, entry.rhythm)):
                return PARAM if scale != 1.0 else 0.0
    return (NOTE / 2.0) * len(shape.rhythm)


def cost(shape: Shape, vocabulary: tuple[Shape, ...]) -> tuple[float, str | None]:
    """The cheapest way to describe ``shape``, and what it leaned on."""
    best, source = shape.literal(), None
    for entry in vocabulary:
        if entry.label == shape.label:
            continue    # a thing cannot be a reference to itself
        found = match(shape, entry)
        if found is None:
            continue
        name, params = found
        priced = HEADER + REFERENCE + params + rhythm_cost(shape, entry)
        if priced < best:
            best, source = priced, f"{entry.label} {name}"
    return best, source


def describe(shapes: tuple[Shape, ...], vocabulary: tuple[Shape, ...]) -> float:
    return sum(cost(shape, vocabulary)[0] for shape in shapes)


def describe_in_order(shapes: tuple[Shape, ...], prior: tuple[Shape, ...]) -> float:
    """Cost of describing each shape using only what existed before it.

    This is what the opening actually cost *at the time*. The first version
    priced the opening against all of itself at once, which quietly let bar
    four refer to bar forty — and since a lineage makes consecutive phrases
    near-identical, that made the opening look almost fully compressed before
    the piece had even happened, and left hindsight nothing to buy.
    """
    return sum(cost(shape, prior + shapes[:index])[0]
               for index, shape in enumerate(shapes))


def literal(shapes: tuple[Shape, ...]) -> float:
    return sum(shape.literal() for shape in shapes)


# ── the measurement itself ────────────────────────────────────────────────

def describe_with_hindsight(opening: tuple[Shape, ...], prior: tuple[Shape, ...],
                            later: tuple[Shape, ...]) -> float:
    """Cost of the opening once the rest of the piece is available to refer to.

    Each phrase still only refers backwards *within the opening* — its
    neighbours explaining it is not hindsight, it is just a lineage — plus
    everything the piece went on to become. So the difference against
    :func:`describe_in_order` is precisely what the ending gave back to the
    beginning, and nothing else.
    """
    return sum(cost(shape, prior + opening[:index] + later)[0]
               for index, shape in enumerate(opening))


def reveal(opening: tuple[Shape, ...], prior: tuple[Shape, ...],
           later: tuple[Shape, ...]) -> Reveal:
    """Describe the opening twice: as it was written, and knowing how it ends.

    ``prior`` is whatever existed before the opening began — in practice the
    germ. ``later`` is everything the piece played afterwards.
    """
    return Reveal(
        then=describe_in_order(opening, prior),
        now=describe_with_hindsight(opening, prior, later),
        literal=literal(opening),
        phrases=len(opening),
        vocabulary_then=len(prior) + len(opening),
        vocabulary_now=len(prior) + len(opening) + len(later),
    )


def baseline(opening: tuple[Shape, ...], prior: tuple[Shape, ...]) -> tuple[float, ...]:
    """What each opening phrase cost to write, given only what preceded it.

    These are the numbers hindsight is trying to beat, so they are what a
    candidate should be scored against. Computed once per movement rather than
    once per candidate — the past does not change while an audition runs.
    """
    return tuple(cost(shape, prior + opening[:index])[0]
                 for index, shape in enumerate(opening))


def explains(candidate: Shape, past: tuple[Shape, ...],
             paid: tuple[float, ...] = ()) -> float:
    """How much of the piece's own past this one shape would account for.

    The first version of this asked the marginal question — how much *cheaper*
    the past gets if this candidate joins the vocabulary — and it was flat zero
    everywhere. The reason is structural and worth keeping: every phrase
    descends from the one before it, so by the second half the vocabulary is
    dense with near neighbours and one more entry saves nothing.

    The useful question is not marginal, it is absolute: **if you had to
    explain the opening using this one shape as the key, how far would it get
    you?** That is what "five anomalies turn out to be one pattern rotated
    five ways" actually means, and unlike the marginal version it tells one
    candidate from another.

    ``paid`` is what each past phrase actually cost at the time
    (:func:`baseline`). Scoring against that rather than against the literal
    cost is what makes this criterion aim at the same number the measurement
    reports: a phrase that was already cheap when it was written has nothing
    left for a revelation to buy, however neatly a candidate matches it.

    Returned as a share of what the past cost, so it is comparable across
    pieces and lengths.
    """
    if not past:
        return 0.0
    costs = paid if len(paid) == len(past) else tuple(shape.literal() for shape in past)
    total = sum(costs)
    if total <= 0:
        return 0.0

    saved = 0.0
    for shape, already in zip(past, costs):
        found = match(shape, candidate)
        if found is None:
            continue
        _, params = found
        priced = HEADER + REFERENCE + params + rhythm_cost(shape, candidate)
        saved += max(0.0, already - priced)
    return min(1.0, saved / total)
