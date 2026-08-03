"""What the piece is trying to do to itself.

Everything else in this engine reacts. The critic hears a problem and corrects
it; the ledger notices a debt and settles it; the mixer finds a buried voice
and lifts it. All of that is a piece *responding*, and a piece that only
responds has no plan — which is exactly what was missing when the question was
"how does a machine know what something is for".

An intention here is not an emotion and not a destination. It is a **target
shape for the composer's own internal state over time**: where its unrest
should be at the halfway point, how much it should owe by the end, how torn its
decisions should be in the middle. Build, drop, tension and release are not
named anywhere in this file — they are what those curves produce.

Two rules keep it honest.

**It constrains rather than prefers.** There is no parallel scorer — fifteen
criteria and a sixteenth objective on top would be a piece that sounds like the
average of everything. But nor is an intent another opinion among the fifteen,
which is what the first two versions of this file tried and what neither could
make work. It withholds permission to settle a debt; it makes the audition
imagine more lines than it wanted to; it puts the voices further apart than the
mixer asked. Every one of those bites where the music is built, and nothing
downstream gets a vote.

**It knows which way its own levers run.** Each pairing was measured before it
was written down, and each is monotone, which is what lets the plan hold a
lever on the strength of the shape alone instead of watching the output and
chasing the difference. Chasing was tried twice and measured at nothing both
times; the arithmetic for why is in :func:`pull`.

**It can quit.** When the measurements sit on the target for two movements
running *and* every decision is being won by a mile, the intent is being
satisfied trivially — the piece has stopped being about anything — and it
swaps for another. That is the same second-order move as plasticity, one
storey up: the state evolves, the rule that updates the state evolves, and now
the thing the rule is aiming at evolves too.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from compex import rng
from compex.generate.mood import Mood

ENABLED = True          # the whole mechanism, off
REACH = 0.5             # how hard an intent may pull a weight per listen-back
SETTLED = 0.06          # error below this counts as sitting on the target
DECISIVE = 0.08         # margins wider than this mean nothing is being decided
BORED_AFTER = 2         # movements of both before the intent gives up

#: The internal quantities an intent can aim at, and the lever that reaches
#: each one. Every pairing here was **measured** before it was written down —
#: an intent aiming at something it cannot move would be exactly the theatre
#: this project keeps finding and deleting.
#:
#: Two candidates were cut for failing that test. ``unrest`` is a running mean
#: of how badly the principles are doing, and no lever moves it directly.
#: ``motion`` — spectral flux — could not be moved by the timbre drift at all:
#: measured across three pieces, drift at 0.0 gave 0.584 and at 1.2 gave 0.569,
#: which is noise in the wrong direction. Note changes dominate the flux, so
#: the instruments can travel a long way underneath and the measure barely
#: notices.
#: A third candidate was cut, and it took the first version of this whole file
#: with it. Those levers were **taste weights** — care more about settling, be
#: more curious, want more space — and none of them could steer anything. Over
#: twenty pieces, matched seed for seed with the plan switched off, the state
#: followed its target shape *worse* with the plan on: a paired difference of
#: −0.036 ± 0.030 in correlation, which is nothing, pointing the wrong way.
#:
#: The reason is worth keeping. A weight competes with fourteen other weights
#: and then gets retuned by the critic the moment the movement ends, so its
#: whole authority is a few percent of a quantity that swings by a factor of
#: three on its own. A controller whose hand is that much weaker than the thing
#: it is holding is not a controller.
#:
#: So an intention here is a **constraint, not a preference**. It does not ask
#: the composer to care more about closing its debts; it withholds permission
#: to close them. It does not nudge curiosity up; it makes the audition imagine
#: more lines. It does not want more space; it moves the voices apart. Every
#: one of these bites at the point the music is built, where nothing downstream
#: can politely disagree.
LEVERS: dict[str, str] = {
    "pressure": "settling",    # permission to close what was opened
    "margin": "field",         # how many lines the audition has to choose from
    "crowding": "spread",      # how far apart the voices are put
}
VARIABLES: tuple[str, ...] = tuple(LEVERS)

#: How far along its range the plan is allowed to hold a lever. At 0.9 a curve
#: asking for the top of a quantity holds its lever near the bottom of what the
#: bounds permit, which is as much authority as there is to give.
#:
#: Chosen on eight pieces that are not in the test that checks this works, for
#: the obvious reason: the first time round the gain was tuned on the same four
#: pieces the improvement was then measured on, and a result like that is
#: measuring the tuning. On the held-out eight, agreement with the target shape
#: went −0.008 with the plan off, to +0.078 at 0.5, to +0.204 at 0.9.
GAIN = 0.9
BIAS_BOUNDS = (0.3, 3.0)

#: Curves are **shapes**, not levels — every point is 0..1, meaning "near the
#: bottom of what this piece can do with that quantity" to "near the top".
#:
#: The first version wrote them as absolute numbers and it could not work. What
#: a piece owes runs to five or six however hard the levers are pushed; the
#: curve was asking for 0.2 and for 8.0, both unreachable, so the plan spent
#: every movement pulling at a wall. Levels are a property of the piece; a plan
#: is a property of the shape. Measuring against the piece's own range is what
#: makes an intent a plan rather than a wish.

#: Movements of watching before the piece's own range is worth quoting. This
#: no longer gates anything the piece does — the levers are held from the first
#: listen-back — but :meth:`Plan.agreement` has to say where a reading sits
#: inside the range that piece actually reached, and a range built from two
#: readings is not a range.
SPAN = 3


@dataclass(frozen=True)
class Curve:
    """A target for one quantity, sampled across the piece."""

    variable: str
    points: tuple[float, ...]

    def at(self, position: float) -> float:
        """The target at this fraction of the way through, interpolated."""
        if not self.points:
            return 0.0
        if len(self.points) == 1:
            return self.points[0]
        span = max(0.0, min(1.0, position)) * (len(self.points) - 1)
        low = int(span)
        high = min(low + 1, len(self.points) - 1)
        blend = span - low
        return self.points[low] * (1.0 - blend) + self.points[high] * blend


@dataclass(frozen=True)
class Intent:
    """One plan: a name, and the shape it wants its own state to take."""

    name: str
    why: str
    curves: tuple[Curve, ...] = ()

    def target(self, variable: str, position: float) -> float | None:
        for curve in self.curves:
            if curve.variable == variable:
                return curve.at(position)
        return None

    def describe(self) -> str:
        return f"{self.name} — {self.why}"


#: The plans available. Each is a shape, not a feeling, and each is reachable
#: with levers that already exist.
INTENTS: tuple[Intent, ...] = (
    Intent(
        "settle",
        "end owing nothing: it takes something on early and puts it down, and "
        "the decisions get easier as it goes",
        (Curve("pressure", (0.5, 0.9, 0.6, 0.3, 0.0)),
         Curve("margin", (0.1, 0.3, 0.5, 0.7, 1.0)),
         Curve("crowding", (0.8, 0.6, 0.4, 0.2, 0.0))),
    ),
    Intent(
        "carry",
        "take on something it does not put down: what it owes climbs and stays "
        "there, and it never stops having to choose",
        (Curve("pressure", (0.0, 0.4, 0.7, 0.9, 1.0)),
         Curve("margin", (0.6, 0.4, 0.3, 0.1, 0.1)),
         Curve("crowding", (0.2, 0.4, 0.6, 0.8, 0.9))),
    ),
    Intent(
        "unravel",
        "become less sure of itself: the field closes up, the voices pile into "
        "each other, and every decision gets harder than the last",
        (Curve("pressure", (0.1, 0.3, 0.6, 0.8, 1.0)),
         Curve("margin", (1.0, 0.7, 0.4, 0.2, 0.0)),
         Curve("crowding", (0.0, 0.3, 0.6, 0.8, 1.0))),
    ),
    Intent(
        "harden",
        "arrive somewhere fixed: it stops accumulating, the voices separate, "
        "and it stops arguing with itself",
        (Curve("pressure", (1.0, 0.7, 0.4, 0.2, 0.0)),
         Curve("margin", (0.0, 0.3, 0.5, 0.8, 1.0)),
         Curve("crowding", (1.0, 0.7, 0.4, 0.2, 0.0))),
    ),
)


@dataclass(frozen=True)
class Reading:
    """What the piece's internal state actually was at one listen-back."""

    position: float
    values: tuple[tuple[str, float], ...] = ()
    targets: tuple[tuple[str, float], ...] = ()

    def error(self, variable: str) -> float:
        measured = dict(self.values).get(variable)
        wanted = dict(self.targets).get(variable)
        if measured is None or wanted is None:
            return 0.0
        return measured - wanted

    def distance(self, ranges: dict[str, tuple[float, float]] | None = None) -> float:
        """How far the state sits from the shape, in the piece's own units.

        Each quantity is placed inside the range that piece actually reached,
        so "high pressure" means high *for this piece* rather than high against
        a number chosen in advance and never available.
        """
        spans = ranges or {}
        errors = []
        for name, wanted in self.targets:
            measured = dict(self.values).get(name)
            if measured is None:
                continue
            low, high = spans.get(name, (measured, measured))
            errors.append(abs(place(measured, low, high) - wanted))
        return sum(errors) / len(errors) if errors else 0.0


def _agree(got: list[float], wanted: list[float]) -> float | None:
    """Pearson correlation, or None when one of the two never moved."""
    count = len(got)
    if count < 3 or count != len(wanted):
        return None
    mean_got = sum(got) / count
    mean_wanted = sum(wanted) / count
    covariance = sum((a - mean_got) * (b - mean_wanted) for a, b in zip(got, wanted))
    spread_got = sum((a - mean_got) ** 2 for a in got) ** 0.5
    spread_wanted = sum((b - mean_wanted) ** 2 for b in wanted) ** 0.5
    if spread_got < 1e-9 or spread_wanted < 1e-9:
        return None
    return covariance / (spread_got * spread_wanted)


def place(value: float, low: float, high: float) -> float:
    """Where a value sits inside a range, 0..1. A flat range sits in the middle."""
    if high - low < 1e-9:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


@dataclass(frozen=True)
class Turn:
    """A moment where the plan itself changed, and why."""

    movement: int
    was: str
    now: str
    because: str

    def describe(self) -> str:
        return f"[{self.movement}] gave up on {self.was} for {self.now} — {self.because}"


@dataclass(frozen=True)
class Plan:
    """The intent, its history, the bias it is applying, and how well it is going."""

    intent: Intent
    readings: tuple[Reading, ...] = ()
    turns: tuple[Turn, ...] = ()
    satisfied_for: int = 0
    #: Cumulative multipliers on the levers. A plan that only nudged once per
    #: movement was undone by the critic's own retuning before it arrived
    #: anywhere — measured, and the reason this is state rather than a return
    #: value.
    bias: tuple[tuple[str, float], ...] = ()
    frozen: tuple[tuple[str, tuple[float, float]], ...] = ()

    def lever(self, name: str) -> float:
        for key, value in self.bias:
            if key == name:
                return value
        return 1.0

    def calibrated(self) -> dict[str, tuple[float, float]]:
        """The range this piece runs to: learned from the watching movements,
        then widened but never narrowed.

        Held completely still it was wrong in one specific way. A quiet piece
        can open owing nothing at all, and three readings of exactly zero are
        not a range — so pressure fell out of the calibration and the strongest
        lever in the file sat switched off for the rest of the piece, silently,
        on exactly the pieces where taking something on would have mattered
        most. Widening fixes that without reintroducing the original problem:
        the setpoint only moves when the piece has genuinely gone somewhere it
        had not been, and it never drifts back toward wherever the state
        happens to be sitting.
        """
        if len(self.readings) < SPAN:
            return {}
        return dict(self.frozen)

    def ranges(self) -> dict[str, tuple[float, float]]:
        """The span each quantity actually covered in this piece."""
        return self._ranges(self.readings)

    @staticmethod
    def _ranges(readings) -> dict[str, tuple[float, float]]:
        seen: dict[str, list[float]] = {}
        for reading in readings:
            for name, value in reading.values:
                seen.setdefault(name, []).append(value)
        return {name: (min(values), max(values)) for name, values in seen.items()
                if len(values) >= 2 and max(values) - min(values) > 1e-9}

    def followed(self) -> float:
        """Mean distance from the target shape. Lower is a plan being kept."""
        if len(self.readings) < 2:
            return 0.0
        spans = self.ranges()
        return sum(reading.distance(spans) for reading in self.readings) / len(self.readings)

    def agreement(self) -> float:
        """How well the state actually tracked the shape it was aiming at.

        Correlation, per quantity, averaged — and it is the correlation rather
        than the distance because distance can be won by cheating. A piece
        whose pressure never moves at all sits in the middle of its own range
        for the whole run and scores a respectable distance from any curve,
        while having followed nothing. Correlation gives a flat line zero,
        which is the honest mark for a plan that did not steer.

        One at the top, zero for a plan that was decoration, negative for a
        piece that did the opposite of what it set out to do.
        """
        spans = self.ranges()
        scores = []
        for name, (floor, ceiling) in spans.items():
            got = [place(dict(reading.values)[name], floor, ceiling)
                   for reading in self.readings if name in dict(reading.values)]
            wanted = [dict(reading.targets)[name] for reading in self.readings
                      if name in dict(reading.targets)]
            score = _agree(got, wanted)
            if score is not None:
                scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def summary(self) -> str:
        if not ENABLED:
            return "no intent — reacting only"
        turns = f", changed its mind {len(self.turns)}x" if self.turns else ""
        return (f"{self.intent.name}: {self.intent.why}"
                f" (followed to {self.agreement():+.2f}{turns})")


def choose(seed: int, mood: Mood) -> Plan:
    """Pick what this piece is trying to do to itself.

    Weighted by mood, because a plan that fought the mood would just be a
    second mood. A tense piece is more likely to unravel; a still one to settle.
    """
    if not ENABLED:
        return Plan(intent=Intent("none", "reacting only"))

    weights = {
        "settle": 1.0 + (1.0 - mood.tension) * 1.4,
        "carry": 0.7 + mood.tension * 1.2,
        "unravel": 0.4 + mood.tension * 1.6 * mood.energy,
        "harden": 0.5 + mood.grit * 1.5,
    }
    total = sum(weights.values())
    target = rng.uniform(seed, "intent", 0) * total
    running = 0.0
    for intent in INTENTS:
        running += weights[intent.name]
        if target <= running:
            return Plan(intent=intent)
    return Plan(intent=INTENTS[0])


def read(plan: Plan, position: float, pressure: float, margin: float,
         crowding: float) -> Reading:
    """Compare where the piece's state actually is against where it should be."""
    values = {"pressure": pressure, "margin": margin, "crowding": crowding}
    targets = {name: plan.intent.target(name, position) for name in VARIABLES}
    return Reading(
        position=position,
        values=tuple(sorted(values.items())),
        targets=tuple(sorted((name, value) for name, value in targets.items()
                             if value is not None)),
    )


def pull(plan: Plan, reading: Reading) -> tuple[tuple[str, float], ...]:
    """Set the levers to where the shape says they should be.

    This is the third design and the first that works, and the reason is that
    the first two were both feedback: measure the state, compare it to the
    target, correct the difference. Feedback is the obvious answer and it is
    the wrong one here, for reasons that only showed up once they were counted.

    A piece gets eight or nine listen-backs. Three go on watching what its own
    range is, which leaves five corrections — and both levers with any reach
    are integer-valued, so a correction smaller than one whole extra candidate
    or one whole octave does nothing at all. Meanwhile the quantity being
    steered swings by three or four times what the lever can shift it. That is
    a controller with a noisy sensor, five moves, a deadband, and an actuator
    weaker than the disturbance. Measured over twenty pieces, matched seed for
    seed: the integral version scored −0.041 ± 0.037 against no plan at all,
    and the proportional version +0.001 ± 0.029. Both are nothing.

    So the plan stops measuring and starts *holding*. Every coupling in
    :data:`LEVERS` was measured to be monotone before it was written down —
    more candidates always narrows the margin, more spacing always reduces
    crowding, less permission always leaves more owed — and when you know the
    sign of the plant you do not need to watch the output to know which way to
    push. The lever traces the target curve directly. Nothing lags, nothing
    hunts, and the five listen-backs go back to being what they were for.

    What the readings are still for is the honest part: :meth:`Plan.agreement`
    reports how much of the shape actually survived contact with the piece, and
    it is free to say "none of it". On the same twenty pieces that killed the
    other two versions it says **+0.136 ± 0.046** better than no plan at all,
    improving in eighteen of them — mean agreement +0.110 with the plan off
    against +0.245 with it holding. That is the whole claim, and it is a claim
    about the state of the composer rather than about the music: a listener
    hears a piece that gets more crowded, not a piece that meant to.
    """
    if not ENABLED:
        return plan.bias

    low, high = BIAS_BOUNDS
    held = dict(plan.bias)
    for variable, lever in LEVERS.items():
        wanted = dict(reading.targets).get(variable)
        if wanted is None:
            continue
        # Every lever runs the opposite way to the thing it holds, so wanting a
        # quantity high means holding its lever low.
        held[lever] = max(low, min(high, 1.0 + (0.5 - wanted) * 2.0 * GAIN))
    return tuple(sorted(held.items()))


def reconsider(plan: Plan, reading: Reading, seed: int, movement: int) -> Plan:
    """Keep the plan, or give it up for being too easily satisfied.

    A piece sitting exactly on its target while winning every decision by a
    mile is not succeeding, it is coasting: nothing is being decided and
    nothing is at stake. That is the moment to want something else.
    """
    seen = plan.readings + (reading,)
    frozen = dict(plan.frozen)
    if len(seen) >= SPAN:
        for name, (floor, ceiling) in Plan._ranges(seen).items():
            was = frozen.get(name)
            frozen[name] = (min(floor, was[0]), max(ceiling, was[1])) if was else (floor, ceiling)
    updated = replace(plan, readings=seen, frozen=tuple(sorted(frozen.items())),
                      bias=pull(plan, reading))
    if not ENABLED:
        return updated

    coasting = (reading.distance(updated.calibrated()) < SETTLED
                and dict(reading.values).get("margin", 0.0) > DECISIVE)
    count = updated.satisfied_for + 1 if coasting else 0
    updated = replace(updated, satisfied_for=count)
    if count < BORED_AFTER:
        return updated

    others = tuple(intent for intent in INTENTS if intent.name != plan.intent.name)
    if not others:
        return updated
    nxt = rng.pick(seed, "intent-turn", movement, others)
    return replace(
        updated,
        intent=nxt,
        satisfied_for=0,
        turns=updated.turns + (Turn(
            movement=movement, was=plan.intent.name, now=nxt.name,
            because="it was being satisfied without anything being decided",
        ),),
    )
