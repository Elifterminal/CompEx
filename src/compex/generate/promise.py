"""What the music has promised, and has not yet done.

Every gesture opens an account. A leap owes a step back the other way. A line
that stops a semitone under a chord tone owes the note above it. A pattern
that keeps landing off the beat owes the beat. A progression that walks a long
way from home owes either a return or a reason.

Before this, the composer judged each phrase against the ones around it and
had no memory of anything it had left hanging — so nothing could be *carried*.
A ledger is what makes carrying possible, and carrying is what the difference
between a solver and a composer rests on. A solver wants its uncertainty gone.
This wants to choose which uncertainty is worth keeping.

The thing that makes any of this computable rather than metaphorical is
**maturity**. A promise ripens: it is worth little immediately after it opens,
most when it has been outstanding long enough for the ear to be waiting, and
then it fades, because a debt nobody remembers is not a debt. Pressure is
strength times maturity, and everything else — *is now the right time? is this
too obvious? has this become the identity of the piece?* — falls out of that
one curve instead of being asserted.

Nothing here decides anything. It records what is owed and can say whether a
proposed phrase would settle it. The deciding stays in the audition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from compex.generate.theory import degree_semitone

#: The whole mechanism, off. With this False no promise is ever opened, the
#: criterion is dropped from the audition entirely, and the composer writes
#: exactly what it wrote before there was a ledger.
ENABLED = True

LEAP_DEGREES = 2       # a move larger than this owes an answer
RIPEN_BEATS = 24.0     # how long a promise takes to become worth settling
FORGET_BEATS = 112.0   # after this it starts fading
FADE_BEATS = 72.0      # how fast it fades once forgetting starts
LIVE_FLOOR = 0.04      # below this maturity a promise is no longer on the books
RIPE = 0.45            # settle below this and you settled it too early
LEADING_WINDOW = 1.35  # semitones: close enough under a resting note to lean on it

DOMAINS = ("pitch", "metre", "harmony")


@dataclass(frozen=True)
class Promise:
    """One outstanding obligation, and what would settle it."""

    kind: str            # leap · leading · syncopation · distance
    domain: str          # pitch · metre · harmony
    strength: float      # how large the gesture that opened it was, 0..1
    opened_at: float     # beats from the top of the piece
    detail: tuple[tuple[str, float], ...] = ()

    def get(self, name: str, default: float = 0.0) -> float:
        for key, value in self.detail:
            if key == name:
                return value
        return default

    def maturity(self, now: float) -> float:
        """0 when it has just opened, 1 once the ear is waiting, fading after.

        The fade is the part people leave out. A promise that can never expire
        makes a composer that is still trying to answer bar four in bar four
        hundred, and it turns the ledger into a monotonically growing pile.
        """
        age = max(0.0, now - self.opened_at)
        ripe = min(1.0, age / RIPEN_BEATS) if RIPEN_BEATS > 0 else 1.0
        if age > FORGET_BEATS:
            ripe *= math.exp(-(age - FORGET_BEATS) / max(FADE_BEATS, 1e-6))
        return ripe

    def pressure(self, now: float) -> float:
        return self.strength * self.maturity(now)

    def forgotten(self, now: float) -> bool:
        """Has this faded past the point of being owed?

        Unripe is not the same as forgotten, and conflating the two is a very
        easy way to build a ledger that is always empty: a promise is worth
        nothing the instant it opens, so a filter on maturity alone deletes
        everything the moment it is written down.
        """
        return (now - self.opened_at) > FORGET_BEATS and self.maturity(now) <= LIVE_FLOOR

    def describe(self, now: float | None = None) -> str:
        tail = f", pressure {self.pressure(now):.2f}" if now is not None else ""
        return (f"{self.kind} ({self.domain}) opened at beat {self.opened_at:g}, "
                f"strength {self.strength:.2f}{tail}")


@dataclass(frozen=True)
class Settlement:
    """A promise, and the moment it was answered."""

    promise: Promise
    at_beat: float
    how: str
    waited: float        # beats it stayed open

    def describe(self) -> str:
        return (f"{self.promise.kind} settled at beat {self.at_beat:g} by {self.how} "
                f"after {self.waited:g} beats")


@dataclass(frozen=True)
class Ledger:
    """Everything owed, and everything already paid. Immutable, like the rest."""

    open: tuple[Promise, ...] = ()
    paid: tuple[Settlement, ...] = ()
    given_up: tuple[Promise, ...] = ()

    def live(self, now: float, domain: str | None = None) -> tuple[Promise, ...]:
        """Promises still worth answering — forgotten ones fall off here."""
        return tuple(p for p in self.open
                     if not p.forgotten(now)
                     and (domain is None or p.domain == domain))

    def pressure(self, now: float, domain: str | None = None) -> float:
        return sum(p.pressure(now) for p in self.live(now, domain))

    def deepest(self, now: float) -> Promise | None:
        """The one that has been carried longest and still counts.

        This is the candidate for being what the piece is *about* — a strong
        promise still open while smaller ones around it keep being settled.
        """
        live = self.live(now)
        return max(live, key=lambda p: p.pressure(now)) if live else None

    def opened(self, promises: tuple[Promise, ...]) -> "Ledger":
        return replace(self, open=self.open + tuple(promises)) if promises else self

    def settle(self, promise: Promise, at_beat: float, how: str) -> "Ledger":
        if promise not in self.open or at_beat <= promise.opened_at:
            # Nothing is settled in the same instant it was promised. Two
            # voices writing at the same beat could otherwise open an
            # obligation and answer it before anyone heard it exist, which
            # reads in the ledger as a debt "settled after 0 beats".
            return self
        remaining = list(self.open)
        remaining.remove(promise)
        return replace(
            self,
            open=tuple(remaining),
            paid=self.paid + (Settlement(promise, at_beat, how, at_beat - promise.opened_at),),
        )

    def forget(self, now: float) -> "Ledger":
        """Drop what has faded, so the ledger cannot grow without bound."""
        return replace(self, open=self.live(now))

    def let_go(self, now: float, count: int) -> "Ledger":
        """Abandon the debts being carried hardest. Not the same as settling.

        Settling is answering something; this is deciding it was not what the
        piece is about after all. It has to be explicit rather than a matter of
        waiting, because a promise does not fade merely by being ignored — the
        maturity curve holds it at full weight for the whole of its term, which
        is what makes carrying one mean anything.

        The deepest go first on purpose. Letting go of a debt nobody was still
        hearing costs nothing; what actually moves a piece is dropping the one
        it had been about.
        """
        if count <= 0:
            return self
        doomed = sorted(self.live(now), key=lambda p: -p.pressure(now))[:count]
        if not doomed:
            return self
        return replace(self, open=tuple(p for p in self.open if p not in doomed),
                       given_up=self.given_up + tuple(doomed))

    def summary(self, now: float) -> str:
        live = self.live(now)
        deepest = self.deepest(now)
        carried = f", carrying {deepest.kind} for {now - deepest.opened_at:g} beats" \
            if deepest else ""
        gone = f", {len(self.given_up)} given up" if self.given_up else ""
        return (f"{len(live)} open ({self.pressure(now):.2f} pressure), "
                f"{len(self.paid)} settled{gone}{carried}")


# ── what opens an account ─────────────────────────────────────────────────

def from_phrase(steps: tuple[int, ...], chord_degree: int, scale: tuple[int, ...],
                at_beat: float, approach: int | None = None) -> tuple[Promise, ...]:
    """Pitch obligations the chosen line has just taken on.

    Only the ones it did not settle itself. A leap answered inside the phrase
    is a completed gesture, not a debt — writing it to the ledger would make
    the composer pay for something it already did.

    ``approach`` matters more than it looks. The step *into* a phrase is where
    the biggest unanswered leaps live, because until now nothing judged it and
    nothing remembered it. Handing it in here is what makes the ledger cover
    the joins rather than only the insides.
    """
    if not ENABLED or len(steps) < 2:
        return ()
    landings = list(steps)
    moves = [b - a for a, b in zip(steps, steps[1:])]
    if approach is not None:
        moves.insert(0, (chord_degree + steps[0]) - approach)
        landings.insert(0, steps[0])
    out: list[Promise] = []

    for index, move in enumerate(moves):
        if abs(move) <= LEAP_DEGREES:
            continue
        nxt = moves[index + 1] if index + 1 < len(moves) else None
        if nxt is not None and move * nxt < 0 and abs(nxt) <= LEAP_DEGREES:
            continue  # answered where it happened
        out.append(Promise(
            kind="leap", domain="pitch",
            strength=min(1.0, abs(move) / 5.0),
            opened_at=at_beat,
            detail=(("degree", float(chord_degree + landings[min(index + 1, len(landings) - 1)])),
                    ("direction", -1.0 if move > 0 else 1.0)),
        ))

    landing = chord_degree + steps[-1]
    # Measured in cents from the tonic rather than by index, because the piece
    # may not be in twelve — a leading tone is anything that sits close enough
    # underneath a resting point to lean on it, whatever the grid is called.
    within = degree_semitone(scale, landing) % 12.0
    if 12.0 - within <= LEADING_WINDOW:
        out.append(_leading(landing, +1, at_beat))
    elif within <= LEADING_WINDOW and within > 1e-9:
        out.append(_leading(landing, -1, at_beat))
    return tuple(out)


def _leading(landing: int, direction: int, at_beat: float) -> Promise:
    return Promise(
        kind="leading", domain="pitch", strength=0.8, opened_at=at_beat,
        detail=(("degree", float(landing + direction)), ("direction", float(direction))),
    )


def from_pattern(slots: tuple[float, ...], weights: tuple[float, ...], subdivision: float,
                 at_beat: float, voice: str, strong: float = 0.55) -> tuple[Promise, ...]:
    """A metric obligation: hits off the beat with nothing on it.

    Syncopation is only syncopation against something. A voice that plays the
    weak positions and never the strong one it implies has displaced a pulse
    that has not landed yet.
    """
    if not ENABLED or not slots:
        return ()
    hit = [index for index, velocity in enumerate(slots) if velocity > 0]
    if not hit or not weights:
        return ()
    # The obligation is to *the* strong position, not to all of them. A voice
    # that plays around the downbeat and never on it has displaced one pulse,
    # however many lesser beats it happens to catch.
    target = max(range(len(weights)), key=lambda index: weights[index])
    if weights[target] < strong or target in hit:
        return ()

    weak_share = len(hit) / max(1, len(slots))
    return (Promise(
        kind="syncopation", domain="metre",
        strength=min(1.0, 0.4 + weak_share),
        opened_at=at_beat,
        detail=(("slot", float(target)), ("subdivision", subdivision)),
    ),)


def from_harmony(degree: int, scale: tuple[int, ...], at_beat: float) -> tuple[Promise, ...]:
    """A harmonic obligation: the progression has walked a long way from home.

    Distance is counted in fifths rather than in scale steps, because that is
    what actually sounds far — the second degree is next door on the keyboard
    and two moves away round the circle.
    """
    if not ENABLED:
        return ()
    distance = fifth_distance(degree, len(scale))
    if distance < 2:
        return ()
    return (Promise(
        kind="distance", domain="harmony",
        strength=min(1.0, distance / 3.0),
        opened_at=at_beat,
        detail=(("degree", float(degree % max(1, len(scale)))),),
    ),)


def fifth_distance(degree: int, span: int) -> int:
    """How many fifths from the tonic this degree sits, either way round."""
    if span <= 1:
        return 0
    fifth = max(1, round(span * 7 / 12))     # 4 steps in a 7-note scale, 3 in a pentatonic
    for turns in range(span):
        if (fifth * turns) % span == degree % span:
            return min(turns, span - turns)
    return 0


# ── what settles one ──────────────────────────────────────────────────────

def would_settle(promise: Promise, steps: tuple[int, ...], chord_degree: int) -> bool:
    """Would this candidate line answer this pitch promise?"""
    if promise.domain != "pitch" or not steps:
        return False
    target = promise.get("degree")
    direction = promise.get("direction")

    if promise.kind == "leading":
        return any(abs((chord_degree + step) - target) < 1e-6 for step in steps)

    # A leap is answered by a step in the other direction landing near where
    # the leap left off — the same rule the critic measures, carried across
    # phrase boundaries instead of dying at the end of the bar.
    moves = [b - a for a, b in zip(steps, steps[1:])]
    for index, move in enumerate(moves):
        if move == 0 or abs(move) > LEAP_DEGREES:
            continue
        if (move > 0) != (direction > 0):
            continue
        if abs((chord_degree + steps[index]) - target) <= 1.0:
            return True
    return False


def would_settle_any(owed: tuple[Promise, ...], steps: tuple[int, ...],
                     chord_degree: int) -> bool:
    """Would this line answer anything currently outstanding?"""
    return any(would_settle(promise, steps, chord_degree) for promise in owed)


def pattern_settles(promise: Promise, slots: tuple[float, ...], subdivision: float) -> bool:
    """Would this candidate pattern land the beat a syncopation displaced?"""
    if promise.domain != "metre" or not slots:
        return False
    if abs(subdivision - promise.get("subdivision", subdivision)) > 1e-9:
        return False
    slot = int(promise.get("slot"))
    return 0 <= slot < len(slots) and slots[slot] > 0


def credit(promise_maturity: float, settling: bool, ripening: float) -> float:
    """Score one candidate's treatment of the ledger, 0..1.

    The ordering is the whole point and it is asserted in the tests:
    **settling early < carrying < settling ripe.** Paying a debt nobody is
    waiting for yet is the definition of too obvious, and it is the failure
    this criterion exists to make expensive.
    """
    if settling:
        if promise_maturity >= RIPE:
            return promise_maturity
        return promise_maturity * 0.4
    return min(0.6, 0.25 + 0.35 * ripening)


def trace(ledger: Ledger, now: float) -> str:
    """A readable account of what the piece owes and paid, for the formula."""
    lines = [f"ledger: {ledger.summary(now)}"]
    lines += [f"    open  {p.describe(now)}" for p in ledger.live(now)]
    lines += [f"    paid  {s.describe()}" for s in ledger.paid]
    return "\n".join(lines)
