"""The deciding mechanism: how the composer changes its mind mid-piece.

The critic measures what was written. This decides what to do about it.

Two things move. **Drives** are the parameters the composer writes with —
how hard it pushes for new material, how far it lets the line leap, how much
dissonance it will accept. Those shift in response to the critic.

And **plasticity** — how strongly the drives respond at all — moves too. When
the same complaints keep coming back the composer reacts harder, because what
it is doing plainly is not working. When most principles are satisfied it
settles down and stops fiddling. That makes this a second-order system: the
state evolves, and the rule that updates the state evolves under it.

Every change is recorded with the principle that caused it, so the piece can
explain itself afterwards rather than just being different.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from compex.generate.critic import Analysis, Verdict
from compex.generate.mood import Mood

#: Which way a drive should move when its principle reads *below* its band.
#: +1 means "below the band wants more of this drive", -1 means the reverse.
POLARITY: dict[str, int] = {
    "novelty": +1,
    "repetition": -1,        # too little repetition means push novelty DOWN
    "post_skip_reversal": +1,
    "tessitura_drift": -1,   # drifting too far means pull HARDER
    "register_spread": +1,
    "dissonance": +1,
    "density": +1,
    "motif_presence": +1,
}

#: How far one unit of error moves each drive before plasticity scales it.
SENSITIVITY: dict[str, float] = {
    "novelty_pressure": 0.30,
    "gap_fill": 0.34,
    "register_pull": 0.26,
    "register_reach": 0.22,
    "dissonance_ceiling": 0.24,
    "density_bias": 0.30,
    "motif_recall": 0.40,
}

BOUNDS: dict[str, tuple[float, float]] = {
    "novelty_pressure": (0.05, 1.0),
    "gap_fill": (0.0, 1.0),
    "register_pull": (0.0, 1.0),
    "register_reach": (0.1, 1.0),
    "dissonance_ceiling": (0.05, 1.0),
    "density_bias": (0.35, 1.9),
    "motif_recall": (0.0, 1.0),
}

PLASTICITY_BOUNDS = (0.15, 1.40)
UNREST_MEMORY = 0.6        # how much of the running unrest survives each movement
SETTLED = 0.22             # unrest below this and the composer stops pushing so hard
FATIGUE_DECAY = 0.65       # how fast a recently-used transform becomes usable again


@dataclass(frozen=True)
class Drives:
    """The parameters the composer writes with. These are what evolve."""

    novelty_pressure: float = 0.50
    gap_fill: float = 0.50
    register_centre: float = 72.0
    register_pull: float = 0.40
    register_reach: float = 0.50
    dissonance_ceiling: float = 0.50
    density_bias: float = 1.00
    motif_recall: float = 0.30
    plasticity: float = 0.50
    unrest: float = 0.00
    fatigue: tuple[tuple[str, float], ...] = ()

    def tired_of(self, kind: str) -> float:
        for name, level in self.fatigue:
            if name == kind:
                return level
        return 0.0

    def with_use(self, kind: str) -> "Drives":
        """Mark a transform as just used, and let everything else recover a little."""
        decayed = {name: level * FATIGUE_DECAY for name, level in self.fatigue}
        decayed[kind] = decayed.get(kind, 0.0) + 1.0
        return replace(self, fatigue=tuple(sorted(
            ((name, level) for name, level in decayed.items() if level > 0.05),
            key=lambda pair: pair[0],
        )))

    def summary(self) -> str:
        return (f"novelty {self.novelty_pressure:.2f} · gap-fill {self.gap_fill:.2f} · "
                f"pull {self.register_pull:.2f} · reach {self.register_reach:.2f} · "
                f"dissonance {self.dissonance_ceiling:.2f} · density {self.density_bias:.2f} · "
                f"recall {self.motif_recall:.2f} · plasticity {self.plasticity:.2f}")


@dataclass(frozen=True)
class Adjustment:
    """One drive moved, and the principle that moved it."""

    drive: str
    before: float
    after: float
    principle: str
    attribution: str
    measured: float

    @property
    def delta(self) -> float:
        return self.after - self.before

    def describe(self) -> str:
        arrow = "up" if self.delta > 0 else "down"
        return (f"{self.drive} {arrow} {abs(self.delta):.3f} "
                f"({self.principle} read {self.measured:.2f}) — {self.attribution}")


@dataclass(frozen=True)
class Evolution:
    """The full record of one decision point, so the piece can explain itself."""

    movement: int
    movement_name: str
    at_beat: float
    analysis: Analysis
    verdicts: tuple[Verdict, ...]
    adjustments: tuple[Adjustment, ...]
    before: Drives
    after: Drives

    @property
    def unhappy(self) -> tuple[Verdict, ...]:
        return tuple(v for v in self.verdicts if not v.satisfied)

    def note(self) -> str:
        """One sentence about what it decided here."""
        if not self.adjustments:
            return "nothing to correct — held its ground"
        biggest = max(self.adjustments, key=lambda a: abs(a.delta))
        others = len(self.adjustments) - 1
        tail = f", and {others} other change{'s' if others > 1 else ''}" if others else ""
        return f"{biggest.describe()}{tail}"


def initial_drives(mood: Mood, root_pitch: int) -> Drives:
    """Where the composer starts before it has heard anything."""
    return Drives(
        novelty_pressure=0.28 + mood.tension * 0.44,
        gap_fill=0.42 + (1.0 - mood.energy) * 0.3,
        register_centre=float(root_pitch + 12),
        register_pull=0.3 + (1.0 - mood.energy) * 0.35,
        register_reach=0.25 + mood.energy * 0.5,
        dissonance_ceiling=0.18 + mood.tension * 0.6,
        density_bias=0.7 + mood.density * 0.6,
        motif_recall=0.25,
        plasticity=0.35 + mood.tension * 0.3,
        unrest=0.0,
        fatigue=(),
    )


def respond(drives: Drives, verdicts: tuple[Verdict, ...],
            analysis: Analysis) -> tuple[Drives, tuple[Adjustment, ...]]:
    """Move the drives toward satisfying the critic, then move the plasticity."""
    values = {name: getattr(drives, name) for name in SENSITIVITY}
    adjustments: list[Adjustment] = []

    for verdict in verdicts:
        if verdict.satisfied:
            continue
        drive = verdict.principle.drive
        if drive not in values:
            continue

        polarity = POLARITY.get(verdict.principle.name, 1)
        delta = polarity * (-verdict.error) * SENSITIVITY[drive] * drives.plasticity

        before = values[drive]
        low, high = BOUNDS[drive]
        after = max(low, min(high, before + delta))
        if abs(after - before) < 1e-4:
            continue

        values[drive] = after
        adjustments.append(Adjustment(
            drive=drive, before=before, after=after,
            principle=verdict.principle.name,
            attribution=verdict.principle.attribution,
            measured=verdict.measured,
        ))

    updated = replace(drives, **values)
    updated = _shift_centre(updated, analysis)
    updated = _reconsider(updated, verdicts)
    return updated, tuple(adjustments)


def _shift_centre(drives: Drives, analysis: Analysis) -> Drives:
    """Let the tessitura centre creep toward where the music actually went.

    Pulling hard toward a centre the piece has abandoned would fight it forever.
    The centre follows, slowly — the pull is a tether, not a cage.
    """
    if analysis.is_empty():
        return drives
    drift = analysis.tessitura_drift * (1.0 - drives.register_pull) * 0.35
    return replace(drives, register_centre=drives.register_centre + drift)


def _reconsider(drives: Drives, verdicts: tuple[Verdict, ...]) -> Drives:
    """The second-order step: change how hard it reacts, based on whether reacting worked.

    Persistent unhappiness means the corrections so far were too timid, so
    plasticity climbs and the next ones bite harder. Once most principles are
    satisfied it relaxes and stops fidgeting with a piece that is working.
    """
    if not verdicts:
        return drives
    now = sum(abs(v.error) for v in verdicts) / len(verdicts)
    unrest = UNREST_MEMORY * drives.unrest + (1.0 - UNREST_MEMORY) * now

    low, high = PLASTICITY_BOUNDS
    plasticity = max(low, min(high, drives.plasticity + (unrest - SETTLED) * 0.55))
    return replace(drives, unrest=unrest, plasticity=plasticity)


def trace(history: tuple[Evolution, ...]) -> str:
    """A readable account of how the piece changed its mind, for the formula."""
    if not history:
        return "no evolution recorded"
    lines = []
    for step in history:
        unhappy = ", ".join(v.principle.name for v in step.unhappy) or "all principles satisfied"
        lines.append(
            f"[{step.movement}] {step.movement_name} @ beat {step.at_beat:g}\n"
            f"    heard: {unhappy}\n"
            f"    did:   {step.note()}\n"
            f"    state: {step.after.summary()}"
        )
    return "\n".join(lines)
