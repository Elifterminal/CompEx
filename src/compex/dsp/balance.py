"""Deciding how loud each voice should be, by measuring what it is doing.

Before this the mix was a table: bass 0.90, lead 0.67, pad 0.34, texture 0.23,
one number per role, written once by me and never checked against anything. It
produced pieces where six drums outweighed everything melodic by more than two
to one, and pads that were technically present and inaudible.

The problem with a table is not that the numbers are wrong. It is that a voice
is buried or not depending on **what else is playing in the same part of the
spectrum**, which the table cannot know: a pad under a noise wash and the same
pad under a sub bass are two different situations and one gain covers both.

So this measures instead. Every voice is rendered once — a representative
note, through its own effects chain, because a reverb or a ring modulator
changes where a voice lives — and its energy is counted per frequency band.
Multiplied by how often that voice plays, that gives what each voice actually
contributes to each band. Then a voice that owns too little of every band it
lives in gets lifted, and one that owns too much of a band gets held back.

It is a mixer, not a composer. It does not decide what is played and it never
sees the critic. That separation is on purpose — but it does mean the
listening loop still cannot hear the mix, which is written down as a known
limit rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

#: Rough critical-band groups: sub, bass, body, presence, air.
BANDS: tuple[tuple[float, float], ...] = (
    (20.0, 120.0),
    (120.0, 400.0),
    (400.0, 1200.0),
    (1200.0, 4000.0),
    (4000.0, 16000.0),
)
BAND_NAMES: tuple[str, ...] = ("sub", "bass", "body", "presence", "air")

#: The share of a band a voice in each role should be able to claim somewhere.
#: Below the floor it is inaudible; above the ceiling it is the only thing you
#: can hear. Both are judgements, and both are checked against measurement.
FLOOR: dict[str, float] = {
    "bass": 0.20, "lead": 0.16, "pad": 0.09, "texture": 0.05, "perc": 0.12,
}
CEILING: dict[str, float] = {
    "bass": 0.72, "lead": 0.62, "pad": 0.55, "texture": 0.40, "perc": 0.70,
}

MAX_LIFT = 2.6      # nothing gets more than this much help
MAX_CUT = 0.35      # nor held back harder than this
ENABLED = True      # the whole stage, off — gains fall back to the role table


@dataclass(frozen=True)
class Presence:
    """What one voice contributes, band by band, and what to do about it."""

    voice: str
    role: str
    energy: tuple[float, ...]
    share: tuple[float, ...] = ()
    trim: float = 1.0
    presence: float = 1.0     # how much this voice plays, against the busiest one
    after: float = 0.0        # what it owns at home once the trims are applied

    @property
    def home(self) -> int:
        """The band where this voice puts most of *its own* energy.

        Not the band where it has the largest share — that was the first
        version and it was nonsense: a bass reed came out "living in the air"
        because its breath noise owned 98% of a band nothing else was using.
        A voice lives where it lives; the question is who else is there.
        """
        return int(np.argmax(self.energy)) if self.energy else 0

    @property
    def best_share(self) -> float:
        return self.share[self.home] if self.share else 0.0

    def floor(self) -> float:
        """The share this voice should own at home, scaled by how much it plays.

        A texture that speaks four times in five minutes is not buried, it is
        occasional. Holding it to the same floor as a pad playing every bar
        would have the mixer shouting down a piece's own sparseness.
        """
        return FLOOR.get(self.role, 0.1) * (self.presence ** 0.5)

    @property
    def buried(self) -> bool:
        """Was it inaudible *before* the mixer did anything?"""
        return self.best_share < self.floor()

    @property
    def still_buried(self) -> bool:
        """And is it still, after? Reporting only the diagnosis would let the
        stage claim credit for problems it merely noticed."""
        return self.after < self.floor()

    def describe(self) -> str:
        band = BAND_NAMES[self.home] if self.share else "?"
        return (f"{self.voice} lives in the {band}, owns {self.best_share * 100:.0f}% of it"
                + (f", lifted {self.trim:.2f}×" if self.trim > 1.02 else
                   f", held back to {self.trim:.2f}×" if self.trim < 0.98 else ""))


@dataclass(frozen=True)
class Mix:
    """The whole decision: who sits where, and what was done about it."""

    voices: tuple[Presence, ...] = ()
    percussive: float = 0.0     # share of total energy that is drums

    def trims(self) -> dict[str, float]:
        return {presence.voice: presence.trim for presence in self.voices}

    def buried(self) -> tuple[str, ...]:
        return tuple(p.voice for p in self.voices if p.buried)

    def still_buried(self) -> tuple[str, ...]:
        return tuple(p.voice for p in self.voices if p.still_buried)

    def summary(self) -> str:
        if not self.voices:
            return "no mix decisions"
        lifted = sum(1 for p in self.voices if p.trim > 1.02)
        cut = sum(1 for p in self.voices if p.trim < 0.98)
        rescued = len(self.buried()) - len(self.still_buried())
        return (f"{len(self.voices)} voices, {self.percussive * 100:.0f}% percussive, "
                f"{lifted} lifted, {cut} held back, "
                f"{rescued} of {len(self.buried())} buried voices brought back")


def band_energy(signal: np.ndarray, sample_rate: int) -> tuple[float, ...]:
    """How much of this sound sits in each band.

    One FFT of the whole thing, which is enough — this is a question about a
    voice's character, not about any particular moment of it.
    """
    if len(signal) < 32:
        return tuple(0.0 for _ in BANDS)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sample_rate)
    return tuple(float(spectrum[(freqs >= low) & (freqs < high)].sum())
                 for low, high in BANDS)


def weigh(energies: dict[str, tuple[str, tuple[float, ...], float]],
          played: dict[str, float] | None = None) -> Mix:
    """Turn per-voice band energies into shares, and shares into trims.

    ``energies`` maps voice id to (role, per-band energy, current gain), the
    energy already scaled by how much that voice plays. ``played`` is the raw
    amount each voice plays, which decides how much presence it is entitled to
    ask for.
    """
    if not energies:
        return Mix()

    names = list(energies)
    totals = [sum(energies[name][1][band] for name in names) or 1.0
              for band in range(len(BANDS))]
    busiest = max((played or {}).values(), default=1.0) or 1.0

    voices: list[Presence] = []
    for name in names:
        role, energy, _ = energies[name]
        share = tuple(energy[band] / totals[band] for band in range(len(BANDS)))
        voices.append(Presence(voice=name, role=role, energy=energy, share=share,
                               presence=min(1.0, (played or {}).get(name, busiest) / busiest)))

    voices = [replace(presence, trim=_trim_for(presence)) for presence in voices]

    # What the trims actually bought, measured the same way as the diagnosis.
    settled = [[presence.energy[band] * presence.trim ** 2 for band in range(len(BANDS))]
               for presence in voices]
    after_totals = [sum(row[band] for row in settled) or 1.0 for band in range(len(BANDS))]
    voices = [replace(presence, after=row[presence.home] / after_totals[presence.home])
              for presence, row in zip(voices, settled)]

    percussive = sum(sum(energies[p.voice][1]) for p in voices if p.role == "perc")
    everything = sum(sum(energies[p.voice][1]) for p in voices) or 1.0
    return Mix(voices=tuple(voices), percussive=percussive / everything)


def _trim_for(presence: Presence) -> float:
    """Lift a voice that owns too little, hold back one that owns too much.

    The square root is doing real work: shares are energy ratios, and a voice
    that is at half the floor does not need four times the gain to get there.
    Overcorrecting is how a mix starts pumping.
    """
    if not ENABLED or not presence.share:
        return 1.0
    floor = presence.floor()
    ceiling = CEILING.get(presence.role, 0.6)
    best = presence.best_share

    if best <= 1e-9:
        return 1.0
    if best < floor:
        return min(MAX_LIFT, (floor / best) ** 0.5)
    if best > ceiling:
        return max(MAX_CUT, (ceiling / best) ** 0.5)
    return 1.0
