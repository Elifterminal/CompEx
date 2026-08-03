"""Listen back to what was actually written, and judge it.

This is the half of the loop that reads rather than writes. After each
movement the composer hands over everything played so far; the critic
measures it and reports where it sits against a set of named aesthetic
principles, each with a target band and a reason for existing.

The principles are not preferences invented here. They are the ones music
perception research keeps finding, and each carries the name it is known by so
the judgement can be argued with:

* **Berlyne's inverted-U** — liking peaks at moderate novelty. Too familiar is
  boring, too novel is noise. So novelty has a *band*, not a direction.
* **Huron's post-skip reversal** — after a large leap, melodies overwhelmingly
  move back by step. It is one of the strongest regularities in the corpus.
* **von Hippel & Huron's regression to the mean** — melodies drift back toward
  their central pitch rather than wandering off.
* **Meyer's expectation** — tension comes from setting up an expectation and
  delaying it, so realised dissonance should track an intended arc rather than
  sit flat.
* **Huron on repetition** — music is far more self-similar than speech. A
  repetition *floor* is as necessary as a novelty ceiling.
* **Developing variation (Schoenberg)** — the germ should stay present but
  transformed. Abandoning it entirely reads as a different piece.

Nothing here mutates anything. It measures, and says what it thinks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from compex.dsp.listen import Heard
from compex.generate.material import Note, Stroke
from compex.generate.mood import Mood
from compex.generate.theory import Motif

LEAP_SEMITONES = 4.0       # above this an interval counts as a skip rather than a step
RECENT_BEATS = 32.0        # how far back "recently" reaches when judging novelty
MIN_NOTES = 6              # below this there is not enough material to judge

#: Roughness of an interval class against the tonic — the standard consonance
#: ordering: unison and fifths calm, seconds and tritones not. Lives here
#: because the composer judges candidate notes with the same ear it judges
#: finished music with; two tables would drift apart the first time either moved.
ROUGHNESS: dict[int, float] = {
    0: 0.0, 7: 0.12, 5: 0.18, 4: 0.22, 3: 0.26, 9: 0.30,
    8: 0.36, 2: 0.52, 10: 0.56, 11: 0.72, 1: 0.86, 6: 0.95,
}


@dataclass(frozen=True)
class Analysis:
    """What the music actually turned out to be, measured rather than intended."""

    note_count: int
    lead_count: int
    novelty: float             # share of recent pitch-interval pairs not heard before
    repetition: float          # share of recent material that echoes something earlier
    post_skip_reversal: float  # share of leaps answered by a step the other way
    tessitura_drift: float     # semitone distance of the recent centre from the overall centre
    register_spread: float     # semitones between the 10th and 90th percentile of pitch
    dissonance: float          # mean interval roughness against the tonic, 0..1
    density: float             # events per beat, normalised against a busy reference
    motif_presence: float      # how much of the germ's contour is still audible
    revelation: float = 0.5    # share of its own opening the piece has since explained
    crowding: float = 0.0      # how much the voices are sitting on top of each other
    heard_roughness: float = 0.04   # beating partials in the rendered sound
    heard_brightness: float = 0.5   # where the energy actually sits
    heard_motion: float = 0.5       # how much the timbre moves, frame to frame

    def is_empty(self) -> bool:
        return self.note_count < MIN_NOTES


@dataclass(frozen=True)
class Principle:
    """One aesthetic criterion, with the band it wants and why."""

    name: str
    attribution: str
    low: float
    high: float
    drive: str
    why: str


@dataclass(frozen=True)
class Verdict:
    """How far one principle is from satisfied, and in which direction."""

    principle: Principle
    measured: float
    error: float   # negative = below the band, positive = above it, 0 = inside

    @property
    def satisfied(self) -> bool:
        return self.error == 0.0

    def describe(self) -> str:
        if self.satisfied:
            return f"{self.principle.name}: {self.measured:.2f} — inside the band"
        direction = "below" if self.error < 0 else "above"
        return (f"{self.principle.name}: {self.measured:.2f} {direction} "
                f"[{self.principle.low:.2f}, {self.principle.high:.2f}]")


#: The measured field each principle reads, and the drive it pushes when unhappy.
PRINCIPLES: tuple[Principle, ...] = (
    Principle(
        "novelty", "Berlyne's inverted-U", 0.28, 0.62, "novelty_pressure",
        "Liking peaks at moderate novelty. Too familiar bores; too novel stops "
        "reading as music at all. This wants a band, not a maximum.",
    ),
    Principle(
        "repetition", "Huron on self-similarity", 0.18, 0.70, "novelty_pressure",
        "Music repeats itself far more than speech does. Without a repetition "
        "floor the piece never accumulates anything to recognise.",
    ),
    Principle(
        "post_skip_reversal", "Huron's post-skip reversal", 0.45, 1.0, "gap_fill",
        "After a large leap, melodies overwhelmingly step back the other way. "
        "Leaping repeatedly in one direction is the clearest tell of a machine "
        "that has never listened to a melody.",
    ),
    Principle(
        "tessitura_drift", "von Hippel & Huron, regression to the mean", 0.0, 5.0,
        "register_pull",
        "Melodies return toward their central pitch. Drift that never comes back "
        "reads as wandering rather than as line.",
    ),
    Principle(
        "register_spread", "melodic arch", 7.0, 26.0, "register_reach",
        "Using one octave is monotonous; using four is incoherent as a single "
        "voice. Somewhere between a fifth and two octaves reads as one instrument.",
    ),
    Principle(
        "dissonance", "Meyer's expectation", 0.12, 0.72, "dissonance_ceiling",
        "Tension has to move to mean anything. Flat consonance is wallpaper; "
        "flat dissonance stops being tense and becomes texture.",
    ),
    Principle(
        "density", "arousal matching", 0.10, 0.92, "density_bias",
        "The realised event rate should track what the mood asked for, not "
        "whatever the note-writing rules happened to produce.",
    ),
    Principle(
        "motif_presence", "Schoenberg's developing variation", 0.22, 0.95, "motif_recall",
        "The germ should stay audible through its transformations. Losing it "
        "entirely means the piece stopped being about anything.",
    ),
    Principle(
        "heard_roughness", "Plomp & Levelt, sensory dissonance", 0.0, 0.09,
        "dissonance_ceiling",
        "Dissonance measured in the sound rather than in the interval. Two "
        "notes a semitone apart are one number to a symbolic critic whether "
        "they are played by flutes or by sawtooth pads, and those are not the "
        "same event. This counts partials close enough together to beat, which "
        "is what roughness physically is.",
    ),
    Principle(
        "heard_motion", "spectral flux", 0.30, 0.85, "timbre_drift",
        "How much the sound itself changes, frame to frame. A piece whose "
        "timbre never moves stops rewarding attention however much its notes "
        "move — and unlike a player with one instrument, this can do something "
        "about it.",
    ),
    Principle(
        "crowding", "auditory stream segregation", 0.0, 0.45, "spacing",
        "Two voices in the same octave playing at the same time are heard as "
        "one thicker voice, not as two. The mixer can only make that thicker "
        "voice louder — moving them apart is the composer's job and nobody "
        "else can do it. Measured in registers rather than in spectra, because "
        "registers are what the composer can actually act on.",
    ),
    Principle(
        "revelation", "minimum description length", 0.04, 1.0, "motif_recall",
        "A piece that ends should have made its own beginning cheaper to "
        "describe than it was at the time — five strange events turning out to "
        "be one thing rotated five ways. Measured in symbols against the "
        "notation the engine writes, which is why it is a number here rather "
        "than a figure of speech. Neutral until a piece is long enough to have "
        "a past worth explaining.",
    ),
)


def analyse(notes: Sequence[Note], strokes: Sequence[Stroke], motif: Motif,
            scale: tuple[float, ...], root_pitch: int, upto_beat: float,
            lead_voices: frozenset[str], revelation: float = 0.5,
            heard: "Heard | None" = None) -> Analysis:
    """Measure everything written before ``upto_beat``.

    ``revelation`` is handed in rather than computed here: it is a property of
    the *descriptions* the piece could write, not of the notes, and the module
    that prices descriptions has no business being imported by the one that
    counts intervals.
    """
    played = [n for n in notes if n.start < upto_beat]
    if len(played) < MIN_NOTES:
        return _neutral(len(played), 0)

    lead = _melodic_line(played, lead_voices)
    if len(lead) < MIN_NOTES:
        # There is a melody, it just has not said enough yet. Reporting the bass
        # and pads as if they were the tune would push every melodic drive on
        # noise — better to stay quiet than to invent a reading.
        return _neutral(len(played), len(lead),
                        density=_density(played, strokes, upto_beat),
                        dissonance=_dissonance(played, scale, root_pitch),
                        revelation=revelation)

    pitches = [n.pitch for n in lead]

    # Split the line into an older and a more recent half rather than using a
    # fixed window: early on, a fixed window swallows everything, leaves nothing
    # to compare against, and reports maximum novelty forever.
    look_back = min(RECENT_BEATS, upto_beat * 0.5)
    recent_from = max(0.0, upto_beat - look_back)
    recent = [n for n in lead if n.start >= recent_from]
    earlier = [n for n in lead if n.start < recent_from]

    return Analysis(
        note_count=len(played),
        lead_count=len(lead),
        novelty=_novelty(recent, earlier),
        repetition=_repetition(recent, earlier),
        post_skip_reversal=_post_skip_reversal(pitches),
        tessitura_drift=_tessitura_drift(pitches),
        register_spread=_spread(pitches),
        dissonance=_dissonance(played, scale, root_pitch),
        density=_density(played, strokes, upto_beat),
        motif_presence=_motif_presence(pitches, motif),
        revelation=revelation,
        crowding=_crowding(played, upto_beat),
        heard_roughness=heard.roughness if heard else 0.04,
        heard_brightness=heard.brightness if heard else 0.5,
        heard_motion=heard.motion if heard else 0.5,
    )


def _neutral(note_count: int, lead_count: int, density: float = 0.3,
             dissonance: float = 0.3, revelation: float = 0.5) -> Analysis:
    """An analysis that sits inside every band, so it applies no pressure at all."""
    return Analysis(note_count=note_count, lead_count=lead_count, novelty=0.45,
                    repetition=0.45, post_skip_reversal=0.7, tessitura_drift=0.0,
                    register_spread=12.0, dissonance=dissonance, density=density,
                    motif_presence=0.6, revelation=revelation)


def _melodic_line(played: Sequence[Note], lead_voices: frozenset[str]) -> list[Note]:
    """The notes that count as the tune.

    Lead voices if the piece has any. If it does not — a sparse mood may write
    no lead at all — the highest-sitting voice stands in, because that is what
    a listener hears as the line.
    """
    lead = [n for n in played if n.voice in lead_voices]
    if lead:
        return sorted(lead, key=lambda n: n.start)

    by_voice: dict[str, list[Note]] = {}
    for note in played:
        by_voice.setdefault(note.voice, []).append(note)
    if not by_voice:
        return []
    highest = max(by_voice, key=lambda v: sum(n.pitch for n in by_voice[v]) / len(by_voice[v]))
    return sorted(by_voice[highest], key=lambda n: n.start)


def judge(analysis: Analysis, mood: Mood) -> tuple[Verdict, ...]:
    """Score the analysis against every principle, with the bands shifted by mood.

    A menacing piece is allowed — expected — to sit at a dissonance a serene one
    would fail on. The principle is the same; where its band sits is not.
    """
    verdicts: list[Verdict] = []
    for principle in PRINCIPLES:
        measured = float(getattr(analysis, principle.name))
        low, high = _band_for(principle, mood)
        if measured < low:
            error = (measured - low) / max(low, 1e-6)
        elif measured > high:
            error = (measured - high) / max(1.0 - high, 1e-6) if high <= 1.0 else \
                    (measured - high) / max(high, 1e-6)
        else:
            error = 0.0
        verdicts.append(Verdict(principle, measured, max(-1.0, min(1.0, error))))
    return tuple(verdicts)


def _band_for(principle: Principle, mood: Mood) -> tuple[float, float]:
    """Shift a principle's target band to suit the mood."""
    low, high = principle.low, principle.high
    if principle.name == "dissonance":
        centre = 0.12 + mood.tension * 0.6
        return max(0.0, centre - 0.22), min(1.0, centre + 0.22)
    if principle.name == "novelty":
        centre = 0.30 + mood.tension * 0.28
        return max(0.0, centre - 0.16), min(1.0, centre + 0.16)
    if principle.name == "density":
        centre = 0.12 + mood.density * 0.75
        return max(0.0, centre - 0.24), min(1.4, centre + 0.24)
    if principle.name == "repetition":
        # Hypnotic wants far more self-similarity than frantic does.
        centre = 0.62 - mood.energy * 0.34
        return max(0.0, centre - 0.24), min(1.0, centre + 0.26)
    if principle.name == "heard_roughness":
        # A serene piece should have almost no beating in it; a shattered one
        # is allowed to grind. The numbers are small because real music mostly
        # avoids clusters — calibrated against tones, where a semitone dyad
        # reads 0.27 and a fifth reads zero.
        return 0.0, 0.03 + mood.tension * 0.16 + mood.grit * 0.06
    if principle.name == "heard_motion":
        centre = 0.42 + mood.energy * 0.25
        return max(0.0, centre - 0.22), min(1.0, centre + 0.25)
    return low, high


# ── measurements ───────────────────────────────────────────────────────────

def _intervals(notes: Sequence[Note]) -> list[int]:
    return [int(round(b.pitch - a.pitch)) for a, b in zip(notes, notes[1:])]


def _bigrams(notes: Sequence[Note]) -> set[tuple[int, int]]:
    """Interval pairs — the smallest unit that carries a shape rather than a pitch."""
    steps = _intervals(notes)
    return {(a, b) for a, b in zip(steps, steps[1:])}


def _novelty(recent: Sequence[Note], earlier: Sequence[Note]) -> float:
    """Share of recent shapes not heard before.

    With nothing to compare against this returns the middle of the band, not
    1.0. Claiming maximum novelty because you have no memory is a measurement
    artefact, and it pins the drive to one end from the first bar.
    """
    fresh, old = _bigrams(recent), _bigrams(earlier)
    if not fresh or not old:
        return 0.45
    return len(fresh - old) / len(fresh)


def _repetition(recent: Sequence[Note], earlier: Sequence[Note]) -> float:
    fresh, old = _bigrams(recent), _bigrams(earlier)
    if not fresh or not old:
        return 0.45
    return len(fresh & old) / len(fresh)


def _post_skip_reversal(pitches: Sequence[float]) -> float:
    steps = [b - a for a, b in zip(pitches, pitches[1:])]
    answered = considered = 0
    for leap, nxt in zip(steps, steps[1:]):
        if abs(leap) <= LEAP_SEMITONES:
            continue
        considered += 1
        if leap * nxt < 0 and abs(nxt) <= LEAP_SEMITONES:
            answered += 1
    return answered / considered if considered else 0.7  # nothing to answer for


def _tessitura_drift(pitches: Sequence[float]) -> float:
    if len(pitches) < 4:
        return 0.0
    overall = sum(pitches) / len(pitches)
    tail = pitches[-max(4, len(pitches) // 4):]
    return abs(sum(tail) / len(tail) - overall)


def _spread(pitches: Sequence[float]) -> float:
    if len(pitches) < 4:
        return 12.0
    ordered = sorted(pitches)
    low = ordered[int(len(ordered) * 0.1)]
    high = ordered[int(len(ordered) * 0.9) - 1 if len(ordered) > 9 else -1]
    return float(high - low)


def _dissonance(notes: Sequence[Note], scale: tuple[int, ...], root_pitch: int) -> float:
    """Roughness of each sounding pitch against the tonic.

    Crude next to a real roughness model, and enough to tell a plagal drift
    from a tritone pile-up. Interpolated rather than rounded to a semitone,
    because the piece is not necessarily in twelve — a neutral third has to
    land between the minor and the major one rather than being filed as
    whichever it is nearer.
    """
    if not notes:
        return 0.3
    from compex.generate.tuning import roughness_at

    total = sum(roughness_at(note.pitch - root_pitch) for note in notes)
    return total / len(notes)


def _density(notes: Sequence[Note], strokes: Sequence[Stroke], upto_beat: float) -> float:
    if upto_beat <= 0:
        return 0.0
    events = len(notes) + len([s for s in strokes if s.start < upto_beat])
    per_beat = events / upto_beat
    return min(1.4, per_beat / 6.0)  # 6 events a beat is about as busy as it gets here


MOTIF_MATCH = 0.75  # how much of the contour must line up to count as the germ


def _motif_presence(pitches: Sequence[float], motif: Motif) -> float:
    """How *often* the germ's contour shows up, in any transposition or inversion.

    Direction only, not exact size — an inversion or an augmentation should
    still count, which is the whole point of developing variation.

    This counts the share of windows that match rather than whether the best
    one does. Best-match saturates at 1.0 almost immediately: over a long line,
    some window always happens to fit a short contour by chance, so the measure
    stops distinguishing a piece built on its motif from one that abandoned it.
    """
    shape = [1 if step > 0 else -1 if step < 0 else 0
             for step in _diff(motif.steps)]
    if not shape or len(pitches) < len(shape) + 1:
        return 0.6
    played = [1 if step > 0 else -1 if step < 0 else 0 for step in _diff(pitches)]

    windows = matches = 0
    for start in range(len(played) - len(shape) + 1):
        window = played[start : start + len(shape)]
        forward = sum(1 for a, b in zip(window, shape) if a == b) / len(shape)
        mirrored = sum(1 for a, b in zip(window, shape) if a == -b) / len(shape)
        windows += 1
        if max(forward, mirrored) >= MOTIF_MATCH:
            matches += 1
    return matches / windows if windows else 0.6


CROWD_WINDOW = 7.0   # semitones — inside this two voices stop being two voices


def _crowding(notes: Sequence[Note], upto_beat: float) -> float:
    """How much of the time two voices are sitting in each other's register.

    For every pair of voices that overlap in time, how close their registers
    are. Two lines a fifth apart are two lines; two lines three semitones
    apart are one thick line and a mixing problem that no gain can fix.

    This is deliberately symbolic. The spectral version of the question lives
    in the mixer and needs audio; this one needs only the notes, which is what
    makes it available *while the piece is still being written*.
    """
    recent = [n for n in notes if n.start >= max(0.0, upto_beat - RECENT_BEATS)]
    by_voice: dict[str, list[float]] = {}
    for note in recent:
        by_voice.setdefault(note.voice, []).append(note.pitch)
    if len(by_voice) < 2:
        return 0.0

    centres = {voice: sum(pitches) / len(pitches) for voice, pitches in by_voice.items()}
    names = sorted(centres)
    close = pairs = 0
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            pairs += 1
            if abs(centres[names[first]] - centres[names[second]]) < CROWD_WINDOW:
                close += 1
    return close / pairs if pairs else 0.0


def _diff(values: Sequence[float]) -> list[float]:
    return [b - a for a, b in zip(values, values[1:])]
