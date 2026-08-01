"""Invent a whole piece from a seed, a runtime and a mood.

Nothing is looked up. Tempo, meter, mode, harmony, the germ motif, the form,
which instruments exist and how they are voiced are all decided here by
walking the theory in :mod:`compex.generate.theory` with the seeded RNG.
Because the RNG is positional rather than sequential, the same three inputs
always produce the same piece, note for note.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from compex import rng
from compex.generate import theory
from compex.generate.mood import Mood
from compex.generate.palette import (
    ROLE_BASS,
    ROLE_LEAD,
    ROLE_PAD,
    ROLE_TEXTURE,
    VoiceSpec,
    build_kit,
    build_voice,
)
from compex.generate.theory import Chord, Motif
from compex.timing import Comb

MIN_BPM, MAX_BPM = 52.0, 168.0
MIN_MOVEMENTS, MAX_MOVEMENTS = 2, 8
SECONDS_PER_MOVEMENT = 16.0

ARCHETYPES: tuple[str, ...] = ("arch", "ramp", "terraced", "return", "through", "erode")

#: Beats between hits for each drum, before density thins or doubles them.
DRUM_PERIODS: dict[str, tuple[float, ...]] = {
    "kick": (1.0, 2.0, 2.0, 4.0),
    "snare": (4.0, 4.0, 8.0),
    "hat": (0.25, 0.5, 0.5, 1.0),
    "tom": (3.0, 5.0, 6.0, 8.0),
    "rim": (1.5, 2.0, 3.0),
    "clap": (4.0, 8.0),
    "ride": (0.5, 1.0, 1.0, 2.0),
    "crash": (8.0, 16.0, 16.0),
    "shaker": (0.25, 0.25, 0.5),
    "cowbell": (2.0, 3.0, 4.0),
    "woodblock": (1.0, 1.5, 2.0),
    "conga": (1.0, 1.5, 2.0, 3.0),
    "snap": (2.0, 4.0),
    "boom": (4.0, 8.0),
    "anvil": (6.0, 8.0, 16.0),
}


@dataclass(frozen=True)
class Movement:
    """One span of the invented form."""

    name: str
    beats: float
    energy: float
    tension: float


@dataclass(frozen=True)
class Note:
    start: float       # beats from the top
    duration: float    # beats
    pitch: float       # MIDI note number
    velocity: float
    voice: str


@dataclass(frozen=True)
class Stroke:
    """A percussion hit."""

    start: float
    velocity: float
    voice: str


@dataclass(frozen=True)
class Composition:
    seed: int
    mood: Mood
    bpm: float
    beats_per_bar: int
    root_pitch: int
    scale_name: str
    scale: tuple[int, ...]
    archetype: str
    progression: tuple[Chord, ...]
    chord_beats: float
    motif: Motif
    movements: tuple[Movement, ...]
    voices: tuple[VoiceSpec, ...]
    combs: tuple[Comb, ...]
    notes: tuple[Note, ...]
    strokes: tuple[Stroke, ...]

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def total_beats(self) -> float:
        return sum(movement.beats for movement in self.movements)

    @property
    def total_seconds(self) -> float:
        return self.total_beats * self.seconds_per_beat

    def voice(self, voice_id: str) -> VoiceSpec | None:
        return next((v for v in self.voices if v.voice_id == voice_id), None)

    def summary(self) -> str:
        pitched = [v for v in self.voices if v.role != "perc"]
        drums = [v for v in self.voices if v.role == "perc"]
        return "\n".join([
            f"{self.bpm:.1f} BPM in {self.beats_per_bar}/4 · {self.scale_name.replace('_', ' ')}"
            f" on {_pitch_name(self.root_pitch)}",
            f"form: {self.archetype} — " + " → ".join(m.name for m in self.movements),
            f"harmony: {len(self.progression)} chords, one per {self.chord_beats:g} beats",
            f"motif: {len(self.motif.steps)} notes, {self.motif.beats:g} beats",
            "voices: " + ", ".join(f"{v.role}/{v.engine}" for v in pitched),
            "kit: " + (", ".join(v.voice_id for v in drums) or "none"),
            f"{len(self.notes)} notes, {len(self.strokes)} strokes",
        ])


def compose(seed: int, duration_s: float, mood: Mood) -> Composition:
    """Invent a piece. Same arguments in, same piece out, always."""
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")
    if not isinstance(mood, Mood):
        raise TypeError(f"mood must be a Mood, got {type(mood).__name__}")

    bpm = _tempo(seed, mood)
    beats_per_bar = _meter(seed, mood)
    root_pitch = 36 + int(rng.uniform(seed, "root", 0) * 12)

    scale_name = theory.pick_scale(seed, mood.valence, mood.tension)
    scale = theory.SCALES[scale_name]

    chord_count = max(2, min(8, 2 + int(mood.density * 6.5)))
    progression = theory.progression(seed, scale, chord_count, mood.tension)
    chord_beats = float(beats_per_bar) * rng.pick(seed, "harmonic-rhythm", 0, (1.0, 1.0, 2.0, 4.0))

    motif = theory.make_motif(seed, scale, mood.energy, mood.tension)
    archetype = _archetype(seed, mood)

    movements = _movements(seed, duration_s, bpm, beats_per_bar, mood, archetype)
    voices = _voices(seed, mood)
    kit = build_kit(seed, mood)
    combs = _combs(seed, kit, beats_per_bar, mood)

    notes: list[Note] = []
    strokes: list[Stroke] = []
    cursor = 0.0
    for index, movement in enumerate(movements):
        written, hit = _write_movement(
            seed, index, movement, cursor, scale, root_pitch, progression,
            chord_beats, motif, voices, combs, kit, mood,
        )
        notes.extend(written)
        strokes.extend(hit)
        cursor += movement.beats

    notes.sort(key=lambda note: (note.start, note.voice))
    strokes.sort(key=lambda stroke: (stroke.start, stroke.voice))

    return Composition(
        seed=seed, mood=mood, bpm=bpm, beats_per_bar=beats_per_bar, root_pitch=root_pitch,
        scale_name=scale_name, scale=scale, archetype=archetype, progression=progression,
        chord_beats=chord_beats, motif=motif, movements=movements,
        voices=tuple(voices) + kit, combs=combs,
        notes=tuple(notes), strokes=tuple(strokes),
    )


# -- the decisions ---------------------------------------------------------

def _tempo(seed: int, mood: Mood) -> float:
    base = MIN_BPM + mood.energy * (MAX_BPM - MIN_BPM)
    jitter = rng.between(seed, "tempo", 0, -9.0, 9.0)
    return round(max(MIN_BPM, min(MAX_BPM, base + jitter)), 1)


def _archetype(seed: int, mood: Mood) -> str:
    """Weight the form archetypes by mood, then draw. Every form stays possible.

    Without this the archetype depends only on the seed, so a serene piece and
    a frantic one at the same seed would share a shape they should not.
    """
    weights = {
        "arch": 1.0 + mood.energy,
        "ramp": 0.5 + 1.8 * mood.energy * mood.valence,
        "terraced": 0.4 + 1.6 * mood.density,
        "return": 0.6 + 1.5 * (1.0 - mood.tension),
        "through": 0.5 + 1.7 * mood.tension,
        "erode": 0.4 + 1.9 * (1.0 - mood.valence) * (1.0 - mood.energy),
    }
    total = sum(weights.values())
    target = rng.uniform(seed, "archetype", 0) * total
    running = 0.0
    for name in ARCHETYPES:
        running += weights[name]
        if target <= running:
            return name
    return ARCHETYPES[-1]


def _meter(seed: int, mood: Mood) -> int:
    pool = [4, 4, 4, 3]
    if mood.tension > 0.55:
        pool.extend((5, 7))
    if mood.tension > 0.78:
        pool.extend((5, 6, 7))
    return int(rng.pick(seed, "meter", 0, tuple(pool)))


def _movements(seed: int, duration_s: float, bpm: float, beats_per_bar: int,
               mood: Mood, archetype: str) -> tuple[Movement, ...]:
    count = max(MIN_MOVEMENTS, min(MAX_MOVEMENTS, round(duration_s / SECONDS_PER_MOVEMENT)))
    total_beats = duration_s * bpm / 60.0

    weights = [rng.between(seed, "mlen", index, 0.72, 1.38) for index in range(count)]
    spread = sum(weights)

    movements: list[Movement] = []
    for index, weight in enumerate(weights):
        bars = max(1, round((total_beats * weight / spread) / beats_per_bar))
        position = index / max(1, count - 1)
        shape = _shape(archetype, position, seed, index)

        energy = _clamp(shape * (0.42 + 0.72 * mood.energy), 0.06, 1.0)
        tension = _clamp(
            mood.tension * (0.62 + 0.5 * position) + rng.between(seed, "mten", index, -0.1, 0.1),
            0.0, 1.0,
        )
        movements.append(Movement(
            name=_movement_name(energy, index, count, tension),
            beats=float(bars * beats_per_bar),
            energy=energy,
            tension=tension,
        ))
    return tuple(movements)


def _shape(archetype: str, position: float, seed: int, index: int) -> float:
    if archetype == "arch":
        return 0.22 + 0.78 * math.sin(math.pi * position)
    if archetype == "ramp":
        return 0.14 + 0.86 * position
    if archetype == "terraced":
        return 0.85 if index % 2 == 0 else 0.3
    if archetype == "return":
        return 0.5 + 0.45 * math.sin(2 * math.pi * position)
    if archetype == "erode":
        return 0.95 - 0.72 * position
    return rng.between(seed, "walk", index, 0.18, 1.0)  # through-composed


def _movement_name(energy: float, index: int, count: int, tension: float) -> str:
    if index == 0:
        return "open"
    if index == count - 1:
        return "leave open" if tension > 0.72 else "settle"
    if energy < 0.26:
        return "still"
    if energy < 0.46:
        return "drift"
    if energy < 0.68:
        return "gather"
    if energy < 0.86:
        return "surge"
    return "break"


def _voices(seed: int, mood: Mood) -> list[VoiceSpec]:
    voices = [build_voice(seed, 0, ROLE_BASS, mood)]
    if mood.density > 0.12:
        voices.append(build_voice(seed, 1, ROLE_LEAD, mood))
    for slot in range(1 + int(mood.density * 2.2)):
        voices.append(build_voice(seed, 2 + slot, ROLE_PAD, mood))
    if mood.density > 0.48:
        voices.append(build_voice(seed, 8, ROLE_TEXTURE, mood))
    return voices


def _combs(seed: int, kit: tuple[VoiceSpec, ...], beats_per_bar: int,
           mood: Mood) -> tuple[Comb, ...]:
    combs: list[Comb] = []
    for index, drum in enumerate(kit):
        pool = DRUM_PERIODS.get(drum.voice_id, (2.0, 4.0))
        period = float(rng.pick(seed, f"comb-{drum.voice_id}", index, pool))
        if mood.density > 0.72 and period > 0.5:
            period /= 2.0
        elif mood.density < 0.28 and period < 8.0:
            period *= 2.0
        offset = 0.0 if drum.voice_id == "kick" else round(
            rng.uniform(seed, f"comb-off-{drum.voice_id}", index) * period * 2
        ) / 2.0
        combs.append(Comb(name=drum.voice_id, period_beats=period, offset_beats=offset % period))
    return tuple(combs)


# -- writing the notes -----------------------------------------------------

def _write_movement(seed, index, movement, origin, scale, root_pitch, progression,
                    chord_beats, motif, voices, combs, kit, mood):
    notes: list[Note] = []
    strokes: list[Stroke] = []
    local_motif = motif if index == 0 else theory.develop(motif, seed, index, movement.tension)

    end = origin + movement.beats
    chord_index = 0
    cursor = origin
    while cursor < end - 1e-6:
        chord = progression[chord_index % len(progression)]
        span = min(chord_beats, end - cursor)
        for voice in voices:
            notes.extend(
                _voice_notes(seed, index, chord_index, voice, chord, scale, root_pitch,
                             cursor, span, movement, local_motif, mood)
            )
        cursor += span
        chord_index += 1

    for drum, comb in zip(kit, combs):
        for beat in comb.hits(origin, end):
            if rng.uniform(seed, f"gate-{drum.voice_id}", int(beat * 4)) > movement.energy + 0.18:
                continue
            velocity = _clamp(
                0.45 + 0.55 * movement.energy
                + rng.between(seed, f"vel-{drum.voice_id}", int(beat * 4), -0.14, 0.14),
                0.1, 1.0,
            )
            strokes.append(Stroke(start=beat, velocity=velocity, voice=drum.voice_id))

    return notes, strokes


def _voice_notes(seed, movement_index, chord_index, voice, chord, scale, root_pitch,
                 start, span, movement, motif, mood) -> list[Note]:
    stream = f"{voice.voice_id}-{movement_index}-{chord_index}"
    base = root_pitch + 12 * voice.octave
    velocity = _clamp(0.35 + 0.65 * movement.energy, 0.08, 1.0)

    if voice.role == ROLE_PAD:
        pitches = [base + step for step in chord.semitones(scale)]
        return [
            Note(start=start, duration=span, pitch=float(pitch), velocity=velocity * 0.7,
                 voice=voice.voice_id)
            for pitch in pitches
        ]

    if voice.role == ROLE_BASS:
        root = float(base + theory.degree_semitone(scale, chord.degree))
        hits = [0.0]
        subdivision = 1.0 if movement.energy < 0.5 else 0.5
        position = subdivision
        while position < span:
            if rng.uniform(seed, f"{stream}-b", int(position * 4)) < movement.energy * 0.55:
                hits.append(position)
            position += subdivision
        return [
            Note(start=start + offset,
                 duration=min(subdivision, span - offset),
                 pitch=root, velocity=velocity, voice=voice.voice_id)
            for offset in hits
        ]

    if voice.role == ROLE_TEXTURE:
        if rng.uniform(seed, f"{stream}-t", 0) > 0.35 + movement.energy * 0.4:
            return []
        pitch = float(base + theory.degree_semitone(scale, chord.degree + 4))
        return [Note(start=start, duration=span * 0.9, pitch=pitch,
                     velocity=velocity * 0.55, voice=voice.voice_id)]

    # lead — speak the motif over this chord, then rest
    if rng.uniform(seed, f"{stream}-rest", 0) > 0.22 + movement.energy * 0.72:
        return []

    notes: list[Note] = []
    offset = 0.0
    for position, (step, length) in enumerate(zip(motif.steps, motif.rhythm)):
        if offset >= span:
            break
        pitch = float(base + theory.degree_semitone(scale, chord.degree + step))
        notes.append(Note(
            start=start + offset,
            duration=min(length, span - offset),
            pitch=pitch,
            velocity=_clamp(velocity + rng.between(seed, f"{stream}-v", position, -0.12, 0.12),
                            0.08, 1.0),
            voice=voice.voice_id,
        ))
        offset += length
    return notes


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _pitch_name(pitch: int) -> str:
    return f"{_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"
