"""Turning decisions into notes.

The composer decides (:mod:`compex.generate.compose`), the critic judges
(:mod:`compex.generate.critic`), the melody and pattern modules choose. This is
where those choices become material a renderer can play: pitches, durations,
velocities, strikes.

It is kept separate because it is the only part with an opinion about voices —
what a pad does with a chord, how a bass reads a rhythm grid — and none of the
deciding should have to know any of that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from compex import rng
from compex.generate import melody, pattern, promise, theory
from compex.generate.evolve import Drives
from compex.generate.material import Movement, Note, Stroke
from compex.generate.melody import Choice, Phrase, Setting, Taste
from compex.generate.palette import (
    ROLE_BASS,
    ROLE_LEAD,
    ROLE_PAD,
    ROLE_TEXTURE,
    VoiceSpec,
)
from compex.generate.pattern import Bed, Grid, Pattern, PatternChoice
from compex.generate.promise import Ledger
from compex.generate.theory import Chord
from compex.measure import Shape


@dataclass(frozen=True)
class Written:
    """One movement's worth of music, and what the composer learned writing it."""

    notes: tuple[Note, ...]
    strokes: tuple[Stroke, ...]
    choices: tuple[Choice, ...]
    lineage: Phrase | None
    heard: frozenset[tuple[int, int]]
    approach: int | None = None   # the scale degree the last phrase ended on
    ledger: Ledger = Ledger()     # what the piece owes after writing this


def choose_patterns(seed: int, index: int, movement: Movement, grids: dict[str, Grid],
                    groove: pattern.Groove, drives: Drives, bar_beats: float,
                    previous: dict[str, Pattern], ledger: Ledger = Ledger(),
                    at_beat: float = 0.0) -> tuple[dict[str, Pattern],
                                                   tuple[PatternChoice, ...], Ledger]:
    """Hold an audition per voice, in a fixed order so each one hears the last.

    Order matters and it is deliberate: whoever goes first gets the obvious
    slots, and everyone after has to find somewhere else to be. That is what
    makes the kit interlock instead of stacking.
    """
    chosen: dict[str, Pattern] = {}
    records: list[PatternChoice] = []
    claimed: set[float] = set()

    for voice, grid in grids.items():
        bed = Bed(
            energy=movement.energy,
            tension=movement.tension,
            drives=drives,
            bar_beats=bar_beats,
            claimed=frozenset(claimed),
            previous=previous.get(voice),
            ledger=ledger,
            now_beat=at_beat,
        )
        record = pattern.invent(seed, index, len(records), grid, groove, bed)
        chosen[voice] = record.pattern
        records.append(record)

        # Settle first, then open: a pattern that lands the displaced beat and
        # promptly displaces another one has done both, and in that order.
        for owed in ledger.live(at_beat, domain="metre"):
            if promise.pattern_settles(owed, record.pattern.slots, record.pattern.subdivision):
                ledger = ledger.settle(owed, at_beat, f"{voice} landed it")
        ledger = ledger.opened(promise.from_pattern(
            record.pattern.slots, grid.weights, record.pattern.subdivision, at_beat, voice))

        for slot, _ in record.pattern.onsets():
            claimed.add(pattern.in_bar(slot * record.pattern.subdivision, bar_beats))

    return chosen, tuple(records), ledger


def write_movement(seed: int, index: int, movement: Movement, origin: float,
                   scale: tuple[int, ...], root_pitch: int,
                   progression: tuple[Chord, ...], chord_beats: float,
                   motif: theory.Motif, voices: list[VoiceSpec], kit: tuple[VoiceSpec, ...],
                   patterns: dict[str, Pattern], drives: Drives, taste: Taste,
                   lineage: Phrase | None, heard: frozenset[tuple[int, int]],
                   approach: int | None = None, ledger: Ledger = Ledger(),
                   position: float = 0.0, past: tuple[Shape, ...] = (),
                   vocabulary: tuple[Shape, ...] = (),
                   past_cost: tuple[float, ...] = ()) -> Written:
    """Write one movement under the current drives, taste and patterns."""
    notes: list[Note] = []
    strokes: list[Stroke] = []
    choices: list[Choice] = []

    end = origin + movement.beats
    chord_index = 0
    cursor = origin

    while cursor < end - 1e-6:
        chord = progression[chord_index % len(progression)]
        span = min(chord_beats, end - cursor)
        for voice in voices:
            if voice.role == ROLE_LEAD:
                written, lineage, heard, approach, ledger, choice = _lead(
                    seed, index, chord_index, voice, chord, scale, root_pitch,
                    cursor, span, movement, motif, drives, taste, lineage, heard,
                    approach, ledger, position, past, vocabulary, past_cost)
                notes.extend(written)
                if choice is not None:
                    choices.append(choice)
                continue
            notes.extend(_voice_notes(seed, index, chord_index, voice, chord, scale,
                                      root_pitch, cursor, span, movement, patterns, drives))
        cursor += span
        chord_index += 1

    gate = movement.energy * drives.density_bias
    for drum in kit:
        figure = patterns.get(drum.voice_id)
        if figure is None:
            continue
        for beat, velocity in figure.hits(origin, end):
            # The pattern says where; the movement still says whether. A quiet
            # movement thins its own groove rather than swapping it for another.
            if rng.uniform(seed, f"gate-{drum.voice_id}", int(beat * 4)) > gate + 0.22:
                continue
            strokes.append(Stroke(
                start=beat,
                velocity=_clamp(velocity * (0.55 + 0.55 * movement.energy)
                                + rng.between(seed, f"vel-{drum.voice_id}",
                                              int(beat * 4), -0.1, 0.1), 0.1, 1.0),
                voice=drum.voice_id,
            ))

    return Written(notes=tuple(notes), strokes=tuple(strokes), choices=tuple(choices),
                   lineage=lineage, heard=heard, approach=approach, ledger=ledger)


# ── the lead: audition, then play the winner ──────────────────────────────

def _lead(seed, index, chord_index, voice, chord, scale, root_pitch, start, span,
          movement, motif, drives, taste, lineage, heard, approach, ledger,
          position=0.0, past=(), vocabulary=(), past_cost=()):
    """Choose a phrase for this chord and write it out.

    The choosing happens in scale degrees, which is why it can be judged before
    a pitch exists. What gets written here is the *voiced* phrase — the one the
    audition actually scored — so nothing between the judgement and the sound
    changes the thing that was judged.
    """
    stream = f"{voice.voice_id}-{index}-{chord_index}"
    rest_gate = 0.22 + movement.energy * 0.72 * drives.density_bias
    if rng.uniform(seed, f"{stream}-rest", 0) > rest_gate:
        return [], lineage, heard, approach, ledger, None

    setting = Setting(
        scale=scale,
        chord_degree=chord.degree,
        chord_size=chord.size,
        span_beats=span,
        energy=movement.energy,
        tension=movement.tension,
        drives=drives,
        germ=tuple(b - a for a, b in zip(motif.steps, motif.steps[1:])),
        heard=heard,
        approach=approach,
        ledger=ledger,
        now_beat=start,
        position=position,
        past=past,
        vocabulary=vocabulary,
        past_cost=past_cost,
    )
    choice = melody.choose(seed, index, chord_index, setting, taste, lineage, motif,
                           at_beat=start)
    phrase = melody.voiced(choice.chosen, drives)

    base = root_pitch + 12 * voice.octave
    velocity = _clamp(0.35 + 0.65 * movement.energy, 0.08, 1.0)

    notes: list[Note] = []
    offset = 0.0
    played = 0
    for position, (step, length) in enumerate(zip(phrase.steps, phrase.rhythm)):
        if offset >= span:
            break
        pitch = float(base + theory.degree_semitone(scale, chord.degree + step))
        played = position
        notes.append(Note(
            start=start + offset,
            duration=min(length, span - offset),
            pitch=pitch,
            velocity=_clamp(velocity + rng.between(seed, f"{stream}-v", position, -0.12, 0.12),
                            0.08, 1.0),
            voice=voice.voice_id,
        ))
        offset += length

    notes = _pull_toward_centre(notes, drives, seed, stream)
    # The ledger moves in the same order the music does: what this phrase
    # answered is settled, and only then does what it opened go on the books.
    for owed in ledger.live(start, domain="pitch"):
        if promise.would_settle(owed, phrase.steps, chord.degree):
            ledger = ledger.settle(owed, start, f"{voice.voice_id} answered it")
    ledger = ledger.opened(
        promise.from_phrase(phrase.steps, chord.degree, scale, start, approach))
    ledger = ledger.forget(start)

    moves = [b - a for a, b in zip(phrase.steps, phrase.steps[1:])]
    heard = heard | {(a, b) for a, b in zip(moves, moves[1:])}
    # Where this phrase actually stopped, so the next audition can judge the
    # step into itself. The lineage stays unvoiced — mutations work on the idea,
    # not on the copy the register drives happened to make of it.
    landed = chord.degree + phrase.steps[played] if notes else approach
    return notes, choice.chosen, heard, landed, ledger, choice


# ── everything that is not the tune ───────────────────────────────────────

def _voice_notes(seed, movement_index, chord_index, voice, chord, scale, root_pitch,
                 start, span, movement, patterns, drives) -> list[Note]:
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
        return _bass(seed, stream, voice, chord, scale, base, start, span,
                     movement, patterns.get(voice.voice_id), drives, velocity)

    if voice.role == ROLE_TEXTURE:
        if rng.uniform(seed, f"{stream}-t", 0) > 0.35 + movement.energy * 0.4:
            return []
        pitch = float(base + theory.degree_semitone(scale, chord.degree + 4))
        return [Note(start=start, duration=span * 0.9, pitch=pitch,
                     velocity=velocity * 0.55, voice=voice.voice_id)]

    return []


def _bass(seed, stream, voice, chord, scale, base, start, span, movement,
          figure: Pattern | None, drives, velocity) -> list[Note]:
    """The bass reads the same kind of grid the drums do.

    It used to roll dice per subdivision, which meant it never played a figure
    twice and so never played a figure at all. Now it has one, chosen the same
    way the kick's is, and the interlock criterion has already made sure the two
    are not fighting for the same slots.
    """
    root = float(base + theory.degree_semitone(scale, chord.degree))
    if figure is None:
        return [Note(start=start, duration=min(1.0, span), pitch=root,
                     velocity=velocity, voice=voice.voice_id)]

    notes: list[Note] = []
    for beat, level in figure.hits(start, start + span):
        offset = beat - start
        # A low voice that changes note on every hit stops being a foundation.
        # Anything off the strong slots gets the fifth instead, which is what a
        # bass player does when they want movement without leaving the chord.
        degree = chord.degree if level > 0.6 else chord.degree + 4
        pitch = float(base + theory.degree_semitone(scale, degree))
        notes.append(Note(
            start=beat,
            duration=min(figure.subdivision * 1.5, span - offset),
            pitch=pitch if degree != chord.degree else root,
            velocity=_clamp(velocity * (0.55 + level * 0.6), 0.08, 1.0),
            voice=voice.voice_id,
        ))
    if not notes:
        notes.append(Note(start=start, duration=min(1.0, span), pitch=root,
                          velocity=velocity, voice=voice.voice_id))
    return notes


def _pull_toward_centre(notes: list[Note], drives: Drives, seed: int,
                        stream: str) -> list[Note]:
    """Octave-shift a phrase back toward the tessitura centre.

    Octaves rather than semitones, so the line comes home without going out of
    key to do it — and the **whole phrase** rather than the stray notes inside
    it. Shifting notes one at a time inserted an unanswered twelfth into the
    middle of a line the composer had just judged for its leaps, which is a
    correction that breaks the thing it is correcting.
    """
    if not notes:
        return notes
    centre = sum(note.pitch for note in notes) / len(notes)
    distance = centre - drives.register_centre
    if abs(distance) <= 7.0:
        return notes
    if rng.uniform(seed, f"{stream}-pull", 0) > drives.register_pull:
        return notes
    shift = -12.0 * (1 if distance > 0 else -1)
    return [replace(note, pitch=note.pitch + shift) for note in notes]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = ["Written", "choose_patterns", "write_movement"]
