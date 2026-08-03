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
from compex.generate.clocks import Clock
from compex.generate.pattern import Bed, Grid, Pattern, PatternChoice
from compex.generate.promise import Ledger
from compex.generate.theory import Chord
from compex.measure import Shape


#: What marks a note as a ghost — a line the composer imagined, judged, and
#: decided against. The renderer strips this to find the voice it belongs to.
GHOST_MARK = "~ghost"
GHOST_VELOCITY = 0.8      # how loud a ghost is against the line that beat it
GHOST_OWED_BOOST = 1.5    # a ghost that would have settled a debt is worth hearing

#: How far apart the spacing drive is allowed to push a voice, in octaves.
SPREAD_LIMIT = (-1, 2)
PITCH_RANGE = (24.0, 102.0)


def spread(voices: list[VoiceSpec], spacing: float) -> dict[str, int]:
    """Octave offsets that move the voices out of each other's register.

    This is the composer's answer to a crowded mix, and it is a different
    answer from the mixer's: the mixer can only make a thick voice louder,
    while moving two lines an octave apart makes them two lines. Ordered by
    where each voice naturally sits, then spread around the middle.
    """
    if not voices:
        return {}
    ordered = sorted(voices, key=lambda voice: (voice.octave, voice.voice_id))
    centre = (len(ordered) - 1) / 2.0
    low, high = SPREAD_LIMIT
    return {
        voice.voice_id: max(low, min(high, int(round((index - centre) * spacing))))
        for index, voice in enumerate(ordered)
    }


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
    ghosts: tuple[Note, ...] = ()  # the lines it decided against, kept to be heard


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
                   past_cost: tuple[float, ...] = (),
                   clocks: dict[str, Clock] | None = None,
                   push: tuple[tuple[str, float], ...] = ()) -> Written:
    """Write one movement under the current drives, taste and patterns."""
    notes: list[Note] = []
    strokes: list[Stroke] = []
    choices: list[Choice] = []
    ghosts: list[Note] = []

    end = origin + movement.beats
    chord_index = 0
    cursor = origin
    pushed = dict(push)
    spacing = spread(voices, drives.spacing * pushed.get("spread", 1.0))

    while cursor < end - 1e-6:
        chord = progression[chord_index % len(progression)]
        span = min(chord_beats, end - cursor)
        for voice in voices:
            if voice.role == ROLE_LEAD:
                written, lineage, heard, approach, ledger, choice, haze = _lead(
                    seed, index, chord_index, voice, chord, scale, root_pitch,
                    cursor, span, movement, motif, drives, taste, lineage, heard,
                    approach, ledger, position, past, vocabulary, past_cost,
                    spacing.get(voice.voice_id, 0),
                    (clocks or {}).get(voice.voice_id),
                    pushed.get("field", 1.0))
                notes.extend(written)
                ghosts.extend(haze)
                if choice is not None:
                    choices.append(choice)
                continue
            notes.extend(_voice_notes(seed, index, chord_index, voice, chord, scale,
                                      root_pitch, cursor, span, movement, patterns, drives,
                                      spacing.get(voice.voice_id, 0),
                                      (clocks or {}).get(voice.voice_id)))
        cursor += span
        chord_index += 1

    gate = movement.energy * drives.density_bias
    for drum in kit:
        figure = patterns.get(drum.voice_id)
        if figure is None:
            continue
        for beat, velocity in _on_clock(figure, origin, end,
                                        (clocks or {}).get(drum.voice_id)):
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
                   lineage=lineage, heard=heard, approach=approach, ledger=ledger,
                   ghosts=tuple(ghosts))


# ── the lead: audition, then play the winner ──────────────────────────────

def _lead(seed, index, chord_index, voice, chord, scale, root_pitch, start, span,
          movement, motif, drives, taste, lineage, heard, approach, ledger,
          position=0.0, past=(), vocabulary=(), past_cost=(), octave=0, clock=None,
          field_size=1.0):
    """Choose a phrase for this chord and write it out.

    The choosing happens in scale degrees, which is why it can be judged before
    a pitch exists. What gets written here is the *voiced* phrase — the one the
    audition actually scored — so nothing between the judgement and the sound
    changes the thing that was judged.
    """
    stream = f"{voice.voice_id}-{index}-{chord_index}"
    rest_gate = 0.22 + movement.energy * 0.72 * drives.density_bias
    if rng.uniform(seed, f"{stream}-rest", 0) > rest_gate:
        return [], lineage, heard, approach, ledger, None, []

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
                           field_size=field_size,
                           at_beat=start)
    phrase = melody.voiced(choice.chosen, drives)

    base = root_pitch + 12 * (voice.octave + octave)
    velocity = _clamp(0.35 + 0.65 * movement.energy, 0.08, 1.0)

    notes: list[Note] = []
    offset = 0.0
    played = 0
    # A voice on its own clock fits more (or less) of its material into the
    # same stretch of the piece. The span it is given does not move; how fast
    # it moves through it does.
    rate = clock.ratio if clock else 1.0
    for position, (step, length) in enumerate(zip(phrase.steps, phrase.rhythm)):
        if offset >= span:
            break
        pitch = _in_range(float(base + theory.degree_semitone(scale, chord.degree + step)))
        played = position
        notes.append(Note(
            start=start + offset / rate,
            duration=min(length / rate, span - offset / rate),
            pitch=pitch,
            velocity=_clamp(velocity + rng.between(seed, f"{stream}-v", position, -0.12, 0.12),
                            0.08, 1.0),
            voice=voice.voice_id,
            timbre=index,
        ))
        offset += length

    notes = _pull_toward_centre(notes, drives, seed, stream)
    # Judged against the ledger as it stood when the audition happened, not
    # after this phrase has paid anything: what the losers *would* have settled
    # is a fact about the moment the choice was made.
    haze = _ghosts(choice, voice, chord, scale, base, start, span, velocity, drives, ledger)

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
    return notes, choice.chosen, heard, landed, ledger, choice, haze


def _ghosts(choice: Choice, voice: VoiceSpec, chord: Chord, scale, base: int,
            start: float, span: float, velocity: float, drives: Drives,
            ledger: Ledger) -> list[Note]:
    """Write out the lines the composer decided against, quietly.

    This is the thing the project is named for. Every phrase you hear beat two
    to six others; until now those were scored and thrown away, so the only
    audible trace of the machine having *had* a choice was that a choice had
    been made. Playing the runners-up underneath, at a level set by how close
    they came, makes the deciding itself audible: when the composer was sure,
    the texture is clean, and when it was genuinely torn, the line blooms into
    a haze of the things it nearly played instead.

    A loser that would have settled an outstanding promise is heard louder than
    its score alone earns. That one is not merely a road not taken — it is the
    piece declining to do something it owes, and that is worth hearing.
    """
    haze: list[Note] = []
    for ghost in choice.ghosts:
        level = ghost.closeness()
        if promise.would_settle_any(ledger.live(start, domain="pitch"),
                                    ghost.phrase.steps, chord.degree):
            level = min(1.0, level * GHOST_OWED_BOOST)
        if level < 0.02:
            continue

        spoken = melody.voiced(ghost.phrase, drives)
        offset = 0.0
        for step, length in zip(spoken.steps, spoken.rhythm):
            if offset >= span:
                break
            haze.append(Note(
                start=start + offset,
                duration=min(length, span - offset),
                pitch=_in_range(float(base + theory.degree_semitone(scale, chord.degree + step))),
                velocity=_clamp(velocity * level * GHOST_VELOCITY, 0.02, 1.0),
                voice=f"{voice.voice_id}{GHOST_MARK}",
            ))
            offset += length
    return haze


# ── everything that is not the tune ───────────────────────────────────────

def _voice_notes(seed, movement_index, chord_index, voice, chord, scale, root_pitch,
                 start, span, movement, patterns, drives, octave=0,
                 clock=None) -> list[Note]:
    stream = f"{voice.voice_id}-{movement_index}-{chord_index}"
    base = root_pitch + 12 * (voice.octave + octave)
    velocity = _clamp(0.35 + 0.65 * movement.energy, 0.08, 1.0)

    if voice.role == ROLE_PAD:
        pitches = [base + step for step in chord.semitones(scale)]
        return [
            Note(start=start, duration=span, pitch=_in_range(float(pitch)),
                 velocity=velocity * 0.7, voice=voice.voice_id, timbre=movement_index)
            for pitch in pitches
        ]

    if voice.role == ROLE_BASS:
        return _bass(seed, stream, voice, chord, scale, base, start, span,
                     movement, patterns.get(voice.voice_id), drives, velocity, clock)

    if voice.role == ROLE_TEXTURE:
        if rng.uniform(seed, f"{stream}-t", 0) > 0.35 + movement.energy * 0.4:
            return []
        pitch = _in_range(float(base + theory.degree_semitone(scale, chord.degree + 4)))
        return [Note(start=start, duration=span * 0.9, pitch=pitch,
                     velocity=velocity * 0.55, voice=voice.voice_id,
                     timbre=movement_index)]

    return []


def _bass(seed, stream, voice, chord, scale, base, start, span, movement,
          figure: Pattern | None, drives, velocity, clock=None) -> list[Note]:
    """The bass reads the same kind of grid the drums do.

    It used to roll dice per subdivision, which meant it never played a figure
    twice and so never played a figure at all. Now it has one, chosen the same
    way the kick's is, and the interlock criterion has already made sure the two
    are not fighting for the same slots.
    """
    root = _in_range(float(base + theory.degree_semitone(scale, chord.degree)))
    if figure is None:
        return [Note(start=start, duration=min(1.0, span), pitch=root,
                     velocity=velocity, voice=voice.voice_id)]

    notes: list[Note] = []
    for beat, level in _on_clock(figure, start, start + span, clock):
        offset = beat - start
        # A low voice that changes note on every hit stops being a foundation.
        # Anything off the strong slots gets the fifth instead, which is what a
        # bass player does when they want movement without leaving the chord.
        degree = chord.degree if level > 0.6 else chord.degree + 4
        pitch = _in_range(float(base + theory.degree_semitone(scale, degree)))
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


def _on_clock(figure: Pattern, start: float, end: float,
              clock: Clock | None) -> tuple[tuple[float, float], ...]:
    """A pattern's hits, on this voice's own clock.

    The grid is generated in the voice's time and then mapped back into the
    piece's, which is what makes three-against-two a *pattern* at a different
    rate rather than a pattern with its hits nudged around.
    """
    if clock is None or clock.anchored:
        return figure.hits(start, end)
    rate = clock.ratio
    return tuple((beat / rate, level)
                 for beat, level in figure.hits(start * rate, end * rate))


def _in_range(pitch: float) -> float:
    """Fold a pitch back inside the range the engines are built for.

    Spacing can push a voice off the end of the keyboard, and an engine asked
    for a 12 Hz fundamental does not fail — it just makes something nobody can
    hear, which is worse.
    """
    low, high = PITCH_RANGE
    while pitch < low:
        pitch += 12.0
    while pitch > high:
        pitch -= 12.0
    return pitch


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


__all__ = ["Written", "choose_patterns", "write_movement"]
