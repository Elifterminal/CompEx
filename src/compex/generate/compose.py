"""Invent a whole piece from a seed, a runtime and a mood — and keep changing it.

Nothing is looked up. Tempo, meter, mode, harmony, the germ motif, the form,
which instruments exist and how they are voiced are all decided by walking the
theory in :mod:`compex.generate.theory` with the seeded RNG.

But the piece is not decided all at once. After each movement the composer
**listens back** to what it actually wrote (:mod:`compex.generate.critic`),
compares it against a set of named aesthetic principles, and **shifts the
drives it writes with** (:mod:`compex.generate.evolve`) before starting the
next one. The harmony is reharmonised under the new drives, the motif is
either developed further or recalled through a lossy memory, and the note
writing itself reads the drives rather than fixed constants.

Two of those decisions are big enough to have their own modules. Every phrase
the lead plays is **auditioned** — several candidates imagined, judged, and one
kept (:mod:`compex.generate.melody`) — and every rhythm is a **pattern the
composer decided on** and keeps revising out of what it has already played
(:mod:`compex.generate.pattern`). Both are governed by weights that start
somewhere derived from the mood and then go wherever the music argues for,
including past where they started.

So the second half of a piece is a consequence of the first half, not just of
the seed. It still comes out identical every time — the analysis is
deterministic and so is everything it feeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from compex import rng
from compex.generate import (
    clocks as clockwork,
)
from compex.generate import (
    critic, intent as planning, melody, pattern, promise, recall, theory, tuning, write,
)
from compex.generate.critic import Analysis, Verdict
from compex.generate import evolve
from compex.generate.evolve import Drives, Evolution, initial_drives, respond, trace
from compex.generate.material import Movement, Note, Stroke
from compex.generate.melody import Choice, Phrase, Taste
from compex.generate.mood import Mood
from compex.generate.palette import (
    DRUM_COLOUR,
    ROLE_BASS,
    ROLE_LEAD,
    ROLE_PAD,
    ROLE_PERC,
    ROLE_TEXTURE,
    VoiceSpec,
    build_kit,
    build_voice,
)
from compex.generate.pattern import Grid, Groove, Pattern, PatternChoice
from compex.generate.promise import Ledger
from compex.generate.theory import Chord, Motif
from compex.generate.clocks import Clock
from compex.generate.intent import Plan
from compex.generate.tuning import Tuning
from compex import measure, remember
from compex.dsp import listen
from compex.remember import Memory
from compex.measure import Reveal, Shape

MIN_BPM, MAX_BPM = 52.0, 168.0
MIN_MOVEMENTS, MAX_MOVEMENTS = 2, 8
SECONDS_PER_MOVEMENT = 16.0

ARCHETYPES: tuple[str, ...] = ("arch", "ramp", "terraced", "return", "through", "erode")

#: How the bass sits in the pattern machinery. It is not a drum, so it has no
#: entry in the kit's colour map — but it needs the same two numbers, and a
#: low, fairly gentle voice is what a bass is.
BASS_CHARACTER = (0.18, 0.35)

#: How much of its own past the composer weighs up when deciding whether a
#: candidate would explain it. All of it would be quadratic in a long piece and
#: the opening is the part worth explaining anyway.
PAST_WINDOW = 24
VOCABULARY_WINDOW = 32


@dataclass(frozen=True)
class Retuning:
    """What one listen-back did to the composer's taste, as opposed to its drives."""

    movement: int
    movement_name: str
    taste: tuple[melody.Shift, ...]
    groove: tuple[pattern.Shift, ...]

    def note(self) -> str:
        moved = [shift.describe() for shift in self.taste]
        moved += [shift.describe() for shift in self.groove]
        return "; ".join(moved) if moved else "taste held"


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
    notes: tuple[Note, ...]
    strokes: tuple[Stroke, ...]
    ghosts: tuple[Note, ...] = ()
    evolution: tuple[Evolution, ...] = ()
    final_drives: Drives = Drives()
    melodies: tuple[Choice, ...] = ()
    patterns: tuple[PatternChoice, ...] = ()
    grids: tuple[Grid, ...] = ()
    retunings: tuple[Retuning, ...] = ()
    tuning: Tuning = Tuning(name="ionian", family="twelve",
                            steps=(0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0))
    clocks: tuple[Clock, ...] = ()
    #: Every state each voice passed through, as (voice, movement, spec).
    #: Instruments drift now, so one spec per voice is no longer the whole truth.
    states: tuple[tuple[str, int, VoiceSpec], ...] = ()
    heard: tuple[listen.Heard, ...] = ()
    memory: Memory = Memory()
    plan: Plan = Plan(intent=planning.INTENTS[0])
    taste: Taste = Taste()
    opening_taste: Taste = Taste()
    groove: Groove = Groove()
    ledger: Ledger = Ledger()
    reveal: Reveal | None = None

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

    def state(self, voice_id: str, timbre: int) -> VoiceSpec | None:
        """The voice as it stood when a given note was written."""
        for name, movement, spec in self.states:
            if name == voice_id and movement == timbre:
                return spec
        return self.voice(voice_id)

    def evolution_trace(self) -> str:
        return trace(self.evolution)

    def melody_trace(self) -> str:
        return melody.trace(self.melodies)

    def pattern_trace(self) -> str:
        return pattern.trace(self.patterns)

    def ledger_trace(self) -> str:
        return promise.trace(self.ledger, self.total_beats)

    def shapes(self) -> tuple[Shape, ...]:
        """Every phrase the piece chose, as material a description could refer to."""
        return tuple(_shape_of(choice) for choice in self.melodies)

    def final_patterns(self) -> tuple[Pattern, ...]:
        """The last pattern each voice settled on."""
        latest: dict[str, Pattern] = {}
        for choice in self.patterns:
            latest[choice.voice] = choice.pattern
        return tuple(latest[voice] for voice in sorted(latest))

    def summary(self) -> str:
        pitched = [v for v in self.voices if v.role != ROLE_PERC]
        drums = [v for v in self.voices if v.role == ROLE_PERC]
        corrections = sum(len(step.adjustments) for step in self.evolution)
        auditioned = sum(choice.considered for choice in self.melodies)
        strayed = self.taste.strayed_from(self.opening_taste)
        moved = sum(1 for grid in self.grids if grid.past_start())
        return "\n".join([
            f"{self.bpm:.1f} BPM in {self.beats_per_bar}/4 · {self.tuning.describe()}"
            f" on {_pitch_name(self.root_pitch)}",
            f"form: {self.archetype} — " + " → ".join(m.name for m in self.movements),
            f"harmony: {len(self.progression)} chords, one per {self.chord_beats:g} beats",
            f"motif: {len(self.motif.steps)} notes, {self.motif.beats:g} beats",
            "voices: " + ", ".join(f"{v.role}/{v.engine}" for v in pitched),
            "kit: " + (", ".join(v.voice_id for v in drums) or "none"),
            f"{len(self.notes)} notes, {len(self.strokes)} strokes",
            f"melody: {len(self.melodies)} phrases chosen out of {auditioned} imagined, "
            f"{_generations(self.melodies)} generations deep",
            "patterns: " + (", ".join(f"{p.voice} |{p.describe()}|"
                                      for p in self.final_patterns()) or "none"),
            f"taste moved on: {', '.join(strayed) if strayed else 'nothing'}"
            f" · {moved} rhythm grid(s) past their starting weights",
            f"promises: {self.ledger.summary(self.total_beats)}",
            "heard: " + (self.heard[-1].describe() if self.heard else "not listened to"),
            f"memory: {self.memory.describe()}",
            f"intent: {self.plan.summary()}",
            f"ghosts: {len(self.ghosts)} notes it decided against, kept audible",
            "clocks: " + clockwork.summary({c.voice: c for c in self.clocks},
                                           float(self.beats_per_bar)),
            f"hindsight: {self.reveal.describe() if self.reveal else 'not measured'}",
            f"evolved: {corrections} corrections over {len(self.evolution)} listen-backs, "
            f"plasticity {self.final_drives.plasticity:.2f}",
        ])


def compose(seed: int, duration_s: float, mood: Mood,
            memory: Memory = Memory()) -> Composition:
    """Invent a piece. Same arguments in, same piece out, always.

    ``memory`` is what the engine carries from the pieces it has already made:
    where its taste ended up, and what it has been reaching for lately. It is
    an *argument* rather than hidden state, which is the only way it can exist
    at all here — determinism is the property everything else rests on, and a
    hidden accumulator would have broken it silently. Same seed, same mood,
    same memory digest, same audio; the digest is printed in the formula.
    """
    if duration_s <= 0:
        raise ValueError(f"duration must be positive, got {duration_s}")
    if not isinstance(mood, Mood):
        raise TypeError(f"mood must be a Mood, got {type(mood).__name__}")

    bpm = _tempo(seed, mood)
    beats_per_bar = _meter(seed, mood)
    root_pitch = 36 + int(rng.uniform(seed, "root", 0) * 12)

    # The twelve-tone grid is a table, not a fact — so it picks its own.
    stale_tunings = {name: remember.bored_of(memory, "tunings", name)
                     for name in ("twelve", "equal", "just")}
    voicing = tuning.derive(seed, mood, bored=stale_tunings)
    scale_name, scale = voicing.name, voicing.steps

    chord_count = max(2, min(8, 2 + int(mood.density * 6.5)))
    progression = theory.progression(seed, scale, chord_count, mood.tension)
    chord_beats = float(beats_per_bar) * rng.pick(seed, "harmonic-rhythm", 0, (1.0, 1.0, 2.0, 4.0))

    motif = theory.make_motif(seed, scale, mood.energy, mood.tension)
    sketch = recall.compress(motif)
    archetype = _archetype(seed, mood)

    movements = _movements(seed, duration_s, bpm, beats_per_bar, mood, archetype)
    total_beats = sum(movement.beats for movement in movements)
    voices = _voices(seed, mood, memory)
    kit = build_kit(seed, mood)
    grids = _grids(seed, kit, voices, beats_per_bar, mood)
    ticking = clockwork.assign(seed, voices, kit, mood)
    lead_voices = frozenset(v.voice_id for v in voices if v.role == ROLE_LEAD)

    drives = initial_drives(mood, root_pitch)
    plan = planning.choose(seed, mood)
    opening_taste = _remembered_taste(melody.initial_taste(mood), memory)
    taste = opening_taste
    groove = _remembered_groove(pattern.initial_groove(mood), memory)

    notes: list[Note] = []
    strokes: list[Stroke] = []
    ghosts: list[Note] = []
    history: list[Evolution] = []
    melodies: list[Choice] = []
    patterns: list[PatternChoice] = []
    retunings: list[Retuning] = []

    current_motif = motif
    states: list[tuple[str, int, VoiceSpec]] = []
    listened: list[listen.Heard] = []
    lineage: Phrase | None = None
    heard_pairs: frozenset[tuple[int, int]] = frozenset()
    approach: int | None = None
    ledger = Ledger()
    shapes: list[Shape] = [Shape(label="germ", steps=motif.steps, rhythm=motif.rhythm)]
    revelation = 0.5
    figures: dict[str, Pattern] = {}
    cursor = 0.0

    for index, movement in enumerate(movements):
        # ── listen back before writing anything new ──────────────────────
        if index > 0:
            # Only ask what the piece has explained about itself once there is
            # enough of it to have been strange in the first place.
            revelation = _revelation(shapes, movements, index, cursor) \
                if index > 1 else 0.5
            # And it listens to the sound, not only to the score. One short
            # window of what the last movement actually became.
            heard = listen.probe(notes, {name: spec for name, _, spec in states} or
                                 {v.voice_id: v for v in voices},
                                 seed, 60.0 / bpm, max(0.0, cursor - movement.beats),
                                 skip=frozenset(drum.voice_id for drum in kit))
            listened.append(heard)
            analysis = critic.analyse(notes, strokes, motif, scale, root_pitch,
                                      cursor, lead_voices, revelation=revelation,
                                      heard=heard)
            verdicts = critic.judge(analysis, mood)
            before = drives
            drives, adjustments = respond(drives, verdicts, analysis)
            history.append(Evolution(
                movement=index, movement_name=movement.name, at_beat=cursor,
                analysis=analysis, verdicts=verdicts, adjustments=adjustments,
                before=before, after=drives,
            ))

            # The same verdicts teach the ear as well as the hand: drives are
            # what it writes with, taste is what it listens for when choosing.
            recent = tuple(c for c in melodies if c.movement == index - 1)
            taste, taste_shifts = melody.retune(taste, opening_taste, verdicts, recent, drives)
            groove, groove_shifts = pattern.retune(groove, verdicts, drives)

            # And then the plan: not another judge of the music, but a target
            # shape for the composer's own state, leaning on the weights that
            # were already deciding things.
            reading = planning.read(
                plan, cursor / max(total_beats, 1e-6),
                pressure=ledger.pressure(cursor),
                margin=(sum(c.margin for c in recent) / len(recent)) if recent else 0.0,
                crowding=analysis.crowding,
            )
            plan = planning.reconsider(plan, reading, seed, index)
            retunings.append(Retuning(index, movement.name, taste_shifts, groove_shifts))

            current_motif, used = _next_motif(current_motif, sketch, seed, index,
                                              movement.tension, drives)
            drives = drives.with_use(used)

        # The instruments are variables too: each movement they are a little
        # further along in becoming something else, under the drift drive.
        playing = [build_voice_state(voice, seed, index, drives.timbre_drift)
                   for voice in voices]
        states.extend((spec.voice_id, index, spec) for spec in playing)

        working = _reharmonise(progression, seed, index, drives, scale)
        ledger = _harmony_promises(ledger, working, scale, cursor,
                                   plan.lever("settling"))

        figures, chosen, ledger = write.choose_patterns(
            seed, index, movement, grids, groove, drives,
            float(beats_per_bar), figures, ledger, cursor)
        patterns.extend(chosen)

        past = tuple(shapes[1:PAST_WINDOW + 1])
        written = write.write_movement(
            seed, index, movement, cursor, scale, root_pitch, working, chord_beats,
            current_motif, playing, kit, figures, drives, taste, lineage, heard_pairs, approach,
            ledger, cursor / max(total_beats, 1e-6),
            past, tuple(shapes[-VOCABULARY_WINDOW:]),
            measure.baseline(past, tuple(shapes[:1])), ticking,
            push=plan.bias,
        )
        notes.extend(written.notes)
        strokes.extend(written.strokes)
        ghosts.extend(written.ghosts)
        melodies.extend(written.choices)
        shapes.extend(_shape_of(choice) for choice in written.choices)
        lineage, heard_pairs, approach = written.lineage, written.heard, written.approach
        ledger = written.ledger

        # What it played is what it now believes belongs there. This is the
        # channel the critic has no part in — the grids learn from the output.
        grids = {voice: pattern.learn(grid, figures[voice], drives.plasticity)
                 if voice in figures else grid
                 for voice, grid in grids.items()}
        cursor += movement.beats

    notes.sort(key=lambda note: (note.start, note.voice))
    strokes.sort(key=lambda stroke: (stroke.start, stroke.voice))
    ghosts.sort(key=lambda note: (note.start, note.voice))

    return Composition(
        seed=seed, mood=mood, bpm=bpm, beats_per_bar=beats_per_bar, root_pitch=root_pitch,
        scale_name=scale_name, scale=scale, archetype=archetype, tuning=voicing,
        progression=progression,
        chord_beats=chord_beats, motif=motif, movements=movements,
        voices=tuple(voices) + kit,
        notes=tuple(notes), strokes=tuple(strokes), ghosts=tuple(ghosts),
        evolution=tuple(history), final_drives=drives,
        melodies=tuple(melodies), patterns=tuple(patterns),
        grids=tuple(grids.values()), retunings=tuple(retunings),
        clocks=tuple(ticking.values()), states=tuple(states),
        heard=tuple(listened),
        taste=taste, opening_taste=opening_taste, groove=groove, memory=memory, plan=plan,
        ledger=ledger.forget(cursor),
        reveal=_reveal_of(tuple(shapes), movements),
    )


# ── how the piece changes its mind ────────────────────────────────────────

def _next_motif(current: Motif, sketch: recall.Sketch, seed: int, index: int,
                tension: float, drives: Drives) -> tuple[Motif, str]:
    """Either develop what we have, or try to remember the germ and fail usefully."""
    if rng.uniform(seed, "recall-gate", index) < drives.motif_recall:
        return recall.remember(sketch, seed, index, drives), "recall"
    return theory.develop(current, seed, index, tension, fatigue=drives.fatigue)


def _reharmonise(base: tuple[Chord, ...], seed: int, index: int, drives: Drives,
                 scale: tuple[int, ...]) -> tuple[Chord, ...]:
    """Re-voice the progression under the current drives.

    Chord heights are drawn against the dissonance the composer will currently
    tolerate, and novelty pressure decides how often a degree gets swapped for
    a neighbour. The harmonic skeleton survives; its flesh moves.
    """
    sizes = theory.chord_sizes(drives.dissonance_ceiling)
    out: list[Chord] = []
    for position, chord in enumerate(base):
        size = rng.pick(seed, f"resize-{index}", position, sizes)
        degree = chord.degree
        if rng.uniform(seed, f"subst-{index}", position) < drives.novelty_pressure * 0.32:
            move = rng.pick(seed, f"substmove-{index}", position, (2, -2, 3, -3, 4))
            degree = (degree + move) % len(scale)
        out.append(Chord(degree=degree, size=size))
    return tuple(out)


def build_voice_state(voice: VoiceSpec, seed: int, movement: int,
                      amount: float) -> VoiceSpec:
    """This voice as it stands in this movement."""
    from compex.generate.palette import drift

    return drift(voice, seed, movement, amount)


def _shape_of(choice: Choice) -> Shape:
    """A chosen phrase as something a later description could point at."""
    return Shape(label=f"m{choice.movement}@{choice.at_beat:g}",
                 steps=choice.chosen.steps, rhythm=choice.chosen.rhythm,
                 at_beat=choice.at_beat)


def _reveal_of(shapes: tuple[Shape, ...], movements: tuple[Movement, ...]) -> Reveal:
    """Describe the opening as it was written, then as the finished piece can.

    The germ is in both vocabularies — it existed before a note did. Everything
    else the piece learned to say about itself is only in the second one, and
    the difference between the two descriptions is what the ending gave back to
    the beginning.

    "The opening" is the first half rather than the first movement. A first
    movement can be two phrases long, and a measurement taken on two phrases is
    a coin toss dressed as a number.
    """
    if not movements:
        return measure.reveal((), (), ())
    half = sum(movement.beats for movement in movements) * 0.5
    germ = shapes[:1]
    written = shapes[1:]
    opening = tuple(shape for shape in written if shape.at_beat < half)[:PAST_WINDOW]
    later = tuple(shape for shape in written if shape.at_beat >= half)
    return measure.reveal(opening, germ, later)


def _revelation(shapes: list[Shape], movements: tuple[Movement, ...], index: int,
                cursor: float) -> float:
    """How much of its own opening the piece has explained by now, 0..1."""
    so_far = _reveal_of(tuple(shapes), movements[:index] or movements)
    return min(1.0, so_far.share)


#: Below this the plan withholds permission to settle: the harmony comes home
#: and the piece keeps owing anyway. Above the second, it stops waiting out the
#: full term on debts that have gone stale and lets them go.
HOLD, LET_GO = 0.8, 1.3

#: How many debts a plan past :data:`LET_GO` abandons per movement, per unit of
#: permission. Advancing the clock instead was tried first and did nothing: a
#: promise holds full weight for its whole term by design, so nothing fades
#: early no matter how far the clock is pushed. Letting go has to be a decision.
LETTING = 4


def _harmony_promises(ledger: Ledger, working: tuple[Chord, ...], scale: tuple[int, ...],
                      at_beat: float, settling: float = 1.0) -> Ledger:
    """Open and settle the obligations a movement's harmony takes on.

    A progression that reaches a long way round the circle of fifths owes
    either a return or a reason. What settles it is *ending* at home — every
    progression here starts there, so passing through on the way out would
    make the debt cancel itself the moment it was taken on.

    ``settling`` is the plan's permission, and it is a permission rather than a
    preference on purpose. Below :data:`HOLD` the piece refuses to close what
    it opened even when the harmony hands it the chance — it comes home and
    keeps owing anyway, which is the only way anything can be made to carry.
    Above :data:`LET_GO` it stops waiting the full term and drops what has gone
    stale, which is not settling: it is giving up on it.
    """
    span = max(1, len(scale))
    if settling > LET_GO:
        ledger = ledger.let_go(at_beat, int((settling - LET_GO) * LETTING))
    if working and working[-1].degree % span == 0 and settling >= HOLD:
        for owed in ledger.live(at_beat, domain="harmony"):
            ledger = ledger.settle(owed, at_beat, "the harmony came home")
        return ledger

    furthest = max(working, key=lambda chord: promise.fifth_distance(chord.degree, span),
                   default=None)
    if furthest is not None:
        ledger = ledger.opened(promise.from_harmony(furthest.degree, scale, at_beat))
    return ledger


# ── the up-front decisions ────────────────────────────────────────────────

def _tempo(seed: int, mood: Mood) -> float:
    base = MIN_BPM + mood.energy * (MAX_BPM - MIN_BPM)
    jitter = rng.between(seed, "tempo", 0, -9.0, 9.0)
    return round(max(MIN_BPM, min(MAX_BPM, base + jitter)), 1)


def _archetype(seed: int, mood: Mood) -> str:
    """Weight the form archetypes by mood, then draw. Every form stays possible."""
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


def _voices(seed: int, mood: Mood, memory: Memory = Memory()) -> list[VoiceSpec]:
    bored = {name: remember.bored_of(memory, "engines", name)
             for name, _ in memory.engines}
    voices = [build_voice(seed, 0, ROLE_BASS, mood, bored=bored)]
    if mood.density > 0.12:
        voices.append(build_voice(seed, 1, ROLE_LEAD, mood, bored=bored))
    for slot in range(1 + int(mood.density * 2.2)):
        voices.append(build_voice(seed, 2 + slot, ROLE_PAD, mood, bored=bored))
    if mood.density > 0.48:
        voices.append(build_voice(seed, 8, ROLE_TEXTURE, mood, bored=bored))
    return voices


def _remembered_taste(fresh: Taste, memory: Memory) -> Taste:
    """Start listening for what past pieces learned to listen for."""
    if memory.is_blank():
        return fresh
    from dataclasses import replace as _replace

    return _replace(fresh, **{
        name: remember.lean(memory, "taste", value, name)
        for name, value in fresh.weights()
    })


def _remembered_groove(fresh: Groove, memory: Memory) -> Groove:
    if memory.is_blank():
        return fresh
    from dataclasses import replace as _replace

    return _replace(fresh, **{
        name: remember.lean(memory, "groove", value, name)
        for name, value in fresh.weights()
    })


def _grids(seed: int, kit: tuple[VoiceSpec, ...], voices: list[VoiceSpec],
           beats_per_bar: int, mood: Mood) -> dict[str, Grid]:
    """A rhythm grid per voice that plays rhythms, drums and bass alike.

    The drums go first so the bass has to find room around them rather than the
    other way round — a kick that has to dodge the bass stops being a kick.
    """
    grids: dict[str, Grid] = {}
    for index, drum in enumerate(kit):
        brightness, aggression = DRUM_COLOUR.get(drum.voice_id, (0.5, 0.5))
        grids[drum.voice_id] = pattern.build_grid(
            drum.voice_id, seed, index, beats_per_bar, mood, brightness, aggression)

    for offset, voice in enumerate(v for v in voices if v.role == ROLE_BASS):
        brightness, aggression = BASS_CHARACTER
        grids[voice.voice_id] = pattern.build_grid(
            voice.voice_id, seed, len(kit) + offset, beats_per_bar, mood,
            brightness, aggression)
    return grids


def _generations(melodies: tuple[Choice, ...]) -> int:
    return max((choice.chosen.generation for choice in melodies), default=0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _pitch_name(pitch: int) -> str:
    return f"{_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"
