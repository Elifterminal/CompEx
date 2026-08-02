"""Render a generated Composition to samples.

Each voice is rendered onto its own scratch bus so its effects chain can be
applied to the whole part — a reverb tail or a delay repeat has to be allowed
to run past the note that caused it, which per-note processing would chop off.
Only one scratch buffer exists at a time, so this costs memory for one voice,
not for all of them.

Identical notes are synthesised once and reused. A three-minute piece can ask
for a couple of thousand notes, most of them repeats.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from compex.config import SAMPLE_RATE
from compex.dsp import balance, drums, effects, engines, filters, fx
from compex.generate.compose import Composition
from compex.generate.palette import ROLE_PAD, ROLE_PERC, VoiceSpec

TAIL_SECONDS = 3.5
REFERENCE_KIT = 3.0    # the kit size the per-drum gains were written for
MIN_KIT_TRIM = 0.45
VARIANTS = 3           # one-shot variants per voice, cycled so repeats still vary
DUCK_DEPTH = 0.42
DUCK_RELEASE = 0.055

ProgressFn = Optional[Callable[[float, str], None]]


def survey(composition: Composition, sample_rate: int = SAMPLE_RATE) -> balance.Mix:
    """Ask every voice what it sounds like, and decide who needs room.

    One representative note per voice, through its own effects chain, weighted
    by how much that voice actually plays. Cheap — a handful of notes — and it
    is the only way to know that this pad is under a noise wash rather than
    under a sub bass, which is the difference between present and inaudible.
    """
    specs = {voice.voice_id: voice for voice in composition.voices}
    played: dict[str, float] = {}
    for note in composition.notes:
        played[note.voice] = played.get(note.voice, 0.0) + note.velocity * note.duration
    for stroke in composition.strokes:
        played[stroke.voice] = played.get(stroke.voice, 0.0) + stroke.velocity * 0.25

    pitched = [n for n in composition.notes]
    middle: dict[str, float] = {}
    for note in pitched:
        middle.setdefault(note.voice, note.pitch)

    kit_size = sum(1 for v in composition.voices if v.role == ROLE_PERC)
    trim = kit_trim(kit_size)

    energies: dict[str, tuple[str, tuple[float, ...], float]] = {}
    for voice_id, weight in played.items():
        spec = specs.get(voice_id)
        if spec is None or weight <= 0:
            continue
        if spec.role == ROLE_PERC:
            sample = drums.render_drum(spec, sample_rate, composition.seed, 0)
            gain = spec.gain * trim
        else:
            count = int(sample_rate * 1.2)
            sample = engines.render_note(spec, _hz(middle.get(voice_id, 60.0)),
                                         count, sample_rate, composition.seed, 0)
            if spec.effects:
                sample = effects.apply_chain(sample, sample_rate, spec.effects)
            gain = spec.gain
        bands = balance.band_energy(sample, sample_rate)
        scale = weight * gain * gain
        energies[voice_id] = (spec.role, tuple(value * scale for value in bands), gain)

    return balance.weigh(energies, played)


def render_composition(composition: Composition, master_gain: float = 0.89,
                       progress: ProgressFn = None) -> np.ndarray:
    """Render ``composition`` to mono float samples in ``[-1, 1]``."""
    if not 0.0 < master_gain <= 1.0:
        raise ValueError(f"master_gain must be in (0, 1], got {master_gain}")

    seconds_per_beat = composition.seconds_per_beat
    total = int((composition.total_seconds + TAIL_SECONDS) * SAMPLE_RATE)
    if total <= 0:
        raise ValueError("composition has no length")

    melodic = np.zeros(total, dtype=np.float64)
    percussion = np.zeros(total, dtype=np.float64)

    _report(progress, 0.03, "listening to each voice")
    mix = survey(composition)

    _render_voices(melodic, composition, seconds_per_beat, progress, mix.trims())

    _report(progress, 0.80, f"placing {len(composition.strokes)} strokes")
    _render_strokes(percussion, composition, seconds_per_beat, mix.trims())

    _report(progress, 0.90, "ducking against the kick")
    melodic *= _sidechain(total, composition, seconds_per_beat)

    _report(progress, 0.95, "mastering")
    master = filters.highpass(melodic + percussion, SAMPLE_RATE, 24.0)
    master = fx.peak_normalise(master, 0.97) * master_gain
    master = fx.fade_edges(fx.limit(master), SAMPLE_RATE, 0.02)

    _report(progress, 1.0, "done")
    return master


def _render_voices(bus: np.ndarray, composition: Composition, seconds_per_beat: float,
                   progress: ProgressFn, trims: dict[str, float] | None = None) -> None:
    """One voice at a time: synthesise its notes, run its effects, fold it in."""
    specs = {voice.voice_id: voice for voice in composition.voices}
    grouped: dict[str, list] = {}
    for note in composition.notes:
        grouped.setdefault(note.voice, []).append(note)

    for position, (voice_id, notes) in enumerate(grouped.items()):
        spec = specs.get(voice_id)
        if spec is None:
            continue  # a voice that vanished between compose and render — skip, do not crash

        _report(progress, 0.05 + 0.7 * position / max(1, len(grouped)),
                f"{spec.engine} ({len(notes)} notes)"
                + (f" → {', '.join(spec.effect_names())}" if spec.effects else ""))

        scratch = np.zeros(len(bus), dtype=np.float64)
        cache: dict[tuple, np.ndarray] = {}
        for order, note in enumerate(notes):
            count = _note_samples(spec, note.duration, seconds_per_beat)
            variant = order % VARIANTS
            key = (round(note.pitch, 2), count, variant)

            sample = cache.get(key)
            if sample is None:
                sample = engines.render_note(
                    spec, _hz(note.pitch), count, SAMPLE_RATE, composition.seed, variant
                )
                cache[key] = sample
            _add_at(scratch, sample * note.velocity, _sample_at(note.start, seconds_per_beat))

        if spec.effects:
            scratch = effects.apply_chain(scratch, SAMPLE_RATE, spec.effects)
        bus += scratch * spec.gain * (trims or {}).get(voice_id, 1.0)


def kit_trim(count: int) -> float:
    """How much to hold the kit back for having this many drums in it.

    Seven drums should not be seven times one drum. Each voice was given a
    gain as if it were playing alone, so a dense kit arrived at roughly twice
    the level of everything melodic put together — measured, not guessed.
    Square root rather than linear because that is roughly how loudness adds
    when the hits are not simultaneous.
    """
    if count <= 0:
        return 1.0
    return max(MIN_KIT_TRIM, min(1.0, (REFERENCE_KIT / count) ** 0.5))


def _render_strokes(bus: np.ndarray, composition: Composition,
                    seconds_per_beat: float, trims: dict[str, float] | None = None) -> None:
    kit = {voice.voice_id: voice for voice in composition.voices if voice.role == ROLE_PERC}
    trim = kit_trim(len(kit))
    cache: dict[tuple[str, int], np.ndarray] = {}

    for position, stroke in enumerate(composition.strokes):
        spec = kit.get(stroke.voice)
        if spec is None:
            continue
        variant = position % VARIANTS
        key = (stroke.voice, variant)
        sample = cache.get(key)
        if sample is None:
            sample = drums.render_drum(spec, SAMPLE_RATE, composition.seed, variant)
            cache[key] = sample
        _add_at(bus, sample * stroke.velocity * spec.gain * trim
                * (trims or {}).get(stroke.voice, 1.0),
                _sample_at(stroke.start, seconds_per_beat))


def _sidechain(total: int, composition: Composition, seconds_per_beat: float) -> np.ndarray:
    """Duck the melodic bus under every kick. Without it the low end turns to mud."""
    duck = np.ones(total, dtype=np.float64)
    kicks = [stroke for stroke in composition.strokes if stroke.voice in {"kick", "boom"}]
    if not kicks:
        return duck

    window = int(0.17 * SAMPLE_RATE)
    shape = 1.0 - DUCK_DEPTH * np.exp(-np.arange(window) / (DUCK_RELEASE * SAMPLE_RATE))
    for stroke in kicks:
        start = _sample_at(stroke.start, seconds_per_beat)
        end = min(total, start + window)
        if end > start:
            duck[start:end] = np.minimum(duck[start:end], shape[: end - start])
    return duck


def _note_samples(spec: VoiceSpec, duration_beats: float, seconds_per_beat: float) -> int:
    """Note length plus whatever tail this engine needs to ring out."""
    tail = spec.get("release", 0.8) if spec.role == ROLE_PAD else min(
        4.0, max(0.15, spec.get("decay", 1.0))
    )
    return max(64, int((duration_beats * seconds_per_beat + tail) * SAMPLE_RATE))


def _hz(pitch: float) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _add_at(bus: np.ndarray, sample: np.ndarray, offset: int) -> None:
    if offset >= len(bus) or len(sample) == 0:
        return
    end = min(len(bus), offset + len(sample))
    bus[offset:end] += sample[: end - offset]


def _sample_at(beat: float, seconds_per_beat: float) -> int:
    return max(0, int(round(beat * seconds_per_beat * SAMPLE_RATE)))


def _report(progress: ProgressFn, fraction: float, message: str) -> None:
    if progress is not None:
        progress(max(0.0, min(1.0, fraction)), message)
