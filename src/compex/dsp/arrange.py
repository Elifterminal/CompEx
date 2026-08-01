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
from compex.dsp import drums, effects, engines, filters, fx
from compex.generate.compose import Composition
from compex.generate.palette import ROLE_PAD, ROLE_PERC, VoiceSpec

TAIL_SECONDS = 3.5
VARIANTS = 3           # one-shot variants per voice, cycled so repeats still vary
DUCK_DEPTH = 0.42
DUCK_RELEASE = 0.055

ProgressFn = Optional[Callable[[float, str], None]]


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

    _render_voices(melodic, composition, seconds_per_beat, progress)

    _report(progress, 0.80, f"placing {len(composition.strokes)} strokes")
    _render_strokes(percussion, composition, seconds_per_beat)

    _report(progress, 0.90, "ducking against the kick")
    melodic *= _sidechain(total, composition, seconds_per_beat)

    _report(progress, 0.95, "mastering")
    mix = filters.highpass(melodic + percussion, SAMPLE_RATE, 24.0)
    mix = fx.peak_normalise(mix, 0.97) * master_gain
    mix = fx.fade_edges(fx.limit(mix), SAMPLE_RATE, 0.02)

    _report(progress, 1.0, "done")
    return mix


def _render_voices(bus: np.ndarray, composition: Composition, seconds_per_beat: float,
                   progress: ProgressFn) -> None:
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
        bus += scratch * spec.gain


def _render_strokes(bus: np.ndarray, composition: Composition,
                    seconds_per_beat: float) -> None:
    kit = {voice.voice_id: voice for voice in composition.voices if voice.role == ROLE_PERC}
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
        _add_at(bus, sample * stroke.velocity * spec.gain,
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
