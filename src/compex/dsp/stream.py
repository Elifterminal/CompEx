"""Render a piece a span at a time, so playback can start before it is finished.

A ten-minute track is twenty-six million samples. Waiting for all of it before
hearing any of it is bad on a desktop and unusable on a phone, and the whole
point of a long track here is to hear the composer change its mind — which you
cannot do if you are staring at a progress bar.

So this renders in spans and hands each one over as it is ready.

The awkward part is that sound does not respect span boundaries. A note struck
near the end of a span rings past it; so does a reverb tail. Each span is
therefore rendered longer than it needs to be, the finished part is handed
over, and the overhang is **carried** into the next span and summed in. Nothing
is clipped at a seam.

The one deliberate difference from the offline renderer: no whole-piece peak
normalisation, because that cannot be computed before the piece exists. Gain is
fixed and the limiter catches the rest, so every span is levelled the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compex.config import SAMPLE_RATE
from compex.dsp import arrange, drums, effects, engines, filters, fx
from compex.dsp.arrange import (
    DUCK_DEPTH,
    DUCK_RELEASE,
    GHOST_CUTOFF,
    TAIL_SECONDS,
    VARIANTS,
    _note_samples,
    kit_trim,
)
from compex.generate.compose import Composition
from compex.generate.palette import ROLE_PERC
from compex.generate.write import GHOST_MARK
from compex.generate.theory import midi_to_hz

DEFAULT_SPAN_SECONDS = 8.0

TARGET_PEAK = 0.92     # where the level stage aims
PRIOR_RATIO = 0.50     # mixes peak at about half the sum of their voice gains
PEAK_DECAY = 0.985     # how fast the assumed peak relaxes when nothing is loud
MAX_BOOST = 2.5        # never amplify quiet material more than this


@dataclass(frozen=True)
class Level:
    """Running level state, so spans come out at a consistent loudness.

    The offline renderer normalises against the whole piece, which is not
    available here — the point of streaming is that the end has not been
    rendered yet. Instead the level stage starts from a prediction (a mix
    peaks at roughly half the sum of its voice gains, measured across every
    theme) and corrects as real peaks arrive. Gain is ramped across a span
    rather than stepped, so a correction is never audible as a jump.

    Without this a dense piece slams the limiter — measured at 11% of samples
    for *frantic* under a fixed gain, which sounds crushed.
    """

    assumed_peak: float
    gain: float


def initial_level(composition: Composition, master_gain: float = 0.89) -> Level:
    bound = sum(
        voice.gain * (0.95 if voice.role == ROLE_PERC else 0.9)
        for voice in composition.voices
    )
    assumed = max(0.05, bound * PRIOR_RATIO)
    return Level(assumed_peak=assumed,
                 gain=min(MAX_BOOST, TARGET_PEAK / assumed) * master_gain)


@dataclass(frozen=True)
class Span:
    """One slice of the timeline, in beats."""

    index: int
    start_beat: float
    end_beat: float
    movement: str

    @property
    def beats(self) -> float:
        return self.end_beat - self.start_beat


def plan(composition: Composition, span_seconds: float = DEFAULT_SPAN_SECONDS) -> tuple[Span, ...]:
    """Cut the piece into spans, never crossing a movement boundary.

    Keeping spans inside movements means each one carries a single movement's
    name, so a player can say which part of the form you are hearing.
    """
    if span_seconds <= 0:
        raise ValueError(f"span_seconds must be positive, got {span_seconds}")

    per_span = span_seconds / composition.seconds_per_beat
    out: list[Span] = []
    cursor = 0.0
    for movement in composition.movements:
        end = cursor + movement.beats
        position = cursor
        while position < end - 1e-6:
            stop = min(position + per_span, end)
            out.append(Span(len(out), position, stop, movement.name))
            position = stop
        cursor = end
    return tuple(out)


def render_span(composition: Composition, span: Span, carry: np.ndarray | None = None,
                master_gain: float = 0.89, level: Level | None = None,
                trims: dict[str, float] | None = None, ghost_gain: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray, Level]:
    """Render one span. Returns its samples, the overhang, and the level state.

    ``trims`` are the mixer's per-voice decisions. They are computed once for
    the whole piece and passed in rather than recomputed per span, because a
    voice's place in the mix is a property of the piece — and because the
    phone and the desktop have to arrive at the same balance or they are not
    playing the same music.
    """
    if not 0.0 < master_gain <= 1.0:
        raise ValueError(f"master_gain must be in (0, 1], got {master_gain}")
    if level is None:
        level = initial_level(composition, master_gain)

    seconds_per_beat = composition.seconds_per_beat
    length = max(1, int(round(span.beats * seconds_per_beat * SAMPLE_RATE)))
    tail = int(TAIL_SECONDS * SAMPLE_RATE)
    total = length + tail

    melodic = np.zeros(total, dtype=np.float64)
    percussion = np.zeros(total, dtype=np.float64)

    if trims is None:
        trims = arrange.survey(composition).trims()
    _voices_in_span(melodic, composition, span, seconds_per_beat, total, trims)
    if ghost_gain > 0 and composition.ghosts:
        melodic += _ghosts_in_span(composition, span, seconds_per_beat, total) * ghost_gain
    _strokes_in_span(percussion, composition, span, seconds_per_beat, total, trims)
    melodic *= _duck(total, composition, span, seconds_per_beat)

    mix = filters.highpass(melodic + percussion, SAMPLE_RATE, 24.0)
    if carry is not None and len(carry):
        overlap = min(len(carry), total)
        mix[:overlap] += carry[:overlap]

    # Split before gain: the overhang has to stay un-gained, or the next span
    # multiplies it a second time and every tail comes out at gain squared.
    body, overhang = mix[:length], mix[length:].copy()

    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    assumed = max(level.assumed_peak * PEAK_DECAY, peak)
    target = min(MAX_BOOST, TARGET_PEAK / max(assumed, 1e-6)) * master_gain
    ramp = np.linspace(level.gain, target, length)

    return fx.limit(body * ramp), overhang, Level(assumed_peak=assumed, gain=target)


def finish(carry: np.ndarray | None, level: Level | None,
           master_gain: float = 0.89) -> np.ndarray:
    """Turn the last overhang into playable audio.

    The carry is deliberately held *before* the level stage so it is not gained
    a second time when it is summed into the next span. That makes it unsafe to
    play as-is — it has no gain and no limiter — so anything finishing a stream
    has to come through here rather than appending the raw array.
    """
    if carry is None or not len(carry):
        return np.zeros(0, dtype=np.float64)
    trimmed = np.trim_zeros(carry, "b")
    if not len(trimmed):
        return trimmed
    return fx.limit(trimmed * (level.gain if level else master_gain))


def _voices_in_span(bus, composition, span, seconds_per_beat, total, trims=None) -> None:
    specs = {v.voice_id: v for v in composition.voices if v.role != ROLE_PERC}
    grouped: dict[str, list] = {}
    for note in composition.notes:
        if span.start_beat <= note.start < span.end_beat:
            grouped.setdefault(note.voice, []).append(note)

    for voice_id, notes in grouped.items():
        spec = specs.get(voice_id)
        if spec is None:
            continue
        scratch = np.zeros(total, dtype=np.float64)
        cache: dict[tuple, np.ndarray] = {}
        for order, note in enumerate(notes):
            count = _note_samples(spec, note.duration, seconds_per_beat)
            variant = order % VARIANTS
            key = (round(note.pitch, 2), count, variant)
            sample = cache.get(key)
            if sample is None:
                sample = engines.render_note(spec, midi_to_hz(note.pitch), count,
                                             SAMPLE_RATE, composition.seed, variant)
                cache[key] = sample
            _add_at(scratch, sample * note.velocity,
                    _offset(note.start, span.start_beat, seconds_per_beat))
        if spec.effects:
            scratch = effects.apply_chain(scratch, SAMPLE_RATE, spec.effects)
        bus += scratch * spec.gain * (trims or {}).get(voice_id, 1.0)


def _ghosts_in_span(composition, span, seconds_per_beat, total) -> np.ndarray:
    """The lines it decided against, for one span, on their own bus.

    Kept identical to the offline path deliberately — a phone that hears a
    different set of ghosts from the desktop is the drift this project keeps
    testing against.
    """
    specs = {v.voice_id: v for v in composition.voices}
    bus = np.zeros(total, dtype=np.float64)

    grouped: dict[str, list] = {}
    for note in composition.ghosts:
        if span.start_beat <= note.start < span.end_beat:
            grouped.setdefault(note.voice, []).append(note)

    for voice_id, notes in grouped.items():
        spec = specs.get(voice_id.replace(GHOST_MARK, ""))
        if spec is None:
            continue
        cache: dict[tuple, np.ndarray] = {}
        for order, note in enumerate(notes):
            count = _note_samples(spec, note.duration, seconds_per_beat)
            variant = order % VARIANTS
            key = (round(note.pitch, 2), count, variant)
            sample = cache.get(key)
            if sample is None:
                sample = engines.render_note(spec, midi_to_hz(note.pitch), count,
                                             SAMPLE_RATE, composition.seed, variant)
                cache[key] = sample
            _add_at(bus, sample * note.velocity,
                    _offset(note.start, span.start_beat, seconds_per_beat))

    return filters.lowpass(bus, SAMPLE_RATE, GHOST_CUTOFF, order=2)


def _strokes_in_span(bus, composition, span, seconds_per_beat, total, trims=None) -> None:
    kit = {v.voice_id: v for v in composition.voices if v.role == ROLE_PERC}
    trim = kit_trim(len(kit))
    cache: dict[tuple[str, int], np.ndarray] = {}
    for order, stroke in enumerate(composition.strokes):
        if not (span.start_beat <= stroke.start < span.end_beat):
            continue
        spec = kit.get(stroke.voice)
        if spec is None:
            continue
        variant = order % VARIANTS
        key = (stroke.voice, variant)
        sample = cache.get(key)
        if sample is None:
            sample = drums.render_drum(spec, SAMPLE_RATE, composition.seed, variant)
            cache[key] = sample
        _add_at(bus, sample * stroke.velocity * spec.gain * trim
                * (trims or {}).get(stroke.voice, 1.0),
                _offset(stroke.start, span.start_beat, seconds_per_beat))


def _duck(total, composition, span, seconds_per_beat) -> np.ndarray:
    """Sidechain, including kicks just before the span whose duck reaches into it."""
    duck = np.ones(total, dtype=np.float64)
    window = int(0.17 * SAMPLE_RATE)
    shape = 1.0 - DUCK_DEPTH * np.exp(-np.arange(window) / (DUCK_RELEASE * SAMPLE_RATE))
    reach = window / SAMPLE_RATE / seconds_per_beat

    for stroke in composition.strokes:
        if stroke.voice not in {"kick", "boom"}:
            continue
        if not (span.start_beat - reach <= stroke.start < span.end_beat):
            continue
        start = _offset(stroke.start, span.start_beat, seconds_per_beat)
        if start >= total:
            continue
        if start < 0:  # a kick from the previous span, still ducking
            trimmed = shape[-start:]
            end = min(total, len(trimmed))
            duck[:end] = np.minimum(duck[:end], trimmed[:end])
            continue
        end = min(total, start + window)
        duck[start:end] = np.minimum(duck[start:end], shape[: end - start])
    return duck


def _offset(beat: float, span_start: float, seconds_per_beat: float) -> int:
    return int(round((beat - span_start) * seconds_per_beat * SAMPLE_RATE))


def _add_at(bus: np.ndarray, sample: np.ndarray, offset: int) -> None:
    if offset >= len(bus) or len(sample) == 0:
        return
    if offset < 0:
        sample = sample[-offset:]
        offset = 0
        if len(sample) == 0:
            return
    end = min(len(bus), offset + len(sample))
    bus[offset:end] += sample[: end - offset]
