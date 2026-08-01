"""Things that hold. Slow in, slow out, spectra that move while they sit there."""

from __future__ import annotations

import numpy as np

from compex import rng
from compex.dsp import filters
from compex.dsp.engines.shared import (
    ar_envelope,
    band_limited_saw,
    base_phase,
    partial_count,
    phase_stack,
    seconds,
)


def pad(spec, freq, count, sample_rate, seed, index):
    """Detuned band-limited saws under a lowpass."""
    layers = max(1, int(spec.get("layers", 3)))
    detune = spec.get("detune", 0.008)

    voice = np.zeros(count, dtype=np.float64)
    for layer in range(layers):
        offset = 1.0 + detune * rng.between(seed, f"pad-{index}", layer, -1.0, 1.0)
        voice += band_limited_saw(freq * offset, count, sample_rate, 14)
    voice /= layers * 2.0

    voice = filters.lowpass(voice, sample_rate, max(spec.get("cutoff", 900.0), 120.0), order=3)
    return voice * ar_envelope(count, sample_rate, spec.get("attack", 0.4),
                               spec.get("release", 0.8))


def string(spec, freq, count, sample_rate, seed, index):
    """Bowed ensemble: slow swell, per-player vibrato, a little bow noise."""
    t = seconds(count, sample_rate)
    players = max(2, int(spec.get("players", 4)))

    numbers = np.arange(1, partial_count(freq, sample_rate, 12) + 1, dtype=np.float64)
    amplitudes = 1.0 / numbers ** spec.get("rolloff", 1.5)

    voice = np.zeros(count, dtype=np.float64)
    for player in range(players):
        detune = 1.0 + spec.get("spread", 0.005) * rng.between(
            seed, f"string-{index}", player, -1.0, 1.0
        )
        rate = rng.between(seed, f"string-vib-{index}", player, 4.5, 6.8)
        vibrato = 1.0 + spec.get("vibrato", 0.004) * np.sin(2 * np.pi * rate * t + player * 1.7)
        voice += phase_stack(base_phase(freq * detune, count, sample_rate, vibrato),
                             numbers, amplitudes)
    voice /= players

    bow = filters.bandpass(rng.noise(seed, f"bow-{index}", count), sample_rate,
                           freq * 3.5, q=1.4)
    envelope = ar_envelope(count, sample_rate, spec.get("attack", 0.3),
                           spec.get("release", 0.45))
    return (voice * 0.4 + bow * spec.get("bow", 0.05)) * envelope


def organ(spec, freq, count, sample_rate, seed, index):
    """Drawbars at octave and fifth ratios, a key click, and a rotating tremolo."""
    t = seconds(count, sample_rate)
    ratios = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    draw = spec.get("drawbars", 0.6)

    voice = np.zeros(count, dtype=np.float64)
    for position, ratio in enumerate(ratios):
        if freq * ratio > sample_rate * 0.45:
            break
        level = rng.between(seed, f"organ-{index}", position, 0.0, 1.0) ** (2.5 - draw * 2.0)
        voice += np.sin(2 * np.pi * freq * ratio * t) * level

    click = rng.noise(seed, f"organ-click-{index}", count) * np.exp(-t / 0.004)
    leslie = 1.0 - spec.get("leslie", 0.2) * (0.5 + 0.5 * np.sin(
        2 * np.pi * spec.get("leslie_rate", 5.5) * t))

    envelope = ar_envelope(count, sample_rate, spec.get("attack", 0.006),
                           spec.get("release", 0.05))
    return (voice * 0.28 + click * 0.25) * leslie * envelope


def supersaw(spec, freq, count, sample_rate, seed, index):
    """A wall of detuned saws. Bright, wide, and completely unsubtle."""
    voices = max(3, int(spec.get("voices", 7)))
    detune = spec.get("detune", 0.012)

    voice = np.zeros(count, dtype=np.float64)
    for slot in range(voices):
        spread = (slot / (voices - 1)) * 2.0 - 1.0
        offset = 1.0 + detune * spread * rng.between(seed, f"saw-{index}", slot, 0.6, 1.4)
        voice += band_limited_saw(freq * offset, count, sample_rate, 12)
    voice /= voices

    cutoff = max(spec.get("cutoff", 4000.0), 200.0)
    voice = filters.lowpass(voice, sample_rate, cutoff, order=2)
    return voice * ar_envelope(count, sample_rate, spec.get("attack", 0.02),
                               spec.get("release", 0.2)) * 0.6


def drone(spec, freq, count, sample_rate, seed, index):
    """A held spectrum that drifts. Nothing arrives; it is already happening."""
    t = seconds(count, sample_rate)
    total = partial_count(freq, sample_rate, spec.get("partials", 9))
    drift = spec.get("drift", 0.08)

    voice = np.zeros(count, dtype=np.float64)
    for partial in range(1, total + 1):
        rate = rng.between(seed, f"drone-{index}", partial, 0.03, 0.28)
        phase = rng.between(seed, f"drone-ph-{index}", partial, 0.0, 6.28)
        breathe = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t + phase)
        wander = 1.0 + drift * 0.01 * np.sin(2 * np.pi * rate * 0.37 * t)
        voice += np.sin(2 * np.pi * freq * partial * wander * t) * breathe / (partial ** 1.4)

    return voice * ar_envelope(count, sample_rate, spec.get("attack", 1.2),
                               spec.get("release", 1.2)) * 0.5


ENGINES = {
    "pad": pad, "string": string, "organ": organ, "supersaw": supersaw, "drone": drone,
}
