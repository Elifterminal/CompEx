"""The foundational engines: weight, harmonics, modulation, strings, air."""

from __future__ import annotations

import numpy as np

from compex import rng
from compex.dsp import filters, fx
from compex.dsp.engines.shared import (
    ad_envelope,
    base_phase,
    partial_count,
    phase_stack,
    seconds,
)


def sub(spec, freq, count, sample_rate, seed, index):
    """A low sine with weight. Deliberately plain — it is felt, not heard."""
    t = seconds(count, sample_rate)
    envelope = ad_envelope(count, sample_rate, 0.006, spec.get("decay", 1.2))
    voice = np.sin(2 * np.pi * freq * t) * envelope
    click = spec.get("click", 0.0)
    if click > 0:
        voice += rng.noise(seed, f"sub-click-{index}", count) * np.exp(-t / 0.004) * click * 0.4
    return fx.drive(voice, 1.0 + spec.get("drive", 1.5))


def additive(spec, freq, count, sample_rate, seed, index):
    """A harmonic stack. Odd-only partials give it a hollow, clarinet-ish body."""
    total = partial_count(freq, sample_rate, spec.get("partials", 6))
    rolloff = spec.get("rolloff", 1.4)
    detune = spec.get("detune", 0.004)

    numbers = np.arange(1, total + 1, dtype=np.float64)
    if spec.get("odd_only", 0.0) > 0.5:
        numbers = numbers[numbers % 2 == 1]

    drift = np.array([1.0 + detune * rng.between(seed, f"add-{index}", int(n), -1.0, 1.0)
                      for n in numbers])
    voice = phase_stack(base_phase(freq, count, sample_rate),
                        numbers * drift, 1.0 / numbers ** rolloff)

    return voice * ad_envelope(count, sample_rate, spec.get("attack", 0.01),
                               spec.get("decay", 1.2))


def pm(spec, freq, count, sample_rate, seed, index):
    """Phase modulation with a swept index — metallic, growling, bell-like by turns."""
    t = seconds(count, sample_rate)
    low, high = spec.get("index_low", 0.5), spec.get("index_high", 5.0)

    wobble = 0.5 + 0.5 * np.sin(2 * np.pi * spec.get("lfo", 1.0) * t)
    depth = rng.between(seed, "pm-depth", index, 0.35, 1.0)
    envelope = ad_envelope(count, sample_rate, 0.004, spec.get("decay", 1.0))
    modulation = low + (high - low) * (depth * wobble + (1 - depth) * envelope)

    modulator = np.sin(2 * np.pi * freq * spec.get("ratio", 1.0) * t)
    return np.sin(2 * np.pi * freq * t + modulation * modulator) * envelope


def pluck(spec, freq, count, sample_rate, seed, index):
    """Karplus-Strong, computed a delay-line at a time so long notes stay fast.

    Each block of L samples depends only on the previous block, so the whole
    string runs in numpy instead of a per-sample Python loop.
    """
    length = max(2, int(round(sample_rate / freq)))
    excitation = rng.noise(seed, f"pluck-{index}", length)
    excitation = excitation / (float(np.max(np.abs(excitation))) or 1.0)

    brightness = spec.get("brightness", 0.7)
    if brightness < 0.99:
        width = max(1, int((1.0 - brightness) * 9))
        excitation = np.convolve(excitation, np.ones(width) / width, mode="same")

    notch = max(1, int(length * spec.get("pick", 0.2)))
    excitation[:notch] *= np.linspace(0.2, 1.0, notch)

    # Map the 0.28..0.5 parameter onto a usable feedback range; 0.5 is lossless.
    damp = 0.470 + 0.0295 * min(1.0, max(0.0, spec.get("damping", 0.45) * 2.0))

    blocks = int(np.ceil(count / length))
    out = np.empty(blocks * length, dtype=np.float64)
    current = excitation
    for block in range(blocks):
        out[block * length : (block + 1) * length] = current
        current = (current + np.roll(current, 1)) * damp

    return out[:count] * np.exp(-seconds(count, sample_rate)
                                / max(spec.get("decay", 1.5), 0.05))


def noise(spec, freq, count, sample_rate, seed, index):
    """Filtered noise that sweeps — wind, wash, breath, depending on the numbers."""
    source = rng.noise(seed, f"noise-{index}", count)
    centre = max(spec.get("centre", 1200.0), 60.0)
    sweep = spec.get("sweep", 1.0)

    if abs(sweep - 1.0) > 0.05 and count > 64:
        curve = np.geomspace(centre, max(centre * sweep, 80.0), count)
        shaped = filters.sweep_lowpass(source, sample_rate, curve)
    else:
        shaped = filters.bandpass(source, sample_rate, centre, q=spec.get("q", 2.0))

    return shaped * ad_envelope(count, sample_rate, spec.get("attack", 0.1),
                                spec.get("decay", 1.0))


ENGINES = {"sub": sub, "additive": additive, "pm": pm, "pluck": pluck, "noise": noise}
