"""Things you blow through, and things that sound like a throat."""

from __future__ import annotations

import numpy as np

from compex import rng
from compex.dsp import filters
from compex.dsp.engines.shared import (
    ad_envelope,
    ar_envelope,
    band_limited_pulse,
    base_phase,
    partial_count,
    phase_stack,
    seconds,
)


def formant(spec, freq, count, sample_rate, seed, index):
    """A buzzy source through three vowel resonances — the closest thing to a voice."""
    t = seconds(count, sample_rate)
    buzz = spec.get("buzz", 0.6)
    total = partial_count(freq, sample_rate, 6 + buzz * 26)

    numbers = np.arange(1, total + 1, dtype=np.float64)
    source = phase_stack(base_phase(freq, count, sample_rate),
                         numbers, 1.0 / numbers ** (1.6 - buzz))

    quality = spec.get("q", 9.0)
    peaks = (
        (max(spec.get("f1", 600.0), 90.0), quality, 1.0),
        (max(spec.get("f2", 1200.0), 200.0), quality * 1.15, 0.62),
        (max(spec.get("f3", 2500.0), 400.0), quality * 1.3, 0.32),
    )
    shaped = filters.formants(source, sample_rate, peaks, dry=0.12)
    return shaped * ad_envelope(count, sample_rate, 0.02, spec.get("decay", 1.1))


def reed(spec, freq, count, sample_rate, seed, index):
    """Pulse wave with a moving duty cycle and breath — clarinet through oboe."""
    t = seconds(count, sample_rate)
    width = spec.get("width", 0.35)
    wobble = width + 0.12 * np.sin(2 * np.pi * spec.get("pwm", 1.4) * t)

    # Duty modulation needs the pulse rebuilt in a few chunks; four is enough to hear.
    voice = np.zeros(count, dtype=np.float64)
    chunks = 4
    edges = np.linspace(0, count, chunks + 1).astype(int)
    for chunk in range(chunks):
        start, end = edges[chunk], edges[chunk + 1]
        if end <= start:
            continue
        duty = float(np.clip(wobble[start:end].mean(), 0.08, 0.92))
        voice[start:end] = band_limited_pulse(freq, end - start, sample_rate, duty)[: end - start]

    breath = filters.highpass(rng.noise(seed, f"reed-{index}", count), sample_rate, 2200.0)
    envelope = ar_envelope(count, sample_rate, spec.get("attack", 0.04),
                           spec.get("release", 0.12))
    return (voice * 0.5 + breath * spec.get("breath", 0.12)) * envelope


def brass(spec, freq, count, sample_rate, seed, index):
    """Brightness that grows with the envelope — the partials arrive as it swells."""
    t = seconds(count, sample_rate)
    total = partial_count(freq, sample_rate, spec.get("partials", 16))
    envelope = ar_envelope(count, sample_rate, spec.get("attack", 0.06),
                           spec.get("release", 0.15))
    bite = spec.get("bite", 1.0)

    # As the envelope opens, the per-partial rolloff flattens and the tone brightens.
    rolloff = 2.6 - 1.9 * np.clip(envelope * bite, 0.0, 1.0)
    voice = np.zeros(count, dtype=np.float64)
    for partial in range(1, total + 1):
        voice += np.sin(2 * np.pi * freq * partial * t) * np.exp(-rolloff * (partial - 1) * 0.35)

    rasp = spec.get("rasp", 0.0)
    if rasp > 0.01:
        voice += rng.noise(seed, f"brass-{index}", count) * rasp * 0.12 * envelope
    return voice * envelope * 0.4


def choir(spec, freq, count, sample_rate, seed, index):
    """Several detuned formant voices with independent vibrato. Never quite in tune."""
    t = seconds(count, sample_rate)
    singers = max(2, int(spec.get("singers", 4)))
    total = partial_count(freq, sample_rate, 10)

    numbers = np.arange(1, total + 1, dtype=np.float64)
    amplitudes = 1.0 / numbers ** 1.7

    source = np.zeros(count, dtype=np.float64)
    for singer in range(singers):
        detune = 1.0 + spec.get("spread", 0.006) * rng.between(
            seed, f"choir-{index}", singer, -1.0, 1.0
        )
        rate = rng.between(seed, f"choir-vib-{index}", singer, 4.2, 6.4)
        vibrato = 1.0 + 0.0035 * np.sin(2 * np.pi * rate * t + singer)
        # Every partial of one singer shares that singer's vibrato, so the
        # fundamental's phase can be accumulated once and scaled.
        source += phase_stack(base_phase(freq * detune, count, sample_rate, vibrato),
                              numbers, amplitudes)
    source /= singers

    quality = spec.get("q", 8.0)
    peaks = (
        (max(spec.get("f1", 500.0), 90.0), quality, 1.0),
        (max(spec.get("f2", 1500.0), 200.0), quality, 0.55),
        (2800.0, quality * 1.2, 0.22),
    )
    shaped = filters.formants(source, sample_rate, peaks, dry=0.2)
    breath = filters.bandpass(rng.noise(seed, f"choir-air-{index}", count),
                              sample_rate, 3200.0, q=0.7)
    envelope = ar_envelope(count, sample_rate, spec.get("attack", 0.25),
                           spec.get("release", 0.5))
    return (shaped + breath * spec.get("air", 0.08)) * envelope


def flute(spec, freq, count, sample_rate, seed, index):
    """Nearly a sine, carried by breath. The noise is most of what you recognise."""
    t = seconds(count, sample_rate)
    vibrato = 1.0 + spec.get("vibrato", 0.004) * np.sin(2 * np.pi * spec.get("rate", 5.0) * t)

    voice = np.sin(2 * np.pi * freq * vibrato * t)
    voice += 0.12 * np.sin(2 * np.pi * freq * 2 * vibrato * t)
    voice += 0.05 * np.sin(2 * np.pi * freq * 3 * vibrato * t)

    air = filters.bandpass(rng.noise(seed, f"flute-{index}", count),
                           sample_rate, freq * 2.4, q=0.8)
    envelope = ar_envelope(count, sample_rate, spec.get("attack", 0.08),
                           spec.get("release", 0.18))
    return (voice + air * spec.get("breath", 0.3)) * envelope


ENGINES = {
    "formant": formant, "reed": reed, "brass": brass, "choir": choir, "flute": flute,
}
