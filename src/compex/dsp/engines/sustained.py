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


def bowed(spec, freq, count, sample_rate, seed, index):
    """One player, not a section: a slow bow, a scrape at the start, real vibrato.

    ``string`` is an ensemble and always sounds like one. A single bowed voice
    that can be heard as a person is a different instrument, and the piece had
    no way to ask for one.
    """
    t = seconds(count, sample_rate)
    depth = spec.get("vibrato", 0.006)
    rate = spec.get("rate", 5.2)
    # Vibrato arrives after the note does, the way a player adds it.
    onset = np.clip((t - spec.get("attack", 0.25)) / 0.6, 0.0, 1.0)
    wobble = 1.0 + depth * onset * np.sin(2 * np.pi * rate * t)

    phase = base_phase(freq, count, sample_rate, wobble)
    total = partial_count(freq, sample_rate, int(spec.get("partials", 11)))
    numbers = np.arange(1, total + 1, dtype=np.float64)
    # A bowed spectrum is odd-leaning and falls off gently.
    amplitudes = (1.0 / numbers ** spec.get("tilt", 1.25)) * np.where(numbers % 2, 1.0, 0.55)
    voice = phase_stack(phase, numbers, amplitudes) / numbers.sum() ** 0.5

    scrape = filters.bandpass(rng.noise(seed, f"bowed-{index}", count),
                              sample_rate, freq * 3.0, q=0.7)
    bite = np.exp(-t / max(spec.get("bite", 0.09), 1e-3)) * spec.get("grip", 0.25)
    return (voice + scrape * bite) * ar_envelope(
        count, sample_rate, spec.get("attack", 0.25), spec.get("release", 0.5))


def shimmer(spec, freq, count, sample_rate, seed, index):
    """Inharmonic partials that hold and beat against each other.

    A bell that never stops. The beating is what makes it move: partials a few
    cents apart drift in and out of phase for as long as the note lasts, so a
    held chord keeps changing without anything being modulated.
    """
    t = seconds(count, sample_rate)
    total = partial_count(freq, sample_rate, int(spec.get("partials", 8)))
    stretch = spec.get("stretch", 1.06)

    voice = np.zeros(count, dtype=np.float64)
    for partial in range(1, total + 1):
        ratio = partial ** stretch
        beat = rng.between(seed, f"shim-{index}", partial, 0.15, 0.9)
        pair = 1.0 + spec.get("spread", 0.004) * rng.between(
            seed, f"shim-sp-{index}", partial, -1.0, 1.0)
        swell = 0.72 + 0.28 * np.sin(2 * np.pi * beat * t
                                     + rng.between(seed, f"shim-ph-{index}", partial, 0.0, 6.28))
        voice += (np.sin(2 * np.pi * freq * ratio * t)
                  + np.sin(2 * np.pi * freq * ratio * pair * t)) * swell / (partial ** 1.15)

    voice = filters.lowpass(voice, sample_rate, max(spec.get("cutoff", 4200.0), 400.0), order=2)
    return voice * ar_envelope(count, sample_rate, spec.get("attack", 0.5),
                               spec.get("release", 1.4)) * 0.35


def tape(spec, freq, count, sample_rate, seed, index):
    """A held tone with flutter and saturation — warm, slightly unwell.

    The pitch instability is the character. A machine that never wavers is the
    tell everyone hears; this wavers on two timescales, slow wow and fast
    flutter, and neither is periodic with the other.
    """
    t = seconds(count, sample_rate)
    wow = spec.get("wow", 0.004) * np.sin(2 * np.pi * spec.get("wow_rate", 0.7) * t
                                          + rng.between(seed, f"tape-{index}", 0, 0.0, 6.28))
    flutter = spec.get("flutter", 0.0015) * np.sin(2 * np.pi * spec.get("flutter_rate", 7.3) * t)
    phase = base_phase(freq, count, sample_rate, 1.0 + wow + flutter)

    total = partial_count(freq, sample_rate, int(spec.get("partials", 7)))
    numbers = np.arange(1, total + 1, dtype=np.float64)
    voice = phase_stack(phase, numbers, 1.0 / numbers ** 1.4) / total ** 0.5

    voice = np.tanh(voice * (1.0 + spec.get("saturation", 1.6)))
    hiss = rng.noise(seed, f"tape-hiss-{index}", count) * spec.get("hiss", 0.012)
    voice = filters.lowpass(voice + hiss, sample_rate,
                            max(spec.get("cutoff", 5200.0), 600.0), order=2)
    return voice * ar_envelope(count, sample_rate, spec.get("attack", 0.09),
                               spec.get("release", 0.35)) * 0.55


def vox(spec, freq, count, sample_rate, seed, index):
    """A held vowel that changes vowel while it is held.

    ``choir`` sings one vowel per note. This travels between two, slowly, which
    is the thing that keeps a sustained voice interesting for four bars instead
    of one.
    """
    t = seconds(count, sample_rate)
    travel = np.clip(t / max(t[-1] if count > 1 else 1.0, 1e-6), 0.0, 1.0)
    first = spec.get("f1", 620.0) + (spec.get("f1b", 380.0) - spec.get("f1", 620.0)) * travel
    second = spec.get("f2", 1200.0) + (spec.get("f2b", 2200.0) - spec.get("f2", 1200.0)) * travel

    total = partial_count(freq, sample_rate, int(spec.get("partials", 22)))
    numbers = np.arange(1, total + 1, dtype=np.float64)
    detune = 1.0 + spec.get("spread", 0.003) * rng.between(seed, f"vox-{index}", 0, -1.0, 1.0)
    phase = base_phase(freq * detune, count, sample_rate)
    raw = phase_stack(phase, numbers, 1.0 / numbers) / total ** 0.5

    # Two moving formants, summed. Filtering per sample would be honest and
    # ruinous; filtering twice at the mean and crossfading is neither audible
    # as a cheat nor slow.
    low = filters.bandpass(raw, sample_rate, float(first.mean()), q=spec.get("q", 7.0))
    high = filters.bandpass(raw, sample_rate, float(second.mean()), q=spec.get("q", 7.0))
    voice = low * (1.0 - 0.35 * travel) + high * (0.45 + 0.35 * travel)

    breath = filters.bandpass(rng.noise(seed, f"vox-air-{index}", count),
                              sample_rate, 2600.0, q=0.8) * spec.get("air", 0.05)
    return (voice + breath) * ar_envelope(count, sample_rate, spec.get("attack", 0.35),
                                          spec.get("release", 0.7)) * 0.8


def aeolian(spec, freq, count, sample_rate, seed, index):
    """Tuned noise: wind with a pitch in it.

    The existing ``noise`` engine is a wash with no note in it, which is why it
    reads as breath rather than as harmony when a piece hands it a chord. This
    is the same idea done as an instrument — narrow resonant bands sitting on
    the harmonics, so it is unmistakably noise and unmistakably in tune.
    """
    source = rng.noise(seed, f"aeol-{index}", count)
    total = max(1, partial_count(freq, sample_rate, int(spec.get("partials", 4))))
    resonance = spec.get("q", 26.0)

    voice = np.zeros(count, dtype=np.float64)
    for partial in range(1, total + 1):
        centre = freq * partial
        if centre > sample_rate * 0.45:
            break
        band = filters.bandpass(source, sample_rate, centre, q=resonance)
        gust = 0.6 + 0.4 * np.sin(2 * np.pi * rng.between(seed, f"aeol-g-{index}", partial,
                                                          0.08, 0.4) * seconds(count, sample_rate))
        voice += band * gust / (partial ** 0.8)

    return voice * ar_envelope(count, sample_rate, spec.get("attack", 0.7),
                               spec.get("release", 1.1)) * 1.6


ENGINES = {
    "pad": pad, "string": string, "organ": organ, "supersaw": supersaw, "drone": drone,
    "bowed": bowed, "shimmer": shimmer, "tape": tape, "vox": vox, "aeolian": aeolian,
}
