"""Things you hit. Short attacks, partials that decay at their own rates."""

from __future__ import annotations

import numpy as np

from compex import rng
from compex.dsp import filters
from compex.dsp.engines.shared import NYQUIST_MARGIN, partial_count, seconds


def bell(spec, freq, count, sample_rate, seed, index):
    """Inharmonic modal synthesis — partials stretched off the harmonic series."""
    t = seconds(count, sample_rate)
    total = partial_count(freq, sample_rate, spec.get("partials", 5))
    stretch = spec.get("inharmonicity", 0.08)
    decay = spec.get("decay", 2.0)

    voice = np.zeros(count, dtype=np.float64)
    for partial in range(1, total + 1):
        ratio = partial * np.sqrt(1.0 + stretch * partial * partial)
        if freq * ratio > sample_rate * 0.5 * NYQUIST_MARGIN:
            break
        wobble = rng.between(seed, f"bell-{index}", partial, 0.55, 1.45)
        voice += (np.sin(2 * np.pi * freq * ratio * t)
                  * np.exp(-t / (decay / (partial ** 0.7) * wobble)) / partial)

    strike = np.exp(-t / max(spec.get("strike", 0.008), 1e-4))
    return voice + rng.noise(seed, f"bell-hit-{index}", count) * strike * 0.06


def mallet(spec, freq, count, sample_rate, seed, index):
    """Marimba-ish: a strong fundamental with the overtone tuned near 4:1, and wood."""
    t = seconds(count, sample_rate)
    decay = spec.get("decay", 0.7)
    tuning = spec.get("overtone", 4.0)

    voice = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    voice += 0.42 * np.sin(2 * np.pi * freq * tuning * t) * np.exp(-t / (decay * 0.32))
    voice += 0.16 * np.sin(2 * np.pi * freq * tuning * 2.6 * t) * np.exp(-t / (decay * 0.16))

    wood = rng.noise(seed, f"mallet-{index}", count) * np.exp(-t / 0.006)
    hardness = spec.get("hardness", 0.5)
    return voice + filters.bandpass(wood, sample_rate, 900.0 + hardness * 2600.0, q=1.1) * 0.35


def tine(spec, freq, count, sample_rate, seed, index):
    """Rhodes-like: a sine body with a bright tine ping on top that dies fast."""
    t = seconds(count, sample_rate)
    decay = spec.get("decay", 1.8)
    ping_ratio = spec.get("ping", 7.0)

    body = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    body += 0.22 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-t / (decay * 0.6))
    ping = np.sin(2 * np.pi * freq * ping_ratio * t) * np.exp(-t / max(decay * 0.06, 0.01))

    bark = spec.get("bark", 0.4)
    return body + ping * bark


def glass(spec, freq, count, sample_rate, seed, index):
    """High, thin, slightly detuned partials with a long shimmer."""
    t = seconds(count, sample_rate)
    total = partial_count(freq, sample_rate, spec.get("partials", 6))
    decay = spec.get("decay", 3.0)
    shimmer = spec.get("shimmer", 3.0)

    voice = np.zeros(count, dtype=np.float64)
    for partial in range(1, total + 1):
        ratio = partial * (1.0 + 0.004 * partial * spec.get("spread", 1.0))
        vibrato = 1.0 + 0.0016 * np.sin(2 * np.pi * (shimmer + partial * 0.7) * t)
        voice += (np.sin(2 * np.pi * freq * ratio * vibrato * t)
                  * np.exp(-t / (decay / partial ** 0.5)) / (partial ** 1.3))

    attack = np.clip(t / 0.012, 0.0, 1.0)
    return voice * attack


def chime(spec, freq, count, sample_rate, seed, index):
    """Tubular-bell ratios — the partials sit near 2 : 3 : 4.2 : 5.4."""
    t = seconds(count, sample_rate)
    decay = spec.get("decay", 3.5)
    ratios = (1.0, 2.0, 3.01, 4.17, 5.43, 6.79)

    voice = np.zeros(count, dtype=np.float64)
    for position, ratio in enumerate(ratios, start=1):
        if freq * ratio > sample_rate * 0.5 * NYQUIST_MARGIN:
            break
        jitter = rng.between(seed, f"chime-{index}", position, 0.995, 1.005)
        voice += (np.sin(2 * np.pi * freq * ratio * jitter * t)
                  * np.exp(-t / (decay / position ** 0.45)) / position)

    strike = rng.noise(seed, f"chime-hit-{index}", count) * np.exp(-t / 0.003)
    return voice + filters.highpass(strike, sample_rate, 3000.0) * spec.get("strike", 0.1)


ENGINES = {"bell": bell, "mallet": mallet, "tine": tine, "glass": glass, "chime": chime}
