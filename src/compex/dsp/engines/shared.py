"""Helpers every engine family uses.

Frequencies are integrated with ``cumsum`` wherever they move, so a pitch or
rate change mid-note never puts a step in the phase.

Partial sums go through :func:`phase_stack`, which broadcasts over partials
instead of looping in Python. When every partial of a voice shares the same
frequency modulation — vibrato, detune, drift — the fundamental's phase can be
accumulated once and simply multiplied, which is what makes that possible.
"""

from __future__ import annotations

import numpy as np

MAX_PARTIALS = 48
NYQUIST_MARGIN = 0.92
MAX_CELLS = 3_000_000  # partials x samples held at once, to bound memory


def seconds(count: int, sample_rate: int) -> np.ndarray:
    return np.arange(count, dtype=np.float64) / sample_rate


def ad_envelope(count: int, sample_rate: int, attack: float, decay: float) -> np.ndarray:
    """Attack ramp into exponential decay — the shape most engines want."""
    t = seconds(count, sample_rate)
    rise = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    return rise * np.exp(-np.maximum(t - attack, 0.0) / max(decay, 1e-3))


def ar_envelope(count: int, sample_rate: int, attack: float, release: float) -> np.ndarray:
    """Attack, hold, release — for anything that sustains rather than decays."""
    if count <= 1:
        return np.ones(count, dtype=np.float64)
    t = seconds(count, sample_rate)
    rise = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    fall = np.clip((t[-1] - t) / max(release, 1e-4), 0.0, 1.0)
    return rise * fall


def partial_count(freq: float, sample_rate: int, wanted: int) -> int:
    """How many partials fit under Nyquist before they start folding back."""
    ceiling = int((sample_rate * 0.5 * NYQUIST_MARGIN) / max(freq, 1e-6))
    return max(1, min(int(wanted), ceiling, MAX_PARTIALS))


def base_phase(freq: float, count: int, sample_rate: int,
               modulation: np.ndarray | None = None) -> np.ndarray:
    """Accumulated phase of the fundamental, optionally frequency-modulated."""
    if modulation is None:
        return 2 * np.pi * freq * seconds(count, sample_rate)
    return 2 * np.pi * np.cumsum(freq * modulation) / sample_rate


def phase_stack(phase: np.ndarray, ratios, amplitudes) -> np.ndarray:
    """Sum ``amp_i * sin(ratio_i * phase)`` over partials, in blocks."""
    ratio_array = np.asarray(ratios, dtype=np.float64)
    amp_array = np.asarray(amplitudes, dtype=np.float64)
    if ratio_array.size == 0:
        return np.zeros(len(phase), dtype=np.float64)

    out = np.zeros(len(phase), dtype=np.float64)
    block = max(1, int(MAX_CELLS // max(len(phase), 1)))
    for start in range(0, ratio_array.size, block):
        chunk_ratios = ratio_array[start : start + block][:, None]
        chunk_amps = amp_array[start : start + block][:, None]
        out += (np.sin(chunk_ratios * phase[None, :]) * chunk_amps).sum(axis=0)
    return out


def band_limited_saw(freq: float, count: int, sample_rate: int,
                     harmonics: int = 16) -> np.ndarray:
    """An additive sawtooth — no aliasing, unlike a naive ramp."""
    total = partial_count(freq, sample_rate, harmonics)
    index = np.arange(1, total + 1, dtype=np.float64)
    return phase_stack(base_phase(freq, count, sample_rate), index, 1.0 / index)


def band_limited_pulse(freq: float, count: int, sample_rate: int, width: float = 0.5,
                       harmonics: int = 16) -> np.ndarray:
    """A pulse wave built from harmonics; ``width`` sets the duty cycle."""
    total = partial_count(freq, sample_rate, harmonics)
    duty = min(max(width, 0.05), 0.95)
    index = np.arange(1, total + 1, dtype=np.float64)
    amplitudes = np.sin(np.pi * index * duty) / index
    # Cosine phase, so shift by a quarter turn and reuse the same machinery.
    phase = base_phase(freq, count, sample_rate) + np.pi / 2
    return phase_stack(phase, index, amplitudes) * 2.0
