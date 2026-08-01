"""Frequency-domain filters.

Everything is done with numpy's FFT rather than a time-domain biquad. Two
reasons: no scipy dependency, and a pure-Python IIR loop over a few million
samples is painfully slow. Offline rendering does not care about the
zero-phase behaviour this gives us.

Time-varying filtering (the riser sweep) goes through :func:`sweep_lowpass`,
which windows the signal into overlapping blocks and filters each one at its
own cutoff.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _frequencies(count: int, sample_rate: int) -> np.ndarray:
    return np.fft.rfftfreq(count, d=1.0 / sample_rate)


def _apply(signal: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    return np.fft.irfft(spectrum * magnitude, n=len(signal))


def lowpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Butterworth magnitude response, applied in the frequency domain."""
    if cutoff_hz <= 0:
        raise ValueError(f"lowpass cutoff must be positive, got {cutoff_hz}")
    if len(signal) == 0:
        return signal
    freqs = _frequencies(len(signal), sample_rate)
    magnitude = 1.0 / np.sqrt(1.0 + (freqs / cutoff_hz) ** (2 * order))
    return _apply(signal, magnitude)


def highpass(signal: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz <= 0:
        raise ValueError(f"highpass cutoff must be positive, got {cutoff_hz}")
    if len(signal) == 0:
        return signal
    freqs = _frequencies(len(signal), sample_rate)
    ratio = cutoff_hz / np.maximum(freqs, _EPS)
    return _apply(signal, 1.0 / np.sqrt(1.0 + ratio ** (2 * order)))


def _resonator_magnitude(freqs: np.ndarray, centre_hz: float, q: float) -> np.ndarray:
    """Standard analogue bandpass magnitude: 1 / sqrt(1 + Q^2 (f/f0 - f0/f)^2)."""
    safe = np.maximum(freqs, _EPS)
    detune = safe / centre_hz - centre_hz / safe
    return 1.0 / np.sqrt(1.0 + (q * detune) ** 2)


def bandpass(signal: np.ndarray, sample_rate: int, centre_hz: float, q: float = 4.0) -> np.ndarray:
    if len(signal) == 0:
        return signal
    freqs = _frequencies(len(signal), sample_rate)
    return _apply(signal, _resonator_magnitude(freqs, centre_hz, q))


def formants(
    signal: np.ndarray,
    sample_rate: int,
    peaks: tuple[tuple[float, float, float], ...],
    dry: float = 0.35,
) -> np.ndarray:
    """Stack resonant peaks in a single FFT pass.

    ``peaks`` is a tuple of ``(centre_hz, q, gain)``. This is what gives the
    growl its vowel-like, metallic character — the equation calls it formant
    distortion.
    """
    if len(signal) == 0 or not peaks:
        return signal
    freqs = _frequencies(len(signal), sample_rate)
    magnitude = np.full_like(freqs, dry, dtype=np.float64)
    for centre_hz, q, gain in peaks:
        if centre_hz <= 0 or q <= 0:
            raise ValueError(f"invalid formant peak: centre={centre_hz}, q={q}")
        magnitude += gain * _resonator_magnitude(freqs, centre_hz, q)
    return _apply(signal, magnitude)


def sweep_lowpass(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_curve: np.ndarray,
    order: int = 3,
    block: int = 2048,
) -> np.ndarray:
    """Time-varying lowpass via overlapping Hann blocks (50% hop sums to unity)."""
    count = len(signal)
    if count == 0:
        return signal
    if len(cutoff_curve) != count:
        raise ValueError(f"cutoff curve length {len(cutoff_curve)} != signal length {count}")

    block = min(block, max(64, count))
    hop = block // 2
    window = np.hanning(block + 1)[:block]
    out = np.zeros(count + block, dtype=np.float64)
    padded = np.concatenate([signal, np.zeros(block, dtype=np.float64)])

    for start in range(0, count, hop):
        centre = min(count - 1, start + hop)
        cutoff = float(max(20.0, cutoff_curve[centre]))
        chunk = padded[start : start + block] * window
        out[start : start + block] += lowpass(chunk, sample_rate, cutoff, order=order)

    return out[:count]
