"""Per-voice effects.

These widen the palette far more cheaply than new engines do: the same pluck
through a long reverb and through a ring modulator are two different
instruments. The composer invents the chain per voice.

Feedback structures (comb, allpass, delay) are computed a delay-line at a
time. Each block depends only on the previous block, so they run in numpy
rather than a per-sample Python loop.
"""

from __future__ import annotations

import numpy as np

from compex.dsp import filters

#: Names the composer can draw from.
EFFECT_NAMES: tuple[str, ...] = (
    "chorus", "delay", "reverb", "tremolo", "ringmod", "wavefold",
)

MAX_FEEDBACK = 0.85


def _feedback_comb(signal: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    out = signal.astype(np.float64).copy()
    step = int(delay_samples)
    if step < 1 or step >= len(out):
        return out
    gain = min(gain, MAX_FEEDBACK)
    for start in range(step, len(out), step):
        end = min(start + step, len(out))
        out[start:end] += gain * out[start - step : start - step + (end - start)]
    return out


def _allpass(signal: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    step = int(delay_samples)
    out = -gain * signal.astype(np.float64)
    if step < 1 or step >= len(signal):
        return out
    for start in range(step, len(signal), step):
        end = min(start + step, len(signal))
        width = end - start
        out[start:end] += (signal[start - step : start - step + width]
                           + gain * out[start - step : start - step + width])
    return out


def _match_level(wet: np.ndarray, dry: np.ndarray) -> np.ndarray:
    """Keep a feedback structure from running away with the level."""
    wet_peak = float(np.max(np.abs(wet))) if len(wet) else 0.0
    dry_peak = float(np.max(np.abs(dry))) if len(dry) else 0.0
    if wet_peak < 1e-9 or dry_peak < 1e-9:
        return wet
    return wet * min(1.0, dry_peak / wet_peak)


def chorus(signal: np.ndarray, sample_rate: int, rate: float = 0.6,
           depth_ms: float = 6.0, mix: float = 0.4) -> np.ndarray:
    """A modulated delay read at a fractional offset — thickens anything."""
    count = len(signal)
    if count < 64:
        return signal
    t = np.arange(count, dtype=np.float64) / sample_rate
    base = 0.012 * sample_rate
    swing = (max(depth_ms, 0.1) / 1000.0) * sample_rate * np.sin(2 * np.pi * rate * t)
    read = np.clip(np.arange(count) - (base + swing), 0, count - 1)
    delayed = np.interp(read, np.arange(count), signal)
    return signal * (1.0 - mix) + delayed * mix


def delay(signal: np.ndarray, sample_rate: int, time_s: float = 0.25,
          feedback: float = 0.35, mix: float = 0.3) -> np.ndarray:
    """Repeating echoes."""
    step = int(max(time_s, 0.01) * sample_rate)
    if step >= len(signal):
        return signal
    wet = _match_level(_feedback_comb(signal, step, feedback), signal)
    return signal * (1.0 - mix) + wet * mix


def reverb(signal: np.ndarray, sample_rate: int, size: float = 0.6,
           damp: float = 0.4, mix: float = 0.3) -> np.ndarray:
    """Schroeder: four parallel combs into two allpasses, then damped."""
    if len(signal) < 1024:
        return signal
    scale = 0.5 + max(0.0, min(1.0, size))
    wet = np.zeros(len(signal), dtype=np.float64)
    for length in (0.0297, 0.0371, 0.0411, 0.0437):
        wet += _feedback_comb(signal, int(length * sample_rate * scale), 0.2 + 0.65 * size)
    wet /= 4.0

    for length in (0.0050, 0.0017):
        wet = _allpass(wet, int(length * sample_rate), 0.7)

    wet = filters.lowpass(wet, sample_rate, 1800.0 + (1.0 - damp) * 9000.0, order=2)
    return signal * (1.0 - mix) + _match_level(wet, signal) * mix


def tremolo(signal: np.ndarray, sample_rate: int, rate: float = 4.0,
            depth: float = 0.5) -> np.ndarray:
    t = np.arange(len(signal), dtype=np.float64) / sample_rate
    shape = 1.0 - max(0.0, min(1.0, depth)) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t))
    return signal * shape


def ringmod(signal: np.ndarray, sample_rate: int, freq: float = 220.0,
            mix: float = 0.5) -> np.ndarray:
    """Multiply by a sine. Everything comes out inharmonic and slightly wrong."""
    t = np.arange(len(signal), dtype=np.float64) / sample_rate
    return signal * (1.0 - mix) + signal * np.sin(2 * np.pi * max(freq, 1.0) * t) * mix


def wavefold(signal: np.ndarray, sample_rate: int, amount: float = 2.0,
             mix: float = 0.6) -> np.ndarray:
    """Fold the waveform back on itself — adds harmonics without simple clipping."""
    folded = np.sin(signal * max(amount, 0.1) * np.pi * 0.5)
    return signal * (1.0 - mix) + folded * mix


_EFFECTS = {
    "chorus": chorus,
    "delay": delay,
    "reverb": reverb,
    "tremolo": tremolo,
    "ringmod": ringmod,
    "wavefold": wavefold,
}


def apply_chain(signal: np.ndarray, sample_rate: int,
                chain: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]) -> np.ndarray:
    """Run ``signal`` through an invented chain of ``(name, params)`` stages."""
    out = signal
    for name, params in chain:
        effect = _EFFECTS.get(name)
        if effect is None:
            raise ValueError(f"unknown effect {name!r}")
        out = effect(out, sample_rate, **{key: float(value) for key, value in params})
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
