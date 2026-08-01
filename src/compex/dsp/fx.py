"""The destructive end of the bass chain: drive, bitcrush, resampling, limiting.

These are the literal terms from the BASS directive — metallic growl,
formant distortion, resampling, bitcrush, subwoofer pressure — implemented
one function per term so the chain in the equation maps onto the code you
can point at.
"""

from __future__ import annotations

import numpy as np


def drive(signal: np.ndarray, amount: float) -> np.ndarray:
    """Soft saturation. ``amount`` is pre-gain into tanh; 1.0 is nearly clean."""
    if amount <= 0:
        raise ValueError(f"drive amount must be positive, got {amount}")
    return np.tanh(signal * amount) / np.tanh(amount)


def bitcrush(signal: np.ndarray, bits: int) -> np.ndarray:
    """Quantise amplitude to ``bits`` of resolution."""
    if not 1 <= bits <= 24:
        raise ValueError(f"bitcrush bits must be 1..24, got {bits}")
    levels = float(2 ** (bits - 1))
    return np.round(signal * levels) / levels


def resample_crush(signal: np.ndarray, hold: int) -> np.ndarray:
    """Sample-rate reduction — hold each sample for ``hold`` frames.

    The aliasing this throws off is the point, not a defect: it is the
    'resampling' term in the chain.
    """
    if hold < 1:
        raise ValueError(f"resample hold must be >= 1, got {hold}")
    if hold == 1 or len(signal) == 0:
        return signal
    kept = signal[::hold]
    return np.repeat(kept, hold)[: len(signal)]


def peak_normalise(signal: np.ndarray, peak: float = 0.97) -> np.ndarray:
    """Scale so the loudest sample sits at ``peak``. Silence passes through."""
    if not 0.0 < peak <= 1.0:
        raise ValueError(f"peak must be in (0, 1], got {peak}")
    highest = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if highest < 1e-12:
        return signal
    return signal * (peak / highest)


def limit(signal: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Final safety net — soft-knee above the ceiling, hard clip at 1.0."""
    if not 0.0 < ceiling <= 1.0:
        raise ValueError(f"ceiling must be in (0, 1], got {ceiling}")
    over = np.abs(signal) > ceiling
    out = signal.copy()
    out[over] = np.sign(signal[over]) * (
        ceiling + (1.0 - ceiling) * np.tanh((np.abs(signal[over]) - ceiling) / (1.0 - ceiling))
    )
    return np.clip(out, -1.0, 1.0)


def fade_edges(signal: np.ndarray, sample_rate: int, seconds: float = 0.004) -> np.ndarray:
    """Short fades so a buffer never starts or ends on a discontinuity."""
    count = len(signal)
    ramp_len = min(int(seconds * sample_rate), count // 2)
    if ramp_len <= 0:
        return signal
    ramp = np.linspace(0.0, 1.0, ramp_len)
    out = signal.copy()
    out[:ramp_len] *= ramp
    out[-ramp_len:] *= ramp[::-1]
    return out
