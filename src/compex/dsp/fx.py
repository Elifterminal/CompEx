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


LOUDNESS_WINDOW = 0.25   # seconds — long enough to average a note, short enough to catch a hit


def short_term_rms(signal: np.ndarray, sample_rate: int,
                   window: float = LOUDNESS_WINDOW) -> float:
    """RMS of the loudest quarter-second.

    Peak says how *tall* a sound is; this says how loud it is. The difference
    is not academic — a slow pad and a plucked note normalised to the same
    peak arrive nowhere near each other, which is most of why the pads in this
    engine were inaudible under everything else.
    """
    if len(signal) == 0:
        return 0.0
    span = min(len(signal), max(1, int(window * sample_rate)))
    if len(signal) <= span:
        return float(np.sqrt(np.mean(signal ** 2)))
    energy = np.concatenate(([0.0], np.cumsum(signal ** 2)))
    windows = energy[span:] - energy[:-span]
    return float(np.sqrt(max(0.0, windows.max()) / span))


#: How much of the old peak-matching survives. 0 is pure loudness, 1 is what
#: the engine used to do.
CREST_TILT = 0.35


def loudness_normalise(signal: np.ndarray, sample_rate: int, target: float,
                       peak: float = 0.9, ceiling: float = 0.99,
                       tilt: float = CREST_TILT) -> np.ndarray:
    """Level a sound by loudness, with some credit still given to its peak.

    Pure peak matching was the old rule and it buried every pad: a slow swell
    and a plucked note reach the same height and nothing like the same volume.
    Pure loudness matching overcorrects the other way — a transient is *heard*
    as louder than its RMS says, so matching RMS makes plucks and hits vanish
    instead. Neither extreme is right, so this takes a weighted geometric mean
    of the two scale factors and then refuses to clip.
    """
    if len(signal) == 0 or target <= 0:
        return signal
    level = short_term_rms(signal, sample_rate)
    height = float(np.max(np.abs(signal)))
    if level <= 1e-9 or height <= 1e-9:
        return signal

    scale = (target / level) ** (1.0 - tilt) * (peak / height) ** tilt
    scaled = signal * scale
    loudest = float(np.max(np.abs(scaled)))
    if loudest > ceiling:
        scaled = scaled * (ceiling / loudest)
    return scaled


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
