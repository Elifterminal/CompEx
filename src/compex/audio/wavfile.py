"""16-bit PCM WAV output using the standard library only.

Deliberately no soundfile/scipy dependency: the render has to survive being
packaged and moved to another machine, and ``wave`` ships with Python.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from compex.config import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH_BYTES

_INT16_PEAK = 32767


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write mono float samples in ``[-1, 1]`` to ``path``. Returns the path written."""
    if samples.ndim != 1:
        raise ValueError(f"expected mono (1-D) samples, got shape {samples.shape}")
    if len(samples) == 0:
        raise ValueError("refusing to write an empty WAV file")
    if not np.all(np.isfinite(samples)):
        raise ValueError("sample buffer contains NaN or inf — the render is broken")

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * _INT16_PEAK).astype("<i2")

    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    return destination
