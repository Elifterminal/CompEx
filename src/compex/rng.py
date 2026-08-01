"""Deterministic, seekable pseudo-randomness.

Every value is a pure function of ``(seed, stream, index)``. There is no
sequential state, so asking for bar 90 before bar 3 gives the same answer as
asking in order, seeking is free, and turning a knob mid-run never desyncs
anything downstream.

blake2b rather than the builtin ``hash()`` because ``hash()`` is salted per
process — the same seed would give different music on every launch, which
would quietly destroy the one property the engine is built around.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence, TypeVar

import numpy as np

T = TypeVar("T")

_DIGEST_BYTES = 8
_UINT64_SCALE = float(1 << 64)


def _digest(seed: int, stream: str, index: int) -> int:
    payload = struct.pack("<qq", int(seed), int(index)) + stream.encode("utf-8")
    raw = hashlib.blake2b(payload, digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(raw, "little")


def uniform(seed: int, stream: str, index: int) -> float:
    """A float in ``[0, 1)`` for this exact position in this exact stream."""
    return _digest(seed, stream, index) / _UINT64_SCALE


def between(seed: int, stream: str, index: int, low: float, high: float) -> float:
    if high < low:
        raise ValueError(f"between(): high {high} < low {low} on stream {stream!r}")
    return low + (high - low) * uniform(seed, stream, index)


def pick(seed: int, stream: str, index: int, options: Sequence[T]) -> T:
    """Choose one option. Uniform, and stable for a given position."""
    if len(options) == 0:
        raise ValueError(f"pick(): empty option set on stream {stream!r}")
    return options[_digest(seed, stream, index) % len(options)]


def noise(seed: int, stream: str, count: int) -> np.ndarray:
    """A reproducible white-noise buffer.

    numpy's PCG64 is deterministic across platforms and versions, so seeding
    it from our own digest keeps noise bit-identical between runs while
    staying fast enough for full-length renders.
    """
    if count < 0:
        raise ValueError(f"noise(): negative count {count}")
    generator = np.random.default_rng(_digest(seed, stream, 0))
    return generator.standard_normal(count).astype(np.float64)
