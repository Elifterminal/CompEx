"""Every synthesis engine, gathered from the family modules.

An engine is a family, not a patch — the composer invents the parameters, so
"pluck" covers everything from a harp to a snapped wire. They all share one
call signature so the arranger can treat them interchangeably.
"""

from __future__ import annotations

import numpy as np

from compex.dsp import fx
from compex.dsp.engines import core, struck, sustained, textural, voiced

ENGINES = {
    **core.ENGINES,
    **struck.ENGINES,
    **voiced.ENGINES,
    **sustained.ENGINES,
    **textural.ENGINES,
}
ENGINE_NAMES: tuple[str, ...] = tuple(sorted(ENGINES))

#: Every note leaves an engine at this peak, so engines are comparable.
NOTE_PEAK = 0.9


def render_note(spec, freq: float, count: int, sample_rate: int,
                seed: int, index: int) -> np.ndarray:
    """Render one note. ``count`` is the length in samples, including release tail."""
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    if freq <= 0:
        raise ValueError(f"note frequency must be positive, got {freq}")

    engine = ENGINES.get(spec.engine)
    if engine is None:
        raise ValueError(f"unknown engine {spec.engine!r} for voice {spec.voice_id!r}")

    voice = np.nan_to_num(engine(spec, freq, count, sample_rate, seed, index),
                          nan=0.0, posinf=0.0, neginf=0.0)
    if len(voice) < count:
        voice = np.pad(voice, (0, count - len(voice)))
    voice = voice[:count]

    # Engines differ in natural level by more than 50x — a self-oscillating
    # resonance is tiny, a sine is not. Normalising here is what lets the
    # composer's per-voice gain mean the same thing whatever engine it picked.
    peak = float(np.max(np.abs(voice))) if len(voice) else 0.0
    if peak > 1e-9:
        voice = voice * (NOTE_PEAK / peak)

    return fx.fade_edges(voice, sample_rate, 0.003)


__all__ = ["ENGINES", "ENGINE_NAMES", "render_note"]
