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

#: Every note leaves an engine at this *loudness*, so engines are comparable.
#: Peak-matching was the old rule and it was quietly wrong: it made a plucked
#: note and a sustained pad nominally equal and audibly nothing of the sort.
NOTE_LOUDNESS = 0.42
#: The height a transient still gets some credit for — see ``fx.CREST_TILT``.
NOTE_PEAK = 0.9
NOTE_CEILING = 0.97


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
    voice = fx.loudness_normalise(voice, sample_rate, NOTE_LOUDNESS,
                                  peak=NOTE_PEAK, ceiling=NOTE_CEILING)

    return fx.fade_edges(voice, sample_rate, 0.003)


__all__ = ["ENGINES", "ENGINE_NAMES", "render_note"]
