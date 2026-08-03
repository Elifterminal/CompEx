"""The one call everything goes through: three inputs in, a finished track out.

Compose, render, and hand back the audio together with the formula describing
what the machine decided. Both the CLI and the UI use this, so there is a
single definition of what a track is.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compex.audio import encode, write_wav
from compex.config import SAMPLE_RATE, Knobs
from compex.dsp.arrange import ProgressFn, render_composition, survey
from compex.dsp.balance import Mix
from compex.generate import Composition, Mood, compose
from compex.remember import Memory, learn
from compex.notation import emit


@dataclass(frozen=True)
class RenderResult:
    samples: np.ndarray
    composition: Composition
    formula: str
    knobs: Knobs
    mood: Mood
    mix: Mix = Mix()
    #: What the engine now remembers, this piece included. Callers that keep a
    #: history save this back; callers that do not simply drop it, and the next
    #: piece starts from nothing exactly as it always did.
    memory: Memory = Memory()

    @property
    def duration_s(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @property
    def fingerprint(self) -> str:
        """Short digest of the audio — two identical tracks share a fingerprint."""
        return hashlib.blake2b(self.samples.tobytes(), digest_size=8).hexdigest()

    def save(self, path: str | Path, audio_format: str = "wav") -> Path:
        """Write the track. ``audio_format`` is 'wav' or 'mp3'."""
        fmt = str(audio_format).lower().strip()
        if fmt not in encode.FORMATS:
            raise ValueError(f"unknown format {audio_format!r}; use one of {encode.FORMATS}")

        wav_path = write_wav(Path(path).with_suffix(".wav"), self.samples)
        if fmt == "wav":
            return wav_path

        mp3_path = encode.to_mp3(wav_path)
        wav_path.unlink(missing_ok=True)  # the WAV was only scaffolding for the encoder
        return mp3_path

    def save_formula(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().with_suffix(".tex")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.formula, encoding="utf-8")
        return destination


def make_track(knobs: Knobs, mood: Mood, progress: ProgressFn = None,
               memory: Memory = Memory()) -> RenderResult:
    """Invent a piece for these settings and render it."""
    composition = compose(knobs.seed, knobs.duration_s, mood, memory)
    samples = render_composition(composition, knobs.master_gain, progress=progress,
                                 ghost_gain=knobs.ghost_gain)
    return RenderResult(
        samples=samples,
        composition=composition,
        formula=emit(composition, knobs.duration_s),
        knobs=knobs,
        mood=mood,
        mix=survey(composition),
        memory=learn(memory, composition),
    )
