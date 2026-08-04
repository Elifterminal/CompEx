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
from compex.dsp import fx
from compex.dsp.arrange import (
    ProgressFn,
    render_composition,
    render_stems,
    survey,
)
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

    def save_stems(self, directory: str | Path, audio_format: str = "wav") -> list[Path]:
        """Write every voice as its own file, at the level it has in the mix.

        Rendered rather than stored, because keeping a stem per voice in memory
        for a ten-minute piece is a gigabyte of float64 nobody asked for. The
        sum of the stems is the master to within the mastering stage — they are
        the buses that actually went in, not the voices re-recorded alone.
        """
        fmt = str(audio_format).lower().strip()
        if fmt not in encode.FORMATS:
            raise ValueError(f"unknown format {audio_format!r}; use one of {encode.FORMATS}")

        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        stems = render_stems(self.composition, self.knobs.master_gain,
                             self.knobs.ghost_gain)

        written: list[Path] = []
        for name, samples in sorted(stems.items()):
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            wav_path = write_wav(target / f"{safe}.wav", fx.peak_normalise(samples, 0.97)
                                 if name != "master" else samples)
            if fmt == "mp3":
                written.append(encode.to_mp3(wav_path))
                wav_path.unlink(missing_ok=True)
            else:
                written.append(wav_path)
        return written

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
