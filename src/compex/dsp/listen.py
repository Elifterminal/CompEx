"""Hearing the piece instead of reading it.

Everything the critic has ever judged, it judged from symbols: intervals,
durations, counts of notes. That is a real limitation and it has been written
on the public page as one. Two notes a semitone apart are one number to a
symbolic critic whether they are played by two flutes or by two sawtooth pads
with eleven partials each, and those are not remotely the same event — the
first is a mild dissonance and the second is a beating mess.

So this renders a short window of what was actually written, at a low sample
rate, and measures it the way a listener's ear would have to:

* **roughness** — sensory dissonance, from partials that land close enough
  together to beat against each other. This is the Plomp–Levelt curve: two
  tones are at their roughest about a quarter of a critical band apart, and
  perfectly smooth when they are far enough apart to be heard separately. It
  is a property of the *sound*, not of the interval, which is exactly why the
  symbolic version could never see it.
* **brightness** — where the energy sits, as a spectral centroid.
* **motion** — how much the spectrum changes from one frame to the next. A
  piece whose timbre never moves is a piece nobody has to keep listening to.

The probe is deliberately small: a few seconds, mono, at a quarter of the
sample rate. This runs *during composition*, so its cost is paid by every
render on every device including a phone, and a perfect measurement nobody can
afford is worth less than a rough one taken every movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from compex.dsp import engines

if TYPE_CHECKING:  # pragma: no cover — types only
    from compex.generate.material import Note
    from compex.generate.palette import VoiceSpec

# This module is imported by the critic, which is imported by the composer, so
# it must not import the composer back. It takes notes and voice specs as plain
# objects and never asks the generate package for anything.

PROBE_RATE = 11_025     # a quarter of the real rate: enough for anything below 5 kHz
PROBE_BEATS = 6.0       # how much of a movement to listen to
PROBE_VOICES = 6        # and how many voices, busiest first
#: 4096 at 11 kHz is 2.7 Hz per bin. It has to be this fine: two tones a
#: semitone apart at 220 Hz are 13 Hz apart, and at a coarser resolution the
#: leakage guard below throws the second one away as an artefact of the first —
#: which silently reported every dissonance in the low register as smooth.
FRAME = 4096
PEAKS = 20              # partials per frame considered for roughness
#: A Hann window puts one local maximum on a pure tone, so leakage is already
#: filtered by looking for maxima at all; this is only a guard against the
#: shoulder of a very loud partial. Two bins, not three — a quarter tone at
#: 220 Hz is six hertz wide, and three bins threw it away as an artefact.
MIN_PEAK_GAP = 2
BRIGHT_LOW, BRIGHT_HIGH = 100.0, 5000.0   # the range brightness is measured across


@dataclass(frozen=True)
class Heard:
    """What a short listen to the piece actually found."""

    roughness: float = 0.35    # 0..1, sensory dissonance from beating partials
    brightness: float = 0.35   # 0..1, where the energy sits
    motion: float = 0.30       # 0..1, how much the spectrum moves
    frames: int = 0

    def describe(self) -> str:
        return (f"roughness {self.roughness:.2f} · brightness {self.brightness:.2f} · "
                f"motion {self.motion:.2f}")


def probe(notes: Sequence["Note"], specs: dict[str, "VoiceSpec"], seed: int,
          seconds_per_beat: float, start_beat: float,
          beats: float = PROBE_BEATS, skip: frozenset[str] = frozenset()) -> Heard:
    """Render a window of the written music and listen to it.

    ``skip`` is how the caller keeps percussion out, and it should. Drums are
    broadband by construction and would dominate every measure here — roughness
    would read as "there is a snare", which is true and useless. What this is
    for is the pitched material, where the difference between a chord and a
    mess lives.
    """
    window = [note for note in notes
              if start_beat <= note.start < start_beat + beats
              and note.voice not in skip
              and specs.get(note.voice) is not None]
    if not window:
        return Heard()

    busiest = _busiest(window)
    window = [note for note in window if note.voice in busiest]

    length = max(FRAME * 2, int(beats * seconds_per_beat * PROBE_RATE))
    bus = np.zeros(length, dtype=np.float64)
    cache: dict[tuple, np.ndarray] = {}

    for order, note in enumerate(window):
        spec = specs[note.voice]
        count = max(64, int(note.duration * seconds_per_beat * PROBE_RATE))
        key = (note.voice, round(note.pitch, 1), count)
        sample = cache.get(key)
        if sample is None:
            sample = engines.render_note(spec, _hz(note.pitch), count, PROBE_RATE, seed, 0)
            cache[key] = sample
        offset = int((note.start - start_beat) * seconds_per_beat * PROBE_RATE)
        end = min(length, offset + len(sample))
        if end > offset:
            bus[offset:end] += sample[: end - offset] * note.velocity

    return listen(bus)


def listen(signal: np.ndarray) -> Heard:
    """Measure roughness, brightness and motion in a rendered buffer."""
    if len(signal) < FRAME * 2:
        return Heard()

    hop = FRAME // 2
    window = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / PROBE_RATE)

    roughness: list[float] = []
    brightness: list[float] = []
    flux: list[float] = []
    previous: np.ndarray | None = None

    for start in range(0, len(signal) - FRAME, hop):
        spectrum = np.abs(np.fft.rfft(signal[start:start + FRAME] * window))
        total = spectrum.sum()
        if total < 1e-9:
            previous = spectrum
            continue

        brightness.append(float((spectrum * freqs).sum() / total))
        roughness.append(_roughness(spectrum, freqs))
        if previous is not None:
            change = np.abs(spectrum - previous).sum() / max(total, 1e-9)
            flux.append(float(min(2.0, change)))
        previous = spectrum

    if not roughness:
        return Heard()

    return Heard(
        roughness=float(min(1.0, np.mean(roughness))),
        brightness=_brightness(float(np.mean(brightness))),
        motion=float(min(1.0, np.mean(flux) if flux else 0.0)),
        frames=len(roughness),
    )


def _brightness(centroid: float) -> float:
    """Spectral centroid on a log scale between 100 Hz and 5 kHz.

    Log because pitch is: the distance from 100 to 200 Hz is the same musical
    distance as 2 kHz to 4 kHz, and a linear map put every piece this engine
    makes into the top fifth of the range.
    """
    if centroid <= BRIGHT_LOW:
        return 0.0
    span = math.log(BRIGHT_HIGH) - math.log(BRIGHT_LOW)
    return float(max(0.0, min(1.0, (math.log(centroid) - math.log(BRIGHT_LOW)) / span)))


def _roughness(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """Sensory dissonance of one frame, from its loudest partials.

    Plomp and Levelt measured this directly: two pure tones are smooth at a
    unison, roughest around a quarter of a critical band apart, and smooth
    again once they are far enough apart to be resolved separately. Summing
    that over every pair of loud partials is what makes a just-tuned chord and
    an equal-tempered one measurably different — which no count of semitones
    can be.
    """
    if len(spectrum) < 8:
        return 0.0
    # Local maxima, not the loudest bins. A single sine leaks across several
    # neighbouring bins, and taking the top twenty of those reads as twenty
    # partials a few hertz apart — which is maximum roughness, from one pure
    # tone. Calibration caught it: a unison was scoring as rough as a semitone.
    rising = spectrum[1:-1] > spectrum[:-2]
    falling = spectrum[1:-1] >= spectrum[2:]
    where = np.flatnonzero(rising & falling) + 1
    if len(where) == 0:
        return 0.0

    order = where[np.argsort(spectrum[where])][::-1][:PEAKS]
    keep: list[int] = []
    for index in order:
        if all(abs(index - already) >= MIN_PEAK_GAP for already in keep):
            keep.append(int(index))
    if len(keep) < 2:
        return 0.0

    amps = spectrum[keep]
    tones = freqs[keep]
    scale = amps.sum()
    if scale < 1e-9:
        return 0.0

    total = 0.0
    for index in range(len(tones)):
        for other in range(index + 1, len(tones)):
            low, high = sorted((tones[index], tones[other]))
            if low < 20.0:
                continue
            # Critical bandwidth, roughly, in the range that matters here.
            band = 1.72 * (low ** 0.65)
            distance = (high - low) / max(band, 1e-6)
            if distance > 1.2:
                continue
            # Peaks at a quarter of a critical band, zero at unison and beyond.
            curve = math.exp(-3.5 * distance) - math.exp(-5.75 * distance)
            total += curve * amps[index] * amps[other]

    return float(min(1.0, 6.0 * total / (scale ** 2)))


def _busiest(notes: Sequence[Any]) -> set[str]:
    counts: dict[str, int] = {}
    for note in notes:
        counts[note.voice] = counts.get(note.voice, 0) + 1
    ranked = sorted(counts, key=lambda voice: -counts[voice])
    return set(ranked[:PROBE_VOICES])


def _hz(pitch: float) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))
