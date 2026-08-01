"""The strange end of the palette: grains, morphs, resonance, and 8-bit."""

from __future__ import annotations

import numpy as np

from compex import rng
from compex.dsp import filters, fx
from compex.dsp.engines.shared import (
    ad_envelope,
    ar_envelope,
    band_limited_pulse,
    band_limited_saw,
    seconds,
)


def granular(spec, freq, count, sample_rate, seed, index):
    """Chop a tone into windowed grains and scatter them. Shimmering, unstable."""
    grain_s = max(spec.get("grain", 0.06), 0.008)
    grain_len = max(32, int(grain_s * sample_rate))
    hop = max(16, int(grain_len * max(spec.get("overlap", 0.5), 0.1)))

    source_len = max(grain_len * 4, int(0.5 * sample_rate))
    source = band_limited_saw(freq, source_len, sample_rate, 16)
    source += rng.noise(seed, f"grain-src-{index}", source_len) * spec.get("dust", 0.1)

    window = np.hanning(grain_len)
    scatter = spec.get("scatter", 0.5)
    out = np.zeros(count + grain_len, dtype=np.float64)

    for position, start in enumerate(range(0, count, hop)):
        jump = int(rng.uniform(seed, f"grain-{index}", position) * scatter
                   * (source_len - grain_len))
        grain = source[jump : jump + grain_len]
        if len(grain) < grain_len:
            break
        level = rng.between(seed, f"grain-lvl-{index}", position, 0.5, 1.0)
        out[start : start + grain_len] += grain * window * level

    return out[:count] * ad_envelope(count, sample_rate, spec.get("attack", 0.05),
                                     spec.get("decay", 1.4)) * 0.5


def wavetable(spec, freq, count, sample_rate, seed, index):
    """Morph between sine, saw and pulse across the note."""
    saw = band_limited_saw(freq, count, sample_rate, 18)
    pulse = band_limited_pulse(freq, count, sample_rate, spec.get("width", 0.35), 18)
    sine = np.sin(2 * np.pi * freq * seconds(count, sample_rate))

    sweep = np.clip(np.linspace(spec.get("start", 0.0), spec.get("end", 1.0), count), 0.0, 1.0)
    # 0 -> sine, 0.5 -> saw, 1 -> pulse
    first = np.clip(sweep * 2.0, 0.0, 1.0)
    second = np.clip(sweep * 2.0 - 1.0, 0.0, 1.0)
    voice = sine * (1 - first) + saw * first * (1 - second) + pulse * second

    return voice * 0.4 * ad_envelope(count, sample_rate, spec.get("attack", 0.01),
                                     spec.get("decay", 1.3))


def feedback(spec, freq, count, sample_rate, seed, index):
    """A resonance driven to the edge of self-oscillation. It howls at the pitch."""
    excite = rng.noise(seed, f"fb-{index}", count)
    envelope_in = np.exp(-seconds(count, sample_rate) / max(spec.get("burst", 0.05), 0.005))
    rung = filters.bandpass(excite * envelope_in, sample_rate, freq,
                            q=max(spec.get("resonance", 40.0), 4.0))

    partial = filters.bandpass(excite * envelope_in, sample_rate, freq * 2.0,
                               q=max(spec.get("resonance", 40.0) * 0.6, 3.0))
    voice = rung + partial * spec.get("overtone", 0.3)

    voice = fx.drive(voice, 1.0 + spec.get("scream", 2.0))
    return voice * ar_envelope(count, sample_rate, 0.01, spec.get("release", 0.3))


def chip(spec, freq, count, sample_rate, seed, index):
    """Square waves and a fast arpeggio. Cheap in the way that is the whole point."""
    steps = (0, 4, 7, 12)
    step_len = max(64, int(max(spec.get("arp", 0.04), 0.008) * sample_rate))
    duty = spec.get("width", 0.5)

    out = np.zeros(count, dtype=np.float64)
    for position, start in enumerate(range(0, count, step_len)):
        end = min(start + step_len, count)
        semitone = steps[position % len(steps)] if spec.get("arpeggio", 1.0) > 0.5 else 0
        pitch = freq * (2.0 ** (semitone / 12.0))
        out[start:end] = band_limited_pulse(pitch, end - start, sample_rate, duty, 12)

    out = fx.bitcrush(out, max(3, int(spec.get("bits", 6))))
    return out * 0.35 * ad_envelope(count, sample_rate, 0.002, spec.get("decay", 0.9))


ENGINES = {
    "granular": granular, "wavetable": wavetable, "feedback": feedback, "chip": chip,
}
