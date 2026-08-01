"""Percussion voices, all parameterised by the composer rather than fixed.

The same ``kick`` function covers a soft heartbeat and a distorted stomp
depending on the tone, decay and drive the composer invented for it.
"""

from __future__ import annotations

import numpy as np

from compex import rng
from compex.dsp import filters, fx
from compex.generate.palette import VoiceSpec

MAX_SECONDS = 1.4

#: Every one-shot leaves at this peak, so drums are comparable to each other.
HIT_PEAK = 0.95


def render_drum(spec: VoiceSpec, sample_rate: int, seed: int, index: int) -> np.ndarray:
    """Render one percussion one-shot."""
    voice = _DRUMS.get(spec.voice_id)
    if voice is None:
        raise ValueError(f"unknown drum {spec.voice_id!r}")

    # Same reasoning as the pitched engines: normalise so the composer's gain
    # means one thing whichever drum it picked.
    sample = np.nan_to_num(voice(spec, sample_rate, seed, index),
                           nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(sample))) if len(sample) else 0.0
    if peak > 1e-9:
        sample = sample * (HIT_PEAK / peak)
    return fx.fade_edges(sample, sample_rate, 0.002)


def _time(count: int, sample_rate: int) -> np.ndarray:
    return np.arange(count, dtype=np.float64) / sample_rate


def _span(sample_rate: int, seconds: float) -> int:
    return max(64, int(min(seconds, MAX_SECONDS) * sample_rate))


def _kick(spec, sample_rate, seed, index):
    decay = spec.get("decay", 0.14)
    tone = spec.get("tone", 0.6)
    count = _span(sample_rate, decay * 4 + 0.15)
    t = _time(count, sample_rate)

    start_hz = 90.0 + tone * 110.0
    end_hz = 34.0 + tone * 18.0
    pitch = end_hz + (start_hz - end_hz) * np.exp(-t / 0.028)
    body = np.sin(2 * np.pi * np.cumsum(pitch) / sample_rate) * np.exp(-t / decay)

    click = rng.noise(seed, f"kick-{index}", count) * np.exp(-t / 0.0035)
    voice = body + filters.highpass(click, sample_rate, 850.0) * spec.get("noisiness", 0.4) * 0.5
    return fx.drive(voice, spec.get("drive", 1.8))


def _snare(spec, sample_rate, seed, index):
    decay = spec.get("decay", 0.16)
    tone = spec.get("tone", 0.6)
    count = _span(sample_rate, decay * 4 + 0.12)
    t = _time(count, sample_rate)

    source = rng.noise(seed, f"snare-{index}", count)
    shell = filters.bandpass(source, sample_rate, 1400.0 + tone * 1800.0, q=0.85)
    crack = filters.highpass(source, sample_rate, 4000.0) * np.exp(-t / 0.026)
    body = np.sin(2 * np.pi * (150.0 + tone * 90.0) * t) * np.exp(-t / 0.05)

    noisiness = spec.get("noisiness", 0.6)
    voice = shell * np.exp(-t / decay) * (0.5 + noisiness) + crack * 0.5 + body * (1.1 - noisiness)
    return fx.drive(voice, spec.get("drive", 1.5))


def _hat(spec, sample_rate, seed, index):
    decay = max(0.012, spec.get("decay", 0.08) * 0.45)
    tone = spec.get("tone", 0.6)
    count = _span(sample_rate, decay * 5 + 0.05)
    t = _time(count, sample_rate)

    source = rng.noise(seed, f"hat-{index}", count)
    voice = filters.highpass(source, sample_rate, 5200.0 + tone * 3500.0) * np.exp(-t / decay)
    return fx.drive(voice, 1.0 + spec.get("drive", 1.4) * 0.35)


def _tom(spec, sample_rate, seed, index):
    decay = spec.get("decay", 0.22)
    tone = spec.get("tone", 0.5)
    count = _span(sample_rate, decay * 4 + 0.15)
    t = _time(count, sample_rate)

    start_hz = 150.0 + tone * 220.0
    pitch = start_hz * (0.62 + 0.38 * np.exp(-t / 0.09))
    body = np.sin(2 * np.pi * np.cumsum(pitch) / sample_rate) * np.exp(-t / decay)
    skin = rng.noise(seed, f"tom-{index}", count) * np.exp(-t / 0.012)
    return fx.drive(body + filters.bandpass(skin, sample_rate, 900.0, q=1.2) * 0.25,
                    spec.get("drive", 1.6))


def _rim(spec, sample_rate, seed, index):
    count = _span(sample_rate, 0.13)
    t = _time(count, sample_rate)
    tone = spec.get("tone", 0.6)

    click = rng.noise(seed, f"rim-{index}", count) * np.exp(-t / 0.0035)
    ping = np.sin(2 * np.pi * (1500.0 + tone * 1400.0) * t) * np.exp(-t / 0.022)
    voice = filters.bandpass(click, sample_rate, 2400.0, q=1.6) * 0.8 + ping * 0.5
    return fx.drive(voice, spec.get("drive", 1.4))


def _clap(spec, sample_rate, seed, index):
    """Several staggered bursts — what makes a clap read as a clap rather than noise."""
    decay = spec.get("decay", 0.14)
    count = _span(sample_rate, decay * 4 + 0.2)
    t = _time(count, sample_rate)

    source = filters.bandpass(rng.noise(seed, f"clap-{index}", count), sample_rate, 1500.0, q=0.8)
    voice = np.zeros(count, dtype=np.float64)
    bursts = 3 + int(rng.uniform(seed, f"clap-n-{index}", 0) * 2)
    for burst in range(bursts):
        offset = int(rng.between(seed, f"clap-off-{index}", burst, 0.004, 0.026) * sample_rate * burst)
        if offset >= count:
            break
        tail = np.exp(-t[: count - offset] / (0.012 if burst < bursts - 1 else decay))
        voice[offset:] += source[: count - offset] * tail * (1.0 - burst * 0.12)
    return fx.drive(voice, spec.get("drive", 1.5))


def _ride(spec, sample_rate, seed, index):
    """Sustained metal — inharmonic partials over a wash that takes its time."""
    decay = max(spec.get("decay", 0.4), 0.25) * 2.4
    count = _span(sample_rate, decay + 0.3)
    t = _time(count, sample_rate)
    tone = spec.get("tone", 0.6)

    voice = filters.highpass(rng.noise(seed, f"ride-{index}", count), sample_rate,
                             3800.0) * np.exp(-t / decay) * 0.55
    for position, ratio in enumerate((1.0, 1.51, 2.13, 2.87, 3.61), start=1):
        partial = 480.0 * (1.0 + tone) * ratio
        voice += np.sin(2 * np.pi * partial * t) * np.exp(-t / (decay / position ** 0.4)) * 0.11
    ping = np.exp(-t / 0.02) * spec.get("noisiness", 0.5)
    return fx.drive(voice + filters.bandpass(
        rng.noise(seed, f"ride-ping-{index}", count), sample_rate, 6000.0, q=1.4) * ping, 1.2)


def _crash(spec, sample_rate, seed, index):
    """Broadband, bright, and slow to leave."""
    decay = max(spec.get("decay", 0.4), 0.3) * 3.0
    count = _span(sample_rate, decay + 0.4)
    t = _time(count, sample_rate)

    source = rng.noise(seed, f"crash-{index}", count)
    body = filters.highpass(source, sample_rate, 1800.0 + spec.get("tone", 0.5) * 2400.0)
    swell = np.clip(t / 0.006, 0.0, 1.0)
    return fx.drive(body * swell * np.exp(-t / decay), 1.15)


def _shaker(spec, sample_rate, seed, index):
    """A rattle. Short, high, and mostly transient."""
    decay = max(0.012, spec.get("decay", 0.06) * 0.4)
    count = _span(sample_rate, decay * 6 + 0.04)
    t = _time(count, sample_rate)
    source = rng.noise(seed, f"shaker-{index}", count)
    voice = filters.bandpass(source, sample_rate, 6500.0 + spec.get("tone", 0.5) * 3000.0, q=0.9)
    return voice * np.exp(-t / decay) * np.clip(t / 0.0015, 0.0, 1.0)


def _cowbell(spec, sample_rate, seed, index):
    """Two detuned square-ish tones that refuse to agree."""
    decay = spec.get("decay", 0.18)
    count = _span(sample_rate, decay * 4 + 0.1)
    t = _time(count, sample_rate)
    tone = spec.get("tone", 0.5)

    low = 480.0 + tone * 180.0
    voice = np.sign(np.sin(2 * np.pi * low * t)) * 0.5
    voice += np.sign(np.sin(2 * np.pi * low * 1.4854 * t)) * 0.35
    voice = filters.bandpass(voice, sample_rate, low * 1.6, q=1.1)
    return fx.drive(voice * np.exp(-t / decay), spec.get("drive", 1.4))


def _woodblock(spec, sample_rate, seed, index):
    """A hard, dry click with just enough body to have a pitch."""
    count = _span(sample_rate, 0.12)
    t = _time(count, sample_rate)
    pitch = 900.0 + spec.get("tone", 0.5) * 1300.0

    body = np.sin(2 * np.pi * pitch * t) * np.exp(-t / 0.018)
    body += np.sin(2 * np.pi * pitch * 2.7 * t) * np.exp(-t / 0.008) * 0.4
    knock = rng.noise(seed, f"wood-{index}", count) * np.exp(-t / 0.002)
    return fx.drive(body + filters.bandpass(knock, sample_rate, pitch * 2, q=1.5) * 0.4,
                    spec.get("drive", 1.3))


def _conga(spec, sample_rate, seed, index):
    """A tuned membrane — pitch drops a little, skin on top."""
    decay = spec.get("decay", 0.28)
    count = _span(sample_rate, decay * 4 + 0.15)
    t = _time(count, sample_rate)
    tone = spec.get("tone", 0.5)

    start_hz = 220.0 + tone * 260.0
    pitch = start_hz * (0.86 + 0.14 * np.exp(-t / 0.05))
    body = np.sin(2 * np.pi * np.cumsum(pitch) / sample_rate) * np.exp(-t / decay)
    body += np.sin(2 * np.pi * np.cumsum(pitch * 1.6) / sample_rate) * np.exp(-t / (decay * 0.4)) * 0.25
    skin = rng.noise(seed, f"conga-{index}", count) * np.exp(-t / 0.008)
    return fx.drive(body + filters.highpass(skin, sample_rate, 1800.0) * 0.3,
                    spec.get("drive", 1.4))


def _snap(spec, sample_rate, seed, index):
    """A finger snap. Almost entirely attack."""
    count = _span(sample_rate, 0.14)
    t = _time(count, sample_rate)
    source = rng.noise(seed, f"snap-{index}", count)
    crack = filters.bandpass(source, sample_rate, 2200.0 + spec.get("tone", 0.5) * 2200.0, q=1.1)
    return fx.drive(crack * np.exp(-t / max(spec.get("decay", 0.05) * 0.35, 0.008)),
                    spec.get("drive", 1.6))


def _boom(spec, sample_rate, seed, index):
    """808-style: a long sub that slides down and sits there."""
    decay = max(spec.get("decay", 0.3), 0.25) * 2.6
    count = _span(sample_rate, decay + 0.2)
    t = _time(count, sample_rate)
    tone = spec.get("tone", 0.5)

    pitch = (28.0 + tone * 20.0) + 120.0 * np.exp(-t / 0.035)
    voice = np.sin(2 * np.pi * np.cumsum(pitch) / sample_rate) * np.exp(-t / decay)
    return fx.drive(voice, 1.0 + spec.get("drive", 1.6) * 0.5)


def _anvil(spec, sample_rate, seed, index):
    """Struck metal. Inharmonic, unpleasant, and it rings for a while."""
    decay = max(spec.get("decay", 0.3), 0.2) * 2.0
    count = _span(sample_rate, decay + 0.25)
    t = _time(count, sample_rate)
    root = 320.0 + spec.get("tone", 0.5) * 700.0

    voice = np.zeros(count, dtype=np.float64)
    for position, ratio in enumerate((1.0, 1.73, 2.41, 3.19, 4.62, 6.03), start=1):
        jitter = rng.between(seed, f"anvil-{index}", position, 0.97, 1.03)
        voice += (np.sin(2 * np.pi * root * ratio * jitter * t)
                  * np.exp(-t / (decay / position ** 0.5)) / position)
    hit = rng.noise(seed, f"anvil-hit-{index}", count) * np.exp(-t / 0.0025)
    return fx.drive(voice + filters.highpass(hit, sample_rate, 2500.0) * 0.5,
                    spec.get("drive", 2.0))


_DRUMS = {
    "kick": _kick,
    "snare": _snare,
    "hat": _hat,
    "tom": _tom,
    "rim": _rim,
    "clap": _clap,
    "ride": _ride,
    "crash": _crash,
    "shaker": _shaker,
    "cowbell": _cowbell,
    "woodblock": _woodblock,
    "conga": _conga,
    "snap": _snap,
    "boom": _boom,
    "anvil": _anvil,
}

DRUM_NAMES: tuple[str, ...] = tuple(sorted(_DRUMS))
