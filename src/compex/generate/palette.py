"""The sound palette the composer draws from.

Twenty-four synthesis engines and fifteen drums, each a family rather than a
patch — the composer invents the parameters, so "pluck" covers everything
from a harp to a snapped wire. On top of that every voice gets an effects
chain, which widens the space far more cheaply than more engines would: the
same pluck through a long reverb and through a ring modulator are two
different instruments.

Engines are chosen by how near their character sits to the mood, with several
near-misses in the hat so the same theme does not instrument itself
identically twice.
"""

from __future__ import annotations

from dataclasses import dataclass

from compex import rng
from compex.dsp.effects import EFFECT_NAMES
from compex.generate.mood import Mood

ROLE_BASS = "bass"
ROLE_LEAD = "lead"
ROLE_PAD = "pad"
ROLE_TEXTURE = "texture"
ROLE_PERC = "perc"

#: engine -> (brightness, grit, motion) — its character, for matching a mood.
ENGINE_COLOUR: dict[str, tuple[float, float, float]] = {
    # core
    "sub": (0.14, 0.42, 0.20),
    "additive": (0.75, 0.15, 0.35),
    "pm": (0.33, 0.82, 0.72),
    "pluck": (0.65, 0.25, 0.62),
    "noise": (0.40, 0.70, 0.42),
    # struck
    "bell": (0.60, 0.30, 0.28),
    "mallet": (0.68, 0.18, 0.55),
    "tine": (0.62, 0.20, 0.40),
    "glass": (0.88, 0.12, 0.30),
    "chime": (0.72, 0.16, 0.22),
    # voiced
    "formant": (0.32, 0.60, 0.52),
    "reed": (0.48, 0.35, 0.45),
    "brass": (0.62, 0.45, 0.60),
    "choir": (0.55, 0.18, 0.25),
    "flute": (0.70, 0.08, 0.30),
    # sustained
    "pad": (0.55, 0.18, 0.14),
    "string": (0.50, 0.15, 0.22),
    "organ": (0.58, 0.30, 0.35),
    "supersaw": (0.80, 0.45, 0.55),
    "drone": (0.30, 0.22, 0.08),
    # textural
    "granular": (0.45, 0.50, 0.66),
    "wavetable": (0.60, 0.48, 0.58),
    "feedback": (0.35, 0.88, 0.62),
    "chip": (0.72, 0.62, 0.75),
}

ENGINES_FOR_ROLE: dict[str, tuple[str, ...]] = {
    ROLE_BASS: ("sub", "pm", "formant", "additive", "pluck", "reed",
                "supersaw", "feedback", "chip", "organ"),
    ROLE_LEAD: ("pluck", "bell", "pm", "additive", "formant", "mallet", "tine",
                "glass", "chime", "reed", "brass", "flute", "chip", "wavetable"),
    ROLE_PAD: ("pad", "additive", "bell", "noise", "string", "choir",
               "organ", "drone", "glass", "supersaw"),
    ROLE_TEXTURE: ("noise", "bell", "pm", "formant", "pluck", "granular",
                   "feedback", "wavetable", "chime", "drone", "chip"),
}

DRUM_VOICES: tuple[str, ...] = (
    "kick", "snare", "hat", "tom", "rim", "clap", "ride", "crash",
    "shaker", "cowbell", "woodblock", "conga", "snap", "boom", "anvil",
)

#: drum -> (brightness, aggression). A serene piece should not reach for an anvil.
DRUM_COLOUR: dict[str, tuple[float, float]] = {
    "kick": (0.25, 0.45),
    "snare": (0.50, 0.50),
    "hat": (0.85, 0.30),
    "tom": (0.30, 0.45),
    "rim": (0.70, 0.28),
    "clap": (0.60, 0.45),
    "ride": (0.80, 0.25),
    "crash": (0.90, 0.70),
    "shaker": (0.90, 0.12),
    "cowbell": (0.65, 0.52),
    "woodblock": (0.72, 0.20),
    "conga": (0.45, 0.30),
    "snap": (0.75, 0.35),
    "boom": (0.08, 0.65),
    "anvil": (0.55, 0.95),
}

#: Vowel formant triples (Hz) the voiced engines pick between.
VOWELS: tuple[tuple[float, float, float], ...] = (
    (730.0, 1090.0, 2440.0),   # ah
    (270.0, 2290.0, 3010.0),   # ee
    (300.0, 870.0, 2240.0),    # oo
    (530.0, 1840.0, 2480.0),   # eh
    (570.0, 840.0, 2410.0),    # oh
)

LFO_RATES: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

CANDIDATE_POOL = 4  # how many near-matching engines go into the hat


@dataclass(frozen=True)
class VoiceSpec:
    """One instrument: engine, role, invented parameters, and its effects chain."""

    voice_id: str
    engine: str
    role: str
    params: tuple[tuple[str, float], ...] = ()
    effects: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    gain: float = 1.0
    octave: int = 0

    def get(self, name: str, default: float = 0.0) -> float:
        for key, value in self.params:
            if key == name:
                return value
        return default

    def as_dict(self) -> dict[str, float]:
        return dict(self.params)

    def effect_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.effects)


def choose_engine(seed: int, index: int, role: str, mood: Mood) -> str:
    """Pick an engine for ``role`` whose character sits near the mood."""
    candidates = ENGINES_FOR_ROLE.get(role)
    if not candidates:
        raise ValueError(f"no engines registered for role {role!r}")

    target = (mood.valence, mood.grit, mood.energy)
    ranked = sorted(
        candidates,
        key=lambda name: sum(
            (component - aim) ** 2 for component, aim in zip(ENGINE_COLOUR[name], target)
        ),
    )
    return rng.pick(seed, f"engine-{role}", index, ranked[: min(CANDIDATE_POOL, len(ranked))])


def build_voice(seed: int, index: int, role: str, mood: Mood,
                engine: str | None = None) -> VoiceSpec:
    """Invent a full instrument for ``role`` — engine, parameters and effects."""
    engine = engine or choose_engine(seed, index, role, mood)
    stream = f"voice-{role}-{index}"
    params = _PARAM_BUILDERS[engine](seed, stream, mood)

    octave = {ROLE_BASS: -2, ROLE_LEAD: 1, ROLE_PAD: 0, ROLE_TEXTURE: 1}.get(role, 0)
    if role == ROLE_LEAD and mood.energy > 0.7:
        octave += 1

    gain = {ROLE_BASS: 0.90, ROLE_LEAD: 0.62, ROLE_PAD: 0.34, ROLE_TEXTURE: 0.24}.get(role, 0.5)

    return VoiceSpec(
        voice_id=f"{role}_{engine}_{index}",
        engine=engine,
        role=role,
        params=params,
        effects=build_effects(seed, stream, role, mood),
        gain=gain * rng.between(seed, f"{stream}-gain", 0, 0.86, 1.12),
        octave=octave,
    )


def build_kit(seed: int, mood: Mood) -> tuple[VoiceSpec, ...]:
    """Choose which drums exist in this piece. Denser moods get more of them."""
    wanted = 2 + int(round(mood.density * 5.0))
    chosen: list[VoiceSpec] = [_drum(seed, "kick", 0, mood)]

    target = (mood.valence, mood.grit)
    pool = [name for name in DRUM_VOICES if name != "kick"]
    for slot in range(min(wanted, len(pool))):
        # Rank what is left by how near it sits to the mood, then draw from the
        # front of that ranking — so the kit belongs to the piece, not to the seed.
        pool.sort(key=lambda name: sum(
            (component - aim) ** 2 for component, aim in zip(DRUM_COLOUR[name], target)
        ))
        name = rng.pick(seed, "kit", slot, tuple(pool[: min(CANDIDATE_POOL, len(pool))]))
        pool.remove(name)
        chosen.append(_drum(seed, name, slot + 1, mood))
    return tuple(chosen)


def _drum(seed: int, name: str, index: int, mood: Mood) -> VoiceSpec:
    stream = f"drum-{name}"
    params = (
        ("decay", rng.between(seed, f"{stream}-decay", index, 0.05, 0.24 + mood.density * 0.2)),
        ("tone", rng.between(seed, f"{stream}-tone", index, 0.25, 1.0)),
        ("drive", 1.2 + mood.grit * 3.6),
        ("noisiness", rng.between(seed, f"{stream}-noise", index, 0.1, 0.9)),
    )
    return VoiceSpec(
        voice_id=name,
        engine=name,
        role=ROLE_PERC,
        params=params,
        gain=rng.between(seed, f"{stream}-gain", index, 0.55, 0.95),
    )


# -- effects ---------------------------------------------------------------

def build_effects(seed: int, stream: str, role: str,
                  mood: Mood) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    """Invent 0-2 effects for a voice, weighted by what it is and how it should feel."""
    if role == ROLE_PERC:
        return ()

    appetite = 0.25 + mood.grit * 0.5 + mood.density * 0.25
    count = sum(
        1 for slot in range(2)
        if rng.uniform(seed, f"{stream}-fxn", slot) < appetite - slot * 0.35
    )
    if count == 0:
        return ()

    weights = _effect_weights(role, mood)
    chain: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    used: set[str] = set()
    for slot in range(count):
        name = _weighted_pick(seed, f"{stream}-fx", slot, weights, used)
        if name is None:
            break
        used.add(name)
        chain.append((name, _EFFECT_PARAMS[name](seed, f"{stream}-{name}", mood)))
    return tuple(chain)


def _effect_weights(role: str, mood: Mood) -> dict[str, float]:
    quiet = 1.0 - mood.energy
    return {
        "reverb": 0.6 + quiet * 1.6 + (0.8 if role in {ROLE_PAD, ROLE_TEXTURE} else 0.0),
        "delay": 0.4 + mood.tension * 1.1 + (0.7 if role == ROLE_LEAD else 0.0),
        "chorus": 0.5 + quiet * 0.9 + (0.6 if role in {ROLE_PAD, ROLE_LEAD} else 0.0),
        "tremolo": 0.3 + mood.energy * 0.8,
        "ringmod": 0.15 + mood.grit * 1.9 + mood.tension * 0.6,
        "wavefold": 0.15 + mood.grit * 2.1,
    }


def _weighted_pick(seed: int, stream: str, index: int, weights: dict[str, float],
                   used: set[str]) -> str | None:
    available = [(name, weight) for name, weight in weights.items() if name not in used]
    total = sum(weight for _, weight in available)
    if total <= 0:
        return None
    target = rng.uniform(seed, stream, index) * total
    running = 0.0
    for name, weight in available:
        running += weight
        if target <= running:
            return name
    return available[-1][0]


def _fx_reverb(seed, stream, mood):
    return (("size", rng.between(seed, f"{stream}-size", 0, 0.25, 0.95)),
            ("damp", rng.between(seed, f"{stream}-damp", 0, 0.15, 0.85)),
            ("mix", rng.between(seed, f"{stream}-mix", 0, 0.15, 0.55)))


def _fx_delay(seed, stream, mood):
    return (("time_s", rng.pick(seed, f"{stream}-t", 0, (0.09, 0.14, 0.2, 0.28, 0.375, 0.5))),
            ("feedback", rng.between(seed, f"{stream}-fb", 0, 0.15, 0.6)),
            ("mix", rng.between(seed, f"{stream}-mix", 0, 0.12, 0.42)))


def _fx_chorus(seed, stream, mood):
    return (("rate", rng.between(seed, f"{stream}-rate", 0, 0.15, 1.8)),
            ("depth_ms", rng.between(seed, f"{stream}-depth", 0, 2.0, 11.0)),
            ("mix", rng.between(seed, f"{stream}-mix", 0, 0.2, 0.55)))


def _fx_tremolo(seed, stream, mood):
    return (("rate", rng.between(seed, f"{stream}-rate", 0, 1.5, 9.0)),
            ("depth", rng.between(seed, f"{stream}-depth", 0, 0.2, 0.75)))


def _fx_ringmod(seed, stream, mood):
    return (("freq", rng.between(seed, f"{stream}-f", 0, 30.0, 620.0)),
            ("mix", rng.between(seed, f"{stream}-mix", 0, 0.15, 0.6)))


def _fx_wavefold(seed, stream, mood):
    return (("amount", rng.between(seed, f"{stream}-amt", 0, 1.2, 3.5 + mood.grit * 3.0)),
            ("mix", rng.between(seed, f"{stream}-mix", 0, 0.25, 0.8)))


_EFFECT_PARAMS = {
    "reverb": _fx_reverb, "delay": _fx_delay, "chorus": _fx_chorus,
    "tremolo": _fx_tremolo, "ringmod": _fx_ringmod, "wavefold": _fx_wavefold,
}


# -- per-engine parameter invention ---------------------------------------

def _pm(seed, stream, mood):
    return (("ratio", rng.pick(seed, f"{stream}-ratio", 0, (0.5, 1.0, 1.0, 2.0, 3.0, 3.5, 5.0))),
            ("index_low", rng.between(seed, f"{stream}-lo", 0, 0.2, 2.0)),
            ("index_high", rng.between(seed, f"{stream}-hi", 0, 2.5, 3.0 + mood.grit * 8.0)),
            ("lfo", rng.pick(seed, f"{stream}-lfo", 0, LFO_RATES)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.25, 2.4)))


def _additive(seed, stream, mood):
    return (("partials", float(2 + int(rng.uniform(seed, f"{stream}-n", 0)
                                       * (4 + mood.valence * 12)))),
            ("rolloff", rng.between(seed, f"{stream}-roll", 0, 0.6, 2.4)),
            ("detune", rng.between(seed, f"{stream}-det", 0, 0.0, 0.012 + mood.tension * 0.02)),
            ("odd_only", 1.0 if rng.uniform(seed, f"{stream}-odd", 0) > 0.68 else 0.0),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.004, 0.35)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.4, 3.0)))


def _pluck(seed, stream, mood):
    return (("damping", rng.between(seed, f"{stream}-damp", 0, 0.28, 0.5)),
            ("brightness", rng.between(seed, f"{stream}-bright", 0, 0.25, 0.99)),
            ("pick", rng.between(seed, f"{stream}-pick", 0, 0.05, 0.5)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.5, 3.2)))


def _bell(seed, stream, mood):
    return (("partials", float(3 + int(rng.uniform(seed, f"{stream}-n", 0) * 6))),
            ("inharmonicity", rng.between(seed, f"{stream}-inh", 0, 0.02,
                                          0.18 + mood.tension * 0.4)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.8, 4.5)),
            ("strike", rng.between(seed, f"{stream}-strike", 0, 0.001, 0.02)))


def _mallet(seed, stream, mood):
    return (("overtone", rng.between(seed, f"{stream}-ot", 0, 3.0, 4.6)),
            ("hardness", rng.between(seed, f"{stream}-hard", 0, 0.1, 1.0)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.25, 1.4)))


def _tine(seed, stream, mood):
    return (("ping", rng.between(seed, f"{stream}-ping", 0, 5.0, 9.5)),
            ("bark", rng.between(seed, f"{stream}-bark", 0, 0.12, 0.7)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.9, 3.0)))


def _glass(seed, stream, mood):
    return (("partials", float(4 + int(rng.uniform(seed, f"{stream}-n", 0) * 6))),
            ("shimmer", rng.between(seed, f"{stream}-shim", 0, 1.5, 7.0)),
            ("spread", rng.between(seed, f"{stream}-spread", 0, 0.3, 2.4)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 1.4, 4.5)))


def _chime(seed, stream, mood):
    return (("decay", rng.between(seed, f"{stream}-decay", 0, 1.8, 5.0)),
            ("strike", rng.between(seed, f"{stream}-strike", 0, 0.02, 0.3)))


def _formant(seed, stream, mood):
    vowel = rng.pick(seed, f"{stream}-vowel", 0, VOWELS)
    return (("f1", vowel[0]), ("f2", vowel[1]), ("f3", vowel[2]),
            ("buzz", rng.between(seed, f"{stream}-buzz", 0, 0.2, 0.95)),
            ("q", rng.between(seed, f"{stream}-q", 0, 5.0, 16.0)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.4, 2.6)))


def _reed(seed, stream, mood):
    return (("width", rng.between(seed, f"{stream}-w", 0, 0.12, 0.5)),
            ("pwm", rng.between(seed, f"{stream}-pwm", 0, 0.3, 4.0)),
            ("breath", rng.between(seed, f"{stream}-br", 0, 0.03, 0.3)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.01, 0.14)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.05, 0.4)))


def _brass(seed, stream, mood):
    return (("partials", float(8 + int(rng.uniform(seed, f"{stream}-n", 0) * 14))),
            ("bite", rng.between(seed, f"{stream}-bite", 0, 0.4, 1.6)),
            ("rasp", rng.between(seed, f"{stream}-rasp", 0, 0.0, 0.2 + mood.grit * 0.5)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.02, 0.2)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.06, 0.4)))


def _choir(seed, stream, mood):
    vowel = rng.pick(seed, f"{stream}-vowel", 0, VOWELS)
    return (("f1", vowel[0]), ("f2", vowel[1]),
            ("singers", float(3 + int(rng.uniform(seed, f"{stream}-n", 0) * 4))),
            ("spread", rng.between(seed, f"{stream}-sp", 0, 0.002, 0.014)),
            ("air", rng.between(seed, f"{stream}-air", 0, 0.02, 0.2)),
            ("q", rng.between(seed, f"{stream}-q", 0, 5.0, 12.0)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.1, 0.7)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.2, 1.0)))


def _flute(seed, stream, mood):
    return (("vibrato", rng.between(seed, f"{stream}-vib", 0, 0.001, 0.009)),
            ("rate", rng.between(seed, f"{stream}-rate", 0, 3.5, 6.5)),
            ("breath", rng.between(seed, f"{stream}-br", 0, 0.1, 0.55)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.03, 0.25)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.08, 0.4)))


def _pad(seed, stream, mood):
    return (("layers", float(2 + int(rng.uniform(seed, f"{stream}-l", 0) * 4))),
            ("detune", rng.between(seed, f"{stream}-det", 0, 0.002, 0.018)),
            ("cutoff", rng.between(seed, f"{stream}-cut", 0, 320.0,
                                   900.0 + mood.valence * 2600.0)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.08, 1.2)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.3, 2.0)))


def _string(seed, stream, mood):
    return (("players", float(3 + int(rng.uniform(seed, f"{stream}-p", 0) * 4))),
            ("spread", rng.between(seed, f"{stream}-sp", 0, 0.002, 0.011)),
            ("vibrato", rng.between(seed, f"{stream}-vib", 0, 0.001, 0.008)),
            ("rolloff", rng.between(seed, f"{stream}-roll", 0, 1.1, 2.2)),
            ("bow", rng.between(seed, f"{stream}-bow", 0, 0.01, 0.14)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.12, 0.8)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.2, 0.9)))


def _organ(seed, stream, mood):
    return (("drawbars", rng.between(seed, f"{stream}-db", 0, 0.2, 1.0)),
            ("leslie", rng.between(seed, f"{stream}-les", 0, 0.0, 0.5)),
            ("leslie_rate", rng.between(seed, f"{stream}-lr", 0, 0.8, 7.5)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.003, 0.05)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.02, 0.25)))


def _supersaw(seed, stream, mood):
    return (("voices", float(5 + int(rng.uniform(seed, f"{stream}-v", 0) * 4))),
            ("detune", rng.between(seed, f"{stream}-det", 0, 0.004, 0.03)),
            ("cutoff", rng.between(seed, f"{stream}-cut", 0, 1200.0,
                                   3000.0 + mood.valence * 6000.0)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.005, 0.2)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.08, 0.6)))


def _drone(seed, stream, mood):
    return (("partials", float(5 + int(rng.uniform(seed, f"{stream}-n", 0) * 8))),
            ("drift", rng.between(seed, f"{stream}-dr", 0, 0.02, 0.3)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.5, 2.5)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.5, 2.5)))


def _granular(seed, stream, mood):
    return (("grain", rng.between(seed, f"{stream}-g", 0, 0.012, 0.14)),
            ("overlap", rng.between(seed, f"{stream}-ov", 0, 0.2, 0.9)),
            ("scatter", rng.between(seed, f"{stream}-sc", 0, 0.1, 1.0)),
            ("dust", rng.between(seed, f"{stream}-du", 0, 0.0, 0.3)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.01, 0.4)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.5, 3.0)))


def _wavetable(seed, stream, mood):
    return (("start", rng.between(seed, f"{stream}-s", 0, 0.0, 0.6)),
            ("end", rng.between(seed, f"{stream}-e", 0, 0.4, 1.0)),
            ("width", rng.between(seed, f"{stream}-w", 0, 0.15, 0.6)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.004, 0.2)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.4, 2.4)))


def _feedback(seed, stream, mood):
    return (("resonance", rng.between(seed, f"{stream}-res", 0, 18.0, 90.0)),
            ("overtone", rng.between(seed, f"{stream}-ot", 0, 0.0, 0.6)),
            ("scream", rng.between(seed, f"{stream}-sc", 0, 0.5, 2.0 + mood.grit * 5.0)),
            ("burst", rng.between(seed, f"{stream}-b", 0, 0.01, 0.2)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.1, 0.8)))


def _chip(seed, stream, mood):
    return (("arp", rng.pick(seed, f"{stream}-arp", 0, (0.02, 0.03, 0.045, 0.06, 0.09))),
            ("width", rng.pick(seed, f"{stream}-w", 0, (0.125, 0.25, 0.5))),
            ("bits", float(3 + int(rng.uniform(seed, f"{stream}-b", 0) * 6))),
            ("arpeggio", 1.0 if rng.uniform(seed, f"{stream}-on", 0) > 0.35 else 0.0),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.3, 1.6)))


def _sub(seed, stream, mood):
    return (("drive", 1.1 + mood.grit * 4.0),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.4, 2.6)),
            ("click", rng.between(seed, f"{stream}-click", 0, 0.0, 0.35)))


def _noise(seed, stream, mood):
    return (("centre", rng.between(seed, f"{stream}-c", 0, 260.0, 5200.0)),
            ("q", rng.between(seed, f"{stream}-q", 0, 0.6, 9.0)),
            ("sweep", rng.between(seed, f"{stream}-sw", 0, 0.4, 3.0)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.01, 0.6)),
            ("decay", rng.between(seed, f"{stream}-decay", 0, 0.3, 2.8)))


_PARAM_BUILDERS = {
    "sub": _sub, "additive": _additive, "pm": _pm, "pluck": _pluck, "noise": _noise,
    "bell": _bell, "mallet": _mallet, "tine": _tine, "glass": _glass, "chime": _chime,
    "formant": _formant, "reed": _reed, "brass": _brass, "choir": _choir, "flute": _flute,
    "pad": _pad, "string": _string, "organ": _organ, "supersaw": _supersaw, "drone": _drone,
    "granular": _granular, "wavetable": _wavetable, "feedback": _feedback, "chip": _chip,
}
