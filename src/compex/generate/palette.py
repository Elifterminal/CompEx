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

from dataclasses import dataclass, replace

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
    # sustaining, added 2026-08-02 because the ones that actually held a note
    # were six and it was audible
    "bowed": (0.44, 0.28, 0.40),
    "shimmer": (0.82, 0.14, 0.34),
    "tape": (0.40, 0.38, 0.30),
    "vox": (0.50, 0.22, 0.44),
    "aeolian": (0.66, 0.34, 0.24),
}

#: How much of its level an engine still has in the middle of a long note,
#: measured rather than asserted: RMS two thirds of the way through against
#: RMS just after the attack. ``tests/test_engines.py`` recomputes every one of
#: these and fails if an engine stops behaving the way the composer believes
#: it does.
#:
#: This exists because of a listening complaint that turned out to be exactly
#: right. Nearly 40% of everything chosen to hold a chord was a struck sound
#: quietly dying under it — a bell, a glass, a noise wash — which is both why
#: the sustained voices sounded alike and why the whole thing read as
#: percussive.
HOLD: dict[str, float] = {
    "shimmer": 1.79, "string": 1.50, "choir": 1.17, "brass": 1.00,
    "flute": 1.00, "tape": 1.00, "organ": 0.97, "bowed": 0.96,
    "reed": 0.95, "pad": 0.92, "vox": 0.91, "drone": 0.89,
    "supersaw": 0.80, "aeolian": 0.62, "sub": 0.56, "bell": 0.47,
    "granular": 0.41, "additive": 0.39, "chip": 0.39, "formant": 0.39,
    "pm": 0.39, "mallet": 0.38, "tine": 0.38, "glass": 0.37,
    "wavetable": 0.37, "chime": 0.36, "noise": 0.25, "pluck": 0.03,
    "feedback": 0.00,
}

#: What a role needs an engine to do. A pad that dies halfway through the bar
#: is not a pad, whatever its name is.
NEEDS_HOLD: dict[str, float] = {ROLE_PAD: 0.60, ROLE_TEXTURE: 0.30}

ENGINES_FOR_ROLE: dict[str, tuple[str, ...]] = {
    ROLE_BASS: ("sub", "pm", "formant", "additive", "pluck", "reed",
                "supersaw", "feedback", "chip", "organ", "tape"),
    ROLE_LEAD: ("pluck", "bell", "pm", "additive", "formant", "mallet", "tine",
                "glass", "chime", "reed", "brass", "flute", "chip", "wavetable",
                "bowed", "vox"),
    ROLE_PAD: ("pad", "string", "choir", "organ", "drone", "supersaw",
               "bowed", "shimmer", "tape", "vox", "aeolian",
               "additive", "bell", "noise", "glass"),
    ROLE_TEXTURE: ("noise", "bell", "pm", "formant", "pluck", "granular",
                   "feedback", "wavetable", "chime", "drone", "chip",
                   "shimmer", "aeolian", "vox"),
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


def airiness(mood: Mood) -> float:
    """How much breath, air and noise this mood should carry, 0..1.

    Breath is a mix decision wearing a synthesis parameter's clothes, and it
    was the one place left in the palette drawing blind. Air is the first thing
    that muddies a crowded arrangement, so a dense piece gets less of it than a
    sparse one however bright it is.
    """
    open_space = 1.0 - 0.75 * mood.density
    return max(0.0, min(1.0, (0.25 + 0.55 * mood.valence) * open_space + 0.12 * mood.grit))


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


#: Parameters that must not be moved by the timbre drive, per engine family.
#: Not a taste judgement — these are the ones where a small relative change is
#: a large structural one: a vowel is a vowel, a partial count is an integer,
#: and a filter that moves under a resonance sweep whistles.
FIXED: frozenset[str] = frozenset({
    "f1", "f2", "f1b", "f2b", "partials", "layers", "singers", "bits", "wave",
})

#: How far a parameter may drift from where it started, as a multiplier.
DRIFT_FLOOR, DRIFT_CEILING = 0.4, 2.6


def drift(spec: VoiceSpec, seed: int, movement: int, amount: float) -> VoiceSpec:
    """The same instrument, a little further along in becoming something else.

    A player is stuck with the instrument they brought. A machine is not, and
    keeping its voices fixed for eight minutes was never a decision — it was
    an assumption inherited from ensembles made of people. The synthesis
    parameters are now moved by a drive like everything else, and the drift is
    cumulative in the movement index, so a voice arrives somewhere rather than
    wobbling in place.

    Structural parameters are held: moving a formant pair by 20% is a different
    vowel, moving a partial count is a different instrument entirely, and
    neither is what "the same voice, changed" means.
    """
    if amount <= 0.0 or not spec.params:
        return spec

    moved: list[tuple[str, float]] = []
    for name, value in spec.params:
        if name in FIXED or value == 0:
            moved.append((name, value))
            continue
        walk = 0.0
        for step in range(1, movement + 1):
            walk += rng.between(seed, f"drift-{spec.voice_id}-{name}", step, -0.22, 0.22)
        factor = max(DRIFT_FLOOR, min(DRIFT_CEILING, 1.0 + walk * amount))
        moved.append((name, value * factor))
    return replace(spec, params=tuple(moved))


def choose_engine(seed: int, index: int, role: str, mood: Mood,
                  bored: dict[str, float] | None = None) -> str:
    """Pick an engine for ``role`` whose character sits near the mood.

    Roles that have to hold a note only draw from engines that actually do.
    The pool used to include struck sounds for pads on the grounds that a bell
    can be a pad if you squint, and the result was a piece where nearly half of
    what should have been sustaining was decaying away under the harmony.
    """
    candidates = ENGINES_FOR_ROLE.get(role)
    if not candidates:
        raise ValueError(f"no engines registered for role {role!r}")

    needed = NEEDS_HOLD.get(role)
    if needed is not None:
        holding = tuple(name for name in candidates if HOLD.get(name, 0.0) >= needed)
        if holding:
            candidates = holding

    # Anything reached for lately is pushed down the ranking. Not banned —
    # a piece may still want the obvious instrument — but an engine that has
    # made the last six pieces has to be a better fit than one that has not.
    stale = bored or {}

    target = (mood.valence, mood.grit, mood.energy)
    # Anything reached for lately is pushed down the ranking. Not banned — a
    # piece may still want the obvious instrument — but an engine that made the
    # last six pieces has to fit better than one that did not.
    stale = bored or {}
    ranked = sorted(
        candidates,
        key=lambda name: sum(
            (component - aim) ** 2 for component, aim in zip(ENGINE_COLOUR[name], target)
        ) + stale.get(name, 0.0),
    )
    return rng.pick(seed, f"engine-{role}", index, ranked[: min(CANDIDATE_POOL, len(ranked))])


def build_voice(seed: int, index: int, role: str, mood: Mood,
                engine: str | None = None,
                bored: dict[str, float] | None = None) -> VoiceSpec:
    """Invent a full instrument for ``role`` — engine, parameters and effects."""
    engine = engine or choose_engine(seed, index, role, mood, bored)
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
    air = airiness(mood)
    return (("width", rng.between(seed, f"{stream}-w", 0, 0.12, 0.5)),
            ("pwm", rng.between(seed, f"{stream}-pwm", 0, 0.3, 4.0)),
            ("breath", 0.02 + air * 0.16 + rng.between(seed, f"{stream}-br", 0, 0.0, 0.05)),
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
            ("air", 0.02 + airiness(mood) * 0.13),
            ("q", rng.between(seed, f"{stream}-q", 0, 5.0, 12.0)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.1, 0.7)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.2, 1.0)))


def _flute(seed, stream, mood):
    # The flute is mostly air by design — "the noise is most of what you
    # recognise" — so this is the parameter that decides whether a lead reads
    # as a flute or as somebody breathing into the microphone.
    air = airiness(mood)
    return (("vibrato", rng.between(seed, f"{stream}-vib", 0, 0.001, 0.009)),
            ("rate", rng.between(seed, f"{stream}-rate", 0, 3.5, 6.5)),
            ("breath", 0.07 + air * 0.26 + rng.between(seed, f"{stream}-br", 0, 0.0, 0.06)),
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


def _bowed(seed, stream, mood):
    return (("partials", float(8 + int(rng.uniform(seed, f"{stream}-n", 0) * 8))),
            ("tilt", rng.between(seed, f"{stream}-tilt", 0, 1.0, 1.7)),
            ("vibrato", rng.between(seed, f"{stream}-vib", 0, 0.002, 0.011)),
            ("rate", rng.between(seed, f"{stream}-rate", 0, 4.0, 6.5)),
            ("grip", 0.1 + mood.grit * 0.45),
            ("bite", rng.between(seed, f"{stream}-bite", 0, 0.04, 0.16)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.12, 0.5)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.2, 0.7)))


def _shimmer(seed, stream, mood):
    return (("partials", float(6 + int(rng.uniform(seed, f"{stream}-n", 0) * 6))),
            # How far the partials are stretched off the harmonic series is the
            # whole character: 1.0 is a chord, 1.15 is a struck bar.
            ("stretch", 1.0 + rng.between(seed, f"{stream}-st", 0, 0.01, 0.06)
             + mood.tension * 0.06),
            ("spread", rng.between(seed, f"{stream}-sp", 0, 0.002, 0.009)),
            ("cutoff", rng.between(seed, f"{stream}-cut", 0, 2200.0, 6500.0)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.25, 0.9)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.6, 1.8)))


def _tape(seed, stream, mood):
    return (("partials", float(5 + int(rng.uniform(seed, f"{stream}-n", 0) * 6))),
            ("wow", rng.between(seed, f"{stream}-wow", 0, 0.001, 0.007)),
            ("wow_rate", rng.between(seed, f"{stream}-wr", 0, 0.4, 1.1)),
            ("flutter", rng.between(seed, f"{stream}-fl", 0, 0.0005, 0.003)),
            ("flutter_rate", rng.between(seed, f"{stream}-fr", 0, 5.5, 9.5)),
            ("saturation", 0.6 + mood.grit * 2.4),
            ("hiss", 0.004 + mood.grit * 0.02),
            ("cutoff", rng.between(seed, f"{stream}-cut", 0, 3000.0, 7000.0)),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.04, 0.2)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.15, 0.6)))


def _vox(seed, stream, mood):
    first, second = rng.pick(seed, f"{stream}-from", 0, VOWELS)[:2]
    third, fourth = rng.pick(seed, f"{stream}-to", 0, VOWELS)[:2]
    return (("f1", first), ("f2", second), ("f1b", third), ("f2b", fourth),
            ("partials", float(16 + int(rng.uniform(seed, f"{stream}-n", 0) * 12))),
            ("spread", rng.between(seed, f"{stream}-sp", 0, 0.001, 0.006)),
            ("q", rng.between(seed, f"{stream}-q", 0, 5.0, 10.0)),
            ("air", 0.02 + airiness(mood) * 0.09),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.2, 0.7)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.3, 1.0)))


def _aeolian(seed, stream, mood):
    return (("partials", float(2 + int(rng.uniform(seed, f"{stream}-n", 0) * 5))),
            # A low q is wind, a high q is a note. Sparse moods get the note.
            ("q", 12.0 + (1.0 - mood.density) * 34.0),
            ("attack", rng.between(seed, f"{stream}-att", 0, 0.35, 1.1)),
            ("release", rng.between(seed, f"{stream}-rel", 0, 0.5, 1.6)))


_PARAM_BUILDERS = {
    "sub": _sub, "additive": _additive, "pm": _pm, "pluck": _pluck, "noise": _noise,
    "bell": _bell, "mallet": _mallet, "tine": _tine, "glass": _glass, "chime": _chime,
    "formant": _formant, "reed": _reed, "brass": _brass, "choir": _choir, "flute": _flute,
    "pad": _pad, "string": _string, "organ": _organ, "supersaw": _supersaw, "drone": _drone,
    "granular": _granular, "wavetable": _wavetable, "feedback": _feedback, "chip": _chip,
    "bowed": _bowed, "shimmer": _shimmer, "tape": _tape, "vox": _vox, "aeolian": _aeolian,
}
