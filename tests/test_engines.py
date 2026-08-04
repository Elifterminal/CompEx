"""Every engine, drum and effect has to render something finite and audible.

With twenty-four engines and fifteen drums it is easy for one to rot
unnoticed, so this sweeps all of them rather than spot-checking.
"""

import unittest

import numpy as np

from compex.dsp import drums, effects, engines
from compex.dsp.engines import core
from compex.dsp import fx
from compex.dsp.engines import ENGINE_NAMES, NOTE_CEILING, NOTE_LOUDNESS
from compex.generate.mood import THEMES
from compex.generate.palette import (
    ENGINES_FOR_ROLE,
    HOLD,
    NEEDS_HOLD,
    ROLE_PAD,
    choose_engine,
)
from compex.generate import THEMES
from compex.generate.palette import (
    DRUM_COLOUR,
    DRUM_VOICES,
    ENGINE_COLOUR,
    ENGINES_FOR_ROLE,
    _drum,
    _EFFECT_PARAMS,
    _PARAM_BUILDERS,
    VoiceSpec,
    build_effects,
    build_kit,
)

SR = 44_100
MOOD = THEMES["restless"]
PITCHES = (55.0, 220.0, 880.0)


def _spec(name: str) -> VoiceSpec:
    return VoiceSpec(voice_id=name, engine=name, role="lead",
                     params=_PARAM_BUILDERS[name](7, f"probe-{name}", MOOD))


class RegistryTests(unittest.TestCase):
    def test_every_engine_has_a_colour(self):
        self.assertEqual(set(ENGINE_COLOUR), set(ENGINE_NAMES))

    def test_every_engine_has_parameters(self):
        self.assertEqual(set(_PARAM_BUILDERS), set(ENGINE_NAMES))

    def test_every_engine_is_reachable_from_some_role(self):
        reachable = {name for names in ENGINES_FOR_ROLE.values() for name in names}
        self.assertEqual(reachable, set(ENGINE_NAMES))

    def test_every_drum_exists_and_has_a_colour(self):
        self.assertEqual(set(DRUM_VOICES), set(drums.DRUM_NAMES))
        self.assertEqual(set(DRUM_COLOUR), set(drums.DRUM_NAMES))

    def test_every_effect_has_parameters(self):
        self.assertEqual(set(_EFFECT_PARAMS), set(effects.EFFECT_NAMES))


class EngineTests(unittest.TestCase):
    def test_every_engine_renders_finite_audio_at_every_register(self):
        for name in ENGINE_NAMES:
            spec = _spec(name)
            for freq in PITCHES:
                voice = engines.render_note(spec, freq, int(0.8 * SR), SR, 7, 0)
                self.assertTrue(np.all(np.isfinite(voice)), f"{name} at {freq}Hz")
                self.assertEqual(len(voice), int(0.8 * SR), name)

    def test_every_engine_makes_a_sound(self):
        for name in ENGINE_NAMES:
            voice = engines.render_note(_spec(name), 220.0, int(0.8 * SR), SR, 7, 0)
            self.assertGreater(float(np.sqrt((voice ** 2).mean())), 0.005, f"{name} is silent")

    def test_engines_come_out_at_a_comparable_level(self):
        """Otherwise the composer's per-voice gain means something different each time.

        Comparable in *loudness*, not in peak. Peak-matching was the original
        rule and it was quietly wrong — a swelling pad and a plucked string
        reach the same height and nothing like the same volume, which is how
        the pads ended up inaudible under everything else.

        The floor is generous because it has to be: a very spiky engine runs
        into the clipping ceiling before it reaches the loudness target, and
        holding it back there is the correct answer rather than a failure.
        """
        rendered = [engines.render_note(_spec(name), 220.0, int(0.8 * SR), SR, 7, 0)
                    for name in ENGINE_NAMES]
        levels = [fx.short_term_rms(voice, SR) for voice in rendered]
        peaks = [float(np.max(np.abs(voice))) for voice in rendered]

        self.assertLessEqual(max(levels), NOTE_LOUDNESS * 1.35)
        self.assertGreater(min(levels), NOTE_LOUDNESS * 0.2)
        self.assertLessEqual(max(peaks), NOTE_CEILING + 1e-6)

    def test_engines_are_deterministic(self):
        for name in ENGINE_NAMES:
            first = engines.render_note(_spec(name), 220.0, 8192, SR, 7, 1)
            second = engines.render_note(_spec(name), 220.0, 8192, SR, 7, 1)
            self.assertTrue(np.array_equal(first, second), name)

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):
            engines.render_note(VoiceSpec("x", "theremin", "lead"), 220.0, 1024, SR, 1, 0)

    def test_bad_frequency_is_rejected(self):
        with self.assertRaises(ValueError):
            engines.render_note(_spec("pluck"), 0.0, 1024, SR, 1, 0)


class DrumTests(unittest.TestCase):
    def test_every_drum_renders_finite_audio(self):
        for name in drums.DRUM_NAMES:
            sample = drums.render_drum(_drum(7, name, 0, MOOD), SR, 7, 0)
            self.assertTrue(np.all(np.isfinite(sample)), name)
            self.assertGreater(float(np.max(np.abs(sample))), 0.1, f"{name} is silent")

    def test_drums_come_out_at_a_comparable_level(self):
        """Loudness again, not peak — same argument as the pitched engines.

        A click and a boom matched by peak are not matched by anything a
        listener has; matched by loudness they are, right up until the click's
        crest factor runs it into the ceiling.
        """
        rendered = [drums.render_drum(_drum(7, name, 0, MOOD), SR, 7, 0)
                    for name in drums.DRUM_NAMES]
        levels = [fx.short_term_rms(sample, SR) for sample in rendered]
        peaks = [float(np.max(np.abs(sample))) for sample in rendered]

        self.assertLessEqual(max(levels), drums.HIT_LOUDNESS * 1.35)
        self.assertGreater(min(levels), drums.HIT_LOUDNESS * 0.2)
        self.assertLessEqual(max(peaks), drums.HIT_CEILING + 1e-6)

    def test_unknown_drum_is_rejected(self):
        with self.assertRaises(ValueError):
            drums.render_drum(VoiceSpec("x", "x", "perc"), SR, 1, 0)


class EffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = np.arange(int(1.5 * SR)) / SR
        cls.dry = np.sin(2 * np.pi * 220 * t) * np.exp(-t / 0.5)

    def test_every_effect_changes_the_signal_without_breaking_it(self):
        for name in effects.EFFECT_NAMES:
            params = _EFFECT_PARAMS[name](7, f"fx-{name}", MOOD)
            wet = effects.apply_chain(self.dry, SR, ((name, params),))
            self.assertTrue(np.all(np.isfinite(wet)), name)
            self.assertEqual(len(wet), len(self.dry), name)
            self.assertFalse(np.allclose(wet, self.dry), f"{name} did nothing")

    def test_feedback_effects_do_not_run_away(self):
        for name in ("delay", "reverb"):
            params = _EFFECT_PARAMS[name](7, f"fx-{name}", MOOD)
            wet = effects.apply_chain(self.dry, SR, ((name, params),))
            self.assertLess(float(np.max(np.abs(wet))), float(np.max(np.abs(self.dry))) * 2.0,
                            f"{name} amplified the signal")

    def test_chains_compose(self):
        chain = tuple((name, _EFFECT_PARAMS[name](7, f"fx-{name}", MOOD))
                      for name in ("chorus", "delay", "reverb"))
        wet = effects.apply_chain(self.dry, SR, chain)
        self.assertTrue(np.all(np.isfinite(wet)))

    def test_empty_chain_is_a_no_op(self):
        self.assertTrue(np.array_equal(effects.apply_chain(self.dry, SR, ()), self.dry))

    def test_unknown_effect_is_rejected(self):
        with self.assertRaises(ValueError):
            effects.apply_chain(self.dry, SR, (("phaser", ()),))


class SelectionTests(unittest.TestCase):
    def test_invented_effects_are_real_and_unique(self):
        for index in range(30):
            chain = build_effects(index, f"s{index}", "lead", MOOD)
            names = [name for name, _ in chain]
            self.assertLessEqual(len(chain), 2)
            self.assertEqual(len(names), len(set(names)), "an effect was used twice")
            for name in names:
                self.assertIn(name, effects.EFFECT_NAMES)

    def test_percussion_gets_no_effects(self):
        self.assertEqual(build_effects(1, "s", "perc", MOOD), ())

    def test_gritty_moods_reach_for_destructive_effects(self):
        clean = {name for seed in range(40)
                 for name, _ in build_effects(seed, f"c{seed}", "lead", THEMES["serene"])}
        dirty = {name for seed in range(40)
                 for name, _ in build_effects(seed, f"d{seed}", "lead", THEMES["shattered"])}
        self.assertTrue({"ringmod", "wavefold"} & dirty)
        self.assertFalse({"ringmod", "wavefold"} <= clean)

    def test_kits_match_the_mood(self):
        gentle = {v.voice_id for v in build_kit(6021, THEMES["serene"])}
        harsh = {v.voice_id for v in build_kit(6021, THEMES["shattered"])}
        self.assertNotIn("anvil", gentle)
        self.assertIn("kick", gentle)
        self.assertNotEqual(gentle, harsh)


if __name__ == "__main__":
    unittest.main()


class HoldTests(unittest.TestCase):
    """The HOLD table is data about behaviour, so it has to be checked against behaviour.

    A composer that believes an engine sustains will hand it a chord to hold.
    If the engine has quietly become a struck sound, nothing fails — the music
    just goes hollow, which is exactly what happened and what Lee heard.
    """

    PARAMS = dict(decay=1.5, release=0.15, attack=0.12, layers=3.0, detune=0.01,
                  bright=0.6, tone=0.6, drive=1.5, noisiness=0.5, centre=800.0, q=1.0,
                  sweep=0.2, partials=10.0, bite=0.8, rasp=0.1, breath=0.2, air=0.1,
                  vibrato=0.004, rate=5.0, f1=530.0, f2=1840.0, singers=4.0,
                  spread=0.006, ratio=1.4, index=2.0, width=0.3, pwm=1.0, grain=0.05,
                  jitter=0.3, feedback=0.5, wave=0.5, morph=0.4, duty=0.5, bits=8.0,
                  sub=0.3, cutoff=1200.0)

    def measure(self, name: str) -> float:
        spec = VoiceSpec("x", name, "pad", tuple(self.PARAMS.items()))
        count = int(3.0 * SR)
        voice = engines.render_note(spec, 220.0, count, SR, 3, 0)
        early = float(np.sqrt((voice[int(0.15 * count):int(0.25 * count)] ** 2).mean()))
        late = float(np.sqrt((voice[int(0.60 * count):int(0.75 * count)] ** 2).mean()))
        return late / max(early, 1e-9)

    def test_every_engine_has_a_measured_hold(self):
        self.assertEqual(set(HOLD), set(ENGINE_NAMES))

    def test_the_table_still_matches_the_engines(self):
        for name in ENGINE_NAMES:
            self.assertAlmostEqual(self.measure(name), HOLD[name], delta=0.06,
                                   msg=f"{name} no longer behaves the way the table says")

    def test_the_engines_a_pad_can_use_actually_hold(self):
        for name in ENGINES_FOR_ROLE[ROLE_PAD]:
            if HOLD[name] >= NEEDS_HOLD[ROLE_PAD]:
                break
        else:
            self.fail("no pad engine holds a note")

    def test_a_pad_is_never_given_a_struck_engine(self):
        """38% of pads used to be bells and glasses dying under the harmony."""
        for seed in range(25):
            for mood in THEMES.values():
                engine = choose_engine(seed, 2, ROLE_PAD, mood)
                self.assertGreaterEqual(HOLD[engine], NEEDS_HOLD[ROLE_PAD],
                                        f"{engine} decays and was chosen as a pad")

    def test_there_are_enough_sustaining_engines_to_choose_between(self):
        """The complaint was that sustained voices all sounded alike."""
        holding = [name for name in ENGINES_FOR_ROLE[ROLE_PAD]
                   if HOLD[name] >= NEEDS_HOLD[ROLE_PAD]]
        self.assertGreaterEqual(len(holding), 10, f"only {len(holding)} pad engines hold")


class PluckDelayLineTests(unittest.TestCase):
    """A string shorter than its own smoothing kernel.

    ``np.convolve(..., mode="same")`` returns the longer of its two arguments,
    not the length of the first, so a dark note high enough to give a
    four-sample delay line came back six samples long and took the render down
    with it. Unreachable at 48 kHz: it needs the 11 kHz rate the listening
    probe runs at, which is how it survived from the day that probe shipped.
    Reference cases with known answers, because a piece of music has no known
    answer to check against.
    """

    def test_a_string_shorter_than_its_kernel_still_renders(self):
        for freq in (2000.0, 2756.0, 4000.0, 5512.0):
            out = core.pluck({"brightness": 0.0, "decay": 1.0}, freq, 4096, 11025, 7, 0)
            self.assertEqual(len(out), 4096, freq)
            self.assertTrue(np.all(np.isfinite(out)), freq)

    def test_it_still_works_where_it_always_did(self):
        for freq, rate in ((55.0, 48000), (440.0, 48000), (110.0, 11025)):
            out = core.pluck({"brightness": 0.3, "decay": 1.0}, freq, 4096, rate, 7, 0)
            self.assertEqual(len(out), 4096)
            self.assertGreater(float(np.max(np.abs(out))), 0.0)

    def test_the_probe_rate_survives_every_pitch_the_engine_can_write(self):
        """The actual failure path: the listening probe, at the top of the range."""
        for midi in range(24, 121, 8):
            freq = 440.0 * 2 ** ((midi - 69) / 12)
            out = core.pluck({"brightness": 0.1, "decay": 1.2}, freq, 2048, 11025, 3, 1)
            self.assertTrue(np.all(np.isfinite(out)), midi)
