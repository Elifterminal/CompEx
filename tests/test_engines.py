"""Every engine, drum and effect has to render something finite and audible.

With twenty-four engines and fifteen drums it is easy for one to rot
unnoticed, so this sweeps all of them rather than spot-checking.
"""

import unittest

import numpy as np

from compex.dsp import drums, effects, engines
from compex.dsp.engines import ENGINE_NAMES, NOTE_PEAK
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
        """Otherwise the composer's per-voice gain means something different each time."""
        peaks = [float(np.max(np.abs(engines.render_note(_spec(name), 220.0,
                                                         int(0.8 * SR), SR, 7, 0))))
                 for name in ENGINE_NAMES]
        self.assertLessEqual(max(peaks), NOTE_PEAK + 1e-6)
        self.assertGreater(min(peaks), NOTE_PEAK * 0.8)

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
        peaks = [float(np.max(np.abs(drums.render_drum(_drum(7, name, 0, MOOD), SR, 7, 0))))
                 for name in drums.DRUM_NAMES]
        self.assertLessEqual(max(peaks), drums.HIT_PEAK + 1e-6)
        self.assertGreater(min(peaks), drums.HIT_PEAK * 0.8)

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
