"""End to end: knobs in, audio out, formula that reproduces it."""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from compex.audio import encode
from compex.config import SAMPLE_RATE, KnobError, Knobs
from compex.generate import THEMES
from compex.notation import ReentryError, read_formula
from compex.pipeline import make_track

FAST = Knobs(seed=1203, duration_s=20.0)
MOOD = THEMES["melancholy"]


class KnobTests(unittest.TestCase):
    def test_rejects_unknown_knob(self):
        with self.assertRaises(KnobError):
            Knobs.parse({"tempo": 120})

    def test_rejects_out_of_range(self):
        with self.assertRaises(KnobError):
            Knobs.parse({"master_gain": 9})

    def test_rejects_non_numeric(self):
        with self.assertRaises(KnobError):
            Knobs.parse({"seed": "banana"})

    def test_knobs_are_immutable(self):
        knobs = Knobs()
        with self.assertRaises(Exception):
            knobs.seed = 5
        self.assertEqual(knobs.with_(seed=9).seed, 9)
        self.assertEqual(knobs.seed, 1203)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = make_track(FAST, MOOD)

    def test_output_is_finite_and_bounded(self):
        self.assertTrue(np.all(np.isfinite(self.result.samples)))
        self.assertLessEqual(float(np.max(np.abs(self.result.samples))), 1.0)

    def test_output_is_not_silent(self):
        self.assertGreater(float(np.sqrt((self.result.samples ** 2).mean())), 0.005)

    def test_same_inputs_give_identical_audio(self):
        again = make_track(FAST, MOOD)
        self.assertTrue(np.array_equal(self.result.samples, again.samples))
        self.assertEqual(self.result.fingerprint, again.fingerprint)

    def test_seed_changes_the_audio(self):
        other = make_track(FAST.with_(seed=4242), MOOD)
        self.assertNotEqual(self.result.fingerprint, other.fingerprint)

    def test_mood_changes_the_audio(self):
        other = make_track(FAST, THEMES["euphoric"])
        self.assertNotEqual(self.result.fingerprint, other.fingerprint)

    def test_every_theme_renders_without_blowing_up(self):
        for name, mood in THEMES.items():
            result = make_track(Knobs(seed=77, duration_s=12.0), mood)
            self.assertTrue(np.all(np.isfinite(result.samples)), name)
            self.assertGreater(len(result.samples), 0, name)


class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = make_track(FAST, MOOD)

    def test_formula_records_the_inputs(self):
        formula = read_formula(self.result.formula)
        self.assertEqual(formula.seed, FAST.seed)
        self.assertAlmostEqual(formula.runtime_s, FAST.duration_s, places=1)

    def test_formula_reproduces_the_exact_track(self):
        formula = read_formula(self.result.formula)
        again = make_track(
            Knobs(seed=formula.seed, duration_s=formula.runtime_s), formula.mood
        )
        self.assertEqual(self.result.fingerprint, again.fingerprint)

    def test_formula_mentions_what_it_decided(self):
        text = self.result.formula
        for marker in ("SEED", "MOOD", "BPM", "METER", "FORM", "H(n)", "\\Sigma"):
            self.assertIn(marker, text)

    def test_formula_round_trips_for_every_theme(self):
        for name, mood in THEMES.items():
            result = make_track(Knobs(seed=31, duration_s=12.0), mood)
            recovered = read_formula(result.formula)
            self.assertEqual(recovered.mood, mood, name)

    def test_a_formula_without_a_seed_is_rejected(self):
        with self.assertRaises(ReentryError):
            read_formula("\\mathrm{BPM}=120")

    def test_empty_formula_is_rejected(self):
        with self.assertRaises(ReentryError):
            read_formula("   ")


class SaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = make_track(FAST, MOOD)

    def test_writes_a_real_wav(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.result.save(Path(folder) / "t", "wav")
            self.assertEqual(path.suffix, ".wav")
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getframerate(), SAMPLE_RATE)
                self.assertEqual(handle.getnframes(), len(self.result.samples))

    @unittest.skipUnless(encode.available(), "ffmpeg not installed")
    def test_writes_an_mp3_and_cleans_up_the_wav(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.result.save(Path(folder) / "t", "mp3")
            self.assertEqual(path.suffix, ".mp3")
            self.assertGreater(path.stat().st_size, 1000)
            self.assertFalse((Path(folder) / "t.wav").exists())

    def test_unknown_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                self.result.save(Path(folder) / "t", "flac")

    def test_saves_the_formula_beside_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.result.save_formula(Path(folder) / "t")
            self.assertEqual(path.suffix, ".tex")
            self.assertIn("SEED", path.read_text(encoding="utf-8"))

    def test_two_saves_of_one_seed_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            first = make_track(FAST, MOOD).save(Path(folder) / "a", "wav")
            second = make_track(FAST, MOOD).save(Path(folder) / "b", "wav")
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
