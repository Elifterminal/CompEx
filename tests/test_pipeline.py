"""End to end: knobs in, audio out, formula that reproduces it."""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from datetime import date

from compex import library, report
from compex.audio import encode
from compex.config import SAMPLE_RATE, KnobError, Knobs
from compex.generate import THEMES, compose
from compex.notation import ReentryError, read_formula
from compex.dsp.arrange import render_stems
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


class StemTests(unittest.TestCase):
    """Stems have to be the buses that actually went in, not a re-record.

    If a stem were the voice rendered in isolation it would be a different
    piece that happens to sound similar — no trims, no duck — and useless for
    putting the part into anything else.
    """

    def test_the_stems_sum_to_the_master(self):
        piece = compose(249984309, 30.0, THEMES["melancholy"])
        stems = render_stems(piece, 0.89, 0.45)
        summed = sum(bus for name, bus in stems.items() if name != "master")
        master = stems["master"]
        # The master is peak-normalised and limited, so compare shape.
        self.assertGreater(float(np.corrcoef(summed, master)[0, 1]), 0.99)

    def test_every_voice_that_plays_gets_a_stem(self):
        piece = compose(249984309, 30.0, THEMES["melancholy"])
        stems = render_stems(piece, 0.89, 0.0)
        playing = {note.voice for note in piece.notes} | {s.voice for s in piece.strokes}
        for voice in playing:
            self.assertIn(voice, stems, voice)
        self.assertNotIn("ghosts", stems)   # switched off

    def test_a_stem_holds_only_its_own_voice(self):
        piece = compose(249984309, 30.0, THEMES["melancholy"])
        stems = render_stems(piece, 0.89, 0.0)
        quiet = [name for name, bus in stems.items()
                 if name != "master" and not np.any(bus)]
        self.assertEqual(quiet, [])
        # Two different voices cannot be the same signal.
        names = [n for n in stems if n != "master"]
        self.assertFalse(np.array_equal(stems[names[0]], stems[names[1]]))

    def test_they_are_written_to_disk(self):
        import tempfile

        result = make_track(Knobs(seed=249984309, duration_s=20.0, ghost_gain=0.45),
                            THEMES["melancholy"])
        with tempfile.TemporaryDirectory() as where:
            paths = result.save_stems(where)
            self.assertTrue(paths)
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 1000)

    def test_the_ghost_stem_follows_the_knob(self):
        piece = compose(249984309, 30.0, THEMES["melancholy"])
        self.assertIn("ghosts", render_stems(piece, 0.89, 0.45))
        self.assertNotIn("ghosts", render_stems(piece, 0.89, 0.0))


class LibraryTests(unittest.TestCase):
    """Three parallel trees, the same folder name in each.

    The point of the layout is that finding a track tells you where its stems
    and its formula are, so the tests are mostly about the name being the key
    and being the same in all three.
    """

    def test_the_name_carries_date_mood_seed_and_fingerprint(self):
        name = library.track_name("melancholy", 249984309, "bb6f655e1199bda6",
                                  when=date(2026, 8, 4))
        self.assertEqual(name, "2026-08-04_melancholy_seed249984309_bb6f655e")

    def test_a_mood_with_no_safe_name_still_produces_a_writable_path(self):
        name = library.track_name("some/odd name", 7, "abcdef0123456789",
                                  when=date(2026, 8, 4))
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)

    def test_it_files_the_track_its_formula_and_its_stems_under_one_name(self):
        import tempfile
        from unittest.mock import patch

        result = make_track(Knobs(seed=771, duration_s=20.0, ghost_gain=0.45),
                            THEMES["wistful"])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch("compex.library.TRACKS_DIR", root / "Tracks"), \
                 patch("compex.library.STEMS_DIR", root / "Stems"), \
                 patch("compex.library.TRACK_META_DIR", root / "TrackMeta"):
                written = library.save(result, "wav", stems=True,
                                       report_json={"movements": []})
            name = written["name"]
            for tree in ("Tracks", "Stems", "TrackMeta"):
                self.assertTrue((root / tree / name).is_dir(), tree)
            self.assertTrue((root / "Tracks" / name / f"{name}.wav").exists())
            self.assertTrue((root / "TrackMeta" / name / "formula.tex").exists())
            self.assertTrue((root / "TrackMeta" / name / "report.json").exists())
            self.assertTrue(written["stems"])

    def test_stems_are_optional_because_they_are_the_expensive_part(self):
        import tempfile
        from unittest.mock import patch

        result = make_track(Knobs(seed=771, duration_s=20.0), THEMES["wistful"])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch("compex.library.TRACKS_DIR", root / "Tracks"), \
                 patch("compex.library.STEMS_DIR", root / "Stems"), \
                 patch("compex.library.TRACK_META_DIR", root / "TrackMeta"):
                written = library.save(result, "wav", stems=False)
            self.assertEqual(written["stems"], [])
            self.assertFalse((root / "Stems" / written["name"]).exists())

    def test_the_report_written_beside_a_track_is_the_whole_report(self):
        result = make_track(Knobs(seed=771, duration_s=20.0), THEMES["wistful"])
        sections = report.everything(result, 0.35)
        for expected in ("movements", "plan", "ledger", "ghosts", "mix", "tuning"):
            self.assertIn(expected, sections)


class SaveEndpointTests(unittest.TestCase):
    """One button files all three artifacts; Make no longer files anything.

    The split matters: composing and filing used to be the same action, so
    every experiment left a folder behind. Now Make is a preview and Save is a
    decision.
    """

    def setUp(self):
        from compex.ui import server

        self.server = server
        server._LAST.clear()

    def _handler(self):
        from compex.ui.server import CompexHandler

        return CompexHandler.__new__(CompexHandler)

    def test_saving_before_making_refuses_rather_than_guessing(self):
        with self.assertRaises(ValueError) as caught:
            self.server.CompexHandler._save(self._handler(), {"format": "wav"})
        self.assertIn("make a track first", str(caught.exception))

    def test_an_unknown_format_is_refused(self):
        with self.assertRaises(ValueError):
            self.server.CompexHandler._save(self._handler(), {"format": "flac"})

    def test_save_files_the_track_its_meta_and_every_stem(self):
        import tempfile
        from unittest.mock import patch

        result = make_track(Knobs(seed=42, duration_s=15.0, ghost_gain=0.4),
                            THEMES["serene"])
        self.server._LAST["result"] = result
        self.server._LAST["knobs"] = Knobs(seed=42, duration_s=15.0, ghost_gain=0.4)

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch("compex.library.TRACKS_DIR", root / "Tracks"), \
                 patch("compex.library.STEMS_DIR", root / "Stems"), \
                 patch("compex.library.TRACK_META_DIR", root / "TrackMeta"):
                out = self.server.CompexHandler._save(self._handler(), {"format": "wav"})

            self.assertTrue(out["ok"])
            name = out["name"]
            self.assertTrue((root / "Tracks" / name / f"{name}.wav").exists())
            self.assertTrue((root / "TrackMeta" / name / "formula.tex").exists())
            self.assertTrue((root / "TrackMeta" / name / "report.json").exists())
            self.assertGreater(len(out["stems"]), 1)
            self.assertTrue(any("ghosts" in one for one in out["stems"]))
