"""The mixer: does it measure what is audible, and does it fix what it finds?

The failure this exists to prevent is the one Lee heard — voices that are
technically present and inaudible, and a kit that outweighs everything else
because nobody counted. So the tests are about the two things a mixer has to
get right: knowing *where* a voice lives, and knowing whether anyone else is
already there.
"""

import unittest
from unittest.mock import patch

import numpy as np

from compex.config import SAMPLE_RATE
from compex.dsp import arrange, fx
from compex.dsp.balance import (
    BAND_NAMES,
    BANDS,
    FLOOR,
    MAX_LIFT,
    Presence,
    band_energy,
    weigh,
)
from compex.generate import THEMES, compose


def tone(freq: float, seconds: float = 1.0, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * freq * t)


class LoudnessTests(unittest.TestCase):
    def test_a_pad_and_a_pluck_end_up_comparable(self):
        """Peak matching made these differ by 4x in loudness. That was the bug."""
        held = tone(220.0, 2.0)
        plucked = tone(220.0, 2.0) * np.exp(-np.arange(2 * SAMPLE_RATE) / (0.15 * SAMPLE_RATE))

        held = fx.loudness_normalise(held, SAMPLE_RATE, 0.42)
        plucked = fx.loudness_normalise(plucked, SAMPLE_RATE, 0.42)
        ratio = fx.short_term_rms(held, SAMPLE_RATE) / fx.short_term_rms(plucked, SAMPLE_RATE)
        self.assertLess(ratio, 4.0, f"a held note is still {ratio:.1f}x the loudness of a pluck")

    def test_nothing_is_normalised_into_clipping(self):
        spiky = np.zeros(SAMPLE_RATE)
        spiky[0] = 1.0
        out = fx.loudness_normalise(spiky, SAMPLE_RATE, 0.42, ceiling=0.99)
        self.assertLessEqual(float(np.max(np.abs(out))), 0.99 + 1e-9)

    def test_silence_survives(self):
        quiet = np.zeros(1000)
        self.assertTrue(np.array_equal(fx.loudness_normalise(quiet, SAMPLE_RATE, 0.42), quiet))

    def test_the_loudest_window_is_what_gets_measured(self):
        signal = np.concatenate([tone(440.0, 0.5, 0.1), tone(440.0, 0.5, 1.0)])
        self.assertGreater(fx.short_term_rms(signal, SAMPLE_RATE),
                           float(np.sqrt(np.mean(signal ** 2))))


class KitTests(unittest.TestCase):
    def test_seven_drums_are_not_seven_times_one_drum(self):
        self.assertEqual(arrange.kit_trim(3), 1.0)
        self.assertLess(arrange.kit_trim(7), arrange.kit_trim(4))
        self.assertGreaterEqual(arrange.kit_trim(20), 0.45)

    def test_an_empty_kit_changes_nothing(self):
        self.assertEqual(arrange.kit_trim(0), 1.0)


class BandTests(unittest.TestCase):
    def test_a_low_tone_lands_in_a_low_band(self):
        energy = band_energy(tone(60.0, 0.5), SAMPLE_RATE)
        self.assertEqual(BAND_NAMES[int(np.argmax(energy))], "sub")

    def test_a_high_tone_lands_in_a_high_band(self):
        energy = band_energy(tone(6000.0, 0.5), SAMPLE_RATE)
        self.assertEqual(BAND_NAMES[int(np.argmax(energy))], "air")

    def test_something_too_short_to_analyse_reports_nothing(self):
        self.assertEqual(band_energy(np.zeros(4), SAMPLE_RATE), tuple(0.0 for _ in BANDS))


class WeighingTests(unittest.TestCase):
    def test_a_voice_lives_where_its_own_energy_is_not_where_it_has_a_share(self):
        """A bass owning 98% of an empty air band does not live in the air."""
        loud_low = Presence(voice="bass", role="bass",
                            energy=(100.0, 20.0, 1.0, 0.0, 0.5),
                            share=(0.9, 0.4, 0.05, 0.0, 0.98))
        self.assertEqual(BAND_NAMES[loud_low.home], "sub")
        self.assertEqual(loud_low.best_share, 0.9)

    def test_a_buried_voice_is_lifted(self):
        decided = weigh({
            "pad": ("pad", (0.0, 1.0, 1.0, 0.0, 0.0), 0.3),
            "wash": ("texture", (0.0, 200.0, 200.0, 0.0, 0.0), 0.3),
        }, {"pad": 1.0, "wash": 1.0})
        pad = next(v for v in decided.voices if v.voice == "pad")
        self.assertTrue(pad.buried)
        self.assertGreater(pad.trim, 1.0)

    def test_a_voice_that_owns_everything_is_held_back(self):
        decided = weigh({
            "hog": ("bass", (100.0, 0.0, 0.0, 0.0, 0.0), 0.9),
            "other": ("pad", (0.2, 0.0, 0.0, 0.0, 0.0), 0.3),
        }, {"hog": 1.0, "other": 1.0})
        hog = next(v for v in decided.voices if v.voice == "hog")
        self.assertLess(hog.trim, 1.0)

    def test_something_that_barely_plays_is_not_forced_forward(self):
        """Sparseness is a decision the composer made. The mixer does not overrule it."""
        occasional = Presence(voice="texture", role="texture",
                              energy=(0.0, 1.0, 0.0, 0.0, 0.0),
                              share=(0.0, 0.02, 0.0, 0.0, 0.0), presence=0.02)
        constant = Presence(voice="pad", role="pad",
                            energy=(0.0, 1.0, 0.0, 0.0, 0.0),
                            share=(0.0, 0.02, 0.0, 0.0, 0.0), presence=1.0)
        self.assertLess(occasional.floor(), constant.floor())

    def test_nothing_is_lifted_past_the_limit(self):
        decided = weigh({
            "ghost": ("lead", (1.0, 0.0, 0.0, 0.0, 0.0), 0.5),
            "wall": ("pad", (10_000.0, 0.0, 0.0, 0.0, 0.0), 0.5),
        }, {"ghost": 1.0, "wall": 1.0})
        ghost = next(v for v in decided.voices if v.voice == "ghost")
        self.assertLessEqual(ghost.trim, MAX_LIFT)
        self.assertTrue(ghost.still_buried,
                        "a voice this far down cannot be rescued by gain, and saying "
                        "otherwise would be the mixer taking credit for a problem it noticed")

    def test_the_stage_can_be_switched_off(self):
        with patch("compex.dsp.balance.ENABLED", False):
            decided = weigh({
                "pad": ("pad", (0.0, 1.0, 0.0, 0.0, 0.0), 0.3),
                "wash": ("texture", (0.0, 500.0, 0.0, 0.0, 0.0), 0.3),
            }, {"pad": 1.0, "wash": 1.0})
        self.assertTrue(all(voice.trim == 1.0 for voice in decided.voices))

    def test_an_empty_piece_decides_nothing(self):
        self.assertEqual(weigh({}).voices, ())


class PieceTests(unittest.TestCase):
    def test_a_real_piece_gets_a_mix_decision_per_voice(self):
        piece = compose(7788, 90.0, THEMES["hypnotic"])
        decided = arrange.survey(piece)
        sounding = {note.voice for note in piece.notes} | {s.voice for s in piece.strokes}
        self.assertEqual({voice.voice for voice in decided.voices}, sounding)

    def test_every_voice_lands_somewhere_and_is_trimmed_sanely(self):
        decided = arrange.survey(compose(7788, 90.0, THEMES["menacing"]))
        for voice in decided.voices:
            self.assertTrue(0.0 <= voice.best_share <= 1.0, voice.voice)
            self.assertTrue(0.3 <= voice.trim <= MAX_LIFT, f"{voice.voice} {voice.trim}")
            self.assertIn(voice.role, set(FLOOR) | {"perc"})

    def test_the_kit_no_longer_outweighs_everything_else(self):
        """Measured at 2.4x before any of this existed. That was the complaint."""
        for theme in ("hypnotic", "menacing", "frantic"):
            piece = compose(7788, 90.0, THEMES[theme])
            self.assertLess(arrange.survey(piece).percussive, 0.55, theme)

    def test_deciding_the_mix_is_deterministic(self):
        piece = compose(7788, 90.0, THEMES["frantic"])
        self.assertEqual(arrange.survey(piece).trims(), arrange.survey(piece).trims())

    def test_the_phone_and_the_desktop_use_the_same_trims(self):
        """They render by different paths. A mix that differs between them is a bug."""
        from compex.web import Session

        piece = compose(7788, 60.0, THEMES["hypnotic"])
        session = Session(7788, 60.0, "hypnotic")
        self.assertEqual(session.mix.trims(), arrange.survey(piece).trims())


if __name__ == "__main__":
    unittest.main()
