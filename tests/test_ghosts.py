"""The ghost layer: is the machine's uncertainty actually audible?

This is the thing the project is named for, so the tests are about the claim
rather than about the plumbing. A ghost has to be *quieter than what beat it*,
*louder when the decision was close*, and *absent when the composer was sure* —
otherwise it is not uncertainty made audible, it is just a second part.

And the switch has to be real: at zero the engine must produce exactly the
music it produced before ghosts existed, or the whole idea is unfalsifiable.
"""

import unittest

import numpy as np

from compex.config import Knobs
from compex.dsp.arrange import render_composition
from compex.generate import THEMES, compose
from compex.generate.melody import Ghost, Phrase
from compex.generate.write import GHOST_MARK
from compex.pipeline import make_track

PHRASE = Phrase(steps=(0, 2, 1, 4), rhythm=(1.0, 0.5, 0.5, 1.0), origin="test")


class ClosenessTests(unittest.TestCase):
    def test_a_near_miss_is_heard_and_a_rout_is_not(self):
        near = Ghost(PHRASE, score=0.60, margin=0.01)
        beaten = Ghost(PHRASE, score=0.20, margin=0.41)
        self.assertGreater(near.closeness(), 0.9)
        self.assertLess(beaten.closeness(), 0.1)

    def test_closeness_falls_off_with_the_margin(self):
        margins = [0.0, 0.05, 0.1, 0.3]
        levels = [Ghost(PHRASE, 0.5, margin).closeness() for margin in margins]
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_a_tie_is_heard_at_full_strength(self):
        self.assertAlmostEqual(Ghost(PHRASE, 0.5, 0.0).closeness(), 1.0)


class KeepingTests(unittest.TestCase):
    def test_the_losers_are_kept_not_just_counted(self):
        piece = compose(7788, 90.0, THEMES["hypnotic"])
        haunted = [choice for choice in piece.melodies if choice.ghosts]
        self.assertTrue(haunted, "every audition threw its losers away")
        for choice in haunted:
            for ghost in choice.ghosts:
                self.assertLessEqual(ghost.score, choice.score)
                self.assertGreaterEqual(ghost.margin, 0.0)

    def test_only_the_nearest_few_are_kept(self):
        from compex.generate.melody import GHOSTS_KEPT

        piece = compose(7788, 120.0, THEMES["frantic"])
        for choice in piece.melodies:
            self.assertLessEqual(len(choice.ghosts), GHOSTS_KEPT)

    def test_ghost_notes_are_marked_as_ghosts(self):
        piece = compose(7788, 90.0, THEMES["hypnotic"])
        self.assertTrue(piece.ghosts)
        for note in piece.ghosts:
            self.assertIn(GHOST_MARK, note.voice)

    def test_ghosts_are_quieter_than_what_beat_them(self):
        piece = compose(7788, 90.0, THEMES["hypnotic"])
        lead = max((note.velocity for note in piece.notes
                    if GHOST_MARK not in note.voice), default=0.0)
        self.assertGreater(lead, max(note.velocity for note in piece.ghosts))

    def test_the_critic_never_hears_them(self):
        """They are texture. Judging the piece on lines it decided against would
        make every measurement an average of what it did and did not do."""
        piece = compose(7788, 90.0, THEMES["hypnotic"])
        for note in piece.notes:
            self.assertNotIn(GHOST_MARK, note.voice)


class RenderTests(unittest.TestCase):
    def test_silence_at_zero_is_the_engine_without_ghosts(self):
        piece = compose(2026, 45.0, THEMES["menacing"])
        without = render_composition(piece, 0.89, None, 0.0)
        stripped = compose(2026, 45.0, THEMES["menacing"])
        object.__setattr__(stripped, "ghosts", ())
        self.assertTrue(np.array_equal(without, render_composition(stripped, 0.89, None, 0.6)))

    def test_turning_them_up_changes_the_audio(self):
        piece = compose(2026, 45.0, THEMES["menacing"])
        quiet = render_composition(piece, 0.89, None, 0.0)
        loud = render_composition(piece, 0.89, None, 0.6)
        self.assertFalse(np.array_equal(quiet, loud))

    def test_nothing_clips(self):
        for gain in (0.0, 0.35, 1.0):
            samples = render_composition(compose(2026, 45.0, THEMES["frantic"]),
                                         0.89, None, gain)
            self.assertLessEqual(float(np.max(np.abs(samples))), 1.0)

    def test_an_out_of_range_gain_is_refused(self):
        piece = compose(2026, 30.0, THEMES["serene"])
        with self.assertRaises(ValueError):
            render_composition(piece, 0.89, None, 1.5)

    def test_the_knob_reaches_the_render(self):
        off = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.0), THEMES["hypnotic"])
        on = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.5), THEMES["hypnotic"])
        self.assertNotEqual(off.fingerprint, on.fingerprint)

    def test_it_stays_deterministic(self):
        first = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.4), THEMES["hypnotic"])
        second = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.4), THEMES["hypnotic"])
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_the_phone_hears_the_same_ghosts_as_the_desktop(self):
        from compex.web import Session

        session = Session(2026, 45.0, "hypnotic", 8.0, 0.89, 0.4)
        piece = compose(2026, 45.0, THEMES["hypnotic"])
        self.assertEqual(session.composition.ghosts, piece.ghosts)
        self.assertEqual(session.ghost_gain, 0.4)


class FormulaTests(unittest.TestCase):
    def test_the_formula_says_what_it_nearly_played(self):
        result = make_track(Knobs(seed=7788, duration_s=60.0, ghost_gain=0.4),
                            THEMES["hypnotic"])
        self.assertIn("GHOSTS", result.formula)
        self.assertIn("Gamma", result.formula.replace("\\", ""))


if __name__ == "__main__":
    unittest.main()
