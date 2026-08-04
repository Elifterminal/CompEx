"""The ghost layer: is the machine's uncertainty actually audible?

This is the thing the project is named for, so the tests are about the claim
rather than about the plumbing. A ghost has to be *quieter than what beat it*,
*louder when the decision was close*, and *absent when the composer was sure* —
otherwise it is not uncertainty made audible, it is just a second part.

And the switch has to be real: at zero the engine must produce exactly the
music it produced before ghosts existed, or the whole idea is unfalsifiable.
"""

import unittest
from unittest.mock import patch

import numpy as np

from compex.config import Knobs
from compex.dsp.arrange import render_composition
from compex.generate import THEMES, compose
from compex.generate.melody import Ghost, Phrase
from compex.generate.pattern import GHOSTS_KEPT as PATTERN_GHOSTS_KEPT
from compex.generate.pattern import Ghost as PatternGhost
from compex.generate.pattern import Pattern
from compex.generate.write import GHOST_MARK, _ghost_strokes
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
        """A piece with nothing turned down renders the same at any gain.

        Both kinds have to be stripped. When the rhythm ghosts landed this test
        failed with only the melodic ones removed, which is exactly what it is
        for — the gain knob has to reach *everything* the mechanism adds, or
        "off" quietly stops meaning off.
        """
        piece = compose(2026, 45.0, THEMES["menacing"])
        without = render_composition(piece, 0.89, None, 0.0)
        stripped = compose(2026, 45.0, THEMES["menacing"])
        object.__setattr__(stripped, "ghosts", ())
        object.__setattr__(stripped, "ghost_strokes", ())
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


class RhythmGhostTests(unittest.TestCase):
    """The kit's runners-up, which needed a different rule from the melodic ones.

    A rejected phrase is a different line. A rejected groove is mostly the
    *same* groove — two patterns drawn off one grid agree about the downbeat
    and argue about the rest — so playing a loser whole would re-strike hits
    that are already sounding. Only the disagreement is a ghost, and these
    tests are mostly about that distinction holding.
    """

    def test_only_the_slots_where_it_disagreed_are_kept(self):
        winner = Pattern(voice="hat", slots=(1.0, 0.0, 0.5, 0.0),
                         subdivision=0.5, origin="w")
        loser = Pattern(voice="hat", slots=(1.0, 0.7, 0.0, 0.6),
                        subdivision=0.5, origin="l")
        ghost = PatternGhost(pattern=loser, score=0.4, margin=0.05)
        self.assertEqual(ghost.instead_of(winner), (1, 3))

    def test_a_loser_identical_to_the_winner_leaves_nothing_behind(self):
        same = Pattern(voice="kick", slots=(1.0, 0.0, 1.0, 0.0),
                       subdivision=0.5, origin="same")
        self.assertEqual(PatternGhost(same, 0.4, 0.01).instead_of(same), ())

    def test_a_loser_on_a_different_grid_is_kept_whole(self):
        """Nothing lines up, so nothing is a doubling."""
        winner = Pattern(voice="hat", slots=(1.0, 0.0), subdivision=0.5, origin="w")
        loser = Pattern(voice="hat", slots=(1.0, 0.8, 0.4), subdivision=0.3333, origin="l")
        self.assertEqual(PatternGhost(loser, 0.4, 0.02).instead_of(winner), (0, 1, 2))

    def test_more_than_half_of_a_rejected_groove_is_usually_a_doubling(self):
        """The number that justifies the rule. If it were near zero, keeping
        only the difference would be pointless complexity."""
        piece = compose(2026, 120.0, THEMES["menacing"])
        kept = sum(len(g.instead_of(c.pattern)) for c in piece.patterns for g in c.ghosts)
        whole = sum(g.pattern.hit_count for c in piece.patterns for g in c.ghosts)
        self.assertGreater(whole, 0)
        self.assertLess(kept / whole, 0.75, "the difference rule is buying nothing")

    def test_the_kit_keeps_its_losers(self):
        piece = compose(2026, 120.0, THEMES["menacing"])
        haunted = [choice for choice in piece.patterns if choice.ghosts]
        self.assertTrue(haunted)
        for choice in haunted:
            self.assertLessEqual(len(choice.ghosts), PATTERN_GHOSTS_KEPT)
            for ghost in choice.ghosts:
                self.assertGreaterEqual(ghost.margin, 0.0)
                self.assertLessEqual(ghost.score, choice.score)

    def test_a_ghost_stroke_is_quieter_than_the_hit_that_beat_it(self):
        piece = compose(2026, 120.0, THEMES["menacing"])
        self.assertTrue(piece.ghost_strokes)
        loudest_ghost = max(stroke.velocity for stroke in piece.ghost_strokes)
        loudest_real = max(stroke.velocity for stroke in piece.strokes)
        self.assertLess(loudest_ghost, loudest_real)

    def test_every_ghost_stroke_names_a_drum_that_exists(self):
        piece = compose(2026, 120.0, THEMES["frantic"])
        kit = {voice.voice_id for voice in piece.voices if voice.role == "perc"}
        for stroke in piece.ghost_strokes:
            self.assertIn(GHOST_MARK, stroke.voice)
            self.assertIn(stroke.voice.replace(GHOST_MARK, ""), kit)

    def test_they_are_audible(self):
        """Silencing them has to change the file, or none of this is real."""
        loud = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.6),
                          THEMES["menacing"]).samples
        with patch("compex.generate.write.GHOST_STROKE_TRIM", 0.0):
            without = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.6),
                                 THEMES["menacing"]).samples
        self.assertFalse(np.array_equal(loud, without))

    def test_the_switch_still_switches_everything_off(self):
        """At zero the engine must write what it wrote before rhythm ghosts."""
        with patch("compex.generate.write.GHOST_STROKE_TRIM", 0.0):
            bare = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.0),
                              THEMES["menacing"]).samples
        with_them = make_track(Knobs(seed=2026, duration_s=45.0, ghost_gain=0.0),
                               THEMES["menacing"]).samples
        np.testing.assert_array_equal(bare, with_them)

    def test_the_shadow_is_never_louder_than_the_kit_casting_it(self):
        """The anti-jazz rule, and the reason it is enforced by level and not
        by count: a shadow busier than what was played stops reading as doubt
        and starts reading as a second drummer."""
        for theme in ("serene", "menacing", "frantic", "weightless"):
            piece = compose(5150, 120.0, THEMES[theme])
            shadow = sum(stroke.velocity for stroke in piece.ghost_strokes)
            kit = sum(stroke.velocity for stroke in piece.strokes)
            self.assertLess(shadow, kit * 0.5, theme)

    def test_a_groove_that_wanted_the_whole_bar_is_held_down_for_it(self):
        """Two ghosts equally close, one of them far busier: the busy one is
        heard more quietly per hit, not louder for having more of them."""
        winner = Pattern(voice="hat", slots=(1.0, 0.0, 0.0, 0.0),
                         subdivision=0.5, origin="w")
        modest = Pattern(voice="hat", slots=(1.0, 1.0, 0.0, 0.0),
                         subdivision=0.5, origin="m")
        greedy = Pattern(voice="hat", slots=(1.0, 1.0, 1.0, 1.0),
                         subdivision=0.5, origin="g")
        movement = compose(11, 30.0, THEMES["serene"]).movements[0]
        quiet = _ghost_strokes(winner, (PatternGhost(modest, 0.5, 0.01),),
                               "hat", 0.0, 8.0, movement)
        loud = _ghost_strokes(winner, (PatternGhost(greedy, 0.5, 0.01),),
                              "hat", 0.0, 8.0, movement)
        self.assertGreater(max(s.velocity for s in quiet),
                           max(s.velocity for s in loud))

    def test_they_do_not_touch_what_was_actually_played(self):
        piece = compose(2026, 90.0, THEMES["hypnotic"])
        for stroke in piece.strokes:
            self.assertNotIn(GHOST_MARK, stroke.voice)
