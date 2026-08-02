"""The audition: does it judge honestly, choose consistently, and learn?

The property that matters here is the same one that matters for the listening
loop — a criterion has to be able to change what gets played. A taste weight
that never flips a decision is decoration, and decoration that claims to be
deciding is worse than nothing.
"""

import unittest
from unittest.mock import patch

from compex.generate import THEMES, compose, melody
from compex.generate.critic import PRINCIPLES, Analysis, judge
from compex.generate.evolve import Drives, initial_drives
from compex.generate.melody import (
    CRITERIA,
    TAUGHT_BY,
    Phrase,
    Setting,
    Taste,
    assess,
    choose,
    initial_taste,
    retune,
)
from compex.generate.theory import SCALES, Motif, make_motif

SCALE = SCALES["ionian"]
MOTIF = Motif(steps=(0, 2, 1, 4), rhythm=(1.0, 0.5, 0.5, 1.0))


def a_setting(**changes) -> Setting:
    base = dict(
        scale=SCALE, chord_degree=0, chord_size=3, span_beats=4.0,
        energy=0.5, tension=0.4, drives=initial_drives(THEMES["wistful"], 48),
        germ=(2, -1, 3),
    )
    base.update(changes)
    return Setting(**base)


class PhraseTests(unittest.TestCase):
    def test_steps_and_rhythm_stay_paired(self):
        with self.assertRaises(ValueError):
            Phrase(steps=(0, 1), rhythm=(1.0,), origin="test")

    def test_an_empty_phrase_is_rejected(self):
        with self.assertRaises(ValueError):
            Phrase(steps=(), rhythm=(), origin="test")


class JudgingTests(unittest.TestCase):
    def test_every_criterion_is_scored_and_in_range(self):
        phrase = Phrase(steps=(0, 2, 1, 4), rhythm=(1.0, 0.5, 0.5, 1.0), origin="test")
        scores = assess(phrase, a_setting())
        self.assertEqual(set(scores), set(CRITERIA))
        for name, value in scores.items():
            self.assertTrue(0.0 <= value <= 1.0, f"{name} = {value}")

    def test_an_unanswered_leap_scores_worse_than_an_answered_one(self):
        """Huron's post-skip reversal, applied before anyone has to hear it."""
        setting = a_setting()
        climbing = Phrase(steps=(0, 5, 10, 15), rhythm=(1.0,) * 4, origin="test")
        answered = Phrase(steps=(0, 5, 4, 3), rhythm=(1.0,) * 4, origin="test")
        self.assertLess(assess(climbing, setting)["answer"],
                        assess(answered, setting)["answer"])

    def test_landing_on_a_chord_tone_cadences_better_than_landing_off_one(self):
        setting = a_setting()
        lands = Phrase(steps=(0, 1, 3, 2), rhythm=(0.5, 0.5, 0.5, 2.0), origin="test")
        hangs = Phrase(steps=(0, 1, 3, 1), rhythm=(0.5, 0.5, 0.5, 2.0), origin="test")
        self.assertGreater(assess(lands, setting)["cadence"],
                           assess(hangs, setting)["cadence"])

    def test_a_phrase_that_overruns_its_slot_loses_gait(self):
        short = a_setting(span_beats=2.0)
        overrunning = Phrase(steps=(0, 1, 2, 3), rhythm=(2.0, 2.0, 2.0, 2.0), origin="test")
        fitting = Phrase(steps=(0, 1, 2, 3), rhythm=(0.5, 0.5, 0.5, 0.5), origin="test")
        self.assertLess(assess(overrunning, short)["gait"], assess(fitting, short)["gait"])

    def test_freshness_is_measured_against_what_was_already_played(self):
        moves = Phrase(steps=(0, 2, 4, 6), rhythm=(1.0,) * 4, origin="test")
        stale = a_setting(heard=frozenset({(2, 2)}))
        virgin = a_setting(heard=frozenset({(-3, 5)}))
        self.assertNotEqual(assess(moves, stale)["freshness"],
                            assess(moves, virgin)["freshness"])


class ChoosingTests(unittest.TestCase):
    def test_the_winner_is_the_highest_scoring_candidate(self):
        setting = a_setting()
        taste = initial_taste(THEMES["restless"])
        choice = choose(11, 1, 0, setting, taste, None, MOTIF)
        for _, score in choice.rejected:
            self.assertGreaterEqual(choice.score, score)

    def test_choosing_is_deterministic(self):
        setting = a_setting()
        taste = initial_taste(THEMES["restless"])
        first = choose(11, 1, 0, setting, taste, None, MOTIF)
        second = choose(11, 1, 0, setting, taste, None, MOTIF)
        self.assertEqual(first.chosen, second.chosen)
        self.assertEqual(first.score, second.score)

    def test_taste_can_change_which_line_wins(self):
        """If no weighting ever flips a decision, the weights are decoration.

        Scanned rather than asserted on one audition: most fields contain a
        candidate that wins on every criterion, and no weighting saves the
        others. The claim is only that taste decides when the field is close,
        so the honest test is how often it decides across many fields.
        """
        parent = Phrase(steps=(0, 4, 8, 3), rhythm=(0.5, 0.5, 1.0, 1.0), origin="seed")
        careful = Taste(line=3.0, answer=3.0, cadence=2.0, freshness=0.05, span=0.05)
        reckless = Taste(line=0.05, answer=0.05, cadence=0.05, freshness=3.0, span=3.0)

        flips = 0
        for seed in range(30):
            setting = a_setting(chord_degree=seed % 7, heard=frozenset({(2, -1), (-1, 3)}))
            flips += (choose(seed, 1, 0, setting, careful, parent, MOTIF).chosen
                      != choose(seed, 1, 0, setting, reckless, parent, MOTIF).chosen)
        self.assertGreaterEqual(flips, 3, f"taste flipped only {flips} of 30 auditions")

    def test_an_unsettled_composer_imagines_more_lines(self):
        setting = a_setting(drives=Drives(unrest=0.0))
        restless = a_setting(drives=Drives(unrest=1.0))
        taste = Taste(curiosity=0.2)
        self.assertLess(choose(11, 1, 0, setting, taste, None, MOTIF).considered,
                        choose(11, 1, 0, restless, taste, None, MOTIF).considered)

    def test_the_audition_is_a_switch_that_can_be_turned_off(self):
        with patch("compex.generate.melody.CANDIDATES_MIN", 1), \
             patch("compex.generate.melody.CANDIDATES_MAX", 1):
            choice = choose(11, 1, 0, a_setting(), Taste(curiosity=1.0), None, MOTIF)
        self.assertEqual(choice.considered, 1)
        self.assertEqual(choice.rejected, ())

    def test_a_mutation_stays_related_to_its_parent(self):
        """Mutations that rewrite the phrase would leave nothing to accumulate."""
        setting = a_setting()
        parent = Phrase(steps=(0, 2, 1, 4), rhythm=(1.0, 0.5, 0.5, 1.0), origin="seed")
        field = melody.propose(11, 1, 0, setting, Taste(), parent, MOTIF)
        children = [p for p in field if p.generation > parent.generation]
        self.assertTrue(children)
        for child in children:
            self.assertLessEqual(abs(len(child.steps) - len(parent.steps)),
                                 len(parent.steps), child.origin)


class LearningTests(unittest.TestCase):
    def test_every_taught_criterion_exists(self):
        for principle, taught in TAUGHT_BY.items():
            for criterion, polarity in taught:
                self.assertIn(criterion, CRITERIA, principle)
                self.assertIn(polarity, (-1, 1), principle)
        named = {p.name for p in PRINCIPLES}
        for principle in TAUGHT_BY:
            self.assertIn(principle, named)

    def test_a_failing_principle_makes_its_criterion_matter_more(self):
        taste = Taste()
        # Leaps never answered, motif abandoned: answer and kinship should climb.
        bad = Analysis(200, 100, 0.45, 0.45, 0.0, 1.0, 12.0, 0.3, 0.4, 0.0)
        moved, shifts = retune(taste, taste, judge(bad, THEMES["wistful"]), (),
                               Drives(plasticity=1.0))
        self.assertGreater(moved.answer, taste.answer)
        self.assertGreater(moved.kinship, taste.kinship)
        self.assertTrue(any(s.criterion == "answer" for s in shifts))

    def test_one_principle_can_teach_more_than_one_criterion(self):
        """Unanswered leaps want the phrase to answer them *and* the ledger to remember them."""
        taste = Taste()
        bad = Analysis(200, 100, 0.45, 0.45, 0.0, 1.0, 12.0, 0.3, 0.4, 0.5)
        moved, _ = retune(taste, taste, judge(bad, THEMES["wistful"]), (),
                          Drives(plasticity=1.0))
        self.assertGreater(moved.answer, taste.answer)
        self.assertGreater(moved.promise, taste.promise)

    def test_a_criterion_switched_off_cannot_be_revived_by_teaching(self):
        taste = Taste(promise=0.0)
        bad = Analysis(200, 100, 0.45, 0.45, 0.0, 1.0, 12.0, 0.3, 0.4, 0.5)
        moved, _ = retune(taste, taste, judge(bad, THEMES["wistful"]), (),
                          Drives(plasticity=1.0))
        self.assertEqual(moved.promise, 0.0)

    def test_weights_can_end_up_past_where_they_started(self):
        start = initial_taste(THEMES["serene"])
        taste = start
        bad = Analysis(200, 100, 0.45, 0.45, 0.0, 1.0, 12.0, 0.3, 0.4, 0.0)
        for _ in range(6):
            taste, _ = retune(taste, start, judge(bad, THEMES["serene"]), (),
                              Drives(plasticity=1.0))
        self.assertTrue(taste.strayed_from(start),
                        "the same complaint six times over moved nothing")

    def test_it_thinks_harder_when_unsettled(self):
        taste = Taste(curiosity=0.2)
        calm, _ = retune(taste, taste, (), (), Drives(unrest=0.0))
        rattled, _ = retune(taste, taste, (), (), Drives(unrest=1.0))
        self.assertLess(calm.curiosity, rattled.curiosity)


class PieceTests(unittest.TestCase):
    def test_a_piece_auditions_its_melodies_and_says_so(self):
        piece = compose(4242, 120.0, THEMES["restless"])
        self.assertTrue(piece.melodies)
        for choice in piece.melodies:
            self.assertGreaterEqual(choice.considered, 1)
            self.assertTrue(choice.note())

    def test_the_melody_has_a_lineage(self):
        """Later phrases descend from earlier ones rather than restarting."""
        piece = compose(4242, 180.0, THEMES["restless"])
        deepest = max(choice.chosen.generation for choice in piece.melodies)
        self.assertGreater(deepest, 1, "no phrase ever descended from another")

    def test_the_formula_carries_the_auditions(self):
        from compex.config import Knobs
        from compex.pipeline import make_track

        result = make_track(Knobs(seed=4242, duration_s=90.0), THEMES["restless"])
        self.assertIn("MELODY", result.formula)
        self.assertIn("Psi", result.formula.replace("\\", ""))

    def test_choosing_does_not_break_determinism(self):
        self.assertEqual(compose(4242, 90.0, THEMES["restless"]),
                         compose(4242, 90.0, THEMES["restless"]))

    def test_every_theme_still_writes_playable_notes(self):
        for name, mood in THEMES.items():
            piece = compose(31, 60.0, mood)
            self.assertTrue(all(note.duration > 0 for note in piece.notes), name)
            self.assertTrue(all(0.0 < note.velocity <= 1.0 for note in piece.notes), name)


if __name__ == "__main__":
    unittest.main()
