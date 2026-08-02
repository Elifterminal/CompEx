"""Hindsight: does the piece make its own beginning cheaper to describe?

The claim under test is narrow and worth stating exactly. Describing the
opening using only what existed while it was written costs some number of
symbols. Describing it again with the rest of the piece available costs no
more, and sometimes less. The difference is what the ending gave back.

Two ways this goes wrong and both have tests here: pricing a reference as
free, which would make every piece look like a masterpiece of construction;
and letting the opening explain itself, which is a lineage rather than a
revelation and quietly saturates the measurement at zero headroom.
"""

import unittest
from unittest.mock import patch

from compex.generate import THEMES, compose
from compex.measure import (
    NOTE,
    Shape,
    baseline,
    cost,
    describe_in_order,
    describe_with_hindsight,
    explains,
    literal,
    match,
    reveal,
)

GERM = Shape("germ", steps=(0, 2, 1, 4), rhythm=(1.0, 0.5, 0.5, 1.0))


def shape(label, steps, rhythm=None, at=0.0) -> Shape:
    return Shape(label, tuple(steps), tuple(rhythm or [1.0] * len(steps)), at)


class MatchingTests(unittest.TestCase):
    def test_the_same_shape_transposed_is_one_fact(self):
        found = match(shape("a", (5, 7, 6, 9)), GERM)
        self.assertIsNotNone(found)
        name, price = found
        self.assertEqual(name, "transposed")
        self.assertLess(price, NOTE * 4)

    def test_inversion_and_retrograde_are_recognised(self):
        self.assertEqual(match(shape("a", (0, -2, -1, -4)), GERM)[0], "inverted")
        self.assertEqual(match(shape("a", (4, 1, 2, 0)), GERM)[0], "retrograde")

    def test_a_near_miss_costs_more_than_an_exact_one(self):
        exact = match(shape("a", (0, 2, 1, 4)), GERM)[1]
        wonky = match(shape("b", (0, 2, 1, 7)), GERM)[1]
        self.assertLess(exact, wonky)

    def test_something_unrelated_is_cheaper_to_spell_out(self):
        stranger = shape("x", (0, 6, -5, 6, -6))
        priced, source = cost(stranger, (GERM,))
        self.assertEqual(priced, stranger.literal())
        self.assertIsNone(source)

    def test_a_shape_cannot_be_a_reference_to_itself(self):
        priced, source = cost(GERM, (GERM,))
        self.assertEqual(priced, GERM.literal())
        self.assertIsNone(source)

    def test_two_notes_is_not_a_match_it_is_a_coincidence(self):
        self.assertIsNone(match(shape("a", (0, 2)), GERM))


class DescriptionTests(unittest.TestCase):
    def test_describing_in_order_cannot_look_forwards(self):
        """The first phrase cannot be explained by the fourth. That is not hindsight."""
        opening = (shape("one", (0, 3, 2, 5), at=0.0),
                   shape("two", (0, 3, 2, 5), at=4.0))
        causal = describe_in_order(opening, (GERM,))
        loose = sum(cost(s, (GERM,) + opening)[0] for s in opening)
        self.assertGreater(causal, loose)

    def test_hindsight_never_costs_more(self):
        opening = (shape("one", (0, 6, -3, 4), at=0.0),
                   shape("two", (0, 1, 5, -2), at=4.0))
        later = (shape("late", (0, 6, -3, 4), at=40.0),)
        self.assertLessEqual(describe_with_hindsight(opening, (GERM,), later),
                             describe_in_order(opening, (GERM,)))

    def test_a_late_return_makes_the_opening_cheaper(self):
        opening = (shape("one", (0, 6, -3, 4), at=0.0),)
        nothing = reveal(opening, (GERM,), ())
        returning = reveal(opening, (GERM,), (shape("late", (2, 8, -1, 6), at=40.0),))
        self.assertEqual(nothing.saved, 0.0)
        self.assertGreater(returning.saved, 0.0)
        self.assertGreater(returning.share, 0.0)

    def test_nothing_is_revealed_by_unrelated_material(self):
        opening = (shape("one", (0, 6, -3, 4), at=0.0),)
        unrelated = (shape("late", (0, 1, 1, 1, 1), at=40.0),)
        self.assertEqual(reveal(opening, (GERM,), unrelated).saved, 0.0)

    def test_the_share_is_against_spelling_it_all_out(self):
        opening = (shape("one", (0, 6, -3, 4), at=0.0),)
        measured = reveal(opening, (GERM,), (shape("late", (0, 6, -3, 4), at=40.0),))
        self.assertEqual(measured.literal, literal(opening))
        self.assertLessEqual(measured.share, 1.0)


class ExplainingTests(unittest.TestCase):
    def test_a_candidate_that_matches_the_past_explains_more_than_one_that_does_not(self):
        past = (shape("one", (0, 6, -3, 4), at=0.0), shape("two", (0, 6, -3, 4), at=8.0))
        paid = baseline(past, (GERM,))
        matching = shape("cand", (1, 7, -2, 5))
        stranger = shape("cand", (0, 1, 0, 1))
        self.assertGreater(explains(matching, past, paid), explains(stranger, past, paid))

    def test_a_phrase_that_was_already_cheap_leaves_nothing_to_buy(self):
        """Explaining something the piece had already explained is not a revelation."""
        past = (shape("one", (0, 2, 1, 4), (1.0, 0.5, 0.5, 1.0)),)   # the germ exactly
        paid = baseline(past, (GERM,))
        self.assertEqual(explains(shape("cand", (0, 2, 1, 4), (1.0, 0.5, 0.5, 1.0)),
                                  past, paid), 0.0)

    def test_explaining_nothing_scores_zero(self):
        self.assertEqual(explains(shape("cand", (0, 1, 2)), (), ()), 0.0)


class PieceTests(unittest.TestCase):
    def test_a_piece_measures_what_its_ending_explained(self):
        piece = compose(7788, 300.0, THEMES["hypnotic"])
        self.assertIsNotNone(piece.reveal)
        self.assertGreater(piece.reveal.phrases, 0)
        self.assertGreaterEqual(piece.reveal.now, 0.0)
        self.assertLessEqual(piece.reveal.now, piece.reveal.then)

    def test_it_reaches_back_for_specific_earlier_phrases(self):
        piece = compose(7788, 300.0, THEMES["hypnotic"])
        origins = {choice.chosen.origin for choice in piece.melodies}
        self.assertTrue(any(origin.startswith("recall") for origin in origins),
                        f"nothing was ever recalled: {sorted(origins)}")

    def test_caring_about_it_raises_it(self):
        """A criterion that cannot move its own measurement is decoration."""
        from dataclasses import replace

        from compex.generate import melody

        real = melody.initial_taste

        def forced(weight):
            return lambda mood: replace(real(mood), reveal=weight)

        indifferent, caring = [], []
        for theme in ("serene", "menacing", "frantic"):
            with patch("compex.generate.melody.initial_taste", forced(0.0)):
                indifferent.append(compose(7788, 300.0, THEMES[theme]).reveal.share)
            with patch("compex.generate.melody.initial_taste", forced(1.5)):
                caring.append(compose(7788, 300.0, THEMES[theme]).reveal.share)
        self.assertGreater(sum(caring), sum(indifferent),
                           f"caring changed nothing: {caring} vs {indifferent}")

    def test_the_formula_carries_the_measurement(self):
        from compex.config import Knobs
        from compex.pipeline import make_track

        result = make_track(Knobs(seed=7788, duration_s=180.0), THEMES["hypnotic"])
        self.assertIn("HINDSIGHT", result.formula)

    def test_it_stays_deterministic(self):
        self.assertEqual(compose(4242, 180.0, THEMES["wistful"]),
                         compose(4242, 180.0, THEMES["wistful"]))


if __name__ == "__main__":
    unittest.main()
