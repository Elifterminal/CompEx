"""The ledger: does it record honestly, ripen sensibly, and change the music?

Three claims are on trial. That an obligation is detected only when the music
actually left something hanging. That "too soon" has a number behind it — the
ordering *settling early < carrying < settling ripe* is the whole mechanism,
and if it inverts the composer will pay every debt the moment it opens. And
that the thing is falsifiable: switched off, the engine writes exactly what it
wrote before there was a ledger; switched on, the piece is different.
"""

import unittest
from unittest.mock import patch

from compex.generate import THEMES, compose
from compex.generate.melody import initial_taste
from compex.generate.promise import (
    FORGET_BEATS,
    RIPE,
    RIPEN_BEATS,
    Ledger,
    Promise,
    credit,
    fifth_distance,
    from_harmony,
    from_pattern,
    from_phrase,
    pattern_settles,
    would_settle,
)
from compex.generate.theory import SCALES

IONIAN = SCALES["ionian"]


def a_promise(kind="leap", domain="pitch", at=0.0, **detail) -> Promise:
    return Promise(kind=kind, domain=domain, strength=0.8, opened_at=at,
                   detail=tuple(sorted(detail.items())))


class RipeningTests(unittest.TestCase):
    def test_a_fresh_promise_is_worth_nothing_yet(self):
        self.assertEqual(a_promise().maturity(0.0), 0.0)

    def test_it_ripens_with_age(self):
        owed = a_promise()
        self.assertLess(owed.maturity(RIPEN_BEATS * 0.25), owed.maturity(RIPEN_BEATS * 0.75))
        self.assertEqual(owed.maturity(RIPEN_BEATS), 1.0)

    def test_it_fades_once_nobody_is_waiting_any_more(self):
        owed = a_promise()
        self.assertGreater(owed.maturity(FORGET_BEATS), owed.maturity(FORGET_BEATS * 4))

    def test_unripe_is_not_the_same_as_forgotten(self):
        """The bug this test exists for emptied the ledger on every write.

        A promise is worth nothing the instant it opens, so a filter on
        maturity alone deletes everything the moment it is written down.
        """
        fresh = a_promise()
        self.assertFalse(fresh.forgotten(0.0))
        self.assertFalse(fresh.forgotten(RIPEN_BEATS))
        self.assertTrue(fresh.forgotten(FORGET_BEATS * 10))

    def test_a_fresh_promise_stays_on_the_books(self):
        ledger = Ledger().opened((a_promise(at=12.0),))
        self.assertEqual(len(ledger.live(12.0)), 1)


class DetectionTests(unittest.TestCase):
    def test_an_unanswered_leap_opens_an_account(self):
        opened = from_phrase((0, 4, 5, 6), 0, IONIAN, 8.0)
        self.assertTrue(any(p.kind == "leap" for p in opened))

    def test_a_leap_the_phrase_answered_itself_does_not(self):
        """Paying for something you already did is the obvious way to get this wrong."""
        opened = from_phrase((0, 4, 3, 2), 0, IONIAN, 8.0)
        self.assertFalse(any(p.kind == "leap" for p in opened))

    def test_the_step_into_a_phrase_counts(self):
        """Joins are where the big unanswered leaps live."""
        without = from_phrase((0, 1, 2), 0, IONIAN, 8.0)
        withjoin = from_phrase((0, 1, 2), 0, IONIAN, 8.0, approach=-6)
        self.assertEqual(len(without), 0)
        self.assertTrue(any(p.kind == "leap" for p in withjoin))

    def test_a_line_stopping_under_the_tonic_owes_the_note_above(self):
        opened = from_phrase((0, 2, 6), 0, IONIAN, 4.0)   # degree 6 is the leading tone
        leading = [p for p in opened if p.kind == "leading"]
        self.assertTrue(leading)
        self.assertEqual(leading[0].get("direction"), 1.0)

    def test_a_pattern_that_never_lands_the_downbeat_owes_it(self):
        weights = (1.0, 0.2, 0.6, 0.2)
        self.assertTrue(from_pattern((0.0, 0.8, 0.0, 0.7), weights, 0.5, 0.0, "hat"))
        self.assertFalse(from_pattern((0.9, 0.8, 0.0, 0.7), weights, 0.5, 0.0, "hat"))

    def test_walking_far_from_home_owes_a_return(self):
        near = from_harmony(4, IONIAN, 0.0)      # one fifth away
        far = from_harmony(2, IONIAN, 0.0)       # two fifths away
        self.assertFalse(near)
        self.assertTrue(far)

    def test_distance_is_counted_in_fifths_not_scale_steps(self):
        """The third degree is next door on a keyboard and three moves round the circle."""
        self.assertEqual(fifth_distance(0, 7), 0)
        self.assertEqual(fifth_distance(4, 7), 1)     # the dominant is one fifth out
        self.assertEqual(fifth_distance(2, 7), 3)     # the mediant is a long way out
        self.assertGreater(fifth_distance(2, 7), fifth_distance(4, 7))

    def test_nothing_is_detected_with_the_mechanism_switched_off(self):
        with patch("compex.generate.promise.ENABLED", False):
            self.assertEqual(from_phrase((0, 4, 5, 6), 0, IONIAN, 8.0), ())
            self.assertEqual(from_harmony(2, IONIAN, 0.0), ())


class SettlingTests(unittest.TestCase):
    def test_a_step_back_the_other_way_settles_a_leap(self):
        owed = a_promise(degree=4.0, direction=-1.0)
        self.assertTrue(would_settle(owed, (4, 3, 2), 0))

    def test_continuing_in_the_same_direction_does_not(self):
        owed = a_promise(degree=4.0, direction=-1.0)
        self.assertFalse(would_settle(owed, (4, 5, 6), 0))

    def test_a_leading_tone_is_settled_by_the_note_it_points_at(self):
        owed = a_promise(kind="leading", degree=7.0, direction=1.0)
        self.assertTrue(would_settle(owed, (5, 7), 0))
        self.assertFalse(would_settle(owed, (5, 4), 0))

    def test_a_pattern_settles_a_metric_debt_by_landing_the_slot(self):
        owed = a_promise(kind="syncopation", domain="metre", slot=0.0, subdivision=0.5)
        self.assertTrue(pattern_settles(owed, (0.9, 0.0, 0.5, 0.0), 0.5))
        self.assertFalse(pattern_settles(owed, (0.0, 0.8, 0.0, 0.7), 0.5))

    def test_settling_early_scores_worse_than_carrying_scores_worse_than_settling_ripe(self):
        """The ordering *is* the mechanism. Invert it and every debt is paid on sight."""
        early = credit(RIPE * 0.4, settling=True, ripening=RIPE * 0.4)
        carrying = credit(0.0, settling=False, ripening=0.6)
        ripe = credit(0.9, settling=True, ripening=0.9)
        self.assertLess(early, carrying)
        self.assertLess(carrying, ripe)


class LedgerTests(unittest.TestCase):
    def test_settling_moves_a_promise_from_open_to_paid(self):
        owed = a_promise(at=0.0)
        ledger = Ledger().opened((owed,)).settle(owed, 40.0, "answered")
        self.assertEqual(ledger.open, ())
        self.assertEqual(len(ledger.paid), 1)
        self.assertEqual(ledger.paid[0].waited, 40.0)

    def test_settling_something_it_does_not_owe_changes_nothing(self):
        ledger = Ledger().opened((a_promise(at=0.0),))
        self.assertEqual(ledger.settle(a_promise(at=99.0), 10.0, "?"), ledger)

    def test_the_deepest_promise_is_the_one_it_is_carrying(self):
        old = a_promise(at=0.0)
        new = a_promise(at=100.0)
        ledger = Ledger().opened((old, new))
        self.assertEqual(ledger.deepest(110.0), old)

    def test_forgetting_drops_only_what_faded(self):
        old = a_promise(at=0.0)
        new = a_promise(at=FORGET_BEATS * 3.0)
        ledger = Ledger().opened((old, new)).forget(FORGET_BEATS * 3.5)
        self.assertNotIn(old, ledger.open)
        self.assertIn(new, ledger.open)


class PieceTests(unittest.TestCase):
    def test_a_long_piece_opens_and_settles_real_obligations(self):
        piece = compose(7788, 180.0, THEMES["menacing"])
        self.assertTrue(piece.ledger.paid, "nothing was ever owed or answered")
        self.assertTrue(any(s.waited > 8.0 for s in piece.ledger.paid),
                        "every debt was settled instantly — nothing was carried")

    def test_the_ledger_does_not_grow_without_bound(self):
        short = compose(7788, 60.0, THEMES["frantic"])
        long = compose(7788, 480.0, THEMES["frantic"])
        self.assertLess(len(long.ledger.open), len(short.ledger.open) + 40)

    def test_switching_it_off_empties_the_ledger_and_the_criterion(self):
        with patch("compex.generate.promise.ENABLED", False):
            piece = compose(7788, 120.0, THEMES["menacing"])
            self.assertEqual(piece.ledger.open, ())
            self.assertEqual(piece.ledger.paid, ())
            self.assertEqual(initial_taste(THEMES["menacing"]).promise, 0.0)

    def test_switching_it_on_changes_the_music(self):
        """If carrying obligations never changes a note, this is theatre."""
        with patch("compex.generate.promise.ENABLED", False):
            without = compose(7788, 120.0, THEMES["menacing"])
        with_ledger = compose(7788, 120.0, THEMES["menacing"])
        self.assertNotEqual(without.notes, with_ledger.notes)

    def test_it_stays_deterministic(self):
        self.assertEqual(compose(7788, 120.0, THEMES["menacing"]),
                         compose(7788, 120.0, THEMES["menacing"]))

    def test_the_formula_carries_the_ledger(self):
        from compex.config import Knobs
        from compex.pipeline import make_track

        result = make_track(Knobs(seed=7788, duration_s=120.0), THEMES["menacing"])
        self.assertIn("OWED", result.formula)


if __name__ == "__main__":
    unittest.main()
