"""What the piece is trying to do to itself.

Everything else in the engine reacts to something. This is the only part that
wants something in advance, and it is therefore the part most likely to be
theatre — a plan is very easy to *state* and very hard to *keep*. So the
load-bearing test in this file is the one that switches the plan off, composes
the same pieces again, and checks that the state tracked its target shape
better with the plan on. Two earlier versions of this mechanism failed exactly
that test and were deleted rather than shipped.

The rest of the tests guard the three properties that made the third version
work: the levers are constraints rather than preferences, the plan watches
before it steers, and the score it reports about itself cannot be won by
standing still.
"""

import unittest
from unittest.mock import patch

from compex.generate import THEMES, compose
from compex.generate.intent import (
    BIAS_BOUNDS,
    INTENTS,
    LEVERS,
    SPAN,
    VARIABLES,
    Curve,
    Intent,
    Plan,
    choose,
    place,
    read,
    reconsider,
)
from compex.generate.melody import CANDIDATES_MAX, CANDIDATES_MIN, Taste, _how_many
from compex.generate.promise import Ledger, Promise
from compex.generate.write import spread


def run(plan: Plan, values: list[dict], seed: int = 7) -> Plan:
    """Feed a plan a run of readings, as a piece would."""
    for step, row in enumerate(values):
        position = (step + 1) / (len(values) + 1)
        plan = reconsider(plan, read(plan, position, **row), seed, step)
    return plan


def flat(count: int = 8) -> list[dict]:
    return [{"pressure": 3.0, "margin": 0.05, "crowding": 0.2} for _ in range(count)]


def rising(count: int = 8) -> list[dict]:
    return [{"pressure": step, "margin": 0.01 * step, "crowding": 0.05 * step}
            for step in range(count)]


class ShapeTests(unittest.TestCase):
    """Curves are shapes in the piece's own units, so every point is 0..1."""

    def test_every_target_is_a_position_in_a_range(self):
        for intent in INTENTS:
            for curve in intent.curves:
                for point in curve.points:
                    self.assertTrue(0.0 <= point <= 1.0, f"{intent.name}/{curve.variable}")

    def test_every_intent_aims_at_everything_it_can_reach(self):
        for intent in INTENTS:
            self.assertEqual({curve.variable for curve in intent.curves}, set(VARIABLES),
                             intent.name)

    def test_a_curve_interpolates_between_its_points(self):
        curve = Curve("pressure", (0.0, 1.0, 0.0))
        self.assertAlmostEqual(curve.at(0.0), 0.0)
        self.assertAlmostEqual(curve.at(0.5), 1.0)
        self.assertAlmostEqual(curve.at(1.0), 0.0)
        self.assertAlmostEqual(curve.at(0.25), 0.5)

    def test_a_position_outside_the_piece_is_clamped(self):
        curve = Curve("margin", (0.2, 0.8))
        self.assertAlmostEqual(curve.at(-1.0), 0.2)
        self.assertAlmostEqual(curve.at(9.0), 0.8)

    def test_it_asks_for_nothing_it_cannot_reach(self):
        """A curve naming a variable with no lever is a wish, not a plan."""
        for intent in INTENTS:
            for curve in intent.curves:
                self.assertIn(curve.variable, LEVERS, f"{intent.name} aims at nothing")


class HoldingTests(unittest.TestCase):
    """The levers trace the shape; they do not chase the measurement."""

    def test_the_levers_are_held_from_the_first_listen_back(self):
        """Waiting to be sure was the previous design, and it cost the piece
        three of its eight chances to do anything."""
        plan = Plan(intent=INTENTS[1])
        plan = reconsider(plan, read(plan, 0.05, **rising(3)[0]), 7, 0)
        self.assertTrue(plan.bias)

    def test_a_lever_runs_opposite_to_the_thing_it_holds(self):
        wants_it_high = Intent("high", "", (Curve("crowding", (1.0,)),
                                            Curve("pressure", (1.0,)),
                                            Curve("margin", (1.0,))))
        wants_it_low = Intent("low", "", (Curve("crowding", (0.0,)),
                                          Curve("pressure", (0.0,)),
                                          Curve("margin", (0.0,))))
        high = dict(run(Plan(intent=wants_it_high), flat(4)).bias)
        low = dict(run(Plan(intent=wants_it_low), flat(4)).bias)
        for lever in LEVERS.values():
            self.assertLess(high[lever], low[lever], lever)

    def test_the_lever_traces_the_curve_it_was_given(self):
        climbing = Intent("climb", "", tuple(Curve(name, (0.0, 1.0))
                                             for name in VARIABLES))
        held = []
        plan = Plan(intent=climbing)
        for step in range(6):
            plan = reconsider(plan, read(plan, step / 5, **flat(6)[step]), 7, step)
            held.append(plan.lever("spread"))
        self.assertEqual(held, sorted(held, reverse=True))

    def test_the_range_widens_and_never_narrows(self):
        plan = run(Plan(intent=INTENTS[1]), rising(9))
        widened = dict(plan.calibrated())
        self.assertIn("pressure", widened)
        floor, ceiling = widened["pressure"]
        self.assertLessEqual(floor, 0.0)
        self.assertGreaterEqual(ceiling, 8.0)

    def test_a_quantity_that_starts_flat_is_picked_up_when_it_moves(self):
        """The bug this caught: pressure opening at zero switched off the
        strongest lever in the file for the whole piece, silently."""
        readings = flat(3) + [{"pressure": 6.0, "margin": 0.05, "crowding": 0.2}] * 4
        for index, row in enumerate(readings):
            row["pressure"] = 0.0 if index < 3 else 6.0
        plan = run(Plan(intent=INTENTS[1]), readings)
        self.assertIn("pressure", plan.calibrated())

    def test_the_levers_stay_inside_their_bounds(self):
        low, high = BIAS_BOUNDS
        for values in (rising(9), flat(9), rising(9)[::-1]):
            plan = run(Plan(intent=INTENTS[2]), values)
            for name, value in plan.bias:
                self.assertTrue(low <= value <= high, f"{name}={value}")


class AuthorityTests(unittest.TestCase):
    """An intention is a constraint. Each lever must actually move its variable.

    These are the tests that the previous version of this file could not pass:
    its levers were taste weights, and a taste weight competing with fourteen
    others and then retuned by the critic has no authority over anything.
    """

    def test_the_plan_can_widen_and_narrow_the_field_of_an_audition(self):
        taste, drives = Taste(), compose(11, 30.0, THEMES["hypnotic"]).evolution[0].after
        wide = _how_many(taste, drives, 2.5)
        narrow = _how_many(taste, drives, 0.2)
        self.assertGreater(wide, narrow)
        for push in (0.0, 0.5, 1.0, 3.0):
            self.assertTrue(CANDIDATES_MIN <= _how_many(taste, drives, push) <= CANDIDATES_MAX)

    def test_the_plan_can_push_the_voices_apart(self):
        voices = compose(11, 30.0, THEMES["menacing"]).voices[:4]
        close = spread(voices, 0.4)
        far = spread(voices, 2.0)
        self.assertGreater(max(far.values()) - min(far.values()),
                           max(close.values()) - min(close.values()))

    def test_letting_go_is_not_settling(self):
        promise = Promise(kind="distance", domain="harmony", strength=1.0, opened_at=0.0)
        ledger = Ledger(open=(promise,))
        gone = ledger.let_go(10.0, 1)
        self.assertEqual(gone.open, ())
        self.assertEqual(gone.paid, ())              # nothing was answered
        self.assertEqual(gone.given_up, (promise,))

    def test_it_lets_go_of_what_it_was_carrying_hardest_first(self):
        """Dropping a debt nobody was hearing costs the piece nothing."""
        deep = Promise(kind="distance", domain="harmony", strength=1.0, opened_at=0.0)
        slight = Promise(kind="leading", domain="melody", strength=0.2, opened_at=0.0)
        gone = Ledger(open=(slight, deep)).let_go(40.0, 1)
        self.assertEqual(gone.given_up, (deep,))
        self.assertEqual(gone.open, (slight,))

    def test_letting_go_of_nothing_changes_nothing(self):
        ledger = Ledger(open=(Promise(kind="leading", domain="melody",
                                      strength=0.5, opened_at=0.0),))
        self.assertEqual(ledger.let_go(10.0, 0), ledger)
        self.assertEqual(Ledger().let_go(10.0, 3), Ledger())


class AgreementTests(unittest.TestCase):
    """The score a plan gives itself has to be one it can fail."""

    def test_a_piece_that_never_moved_scores_zero_rather_than_well(self):
        """Distance rewards a flat line for sitting in the middle of the
        target. Correlation does not, which is why it is the reported one."""
        plan = run(Plan(intent=INTENTS[1]), flat(8))
        self.assertEqual(plan.agreement(), 0.0)

    def test_following_the_shape_scores_high_and_fighting_it_scores_low(self):
        climbing = Intent("climb", "everything rises",
                          tuple(Curve(name, (0.0, 0.25, 0.5, 0.75, 1.0))
                                for name in VARIABLES))
        with_it = run(Plan(intent=climbing), rising(8))
        against_it = run(Plan(intent=climbing), rising(8)[::-1])
        self.assertGreater(with_it.agreement(), 0.9)
        self.assertLess(against_it.agreement(), -0.9)

    def test_it_reports_the_number_whatever_it_says(self):
        piece = compose(2026, 60.0, THEMES["menacing"])
        self.assertIn("followed to", piece.plan.summary())

    def test_where_a_value_sits_in_a_range(self):
        self.assertAlmostEqual(place(5.0, 0.0, 10.0), 0.5)
        self.assertAlmostEqual(place(-3.0, 0.0, 10.0), 0.0)
        self.assertAlmostEqual(place(30.0, 0.0, 10.0), 1.0)
        self.assertAlmostEqual(place(4.0, 4.0, 4.0), 0.5)      # a flat range has no top


class QuittingTests(unittest.TestCase):
    def test_an_intent_that_is_being_satisfied_trivially_is_given_up(self):
        """Sitting on the target while winning every decision by a mile is not
        succeeding — nothing is being decided."""
        easy = Intent("easy", "nothing at stake",
                      tuple(Curve(name, (0.5,)) for name in VARIABLES))
        middling = [{"pressure": 3.0, "margin": 0.9, "crowding": 0.2}] * 3
        moving = [{"pressure": p, "margin": 0.9, "crowding": 0.2} for p in (1.0, 5.0, 3.0)]
        plan = run(Plan(intent=easy), moving + middling * 3)
        self.assertTrue(plan.turns, "it never noticed it had stopped trying")
        self.assertNotEqual(plan.intent.name, "easy")

    def test_a_piece_working_hard_keeps_its_plan(self):
        plan = run(Plan(intent=INTENTS[2]), rising(9))
        self.assertFalse(plan.turns)


class WiringTests(unittest.TestCase):
    def test_the_mood_leans_the_choice_without_deciding_it(self):
        chosen = {choose(seed, THEMES["shattered"]).intent.name for seed in range(40)}
        self.assertGreater(len(chosen), 1)

    def test_determinism_survives(self):
        self.assertEqual(compose(2026, 60.0, THEMES["menacing"]),
                         compose(2026, 60.0, THEMES["menacing"]))

    def test_switching_it_off_is_a_real_off_switch(self):
        with patch("compex.generate.intent.ENABLED", False):
            without = compose(2026, 60.0, THEMES["menacing"])
        with patch("compex.generate.intent.GAIN", 0.0):
            neutral = compose(2026, 60.0, THEMES["menacing"])
        self.assertEqual(without.notes, neutral.notes)

    def test_a_piece_carries_its_plan_and_its_readings(self):
        piece = compose(2026, 120.0, THEMES["wistful"])
        self.assertTrue(piece.plan.readings)
        for reading in piece.plan.readings:
            self.assertEqual({name for name, _ in reading.values}, set(VARIABLES))

    def test_the_formula_says_what_the_piece_was_trying_to_do(self):
        from compex.config import Knobs
        from compex.pipeline import make_track

        result = make_track(Knobs(seed=2026, duration_s=90.0), THEMES["menacing"])
        self.assertIn("INTENT", result.formula)


class SteeringTest(unittest.TestCase):
    """The one that decides whether any of this is real.

    Same seeds, same moods, the plan's gain switched off and on. If the state
    does not track its target shape better with the plan working, the plan is
    decoration and belongs in the bin with the two versions before it.
    """

    def test_the_plan_measurably_steers_the_piece(self):
        cases = ((3131, "menacing"), (4747, "frantic"), (202, "hypnotic"), (11, "serene"))
        deltas = []
        for seed, theme in cases:
            scores = []
            for gain in (0.0, 0.6):
                with patch("compex.generate.intent.GAIN", gain):
                    scores.append(compose(seed, 300.0, THEMES[theme]).plan.agreement())
            deltas.append(scores[1] - scores[0])
        self.assertGreater(sum(deltas) / len(deltas), 0.0,
                           f"the plan did not steer: {deltas}")


if __name__ == "__main__":
    unittest.main()
