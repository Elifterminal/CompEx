"""The rhythms: derived from the meter, chosen by judgement, taught by their own output.

Two claims are on trial here. That the metrical hierarchy falls out of the
arithmetic rather than out of a table — including for meters nobody typed out
— and that a grid which keeps choosing a slot ends up believing in that slot
more than the hierarchy ever did.
"""

import unittest
from unittest.mock import patch

from compex.generate import THEMES, compose
from compex.generate.critic import Analysis, judge
from compex.generate.evolve import Drives
from compex.generate.pattern import (
    CRITERIA,
    Bed,
    Grid,
    Pattern,
    assess,
    build_grid,
    in_bar,
    initial_groove,
    invent,
    learn,
    metre_weights,
    retune,
)

CALM = Drives(density_bias=1.0, unrest=0.2, plasticity=0.5)


def a_bed(**changes) -> Bed:
    base = dict(energy=0.5, tension=0.4, drives=CALM, bar_beats=4.0)
    base.update(changes)
    return Bed(**base)


class HierarchyTests(unittest.TestCase):
    def test_the_downbeat_is_the_heaviest_slot(self):
        for meter in (3, 4, 5, 6, 7):
            weights = metre_weights(meter, 0.5)
            self.assertEqual(max(weights), weights[0], f"{meter}/4")

    def test_the_grid_is_as_long_as_the_bar(self):
        self.assertEqual(len(metre_weights(4, 0.25)), 16)
        self.assertEqual(len(metre_weights(7, 0.5)), 14)
        self.assertEqual(len(metre_weights(3, 1.0)), 3)

    def test_beats_outweigh_the_gaps_between_them(self):
        weights = metre_weights(4, 0.25)
        on_beat = [weights[index] for index in range(0, 16, 4)]
        off_beat = [weights[index] for index in range(16) if index % 4]
        self.assertGreater(min(on_beat), max(off_beat))

    def test_odd_meters_are_derived_not_tabulated(self):
        """Nothing lists 7/4. It still has to come out sensible."""
        weights = metre_weights(7, 1.0)
        self.assertEqual(len(weights), 7)
        self.assertTrue(all(weight > 0 for weight in weights))

    def test_a_bright_voice_gets_a_finer_grid_than_a_dark_one(self):
        bright = build_grid("hat", 5, 0, 4, THEMES["hypnotic"], 0.9, 0.3)
        dark = build_grid("kick", 5, 1, 4, THEMES["hypnotic"], 0.2, 0.5)
        self.assertLessEqual(bright.subdivision, dark.subdivision)

    def test_a_new_grid_starts_where_it_started(self):
        grid = build_grid("kick", 5, 0, 4, THEMES["solemn"], 0.25, 0.45)
        self.assertEqual(grid.weights, grid.start)
        self.assertEqual(grid.strayed(), 0.0)
        self.assertEqual(grid.past_start(), 0)


class PatternTests(unittest.TestCase):
    def test_hits_repeat_on_the_cycle_and_keep_their_velocity(self):
        figure = Pattern(voice="kick", slots=(0.9, 0.0, 0.5, 0.0),
                         subdivision=0.5, origin="test")
        hits = figure.hits(0.0, 4.0)
        self.assertEqual([beat for beat, _ in hits], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual([velocity for _, velocity in hits], [0.9, 0.5, 0.9, 0.5])

    def test_the_cycle_is_anchored_to_the_piece_not_to_the_question(self):
        """Asking about bar nine must not restart the pattern at bar nine."""
        figure = Pattern(voice="kick", slots=(0.9, 0.0, 0.0, 0.0),
                         subdivision=1.0, origin="test")
        self.assertEqual(figure.hits(4.0, 8.0), ((4.0, 0.9),))
        self.assertEqual(figure.hits(5.0, 9.0), ((8.0, 0.9),))

    def test_an_empty_pattern_is_rejected(self):
        with self.assertRaises(ValueError):
            Pattern(voice="kick", slots=(), subdivision=0.5, origin="test")


class JudgingTests(unittest.TestCase):
    def setUp(self):
        self.grid = build_grid("kick", 5, 0, 4, THEMES["hypnotic"], 0.25, 0.45)

    def test_every_criterion_is_scored_and_in_range(self):
        figure = Pattern(voice="kick", slots=tuple([0.8] + [0.0] * (len(self.grid.weights) - 1)),
                         subdivision=self.grid.subdivision, origin="test")
        scores = assess(figure, self.grid, a_bed())
        self.assertEqual(set(scores), set(CRITERIA))
        for name, value in scores.items():
            self.assertTrue(0.0 <= value <= 1.0, f"{name} = {value}")

    def test_a_metronome_reads_as_less_of_a_figure_than_a_figure(self):
        slots = len(self.grid.weights)
        even = Pattern(voice="kick", slots=tuple(0.8 if i % 2 == 0 else 0.0 for i in range(slots)),
                       subdivision=self.grid.subdivision, origin="test")
        shaped = list(even.slots)
        if len(shaped) > 3:
            shaped[3] = 0.6
        figure = Pattern(voice="kick", slots=tuple(shaped),
                         subdivision=self.grid.subdivision, origin="test")
        self.assertLess(assess(even, self.grid, a_bed())["figure"],
                        assess(figure, self.grid, a_bed())["figure"])

    def test_colliding_with_what_is_already_playing_scores_worse(self):
        slots = len(self.grid.weights)
        figure = Pattern(voice="hat", slots=tuple(0.8 if i % 2 == 0 else 0.0 for i in range(slots)),
                         subdivision=self.grid.subdivision, origin="test")
        taken = frozenset(in_bar(index * self.grid.subdivision, 4.0)
                          for index in range(0, slots, 2))
        crowded = assess(figure, self.grid, a_bed(claimed=taken))["interlock"]
        clear = assess(figure, self.grid, a_bed(claimed=frozenset({99.0})))["interlock"]
        self.assertLess(crowded, clear)

    def test_a_silent_bar_holds_no_pulse(self):
        figure = Pattern(voice="kick", slots=tuple([0.0] * len(self.grid.weights)),
                         subdivision=self.grid.subdivision, origin="test")
        self.assertEqual(assess(figure, self.grid, a_bed())["pulse"], 0.0)


class InventingTests(unittest.TestCase):
    def setUp(self):
        self.grid = build_grid("kick", 5, 0, 4, THEMES["restless"], 0.25, 0.5)
        self.groove = initial_groove(THEMES["restless"])

    def test_the_winner_is_the_highest_scoring_candidate(self):
        choice = invent(3, 0, 0, self.grid, self.groove, a_bed())
        for _, score in choice.rejected:
            self.assertGreaterEqual(choice.score, score)

    def test_inventing_is_deterministic(self):
        first = invent(3, 0, 0, self.grid, self.groove, a_bed())
        second = invent(3, 0, 0, self.grid, self.groove, a_bed())
        self.assertEqual(first.pattern, second.pattern)

    def test_the_pattern_it_had_is_always_on_the_table(self):
        """Standing pat has to be an option, or a groove can never survive a movement."""
        held = Pattern(voice="kick", slots=tuple([0.9] + [0.0] * (len(self.grid.weights) - 1)),
                       subdivision=self.grid.subdivision, origin="held")
        choice = invent(3, 1, 0, self.grid, self.groove, a_bed(previous=held))
        origins = {choice.pattern.origin} | {origin for origin, _ in choice.rejected}
        self.assertIn("held", origins)

    def test_a_pattern_never_comes_back_silent(self):
        for seed in range(12):
            choice = invent(seed, 0, 0, self.grid, self.groove, a_bed(energy=0.05))
            self.assertGreater(choice.pattern.hit_count, 0, seed)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.grid = build_grid("kick", 5, 0, 4, THEMES["hypnotic"], 0.25, 0.45)

    def test_a_slot_it_keeps_playing_ends_up_heavier_than_the_meter_made_it(self):
        played = Pattern(voice="kick",
                         slots=tuple(0.9 if index in (1, 5) else 0.0
                                     for index in range(len(self.grid.weights))),
                         subdivision=self.grid.subdivision, origin="test")
        grid = self.grid
        for _ in range(8):
            grid = learn(grid, played, plasticity=0.6)
        self.assertGreater(grid.weights[1], self.grid.weights[1])
        self.assertGreater(grid.past_start(), 0,
                           "no slot ever passed the weight the hierarchy gave it")

    def test_a_slot_it_keeps_skipping_fades(self):
        played = Pattern(voice="kick",
                         slots=tuple(0.9 if index == 0 else 0.0
                                     for index in range(len(self.grid.weights))),
                         subdivision=self.grid.subdivision, origin="test")
        grid = learn(learn(self.grid, played, 0.6), played, 0.6)
        self.assertLess(grid.weights[2], self.grid.weights[2])

    def test_learning_is_a_switch_that_can_be_turned_off(self):
        played = Pattern(voice="kick", slots=tuple([0.9] * len(self.grid.weights)),
                         subdivision=self.grid.subdivision, origin="test")
        with patch("compex.generate.pattern.LEARN_RATE", 0.0):
            frozen = learn(self.grid, played, 1.0)
        self.assertEqual(frozen.weights, self.grid.weights)

    def test_weights_never_pass_the_hard_ceiling(self):
        from compex.generate.pattern import WEIGHT_CEILING

        played = Pattern(voice="kick", slots=tuple([1.0] * len(self.grid.weights)),
                         subdivision=self.grid.subdivision, origin="test")
        grid = self.grid
        for _ in range(60):
            grid = learn(grid, played, 1.0)
        self.assertLessEqual(max(grid.weights), WEIGHT_CEILING)

    def test_a_density_complaint_makes_fill_matter_more(self):
        groove = initial_groove(THEMES["solemn"])
        thin = Analysis(200, 100, 0.45, 0.45, 0.7, 1.0, 12.0, 0.3, 0.02, 0.5)
        moved, shifts = retune(groove, judge(thin, THEMES["frantic"]), Drives(plasticity=1.0))
        self.assertGreater(moved.fill, groove.fill)
        self.assertTrue(shifts)


class PieceTests(unittest.TestCase):
    def test_a_piece_records_the_patterns_it_decided_on(self):
        piece = compose(4242, 120.0, THEMES["hypnotic"])
        self.assertTrue(piece.patterns)
        for choice in piece.patterns:
            self.assertGreater(choice.pattern.hit_count, 0)
            self.assertTrue(choice.note())

    def test_the_bass_gets_a_pattern_like_the_drums_do(self):
        piece = compose(4242, 120.0, THEMES["hypnotic"])
        voices = {choice.voice for choice in piece.patterns}
        bass = next(v for v in piece.voices if v.role == "bass")
        self.assertIn(bass.voice_id, voices)

    def test_the_grids_move_over_a_long_piece(self):
        piece = compose(4242, 240.0, THEMES["frantic"])
        self.assertTrue(any(grid.strayed() > 0.01 for grid in piece.grids),
                        "no grid learned anything from what it played")

    def test_the_formula_carries_the_patterns(self):
        from compex.config import Knobs
        from compex.pipeline import make_track

        result = make_track(Knobs(seed=4242, duration_s=90.0), THEMES["hypnotic"])
        self.assertIn("P_{", result.formula)


if __name__ == "__main__":
    unittest.main()
