"""The listening loop: does it measure honestly, decide sensibly, and stay deterministic.

The property that matters most here is that **a drive can actually move the
thing it is supposed to fix.** A loop where the correction cannot affect the
measurement looks alive in the logs and does nothing to the music.
"""

import unittest

from compex.generate import THEMES, compose
from compex.generate.critic import PRINCIPLES, Analysis, analyse, judge
from compex.generate.evolve import (
    BOUNDS,
    POLARITY,
    SENSITIVITY,
    Drives,
    initial_drives,
    respond,
)
from compex.generate.material import Note
from compex.generate.recall import compress, remember
from compex.generate.theory import SCALES, Motif, make_motif

LONG = 120.0


class WiringTests(unittest.TestCase):
    def test_every_principle_names_a_drive_that_exists(self):
        for principle in PRINCIPLES:
            self.assertIn(principle.drive, SENSITIVITY, principle.name)
            self.assertIn(principle.drive, BOUNDS, principle.name)

    def test_every_principle_has_a_polarity(self):
        for principle in PRINCIPLES:
            self.assertIn(principle.name, POLARITY, principle.name)

    def test_every_principle_reads_a_field_the_analysis_has(self):
        for principle in PRINCIPLES:
            self.assertTrue(hasattr(Analysis, "__dataclass_fields__"))
            self.assertIn(principle.name, Analysis.__dataclass_fields__, principle.name)

    def test_every_principle_carries_an_attribution_and_a_reason(self):
        for principle in PRINCIPLES:
            self.assertTrue(principle.attribution.strip(), principle.name)
            self.assertGreater(len(principle.why), 40, principle.name)


class CriticTests(unittest.TestCase):
    def test_too_little_material_applies_no_pressure(self):
        analysis = analyse([], [], make_motif(1, SCALES["ionian"], 0.5, 0.5),
                           SCALES["ionian"], 48, 8.0, frozenset())
        for verdict in judge(analysis, THEMES["serene"]):
            self.assertTrue(verdict.satisfied,
                            f"{verdict.principle.name} complained about an empty piece")

    def test_novelty_is_neutral_with_nothing_to_compare_against(self):
        """Reporting maximum novelty because you have no memory is an artefact."""
        notes = [Note(start=float(i), duration=1.0, pitch=60.0 + i % 3,
                      velocity=0.5, voice="lead_x") for i in range(8)]
        analysis = analyse(notes, [], make_motif(1, SCALES["ionian"], 0.5, 0.5),
                           SCALES["ionian"], 48, 8.0, frozenset({"lead_x"}))
        self.assertLess(analysis.novelty, 1.0)

    def test_a_repeating_line_reads_as_repetitive(self):
        pattern = [0, 2, 4, 2]
        notes = [Note(start=float(i), duration=1.0, pitch=60.0 + pattern[i % 4],
                      velocity=0.5, voice="lead_x") for i in range(40)]
        analysis = analyse(notes, [], make_motif(1, SCALES["ionian"], 0.5, 0.5),
                           SCALES["ionian"], 60, 40.0, frozenset({"lead_x"}))
        self.assertGreater(analysis.repetition, 0.6)
        self.assertLess(analysis.novelty, 0.4)

    def test_a_line_that_only_leaps_upward_fails_post_skip_reversal(self):
        notes = [Note(start=float(i), duration=1.0, pitch=48.0 + i * 7,
                      velocity=0.5, voice="lead_x") for i in range(10)]
        analysis = analyse(notes, [], make_motif(1, SCALES["ionian"], 0.5, 0.5),
                           SCALES["ionian"], 48, 10.0, frozenset({"lead_x"}))
        self.assertEqual(analysis.post_skip_reversal, 0.0)

    def test_bands_move_with_the_mood(self):
        """A menacing piece is allowed a dissonance a serene one would fail on."""
        analysis = Analysis(80, 40, 0.45, 0.45, 0.7, 1.0, 12.0, 0.62, 0.4, 0.5)
        serene = {v.principle.name: v for v in judge(analysis, THEMES["serene"])}
        menacing = {v.principle.name: v for v in judge(analysis, THEMES["menacing"])}
        self.assertFalse(serene["dissonance"].satisfied)
        self.assertTrue(menacing["dissonance"].satisfied)


class DecidingTests(unittest.TestCase):
    def test_drives_stay_inside_their_bounds_under_extreme_complaints(self):
        drives = initial_drives(THEMES["frantic"], 48)
        analysis = Analysis(200, 100, 1.0, 0.0, 0.0, 40.0, 1.0, 1.0, 1.4, 0.0)
        for _ in range(40):
            drives, _ = respond(drives, judge(analysis, THEMES["frantic"]), analysis)
        for name, (low, high) in BOUNDS.items():
            value = getattr(drives, name)
            self.assertGreaterEqual(value, low, name)
            self.assertLessEqual(value, high, name)

    def test_satisfied_music_settles_the_composer_down(self):
        drives = initial_drives(THEMES["serene"], 48)
        calm = Analysis(80, 40, 0.45, 0.45, 0.7, 1.0, 12.0, 0.2, 0.35, 0.5)
        start = drives.plasticity
        for _ in range(5):
            drives, adjustments = respond(drives, judge(calm, THEMES["serene"]), calm)
            self.assertEqual(adjustments, ())
        self.assertLess(drives.plasticity, start)

    def test_persistent_unhappiness_makes_it_react_harder(self):
        drives = initial_drives(THEMES["serene"], 48)
        awful = Analysis(200, 100, 1.0, 0.0, 0.0, 30.0, 1.0, 1.0, 1.4, 0.0)
        start = drives.plasticity
        for _ in range(4):
            drives, _ = respond(drives, judge(awful, THEMES["serene"]), awful)
        self.assertGreater(drives.plasticity, start)

    def test_adjustments_say_which_principle_caused_them(self):
        drives = initial_drives(THEMES["serene"], 48)
        awful = Analysis(200, 100, 1.0, 0.0, 0.0, 30.0, 1.0, 1.0, 1.4, 0.0)
        _, adjustments = respond(drives, judge(awful, THEMES["serene"]), awful)
        self.assertTrue(adjustments)
        for adjustment in adjustments:
            self.assertTrue(adjustment.principle)
            self.assertTrue(adjustment.attribution)
            self.assertNotEqual(adjustment.before, adjustment.after)

    def test_fatigue_decays_and_accumulates(self):
        drives = Drives().with_use("invert")
        self.assertGreater(drives.tired_of("invert"), 0.9)
        drives = drives.with_use("retrograde")
        self.assertLess(drives.tired_of("invert"), 0.9)
        self.assertGreater(drives.tired_of("retrograde"), 0.9)


class RecallTests(unittest.TestCase):
    def test_the_sketch_throws_information_away(self):
        motif = make_motif(11, SCALES["dorian"], 0.6, 0.4)
        sketch = compress(motif)
        self.assertLess(sketch.fidelity_against(motif), 1.0)

    def test_recall_differs_from_the_original(self):
        """A remembered theme is a reconstruction, not a copy."""
        motif = Motif(steps=(0, 3, 5, 2, -1), rhythm=(1.0, 0.5, 0.5, 2.0, 1.0))
        sketch = compress(motif)
        drives = initial_drives(THEMES["menacing"], 48)
        recalled = remember(sketch, 7, 1, drives)
        self.assertNotEqual(recalled.steps, motif.steps)

    def test_recall_keeps_the_contour(self):
        motif = Motif(steps=(0, 3, 5, 2, -1), rhythm=(1.0, 0.5, 0.5, 2.0, 1.0))
        recalled = remember(compress(motif), 7, 1, initial_drives(THEMES["serene"], 48))
        original = [b - a for a, b in zip(motif.steps, motif.steps[1:])]
        rebuilt = [b - a for a, b in zip(recalled.steps, recalled.steps[1:])]
        for was, now in zip(original, rebuilt):
            if was and now:
                self.assertEqual(was > 0, now > 0, "the direction of a step changed")

    def test_who_is_remembering_changes_what_is_remembered(self):
        motif = Motif(steps=(0, 4, 7, 3, -2), rhythm=(1.0, 1.0, 0.5, 0.5, 2.0))
        sketch = compress(motif)
        narrow = remember(sketch, 7, 1, Drives(register_reach=0.05, gap_fill=0.0))
        wide = remember(sketch, 7, 1, Drives(register_reach=1.0, gap_fill=0.0))
        self.assertNotEqual(narrow.steps, wide.steps)

    def test_recall_is_deterministic(self):
        sketch = compress(make_motif(3, SCALES["aeolian"], 0.5, 0.5))
        drives = initial_drives(THEMES["hypnotic"], 48)
        self.assertEqual(remember(sketch, 5, 2, drives), remember(sketch, 5, 2, drives))


class LoopTests(unittest.TestCase):
    def test_a_piece_records_its_own_evolution(self):
        piece = compose(7788, LONG, THEMES["menacing"])
        self.assertGreater(len(piece.evolution), 1)
        for step in piece.evolution:
            self.assertEqual(len(step.verdicts), len(PRINCIPLES))
            self.assertTrue(step.note())

    def test_evolving_does_not_break_determinism(self):
        self.assertEqual(compose(7788, LONG, THEMES["frantic"]),
                         compose(7788, LONG, THEMES["frantic"]))

    def test_the_drives_actually_end_up_different(self):
        piece = compose(7788, LONG, THEMES["menacing"])
        start = piece.evolution[0].before
        self.assertNotEqual(start, piece.final_drives)

    def test_a_correction_can_move_the_thing_it_measures(self):
        """The loop must be able to close, or it is theatre.

        Narrow range is the clearest case: the reach drive should widen the
        line until register_spread climbs out of the failing region.
        """
        piece = compose(7788, 180.0, THEMES["serene"])
        spreads = [s.analysis.register_spread for s in piece.evolution]
        complained_early = any(s < 7.0 for s in spreads[:2])
        recovered = any(s >= 7.0 for s in spreads[2:])
        self.assertTrue(complained_early and recovered,
                        f"reach never answered the complaint: {spreads}")

    def test_every_theme_survives_the_loop(self):
        for name, mood in THEMES.items():
            piece = compose(4242, 90.0, mood)
            self.assertGreater(len(piece.notes), 0, name)
            self.assertTrue(all(0.0 < n.velocity <= 1.0 for n in piece.notes), name)

    def test_the_formula_carries_the_evolution(self):
        from compex.config import Knobs
        from compex.pipeline import make_track

        result = make_track(Knobs(seed=7788, duration_s=90.0), THEMES["menacing"])
        self.assertIn("EVOLUTION", result.formula)
        self.assertIn("Theta", result.formula.replace("\\", ""))


if __name__ == "__main__":
    unittest.main()
