"""Tunings and clocks: is the machine actually free of the grid, and still coherent?

Two capabilities that only a machine has. Twelve notes to the octave is a fact
about keyboards, and a single shared tempo is a fact about how many pulses a
person can hold at once. Neither is a fact about this engine.

The tests are about the difference between *differently tuned* and *out of
tune*, and between polytempo and drift: a tuning has to be steppable and
derived from something, and two clocks have to meet again.
"""

import unittest
from unittest.mock import patch

from compex.generate import THEMES, clocks, compose, tuning
from compex.generate.clocks import Clock
from compex.generate.palette import ROLE_BASS, ROLE_LEAD, ROLE_PAD, build_kit, build_voice
from compex.generate.tuning import MAX_GAP, MIN_STEP, OCTAVE, Tuning, roughness_at


class TuningTests(unittest.TestCase):
    def test_a_tuning_starts_on_its_tonic_and_ascends(self):
        with self.assertRaises(ValueError):
            Tuning(name="x", family="just", steps=(1.0, 2.0))
        with self.assertRaises(ValueError):
            Tuning(name="x", family="just", steps=(0.0, 3.0, 2.0))

    def test_every_family_is_reachable(self):
        seen = {tuning.derive(seed, mood).family
                for seed in range(40) for mood in THEMES.values()}
        self.assertEqual(seen, {"twelve", "equal", "just"})

    def test_a_serene_piece_mostly_stays_where_the_ear_is_comfortable(self):
        """Strangeness is a decision, and a calm piece should rarely make it."""
        families = [tuning.derive(seed, THEMES["serene"]).family for seed in range(40)]
        self.assertGreater(families.count("twelve") / len(families), 0.7)

    def test_a_shattered_piece_usually_leaves_it(self):
        families = [tuning.derive(seed, THEMES["shattered"]).family for seed in range(40)]
        self.assertGreater(1 - families.count("twelve") / len(families), 0.5)

    def test_no_two_degrees_are_the_same_degree(self):
        for seed in range(30):
            for mood in THEMES.values():
                steps = tuning.derive(seed, mood).steps
                for low, high in zip(steps, steps[1:]):
                    self.assertGreaterEqual(high - low, MIN_STEP - 1e-9)

    def test_no_hole_too_wide_to_step_through(self):
        """A scale with a fourth of an octave missing is a chord, not a scale."""
        for seed in range(30):
            for mood in THEMES.values():
                steps = tuning.derive(seed, mood).steps
                gaps = [high - low for low, high in zip(steps, list(steps[1:]) + [OCTAVE])]
                self.assertLess(max(gaps), MAX_GAP + MIN_STEP + 1e-6,
                                f"{tuning.derive(seed, mood).name}: {steps}")

    def test_just_intervals_really_are_the_ratios_it_names(self):
        found = None
        for seed in range(60):
            candidate = tuning.just(seed, THEMES["shattered"])
            if "3/2" in candidate.detail:
                found = candidate
                break
        self.assertIsNotNone(found, "a fifth never turned up in sixty just tunings")
        self.assertTrue(any(abs(step - 7.019550) < 1e-4 for step in found.steps),
                        f"3/2 is 701.955 cents and is not in {found.cents()}")

    def test_an_equal_tuning_is_actually_equal(self):
        found = tuning.equal(3, THEMES["menacing"])
        divisions = int(found.name.split("-")[0])
        for step in found.steps:
            position = step / (OCTAVE / divisions)
            self.assertAlmostEqual(position, round(position), places=6)

    def test_roughness_is_interpolated_not_rounded(self):
        """A neutral third lands between the minor and the major one."""
        minor, neutral, major = roughness_at(3.0), roughness_at(3.5), roughness_at(4.0)
        self.assertTrue(min(minor, major) <= neutral <= max(minor, major))
        self.assertNotEqual(neutral, minor)

    def test_deriving_is_deterministic(self):
        self.assertEqual(tuning.derive(11, THEMES["anxious"]),
                         tuning.derive(11, THEMES["anxious"]))


class ClockTests(unittest.TestCase):
    def test_a_ratio_meets_the_pulse_again(self):
        """Rational, not arbitrary — irrational ratios never come back."""
        self.assertEqual(Clock("v", 3, 2).meets_every(4.0), 8.0)
        self.assertEqual(Clock("v", 1, 1).meets_every(4.0), 4.0)

    def test_the_floor_stays_on_the_pulse(self):
        """You can only hear three-against-two if something is being the two."""
        mood = THEMES["shattered"]
        voices = [build_voice(7, 0, ROLE_BASS, mood), build_voice(7, 1, ROLE_LEAD, mood)]
        kit = build_kit(7, mood)
        assigned = clocks.assign(7, voices, kit, mood)
        self.assertTrue(assigned[voices[0].voice_id].anchored)
        for drum in kit:
            if drum.voice_id in clocks.ANCHORS:
                self.assertTrue(assigned[drum.voice_id].anchored, drum.voice_id)

    def test_a_calm_piece_keeps_one_clock_more_often_than_a_wild_one(self):
        def loose_share(theme: str) -> float:
            mood = THEMES[theme]
            total = off = 0
            for seed in range(25):
                voices = [build_voice(seed, 0, ROLE_BASS, mood),
                          build_voice(seed, 1, ROLE_LEAD, mood),
                          build_voice(seed, 2, ROLE_PAD, mood)]
                assigned = clocks.assign(seed, voices, build_kit(seed, mood), mood)
                total += len(assigned)
                off += len(clocks.loose(assigned))
            return off / total

        self.assertLess(loose_share("serene"), loose_share("shattered"))

    def test_the_mechanism_can_be_switched_off(self):
        mood = THEMES["shattered"]
        voices = [build_voice(7, 1, ROLE_LEAD, mood)]
        with patch("compex.generate.clocks.ENABLED", False):
            assigned = clocks.assign(7, voices, build_kit(7, mood), mood)
        self.assertEqual(clocks.loose(assigned), ())


class PieceTests(unittest.TestCase):
    def test_a_piece_records_what_it_is_tuned_to(self):
        piece = compose(2026, 60.0, THEMES["shattered"])
        self.assertEqual(piece.scale, piece.tuning.steps)
        self.assertIn(piece.tuning.family, ("twelve", "equal", "just"))

    def test_a_microtonal_piece_really_plays_off_the_grid(self):
        for seed in range(40):
            piece = compose(seed, 45.0, THEMES["shattered"])
            if piece.tuning.family == "twelve":
                continue
            off = [note.pitch for note in piece.notes
                   if abs(note.pitch - round(note.pitch)) > 0.05]
            self.assertTrue(off, "a non-twelve tuning produced only twelve-tone pitches")
            return
        self.fail("no piece left twelve in forty seeds")

    def test_voices_on_their_own_clock_land_off_the_shared_grid(self):
        for seed in range(30):
            piece = compose(seed, 90.0, THEMES["frantic"])
            loose = {clock.voice for clock in piece.clocks if not clock.anchored}
            if not loose:
                continue
            starts = [note.start for note in piece.notes if note.voice in loose]
            self.assertTrue(any(abs(start - round(start)) > 0.01 for start in starts),
                            "a voice at 3:2 played only on the pulse")
            return
        self.fail("no voice ever left the pulse in thirty seeds")

    def test_it_stays_deterministic(self):
        self.assertEqual(compose(2026, 60.0, THEMES["shattered"]),
                         compose(2026, 60.0, THEMES["shattered"]))

    def test_the_formula_carries_both(self):
        from compex.config import Knobs
        from compex.pipeline import make_track

        result = make_track(Knobs(seed=2026, duration_s=60.0), THEMES["shattered"])
        self.assertIn("cents", result.formula)
        self.assertIn("CLOCKS", result.formula)

    def test_every_theme_still_renders(self):
        for name, mood in THEMES.items():
            piece = compose(31, 45.0, mood)
            self.assertTrue(piece.notes, name)
            self.assertTrue(all(20.0 <= note.pitch <= 108.0 for note in piece.notes), name)


if __name__ == "__main__":
    unittest.main()
