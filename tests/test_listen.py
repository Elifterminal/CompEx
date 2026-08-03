"""Hearing, drifting instruments, and scales it works out rather than looks up.

The listening tests are calibration tests, and they exist because calibration
caught two real errors: a unison scoring as rough as a semitone (leakage read
as twenty partials), and every low-register dissonance reading as smooth (the
analysis window too coarse to resolve a thirteen-hertz beat). Reference signals
with known answers are the only way to find that class of mistake — a piece of
music has no known answer to check against.
"""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from compex.dsp.listen import PROBE_RATE, Heard, listen, probe
from compex.generate import THEMES, compose, evolve, tuning
from compex.generate.critic import PRINCIPLES, judge
from compex.generate.palette import (
    DRIFT_CEILING,
    DRIFT_FLOOR,
    FIXED,
    ROLE_PAD,
    build_voice,
    drift,
)
from compex.generate.tuning import MAX_GAP, MIN_STEP, OCTAVE


def tone(freq: float, seconds: float = 2.0, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * PROBE_RATE)) / PROBE_RATE
    return amplitude * np.sin(2 * np.pi * freq * t)


def heard(signal: np.ndarray) -> Heard:
    return listen(signal / max(1e-9, float(np.max(np.abs(signal)))))


class RoughnessTests(unittest.TestCase):
    """Plomp and Levelt in one place: what beats, and what does not."""

    def test_a_unison_is_smooth(self):
        self.assertLess(heard(tone(220.0) + tone(220.0)).roughness, 0.02)

    def test_an_octave_and_a_fifth_are_smooth(self):
        self.assertLess(heard(tone(220.0) + tone(440.0)).roughness, 0.02)
        self.assertLess(heard(tone(220.0) + tone(330.0)).roughness, 0.02)

    def test_a_semitone_is_rough(self):
        self.assertGreater(heard(tone(220.0) + tone(233.0)).roughness, 0.15)

    def test_a_quarter_tone_is_rough_too(self):
        """The window has to be fine enough to see a six-hertz beat."""
        self.assertGreater(heard(tone(220.0) + tone(226.0)).roughness, 0.1)

    def test_a_cluster_is_rougher_than_a_dyad(self):
        dyad = heard(tone(220.0) + tone(233.0)).roughness
        cluster = heard(tone(220.0) + tone(226.0) + tone(233.0) + tone(240.0)).roughness
        self.assertGreater(cluster, dyad)

    def test_thirds_sit_between_the_two(self):
        third = heard(tone(220.0) + tone(275.0)).roughness
        self.assertGreater(third, heard(tone(220.0) + tone(330.0)).roughness)
        self.assertLess(third, heard(tone(220.0) + tone(233.0)).roughness)


class BrightnessTests(unittest.TestCase):
    def test_higher_energy_reads_brighter(self):
        low = heard(tone(120.0)).brightness
        middle = heard(tone(600.0)).brightness
        high = heard(tone(4000.0)).brightness
        self.assertLess(low, middle)
        self.assertLess(middle, high)

    def test_the_range_is_used_rather_than_saturated(self):
        """A linear map put every piece this engine makes in the top fifth."""
        self.assertLess(heard(tone(120.0)).brightness, 0.2)
        self.assertGreater(heard(tone(4000.0)).brightness, 0.8)


class MotionTests(unittest.TestCase):
    def test_a_changing_spectrum_moves_more_than_a_still_one(self):
        still = tone(300.0, 3.0)
        sweeping = np.concatenate([tone(300.0, 1.0), tone(900.0, 1.0), tone(450.0, 1.0)])
        self.assertGreater(heard(sweeping).motion, heard(still).motion)

    def test_silence_reports_nothing_rather_than_guessing(self):
        self.assertEqual(listen(np.zeros(PROBE_RATE)).frames, 0)

    def test_something_too_short_to_analyse_returns_the_neutral_reading(self):
        self.assertEqual(listen(np.zeros(64)), Heard())


class ProbeTests(unittest.TestCase):
    def test_a_piece_is_listened_to_movement_by_movement(self):
        piece = compose(2026, 120.0, THEMES["menacing"])
        self.assertTrue(piece.heard)
        for reading in piece.heard:
            self.assertTrue(0.0 <= reading.roughness <= 1.0)
            self.assertTrue(0.0 <= reading.brightness <= 1.0)

    def test_an_empty_window_is_neutral_not_zero(self):
        self.assertEqual(probe([], {}, 1, 0.5, 0.0), Heard())

    def test_what_it_hears_reaches_the_critic(self):
        piece = compose(2026, 120.0, THEMES["menacing"])
        for step in piece.evolution:
            self.assertTrue(0.0 <= step.analysis.heard_roughness <= 1.0)

    def test_the_bands_move_with_the_mood(self):
        """A serene piece should be allowed almost no beating; a shattered one, some."""
        analysis = compose(2026, 90.0, THEMES["menacing"]).evolution[0].analysis
        rough = replace(analysis, heard_roughness=0.12)
        serene = {v.principle.name: v for v in judge(rough, THEMES["serene"])}
        shattered = {v.principle.name: v for v in judge(rough, THEMES["shattered"])}
        self.assertFalse(serene["heard_roughness"].satisfied)
        self.assertTrue(shattered["heard_roughness"].satisfied)

    def test_every_heard_principle_reads_a_field_that_exists(self):
        from compex.generate.critic import Analysis

        for principle in PRINCIPLES:
            if principle.name.startswith("heard_"):
                self.assertIn(principle.name, Analysis.__dataclass_fields__)


class DriftTests(unittest.TestCase):
    def test_an_instrument_changes_while_it_plays(self):
        voice = build_voice(7, 2, ROLE_PAD, THEMES["menacing"])
        moved = drift(voice, 7, 5, 0.8)
        self.assertNotEqual(dict(voice.params), dict(moved.params))

    def test_it_arrives_somewhere_rather_than_wobbling(self):
        """The walk is cumulative in the movement, so the voice travels."""
        voice = build_voice(7, 2, ROLE_PAD, THEMES["menacing"])

        def travelled(movement: int) -> float:
            moved = dict(drift(voice, 7, movement, 0.8).params)
            start = dict(voice.params)
            return sum(abs(moved[k] - start[k]) / abs(start[k])
                       for k in start if start[k]) / max(1, len(start))

        self.assertGreater(travelled(6), travelled(1))

    def test_structural_parameters_are_held(self):
        """Moving a formant pair is a different vowel, not the same voice changed."""
        voice = build_voice(7, 2, ROLE_PAD, THEMES["euphoric"])
        moved = dict(drift(voice, 7, 6, 1.0).params)
        for name, value in voice.params:
            if name in FIXED:
                self.assertEqual(moved[name], value, name)

    def test_nothing_drifts_out_of_range(self):
        voice = build_voice(7, 2, ROLE_PAD, THEMES["shattered"])
        start = dict(voice.params)
        for movement in range(1, 12):
            moved = dict(drift(voice, 7, movement, 1.0).params)
            for name, value in start.items():
                if name in FIXED or not value:
                    continue
                self.assertGreaterEqual(abs(moved[name]), abs(value) * DRIFT_FLOOR - 1e-9)
                self.assertLessEqual(abs(moved[name]), abs(value) * DRIFT_CEILING + 1e-9)

    def test_zero_drift_leaves_the_instrument_alone(self):
        voice = build_voice(7, 2, ROLE_PAD, THEMES["menacing"])
        self.assertEqual(drift(voice, 7, 8, 0.0), voice)

    def test_the_drive_actually_moves_the_instruments(self):
        def travel(amount: float) -> float:
            real = evolve.initial_drives
            with patch("compex.generate.compose.initial_drives",
                       lambda mood, root: replace(real(mood, root), timbre_drift=amount)):
                piece = compose(2026, 120.0, THEMES["menacing"])
            latest = max(movement for _, movement, _ in piece.states)
            first = {name: spec for name, movement, spec in piece.states if movement == 0}
            last = {name: spec for name, movement, spec in piece.states if movement == latest}
            moved = []
            for name, spec in first.items():
                start, end = dict(spec.params), dict(last[name].params)
                deltas = [abs(end[k] - start[k]) / abs(start[k])
                          for k in start if start[k] and k in end]
                if deltas:
                    moved.append(sum(deltas) / len(deltas))
            return sum(moved) / max(1, len(moved))

        self.assertGreater(travel(1.0), travel(0.0))
        self.assertEqual(travel(0.0), 0.0)

    def test_a_note_remembers_which_instrument_played_it(self):
        piece = compose(2026, 120.0, THEMES["menacing"])
        stamps = {note.timbre for note in piece.notes}
        self.assertGreater(len(stamps), 1)
        for note in piece.notes:
            self.assertIsNotNone(piece.state(note.voice, note.timbre))


class InventedScaleTests(unittest.TestCase):
    def test_it_builds_a_usable_set_out_of_the_grid(self):
        grid = tuple(float(step) for step in range(12))
        for seed in range(20):
            steps = tuning.invent(seed, THEMES["hypnotic"], grid)
            self.assertEqual(steps[0], 0.0)
            self.assertEqual(list(steps), sorted(steps))
            gaps = [high - low for low, high in zip(steps, list(steps[1:]) + [OCTAVE])]
            self.assertLess(max(gaps), MAX_GAP + MIN_STEP + 1e-6)

    def test_a_named_mode_is_recognised_but_not_required(self):
        self.assertEqual(tuning.name_of((0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0)), "ionian")
        self.assertIsNone(tuning.name_of((0.0, 1.0, 2.0, 6.0, 7.0, 9.0)))

    def test_most_of_what_it_invents_has_no_name(self):
        """The table is commentary now, not a source."""
        names = [tuning.twelve(seed, THEMES["menacing"]).name for seed in range(24)]
        self.assertGreater(names.count("unnamed mode") / len(names), 0.5)

    def test_a_set_with_a_hole_in_it_scores_below_one_without(self):
        holed = tuning._score((0.0, 1.0, 2.0), THEMES["hypnotic"])
        stepped = tuning._score((0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0), THEMES["hypnotic"])
        self.assertLess(holed, stepped)


if __name__ == "__main__":
    unittest.main()
