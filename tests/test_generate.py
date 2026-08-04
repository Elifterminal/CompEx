"""The composer: does it decide, does it decide differently, does it stay legal."""

import unittest

from compex.generate import THEMES, Mood, compose
from compex.generate.mood import AXES, MoodError
from compex.generate.palette import ROLE_PERC, build_kit, build_voice, choose_engine
from compex.generate.theory import SCALES, SCALE_COLOUR, Motif, degree_semitone, make_motif

FAST = 40.0


class MoodTests(unittest.TestCase):
    def test_every_theme_is_a_valid_vector(self):
        for name, mood in THEMES.items():
            for axis in AXES:
                value = getattr(mood, axis)
                self.assertTrue(0.0 <= value <= 1.0, f"{name}.{axis} = {value}")

    def test_theme_lookup_is_forgiving_about_case(self):
        self.assertEqual(Mood.from_theme("MENACING"), THEMES["menacing"])

    def test_unknown_theme_is_rejected(self):
        with self.assertRaises(MoodError):
            Mood.from_theme("upbeat-ska")

    def test_unknown_axis_is_rejected(self):
        with self.assertRaises(MoodError):
            Mood.parse({"loudness": 0.5})

    def test_out_of_range_axis_is_rejected(self):
        with self.assertRaises(MoodError):
            Mood.parse({"valence": 4.0})

    def test_nearest_theme_finds_itself(self):
        for name, mood in THEMES.items():
            self.assertEqual(mood.nearest_theme(), name)


class TheoryTests(unittest.TestCase):
    def test_every_scale_has_a_colour(self):
        self.assertEqual(set(SCALES), set(SCALE_COLOUR))

    def test_scales_start_on_the_tonic_and_ascend(self):
        for name, degrees in SCALES.items():
            self.assertEqual(degrees[0], 0, name)
            self.assertEqual(list(degrees), sorted(degrees), name)
            self.assertLess(degrees[-1], 12, name)

    def test_degrees_wrap_octaves_in_both_directions(self):
        scale = SCALES["ionian"]
        self.assertEqual(degree_semitone(scale, 0), 0)
        self.assertEqual(degree_semitone(scale, 7), 12)
        self.assertEqual(degree_semitone(scale, -7), -12)

    def test_motif_steps_and_rhythm_stay_paired(self):
        motif = make_motif(11, SCALES["dorian"], 0.6, 0.4)
        self.assertEqual(len(motif.steps), len(motif.rhythm))
        self.assertGreater(motif.beats, 0)

    def test_mismatched_motif_is_rejected(self):
        with self.assertRaises(ValueError):
            Motif(steps=(0, 1), rhythm=(1.0,))


class PaletteTests(unittest.TestCase):
    def test_engine_chosen_is_legal_for_the_role(self):
        from compex.generate.palette import ENGINES_FOR_ROLE

        for role, allowed in ENGINES_FOR_ROLE.items():
            for index in range(12):
                self.assertIn(choose_engine(index, index, role, THEMES["restless"]), allowed)

    def test_kit_always_contains_a_kick(self):
        for theme in THEMES.values():
            kit = build_kit(7, theme)
            self.assertIn("kick", [voice.voice_id for voice in kit])

    def test_denser_moods_get_bigger_kits(self):
        sparse = build_kit(7, THEMES["desolate"])
        dense = build_kit(7, THEMES["frantic"])
        self.assertGreater(len(dense), len(sparse))

    def test_voice_params_are_immutable(self):
        voice = build_voice(3, 0, "bass", THEMES["menacing"])
        with self.assertRaises(Exception):
            voice.gain = 2.0


class CompositionTests(unittest.TestCase):
    def test_same_inputs_give_the_same_piece(self):
        first = compose(1203, FAST, THEMES["menacing"])
        second = compose(1203, FAST, THEMES["menacing"])
        self.assertEqual(first, second)

    def test_seed_changes_the_piece(self):
        first = compose(1203, FAST, THEMES["menacing"])
        other = compose(99, FAST, THEMES["menacing"])
        self.assertNotEqual(first.notes, other.notes)

    def test_mood_changes_the_piece_at_a_fixed_seed(self):
        calm = compose(1203, FAST, THEMES["serene"])
        wild = compose(1203, FAST, THEMES["frantic"])
        self.assertNotEqual(calm.scale_name, wild.scale_name)
        self.assertLess(calm.bpm, wild.bpm)

    def test_energy_drives_tempo_across_all_themes(self):
        pairs = [(THEMES[name].energy, compose(5, FAST, THEMES[name]).bpm) for name in THEMES]
        low = [bpm for energy, bpm in pairs if energy < 0.3]
        high = [bpm for energy, bpm in pairs if energy > 0.7]
        self.assertLess(sum(low) / len(low), sum(high) / len(high))

    def test_density_drives_how_much_happens(self):
        sparse = compose(5, FAST, THEMES["desolate"])
        dense = compose(5, FAST, THEMES["frantic"])
        self.assertLess(len(sparse.strokes), len(dense.strokes))

    def test_runtime_is_roughly_honoured(self):
        for target in (30.0, 90.0, 180.0):
            piece = compose(21, target, THEMES["hypnotic"])
            self.assertLess(abs(piece.total_seconds - target), target * 0.2 + 4.0)

    def test_notes_are_sorted_and_inside_the_piece(self):
        piece = compose(1203, FAST, THEMES["euphoric"])
        starts = [note.start for note in piece.notes]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(note.start < piece.total_beats + 1e-6 for note in piece.notes))
        self.assertTrue(all(note.duration > 0 for note in piece.notes))

    def test_every_note_names_a_voice_that_exists(self):
        for name in ("serene", "menacing", "frantic", "desolate"):
            piece = compose(1203, FAST, THEMES[name])
            known = {voice.voice_id for voice in piece.voices}
            self.assertTrue({note.voice for note in piece.notes} <= known, name)
            self.assertTrue({stroke.voice for stroke in piece.strokes} <= known, name)

    def test_percussion_voices_are_marked_as_such(self):
        piece = compose(1203, FAST, THEMES["hypnotic"])
        drums = {voice.voice_id for voice in piece.voices if voice.role == ROLE_PERC}
        self.assertTrue({stroke.voice for stroke in piece.strokes} <= drums)

    def test_velocities_stay_in_range(self):
        piece = compose(1203, FAST, THEMES["anxious"])
        for note in piece.notes:
            self.assertTrue(0.0 < note.velocity <= 1.0)
        for stroke in piece.strokes:
            self.assertTrue(0.0 < stroke.velocity <= 1.0)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            compose(1, 0.0, THEMES["serene"])
        with self.assertRaises(TypeError):
            compose(1, FAST, "serene")

    def test_all_themes_produce_something_playable(self):
        for name, mood in THEMES.items():
            piece = compose(808, FAST, mood)
            self.assertGreater(len(piece.notes), 0, f"{name} wrote no notes")
            self.assertGreater(len(piece.movements), 1, f"{name} has no form")


if __name__ == "__main__":
    unittest.main()


class EveryMoodTests(unittest.TestCase):
    """Compose in every theme, not a representative handful.

    This exists because of a crash the suite could not have found. A dark note
    high enough to give a four-sample delay line broke the pluck engine, but
    only at the 11 kHz rate the listening probe runs at — so it needed a
    particular theme, a particular seed and a particular pitch all at once. It
    sat there from the day that probe shipped and was found by sweeping all
    fifteen moods by hand.

    The input space here is small and fully enumerable. Sampling it was the
    mistake.
    """

    def test_every_theme_composes(self):
        for name, mood in sorted(THEMES.items()):
            for seed in (11, 2026, 5150):
                with self.subTest(theme=name, seed=seed):
                    piece = compose(seed, 30.0, mood)
                    self.assertTrue(piece.notes or piece.strokes)

    def test_the_shadow_never_outweighs_the_kit_in_any_mood(self):
        """The anti-jazz guarantee, checked across the whole space rather than
        the moods that happened to get looked at."""
        for name, mood in sorted(THEMES.items()):
            piece = compose(5150, 60.0, mood)
            if not piece.ghost_strokes:
                continue
            shadow = sum(stroke.velocity for stroke in piece.ghost_strokes)
            kit = sum(stroke.velocity for stroke in piece.strokes)
            with self.subTest(theme=name):
                self.assertLess(shadow, kit, name)
