"""Memory across pieces — and the property it was not allowed to break.

The engine has made hundreds of pieces and remembered none of them. Now it
carries taste and boredom forward, which is the one advantage a machine holds
over a person outright: perfect recall of everything it has done, and therefore
the ability to notice it keeps doing the same thing.

The load-bearing test in this file is the boring one. **Determinism survives**,
because the memory is an argument rather than hidden state: same seed, same
mood, same memory, same audio — and an engine with nothing remembered writes
exactly what it wrote before any of this existed.
"""

import json
import unittest

from compex.config import Knobs
from compex.generate import THEMES, compose
from compex.pipeline import make_track
from compex.remember import DECAY, FLOOR, Memory, bored_of, learn


def history(pieces: int, seed: int = 2026, theme: str = "hypnotic") -> Memory:
    memory = Memory()
    for _ in range(pieces):
        memory = learn(memory, compose(seed, 30.0, THEMES[theme], memory))
    return memory


class DeterminismTests(unittest.TestCase):
    def test_nothing_remembered_writes_what_it_always_wrote(self):
        self.assertEqual(compose(2026, 45.0, THEMES["menacing"]),
                         compose(2026, 45.0, THEMES["menacing"], Memory()))

    def test_the_same_memory_gives_the_same_piece(self):
        remembered = history(3)
        self.assertEqual(compose(7788, 45.0, THEMES["serene"], remembered),
                         compose(7788, 45.0, THEMES["serene"], remembered))

    def test_a_different_history_gives_a_different_piece(self):
        """Otherwise it is not remembering, it is only bookkeeping."""
        blank = compose(7788, 45.0, THEMES["hypnotic"], Memory())
        experienced = compose(7788, 45.0, THEMES["hypnotic"], history(5))
        self.assertNotEqual(blank.notes, experienced.notes)

    def test_the_formula_says_which_history_it_was_written_with(self):
        result = make_track(Knobs(seed=2026, duration_s=30.0), THEMES["hypnotic"],
                            memory=history(2))
        self.assertIn("MEMORY", result.formula)
        blank = make_track(Knobs(seed=2026, duration_s=30.0), THEMES["hypnotic"])
        self.assertNotIn("MEMORY", blank.formula)


class LearningTests(unittest.TestCase):
    def test_a_piece_is_folded_in(self):
        memory = learn(Memory(), compose(2026, 30.0, THEMES["hypnotic"]))
        self.assertEqual(memory.pieces, 1)
        self.assertTrue(memory.engines)
        self.assertTrue(memory.taste)

    def test_the_past_decays(self):
        """The last few pieces matter; the first hundred are a rumour."""
        first = learn(Memory(), compose(1, 30.0, THEMES["serene"]))
        engine = max(first.engines, key=lambda pair: pair[1])[0]
        later = first
        for seed in range(2, 6):
            later = learn(later, compose(seed, 30.0, THEMES["frantic"]))
        self.assertLess(later.weight("engines", engine), first.weight("engines", engine) + 1e-9)

    def test_it_does_not_grow_without_bound(self):
        small, large = history(2), history(12)
        self.assertLess(len(large.engines), len(small.engines) + 24)

    def test_the_digest_moves_when_the_memory_does(self):
        one, two = history(1), history(2)
        self.assertNotEqual(one.digest(), two.digest())
        self.assertEqual(two.digest(), Memory.parse(two.to_json()).digest())


class BoredomTests(unittest.TestCase):
    def test_what_it_keeps_using_gets_pushed_down(self):
        remembered = history(4)
        used = max(remembered.engines, key=lambda pair: pair[1])[0]
        unused = "aeolian" if used != "aeolian" else "bowed"
        self.assertGreater(bored_of(remembered, "engines", used),
                           bored_of(remembered, "engines", unused))

    def test_a_blank_memory_is_bored_of_nothing(self):
        self.assertEqual(bored_of(Memory(), "engines", "pad"), 0.0)

    def test_it_stops_reaching_for_the_same_instruments(self):
        """Same seed, same mood, six pieces: the instrumentation should move."""
        memory = Memory()
        seen = []
        for _ in range(6):
            piece = compose(2026, 30.0, THEMES["hypnotic"], memory)
            seen.append(tuple(v.engine for v in piece.voices if v.role != "perc"))
            memory = learn(memory, piece)
        self.assertGreater(len(set(seen)), 3, f"it kept picking the same voices: {seen}")


class ParsingTests(unittest.TestCase):
    """A history is untrusted input — it arrives from a file or a browser."""

    def test_nothing_at_all_is_a_blank_memory(self):
        for raw in (None, "", "not json", "[]", "42"):
            self.assertTrue(Memory.parse(raw).is_blank(), repr(raw))

    def test_it_round_trips(self):
        remembered = history(3)
        self.assertEqual(Memory.parse(remembered.to_json()), remembered)

    def test_junk_entries_are_dropped_rather_than_crashing(self):
        parsed = Memory.parse(json.dumps({
            "pieces": "seven",
            "engines": {"pad": "loud", "bell": 2.0, "drone": float("nan")},
            "taste": "not a table",
        }))
        self.assertEqual(parsed.pieces, 0)
        self.assertEqual(dict(parsed.engines), {"bell": 2.0})
        self.assertEqual(parsed.taste, ())

    def test_a_negative_or_tiny_weight_is_ignored(self):
        parsed = Memory.parse({"pieces": 2, "engines": {"pad": -3.0, "bell": FLOOR / 2}})
        self.assertEqual(parsed.engines, ())

    def test_the_decay_constant_is_a_decay(self):
        self.assertTrue(0.0 < DECAY < 1.0)


class SurfaceTests(unittest.TestCase):
    def test_the_phone_can_hand_its_history_in_and_get_the_new_one_back(self):
        from compex.web import Session, memory_text, start

        mood = json.dumps({"valence": 0.45, "energy": 0.46, "tension": 0.3,
                           "density": 0.62, "grit": 0.3})
        start(2026, 30.0, mood, 8.0, 0.35, "")
        carried = memory_text()
        self.assertEqual(Memory.parse(carried).pieces, 1)

        session = Session(2026, 30.0, "hypnotic", 8.0, 0.89, 0.35, carried)
        self.assertEqual(session.remembered.pieces, 1)
        self.assertEqual(session.learned.pieces, 2)

    def test_a_track_hands_back_what_it_taught(self):
        result = make_track(Knobs(seed=2026, duration_s=30.0), THEMES["serene"])
        self.assertEqual(result.memory.pieces, 1)
        self.assertTrue(result.memory.engines)


if __name__ == "__main__":
    unittest.main()
