"""The seekable RNG is what makes a render reproducible, so it gets tested hard."""

import unittest

from tests import SRC  # noqa: F401  (path setup)

from compex import rng


class UniformTests(unittest.TestCase):
    def test_is_stable_for_a_position(self):
        first = rng.uniform(7, "beta", 42)
        self.assertEqual(first, rng.uniform(7, "beta", 42))

    def test_stays_in_range(self):
        values = [rng.uniform(1, "s", i) for i in range(500)]
        self.assertTrue(all(0.0 <= v < 1.0 for v in values))

    def test_streams_are_independent(self):
        self.assertNotEqual(rng.uniform(3, "beta", 9), rng.uniform(3, "f_LFO", 9))

    def test_seed_changes_everything(self):
        self.assertNotEqual(rng.uniform(1, "s", 5), rng.uniform(2, "s", 5))

    def test_order_of_access_does_not_matter(self):
        forwards = [rng.uniform(11, "s", i) for i in range(64)]
        backwards = [rng.uniform(11, "s", i) for i in reversed(range(64))][::-1]
        self.assertEqual(forwards, backwards)


class BetweenTests(unittest.TestCase):
    def test_respects_bounds(self):
        values = [rng.between(5, "beta", i, 2.0, 9.0) for i in range(400)]
        self.assertTrue(all(2.0 <= v <= 9.0 for v in values))

    def test_rejects_inverted_bounds(self):
        with self.assertRaises(ValueError):
            rng.between(1, "s", 0, 5.0, 1.0)


class PickTests(unittest.TestCase):
    OPTIONS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

    def test_only_returns_declared_options(self):
        drawn = {rng.pick(1, "f_LFO", i, self.OPTIONS) for i in range(300)}
        self.assertTrue(drawn.issubset(set(self.OPTIONS)))

    def test_uses_the_whole_set(self):
        drawn = {rng.pick(1, "f_LFO", i, self.OPTIONS) for i in range(300)}
        self.assertEqual(drawn, set(self.OPTIONS))

    def test_rejects_empty_options(self):
        with self.assertRaises(ValueError):
            rng.pick(1, "s", 0, ())


class NoiseTests(unittest.TestCase):
    def test_is_reproducible(self):
        self.assertTrue((rng.noise(4, "snare", 256) == rng.noise(4, "snare", 256)).all())

    def test_length_and_finiteness(self):
        buffer = rng.noise(4, "snare", 1024)
        self.assertEqual(len(buffer), 1024)
        self.assertTrue(all(abs(v) < 20 for v in buffer[:50]))

    def test_rejects_negative_count(self):
        with self.assertRaises(ValueError):
            rng.noise(1, "s", -1)


if __name__ == "__main__":
    unittest.main()
