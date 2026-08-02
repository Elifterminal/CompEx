"""Streaming playback: does a piece rendered in pieces still add up to the piece.

The player exists so a long track can be heard while it is still being made,
which means the seams have to be inaudible and the level has to be steady
without ever seeing the whole piece.
"""

import json
import unittest

import numpy as np

from compex.dsp.arrange import render_composition
from compex.dsp.stream import (
    MAX_BOOST,
    Level,
    finish,
    initial_level,
    plan,
    render_span,
)
from compex.generate import THEMES, compose
from compex.web import Session, catalogue, formula_text, samples, start, step

SECONDS = 40.0


def drain(piece, span_seconds=8.0):
    carry = level = None
    chunks = []
    for span in plan(piece, span_seconds):
        out, carry, level = render_span(piece, span, carry, level=level)
        chunks.append(out)
    tail = finish(carry, level)
    if len(tail):
        chunks.append(tail)
    return np.concatenate(chunks), level


class PlanTests(unittest.TestCase):
    def test_spans_tile_the_whole_piece(self):
        piece = compose(9042, SECONDS, THEMES["menacing"])
        spans = plan(piece, 8.0)
        self.assertAlmostEqual(spans[0].start_beat, 0.0)
        self.assertAlmostEqual(spans[-1].end_beat, piece.total_beats, places=6)
        for a, b in zip(spans, spans[1:]):
            self.assertAlmostEqual(a.end_beat, b.start_beat, places=6)

    def test_no_span_straddles_a_movement(self):
        piece = compose(9042, SECONDS, THEMES["frantic"])
        edges, cursor = set(), 0.0
        for movement in piece.movements:
            cursor += movement.beats
            edges.add(round(cursor, 6))
        for span in plan(piece, 8.0):
            inside = [e for e in edges if span.start_beat < e < span.end_beat - 1e-9]
            self.assertFalse(inside, f"span {span.index} crosses {inside}")

    def test_rejects_a_nonsense_span_length(self):
        piece = compose(1, SECONDS, THEMES["serene"])
        with self.assertRaises(ValueError):
            plan(piece, 0.0)


class SeamTests(unittest.TestCase):
    def test_streamed_length_matches_the_offline_render(self):
        piece = compose(9042, SECONDS, THEMES["menacing"])
        streamed, _ = drain(piece)
        offline = render_composition(piece)
        self.assertLess(abs(len(streamed) - len(offline)) / len(offline), 0.02)

    def test_it_is_recognisably_the_same_piece(self):
        piece = compose(9042, SECONDS, THEMES["menacing"])
        streamed, _ = drain(piece)
        offline = render_composition(piece)
        n = min(len(streamed), len(offline))
        a = streamed[:n] / (np.max(np.abs(streamed[:n])) or 1)
        b = offline[:n] / (np.max(np.abs(offline[:n])) or 1)
        self.assertGreater(float(np.corrcoef(a, b)[0, 1]), 0.8)

    def test_seams_do_not_click(self):
        """A discontinuity at a span join is the failure this design invites."""
        piece = compose(9042, SECONDS, THEMES["hypnotic"])
        carry = level = None
        chunks = []
        for span in plan(piece, 8.0):
            out, carry, level = render_span(piece, span, carry, level=level)
            chunks.append(out)
        streamed = np.concatenate(chunks)

        steps = np.abs(np.diff(streamed))
        ceiling = float(np.percentile(steps, 99.9)) * 3.0
        position = 0
        for chunk in chunks[:-1]:
            position += len(chunk)
            around = steps[max(0, position - 3):position + 3]
            if len(around):
                self.assertLess(float(np.max(around)), ceiling,
                                f"click at the seam near sample {position}")

    def test_the_carry_is_not_gained_twice(self):
        """Tails were being multiplied by the level once per span they survived."""
        piece = compose(9042, SECONDS, THEMES["serene"])
        spans = plan(piece, 8.0)
        _, carry, level = render_span(piece, spans[0], None)
        self.assertGreater(len(carry), 0)
        # A pre-gain carry has to be louder than the same signal post-gain.
        self.assertGreater(float(np.max(np.abs(carry))), 0.0)
        self.assertLess(level.gain, MAX_BOOST + 1e-9)


class LevelTests(unittest.TestCase):
    def test_dense_pieces_do_not_slam_the_limiter(self):
        """Fixed gain crushed 11% of samples on frantic before this existed."""
        for theme in ("frantic", "shattered", "menacing"):
            streamed, _ = drain(compose(9042, SECONDS, THEMES[theme]))
            crushed = float(np.mean(np.abs(streamed) > 0.985))
            self.assertLess(crushed, 0.01, f"{theme} is being crushed: {crushed:.1%}")

    def test_quiet_pieces_are_not_boosted_without_limit(self):
        piece = compose(9042, SECONDS, THEMES["desolate"])
        streamed, level = drain(piece)
        self.assertLessEqual(level.gain, MAX_BOOST)
        self.assertLessEqual(float(np.max(np.abs(streamed))), 1.0)

    def test_the_initial_guess_is_sane(self):
        for mood in THEMES.values():
            level = initial_level(compose(3, SECONDS, mood))
            self.assertGreater(level.assumed_peak, 0.0)
            self.assertGreater(level.gain, 0.0)
            self.assertLessEqual(level.gain, MAX_BOOST)

    def test_output_never_clips(self):
        for theme in ("serene", "frantic", "shattered"):
            streamed, _ = drain(compose(11, SECONDS, THEMES[theme]))
            self.assertLessEqual(float(np.max(np.abs(streamed))), 1.0, theme)
            self.assertTrue(np.all(np.isfinite(streamed)), theme)


class SessionTests(unittest.TestCase):
    def test_a_session_drains_and_then_stops(self):
        """The tail used to be handed back forever, so a player never finished."""
        session = Session(seed=9042, duration_s=20.0, mood="menacing", span_seconds=6.0)
        seen = 0
        while session.next_span() is not None:
            seen += 1
            self.assertLess(seen, 100, "next_span never ran out")
        self.assertIsNotNone(session.tail())
        self.assertIsNone(session.tail(), "the tail was handed back twice")

    def test_info_is_json_serialisable(self):
        session = Session(seed=1, duration_s=20.0, mood="serene")
        json.dumps(session.info())  # would raise if anything exotic crept in

    def test_samples_come_back_as_float32(self):
        session = Session(seed=1, duration_s=20.0, mood="serene", span_seconds=6.0)
        chunk = session.next_span()
        self.assertEqual(chunk["samples"].dtype, np.float32)


class DriverTests(unittest.TestCase):
    """The exact calls the browser makes."""

    def test_start_step_and_finish(self):
        info = json.loads(start(9042, 20.0, json.dumps("hypnotic"), 6.0))
        self.assertIn("bpm", info)
        self.assertGreater(len(info["movements"]), 0)

        chunks = 0
        while True:
            meta = step()
            if not meta:
                break
            json.loads(meta)
            self.assertIsNotNone(samples())
            chunks += 1
            self.assertLess(chunks, 100, "step() never returned empty")
        self.assertGreater(chunks, 1)
        self.assertIn("SEED", formula_text())

    def test_mood_can_be_axes_rather_than_a_name(self):
        axes = {"valence": 0.2, "energy": 0.9, "tension": 0.8, "density": 0.7, "grit": 0.6}
        info = json.loads(start(5, 20.0, json.dumps(axes), 8.0))
        self.assertEqual(info["mood"]["valence"], 0.2)

    def test_catalogue_has_what_the_interface_needs(self):
        cat = catalogue()
        self.assertEqual(len(cat["axes"]), 5)
        self.assertEqual(len(cat["themes"]), len(THEMES))
        self.assertEqual(cat["sample_rate"], 44100)


class BrowserCompatibilityTests(unittest.TestCase):
    def test_the_engine_imports_without_subprocess(self):
        """Pyodide has no subprocess; importing it at module scope broke the player."""
        import importlib
        import sys
        from importlib.abc import MetaPathFinder

        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "subprocess":
                    raise ImportError("no subprocess (simulating pyodide)")
                return None

        blocker = Block()
        saved = {name: mod for name, mod in sys.modules.items() if name.startswith("compex")}
        for name in list(saved):
            del sys.modules[name]
        sys.meta_path.insert(0, blocker)
        try:
            for module in ("compex.web", "compex.pipeline", "compex.dsp.stream"):
                importlib.import_module(module)
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)


if __name__ == "__main__":
    unittest.main()
