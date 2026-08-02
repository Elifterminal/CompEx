"""The surface the browser talks to.

CompEx runs unmodified in the browser under Pyodide — the same package, not a
port. A JavaScript reimplementation would be three thousand lines of duplicate
engine that drifts from this one the first time either is touched, and then
the phone and the desktop quietly stop making the same music.

Everything here returns plain Python types that survive the JS boundary
without conversion helpers, and audio comes back as a flat float32 array the
caller can drop straight into a Web Audio buffer.
"""

from __future__ import annotations

import json

import numpy as np

from compex import __version__, report
from compex.dsp.stream import DEFAULT_SPAN_SECONDS, Level, finish, plan, render_span
from compex.generate import THEMES, compose
from compex.generate.mood import AXES, Mood, theme_names
from compex.notation import emit

SAMPLE_RATE = 44_100


class Session:
    """One piece, rendered span by span so playback can start early."""

    def __init__(self, seed: int, duration_s: float, mood: dict | str | None,
                 span_seconds: float = DEFAULT_SPAN_SECONDS,
                 master_gain: float = 0.89) -> None:
        self.composition = compose(int(seed), float(duration_s), Mood.parse(mood))
        self.spans = plan(self.composition, float(span_seconds))
        self.master_gain = float(master_gain)
        self._carry: np.ndarray | None = None
        self._level: Level | None = None
        self._position = 0

    # -- what the piece is -------------------------------------------------

    def info(self) -> dict:
        piece = self.composition
        return {
            "version": __version__,
            "seed": piece.seed,
            "theme": piece.mood.nearest_theme(),
            "mood": piece.mood.as_dict(),
            "bpm": round(piece.bpm, 1),
            "meter": piece.beats_per_bar,
            "scale": piece.scale_name.replace("_", " "),
            "archetype": piece.archetype,
            "seconds": round(piece.total_seconds, 2),
            "sample_rate": SAMPLE_RATE,
            "spans": len(self.spans),
            "notes": len(piece.notes),
            "strokes": len(piece.strokes),
            "summary": piece.summary(),
            "voices": [
                {"role": v.role, "engine": v.engine, "effects": list(v.effect_names())}
                for v in piece.voices if v.role != "perc"
            ],
            "kit": [v.voice_id for v in piece.voices if v.role == "perc"],
            "movements": report.movements(self.composition),
            "evolution": self.evolution(),
            "melody": report.melody(self.composition),
            "patterns": report.patterns(self.composition),
            "taste": report.taste(self.composition),
        }

    def evolution(self) -> list[dict]:
        """Where the composer listened back and changed its mind."""
        return report.evolution(self.composition)

    def formula(self) -> str:
        return emit(self.composition, self.composition.total_seconds)

    # -- rendering ---------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._position >= len(self.spans)

    def next_span(self) -> dict | None:
        """Render the next span. Returns None once the piece is finished."""
        if self.done:
            return None
        span = self.spans[self._position]
        samples, self._carry, self._level = render_span(
            self.composition, span, self._carry, self.master_gain, self._level)
        self._position += 1
        return {
            "index": span.index,
            "movement": span.movement,
            "start": round(span.start_beat * self.composition.seconds_per_beat, 3),
            "seconds": round(len(samples) / SAMPLE_RATE, 3),
            "remaining": len(self.spans) - self._position,
            "samples": samples.astype(np.float32),
        }

    def tail(self) -> dict | None:
        """Whatever was still ringing when the last span ended."""
        if not self.done or self._carry is None or not len(self._carry):
            return None
        trimmed = finish(self._carry, self._level, self.master_gain)
        if not len(trimmed):
            return None
        # Consume it. Without this the tail is handed back on every subsequent
        # call, and a player that drains until it gets nothing never stops.
        self._carry = None
        return {
            "index": len(self.spans),
            "movement": "tail",
            "seconds": round(len(trimmed) / SAMPLE_RATE, 3),
            "remaining": 0,
            "samples": trimmed.astype(np.float32),
        }


# ── the driver the browser actually calls ─────────────────────────────────
#
# Keeping the JS side to plain strings and one numpy array avoids wrestling
# with proxy lifetimes for every nested dict. Metadata crosses as JSON; audio
# crosses once, as a buffer.

_session: Session | None = None
_samples: np.ndarray | None = None


def start(seed: int, duration_s: float, mood_json: str,
          span_seconds: float = DEFAULT_SPAN_SECONDS) -> str:
    """Begin a piece. Returns its description as JSON."""
    global _session, _samples
    _samples = None
    _session = Session(seed, duration_s, json.loads(mood_json), span_seconds)
    return json.dumps(_session.info())


def step() -> str:
    """Render the next span. Returns JSON metadata, or "" when the piece is done.

    The audio is left in :func:`samples` rather than returned, so it crosses the
    boundary as one buffer instead of a converted list of a million floats.
    """
    global _samples
    if _session is None:
        raise RuntimeError("start() has not been called")

    chunk = _session.next_span() or _session.tail()
    if chunk is None:
        _samples = None
        return ""
    _samples = chunk.pop("samples")
    return json.dumps(chunk)


def samples() -> np.ndarray | None:
    return _samples


def formula_text() -> str:
    return _session.formula() if _session else ""


def catalogue() -> dict:
    """Everything the interface needs to build its controls."""
    return {
        "version": __version__,
        "axes": list(AXES),
        "themes": [{"name": name, **THEMES[name].as_dict()} for name in theme_names()],
        "sample_rate": SAMPLE_RATE,
    }
