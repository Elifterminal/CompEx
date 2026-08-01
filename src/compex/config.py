"""Engine-wide constants and the render knobs.

There are only three knobs now, deliberately. Everything that used to be a
dial — tempo, key, instrumentation, drive, density — is something the
composer decides for itself. What is left is how long you want it, how loud
it comes out, and which universe you are in.

The emotional theme is not here: it lives in :class:`compex.generate.Mood`,
because it is an input to composition rather than to rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

SAMPLE_RATE = 44_100
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path.home() / "compex" / "out"

#: Guard rails for every numeric knob: name -> (minimum, maximum).
KNOB_RANGES: dict[str, tuple[float, float]] = {
    "seed": (0, 2**31 - 1),
    "duration_s": (8.0, 900.0),
    "master_gain": (0.05, 1.0),
}


class KnobError(ValueError):
    """Raised when a supplied knob value is missing, unparseable or out of range."""


@dataclass(frozen=True)
class Knobs:
    """Immutable render settings. Use :meth:`with_` to change one."""

    seed: int = 1203
    duration_s: float = 90.0
    master_gain: float = 0.89

    def with_(self, **changes: Any) -> "Knobs":
        return replace(self, **changes)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Knobs":
        """Build knobs from untrusted input (UI form, JSON, CLI).

        Unknown keys are rejected rather than dropped, so a typo surfaces as
        an error instead of a render that quietly ignores it.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise KnobError(f"unknown knob(s): {', '.join(sorted(unknown))}")

        values: dict[str, Any] = {}
        for name, value in raw.items():
            if value is None or value == "":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise KnobError(f"knob {name!r} is not a number: {value!r}") from exc
            if number != number:  # NaN
                raise KnobError(f"knob {name!r} is NaN")
            low, high = KNOB_RANGES[name]
            if not low <= number <= high:
                raise KnobError(f"knob {name!r} must be between {low} and {high}, got {number}")
            values[name] = int(round(number)) if name == "seed" else number

        return cls(**values)
