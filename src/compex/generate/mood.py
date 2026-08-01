"""Emotional theme as a small vector rather than a genre list.

A genre list would just be a lookup table — pick "dubstep", get dubstep back.
Axes let the composer land in places between the named themes, which is the
point of letting the computation carry itself. The named themes are only
convenient coordinates in that space; nothing is looked up from them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

AXES = ("valence", "energy", "tension", "density", "grit")


class MoodError(ValueError):
    """Raised when a supplied mood value is unparseable or out of range."""


@dataclass(frozen=True)
class Mood:
    """Where in feeling-space to aim. Every axis runs 0..1.

    valence — dark to bright
    energy  — still to frantic
    tension — resolved to unresolved
    density — sparse to crowded
    grit    — clean to destroyed
    """

    valence: float = 0.50
    energy: float = 0.50
    tension: float = 0.50
    density: float = 0.50
    grit: float = 0.35

    def with_(self, **changes: float) -> "Mood":
        return replace(self, **changes)

    def as_dict(self) -> dict[str, float]:
        return {axis: getattr(self, axis) for axis in AXES}

    def nearest_theme(self) -> str:
        """The named theme this vector sits closest to — for labelling output."""
        best, distance = "", float("inf")
        for name, theme in THEMES.items():
            spread = sum((getattr(self, axis) - getattr(theme, axis)) ** 2 for axis in AXES)
            if spread < distance:
                best, distance = name, spread
        return best

    def describe(self) -> str:
        return ", ".join(f"{axis} {getattr(self, axis):.2f}" for axis in AXES)

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | str | None) -> "Mood":
        """Build a mood from untrusted input: a theme name, or explicit axes."""
        if raw is None:
            return cls()
        if isinstance(raw, str):
            return cls.from_theme(raw)

        unknown = set(raw) - set(AXES)
        if unknown:
            raise MoodError(f"unknown mood axis/axes: {', '.join(sorted(unknown))}")

        values: dict[str, float] = {}
        for axis, value in raw.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise MoodError(f"mood axis {axis!r} is not a number: {value!r}") from exc
            if number != number:
                raise MoodError(f"mood axis {axis!r} is NaN")
            if not 0.0 <= number <= 1.0:
                raise MoodError(f"mood axis {axis!r} must be 0..1, got {number}")
            values[axis] = number
        return cls(**values)

    @classmethod
    def from_theme(cls, name: str) -> "Mood":
        key = str(name).strip().lower().replace(" ", "_")
        if key not in THEMES:
            raise MoodError(f"unknown theme {name!r}; try one of: {', '.join(sorted(THEMES))}")
        return THEMES[key]


#: Named coordinates in mood space. Starting points, not presets to copy.
THEMES: dict[str, Mood] = {
    "serene":      Mood(valence=0.74, energy=0.16, tension=0.14, density=0.26, grit=0.06),
    "tender":      Mood(valence=0.68, energy=0.30, tension=0.26, density=0.34, grit=0.10),
    "wistful":     Mood(valence=0.52, energy=0.26, tension=0.38, density=0.32, grit=0.12),
    "melancholy":  Mood(valence=0.27, energy=0.27, tension=0.46, density=0.35, grit=0.16),
    "desolate":    Mood(valence=0.16, energy=0.11, tension=0.52, density=0.14, grit=0.24),
    "hypnotic":    Mood(valence=0.45, energy=0.46, tension=0.30, density=0.62, grit=0.30),
    "solemn":      Mood(valence=0.34, energy=0.24, tension=0.42, density=0.46, grit=0.14),
    "euphoric":    Mood(valence=0.89, energy=0.83, tension=0.34, density=0.78, grit=0.28),
    "triumphant":  Mood(valence=0.80, energy=0.72, tension=0.38, density=0.72, grit=0.34),
    "restless":    Mood(valence=0.44, energy=0.62, tension=0.62, density=0.58, grit=0.40),
    "anxious":     Mood(valence=0.24, energy=0.68, tension=0.90, density=0.62, grit=0.50),
    "menacing":    Mood(valence=0.11, energy=0.57, tension=0.86, density=0.50, grit=0.78),
    "frantic":     Mood(valence=0.40, energy=0.95, tension=0.80, density=0.88, grit=0.68),
    "shattered":   Mood(valence=0.18, energy=0.74, tension=0.94, density=0.70, grit=0.92),
    "weightless":  Mood(valence=0.62, energy=0.22, tension=0.34, density=0.20, grit=0.10),
}


def theme_names() -> tuple[str, ...]:
    return tuple(sorted(THEMES))
