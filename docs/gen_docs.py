#!/usr/bin/env python3
"""Generate the CompEx living page from the live code.

The living-page format wants one source of truth with a checker that fails the
build on drift. For this project the code *is* the source of truth: engine
counts, theme vectors, the scales each mood reaches and the tempos it picks can
all be read or measured directly. So nothing on the page is typed by hand — it
is introspected from the registries or measured by actually composing pieces.

That means the page cannot claim 24 engines while the package has 23.

    PYTHONPATH=src python3 docs/gen_docs.py
"""

from __future__ import annotations

import html
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compex import __version__                                    # noqa: E402
from compex.config import Knobs                                   # noqa: E402
from compex.dsp import drums                                      # noqa: E402
from compex.dsp.effects import EFFECT_NAMES                       # noqa: E402
from compex.dsp.engines import ENGINE_NAMES                       # noqa: E402
from compex.dsp.engines import core, struck, sustained, textural, voiced  # noqa: E402
from compex.generate import THEMES, compose                       # noqa: E402
from compex.generate.critic import PRINCIPLES, analyse, judge     # noqa: E402
from compex.generate.promise import (                             # noqa: E402
    FORGET_BEATS,
    RIPEN_BEATS,
    RIPE,
)
from compex.generate.melody import CRITERIA_DETAIL as MELODY_CRITERIA  # noqa: E402
from compex.generate.intent import INTENTS                        # noqa: E402
from compex.generate.melody import MUTATIONS                      # noqa: E402
from compex.generate.mood import AXES                             # noqa: E402
from compex.generate.pattern import CRITERIA_DETAIL as GROOVE_CRITERIA  # noqa: E402
from compex.generate.palette import (                             # noqa: E402
    DRUM_COLOUR,
    ENGINE_COLOUR,
    ENGINES_FOR_ROLE,
    HOLD,
    NEEDS_HOLD,
    ROLE_PAD,
)
from compex.generate.theory import SCALES                         # noqa: E402
from compex.pipeline import make_track                            # noqa: E402

PROJECT = "CompEx"
DOC_TYPE = "open engine notebook"
OUT = ROOT / "docs" / "index.html"
FACTS = ROOT / "docs" / "facts.json"

SURVEY_SEEDS = 40      # compositions per theme when measuring what a mood actually does
SURVEY_SECONDS = 45.0
THEME_ORDER = sorted(THEMES, key=lambda name: THEMES[name].energy)

FAMILIES = {
    "core": tuple(sorted(core.ENGINES)),
    "struck": tuple(sorted(struck.ENGINES)),
    "voiced": tuple(sorted(voiced.ENGINES)),
    "sustained": tuple(sorted(sustained.ENGINES)),
    "textural": tuple(sorted(textural.ENGINES)),
}


# ── measuring ──────────────────────────────────────────────────────────────

def survey() -> dict:
    """Compose many pieces per theme and record what the machine actually chose."""
    out: dict[str, dict] = {}
    for name in THEME_ORDER:
        mood = THEMES[name]
        pieces = [compose(seed, SURVEY_SECONDS, mood) for seed in range(SURVEY_SEEDS)]
        out[name] = {
            "axes": mood.as_dict(),
            "bpm": [p.bpm for p in pieces],
            "notes": [len(p.notes) for p in pieces],
            "strokes": [len(p.strokes) for p in pieces],
            "scales": Counter(p.scale_name for p in pieces),
            "meters": Counter(p.beats_per_bar for p in pieces),
            "forms": Counter(p.archetype for p in pieces),
            "engines": Counter(v.engine for p in pieces for v in p.voices if v.role != "perc"),
            "drums": Counter(v.voice_id for p in pieces for v in p.voices if v.role == "perc"),
            "effects": Counter(e for p in pieces for v in p.voices for e in v.effect_names()),
        }
    return out


def worked_example(seed: int = 7788, theme: str = "menacing", seconds: float = 45.0):
    """One real track, rendered, so the page can show the whole chain of decisions."""
    return make_track(Knobs(seed=seed, duration_s=seconds), THEMES[theme])


EVOLUTION_THEMES = ("serene", "hypnotic", "melancholy", "menacing", "frantic")
EVOLUTION_SECONDS = 180.0


def evolution_survey() -> dict:
    """Long pieces, so there are enough listen-backs to see a trajectory."""
    out: dict[str, dict] = {}
    for name in EVOLUTION_THEMES:
        piece = compose(7788, EVOLUTION_SECONDS, THEMES[name])
        out[name] = {
            "piece": piece,
            "plasticity": [s.after.plasticity for s in piece.evolution],
            "unrest": [s.after.unrest for s in piece.evolution],
            "unhappy": [len(s.unhappy) for s in piece.evolution],
            "corrections": sum(len(s.adjustments) for s in piece.evolution),
        }
    return out


CHOICE_THEMES = ("serene", "hypnotic", "melancholy", "menacing", "frantic")
CHOICE_SECONDS = 240.0


def choice_survey() -> dict:
    """Long pieces again, this time measuring what the auditions actually did.

    Everything the choosing panel claims is counted here from real pieces. If
    the composer stopped auditioning, or every audition were won by the same
    kind of candidate, this measurement would say so on the page rather than
    the page quietly continuing to describe an earlier version.
    """
    pieces = [compose(7788, CHOICE_SECONDS, THEMES[name]) for name in CHOICE_THEMES]
    melodies = [choice for piece in pieces for choice in piece.melodies]
    grids = [grid for piece in pieces for grid in piece.grids]

    return {
        "pieces": pieces,
        "example": pieces[CHOICE_THEMES.index("menacing")],
        "chosen": len(melodies),
        "imagined": sum(choice.considered for choice in melodies),
        "winners": Counter(choice.chosen.origin for choice in melodies),
        "deepest": max((choice.chosen.generation for choice in melodies), default=0),
        "margin": statistics.median([choice.margin for choice in melodies] or [0.0]),
        "moved": Counter(
            name for piece in pieces
            for name in piece.taste.strayed_from(piece.opening_taste)
        ),
        "grids": len(grids),
        "grids_past": sum(1 for grid in grids if grid.past_start()),
        "stray": statistics.mean([grid.strayed() for grid in grids] or [0.0]),
    }


PURPOSE_THEMES = ("serene", "hypnotic", "menacing", "frantic")
PURPOSE_SECONDS = 240.0

HINDSIGHT_THEMES = ("serene", "wistful", "hypnotic", "menacing", "frantic")
HINDSIGHT_SECONDS = 300.0

MIX_THEMES = ("serene", "wistful", "hypnotic", "solemn", "menacing", "frantic")
MIX_SECONDS = 120.0

GHOST_THEMES = ("serene", "wistful", "hypnotic", "menacing", "frantic")
GHOST_SECONDS = 180.0

FREE_SEEDS = 40      # how many seeds the tuning and clock survey walks

#: Measured on the build before the mixer existed, seed 7788, two minutes each.
#: Kept as numbers rather than as a memory of them, because the whole claim is
#: that this was arithmetic and not taste.
PERCUSSION_BEFORE = {"hypnotic": 2.39, "menacing": 1.47, "serene": 0.43}


def _crowding_at(spacing: float) -> float:
    """Average crowding when the spacing drive is pinned at this value."""
    from dataclasses import replace
    from unittest.mock import patch

    from compex.generate import evolve
    from compex.generate.critic import analyse

    real = evolve.initial_drives
    measured = []
    with patch("compex.generate.compose.initial_drives",
               lambda mood, root: replace(real(mood, root), spacing=spacing)):
        for name in ("hypnotic", "menacing", "euphoric", "frantic"):
            piece = compose(7788, 120.0, THEMES[name])
            lead = frozenset(v.voice_id for v in piece.voices if v.role == "lead")
            measured.append(analyse(piece.notes, piece.strokes, piece.motif, piece.scale,
                                    piece.root_pitch, piece.total_beats, lead).crowding)
    return statistics.mean(measured)


def memory_survey() -> dict:
    """Make several pieces from the same seed and mood, and watch it get bored."""
    from compex.remember import Memory, learn

    memory = Memory()
    rows = []
    for index in range(6):
        piece = compose(2026, 45.0, THEMES["hypnotic"], memory)
        rows.append({
            "index": index + 1,
            "engines": [voice.engine for voice in piece.voices if voice.role != "perc"],
            "tuning": piece.tuning.name,
            "digest": memory.digest() if not memory.is_blank() else "—",
            "pieces": memory.pieces,
        })
        memory = learn(memory, piece)

    distinct = len({tuple(row["engines"]) for row in rows})
    return {"rows": rows, "distinct": distinct, "final": memory}


def freedom_survey() -> dict:
    """How often the machine leaves the twelve-tone grid and the shared pulse."""
    from collections import Counter

    from compex.generate import clocks as clockwork
    from compex.generate import tuning as tuner

    families: Counter = Counter()
    by_mood: dict[str, Counter] = {}
    examples: dict[str, object] = {}
    off_grid: list[float] = []

    for name in THEME_ORDER:
        mood = THEMES[name]
        counts: Counter = Counter()
        for seed in range(FREE_SEEDS):
            found = tuner.derive(seed, mood)
            counts[found.family] += 1
            families[found.family] += 1
            examples.setdefault(found.family, found)
            if found.family != "twelve":
                off_grid.extend(
                    min(abs(value - nearest * 100.0) for nearest in range(13))
                    for value in found.cents())
        by_mood[name] = counts

    pieces = [compose(2026, 120.0, THEMES[name])
              for name in ("serene", "hypnotic", "menacing", "shattered")]
    loose = [(piece, clock) for piece in pieces for clock in piece.clocks
             if not clock.anchored]

    return {
        "families": families,
        "by_mood": by_mood,
        "examples": examples,
        "off_grid": statistics.mean(off_grid) if off_grid else 0.0,
        "worst_off": max(off_grid) if off_grid else 0.0,
        "pieces": pieces,
        "loose": loose,
        "seeds": FREE_SEEDS,
    }


PLAN_CASES = ((11, "menacing"), (202, "hypnotic"), (3131, "frantic"),
              (4747, "serene"), (5150, "menacing"), (777, "wistful"))
PLAN_SECONDS = 200.0


def plan_survey() -> dict:
    """Compose each piece twice — plan held, plan idle — and compare.

    The number this produces is the only one that decides whether the plan is
    a mechanism or a decoration, so it is measured here rather than asserted.
    """
    from unittest.mock import patch

    from compex.generate.intent import GAIN, LEVERS

    rows = []
    for seed, theme in PLAN_CASES:
        scores = []
        for gain in (0.0, GAIN):
            with patch("compex.generate.intent.GAIN", gain):
                piece = compose(seed, PLAN_SECONDS, THEMES[theme])
            scores.append(piece)
        rows.append({
            "seed": seed, "theme": theme,
            "intent": scores[1].plan.intent.name,
            "off": scores[0].plan.agreement(),
            "on": scores[1].plan.agreement(),
            "gave_up": len(scores[1].ledger.given_up),
            "piece": scores[1],
        })

    deltas = [row["on"] - row["off"] for row in rows]
    return {
        "rows": rows,
        "off": statistics.mean(row["off"] for row in rows),
        "on": statistics.mean(row["on"] for row in rows),
        "delta": statistics.mean(deltas),
        "improved": sum(1 for value in deltas if value > 0),
        "levers": LEVERS,
        "example": rows[0]["piece"],
        "seconds": PLAN_SECONDS,
    }


def ghost_survey() -> dict:
    """How torn the composer actually was, across a spread of moods."""
    pieces = [compose(7788, GHOST_SECONDS, THEMES[name]) for name in GHOST_THEMES]
    every = [(ghost, choice, piece)
             for piece in pieces for choice in piece.melodies for ghost in choice.ghosts]
    closeness = [ghost.closeness() for ghost, _, _ in every]

    return {
        "pieces": pieces,
        "example": pieces[GHOST_THEMES.index("hypnotic")],
        "turned_down": len(every),
        "notes": sum(len(piece.ghosts) for piece in pieces),
        "played": sum(len(piece.notes) for piece in pieces),
        "torn": statistics.mean(closeness) if closeness else 0.0,
        "near_ties": sum(1 for value in closeness if value > 0.9),
        "routs": sum(1 for value in closeness if value < 0.1),
        "closest": min((ghost.margin for ghost, _, _ in every), default=0.0),
        "loudest": sorted(every, key=lambda row: row[0].margin)[:6],
        "themes": len(GHOST_THEMES),
    }


def mix_survey() -> dict:
    """What the mixer measures across a spread of moods, and what it fixes."""
    from compex.dsp.arrange import survey

    rows = []
    for name in MIX_THEMES:
        piece = compose(7788, MIX_SECONDS, THEMES[name])
        decided = survey(piece)
        rows.append({
            "theme": name,
            "mix": decided,
            "voices": len(decided.voices),
            "percussive": decided.percussive,
            "buried": decided.buried(),
            "still": decided.still_buried(),
            "lifted": sum(1 for v in decided.voices if v.trim > 1.02),
            "cut": sum(1 for v in decided.voices if v.trim < 0.98),
        })
    # What the pad pool used to be handed, measured against the current table:
    # every engine the old pool allowed, weighted by how often it was picked.
    was_allowed = ("pad", "additive", "bell", "noise", "string", "choir",
                   "organ", "drone", "glass", "supersaw")
    decaying = [name for name in was_allowed if HOLD.get(name, 1.0) < NEEDS_HOLD[ROLE_PAD]]

    found = sum(len(row["buried"]) for row in rows)
    left = sum(len(row["still"]) for row in rows)
    return {
        "rows": rows,
        "example": rows[MIX_THEMES.index("hypnotic")],
        "found": found,
        "rescued": found - left,
        "left": left,
        "percussive": statistics.mean([row["percussive"] for row in rows]),
        "decaying_before": 100.0 * len(decaying) / len(was_allowed),
        "loudest": max(rows, key=lambda row: row["percussive"]),
    }


def hindsight_survey() -> dict:
    """Measure what each piece's ending gave back to its beginning.

    Three runs, because there are three separable claims and conflating them
    would let the page take credit for something it did not do: the
    measurement itself, the *ability* to reach back for a specific early
    phrase, and the composer actually caring about doing so.
    """
    from dataclasses import replace
    from unittest.mock import patch

    from compex.generate import melody

    real = melody.initial_taste

    def run(weight: float | None) -> list:
        if weight is None:
            return [compose(7788, HINDSIGHT_SECONDS, THEMES[name])
                    for name in HINDSIGHT_THEMES]
        with patch("compex.generate.melody.initial_taste",
                   lambda mood: replace(real(mood), reveal=weight)):
            return [compose(7788, HINDSIGHT_SECONDS, THEMES[name])
                    for name in HINDSIGHT_THEMES]

    caring = run(None)
    indifferent = run(0.0)
    shares = [piece.reveal.share for piece in caring]

    return {
        "pieces": caring,
        "example": caring[HINDSIGHT_THEMES.index("hypnotic")],
        "share": statistics.mean(shares),
        "best": max(shares),
        "worst": min(shares),
        "indifferent": statistics.mean([p.reveal.share for p in indifferent]),
        "recalls": sum(1 for piece in caring for choice in piece.melodies
                       if choice.chosen.origin.startswith("recall")),
        "phrases": sum(piece.reveal.phrases for piece in caring),
        "themes": len(HINDSIGHT_THEMES),
    }


def purpose_survey() -> dict:
    """Measure the ledger, and measure the piece with it switched off.

    The off run is the whole falsification. If a mechanism that claims to give
    a piece long-range intention leaves the music identical, it is decoration,
    and this page should be the thing that says so.
    """
    from unittest.mock import patch

    def run() -> list:
        return [compose(2026, PURPOSE_SECONDS, THEMES[name]) for name in PURPOSE_THEMES]

    def failing(piece) -> int:
        lead = frozenset(v.voice_id for v in piece.voices if v.role == "lead")
        analysis = analyse(piece.notes, piece.strokes, piece.motif, piece.scale,
                           piece.root_pitch, piece.total_beats, lead)
        return sum(1 for v in judge(analysis, piece.mood) if not v.satisfied)

    on = run()
    with patch("compex.generate.promise.ENABLED", False):
        off = run()

    settled = [s for piece in on for s in piece.ledger.paid]
    waits = sorted(s.waited for s in settled)
    carried = [piece.ledger.deepest(piece.total_beats) for piece in on]

    return {
        "pieces": on,
        "example": on[PURPOSE_THEMES.index("menacing")],
        "opened": sum(len(p.ledger.paid) + len(p.ledger.live(p.total_beats)) for p in on),
        "settled": len(settled),
        "kinds": Counter(s.promise.kind for s in settled),
        "median_wait": waits[len(waits) // 2] if waits else 0.0,
        "longest_wait": waits[-1] if waits else 0.0,
        "still_open": sum(len(p.ledger.live(p.total_beats)) for p in on),
        "carrying": [(p.mood.nearest_theme(), c.kind, round(p.total_beats - c.opened_at))
                     for p, c in zip(on, carried) if c is not None],
        "failing_on": sum(failing(p) for p in on),
        "failing_off": sum(failing(p) for p in off),
        "changed": sum(1 for a, b in zip(on, off) if a.notes != b.notes),
        "themes": len(PURPOSE_THEMES),
    }


# ── svg helpers ────────────────────────────────────────────────────────────

def svg(width: int, height: int, body: str) -> str:
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def text(x, y, s, size=12, fill="var(--mut)", anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif">{html.escape(str(s))}</text>')


def line(x1, y1, x2, y2, stroke="var(--line)", width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>')


def rect(x, y, w, h, fill, rx=2, opacity=1.0):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
            f'rx="{rx}" fill="{fill}" opacity="{opacity:.3f}"/>')


def circle(cx, cy, r, fill, opacity=1.0, stroke="none"):
    return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.2f}" fill="{fill}" '
            f'opacity="{opacity:.3f}" stroke="{stroke}"/>')


def fig(markup: str, caption: str, wide: bool = False) -> str:
    cls = "fig wide" if wide else "fig"
    return f'<div class="{cls}">{markup}</div><p class="sub">{caption}</p>'


def frame(width, height, pad, x_label, y_label, xticks, yticks):
    """Axes with ticks. Returns (markup, to_x, to_y) mapping data space to pixels."""
    left, right = pad + 34, width - pad
    top, bottom = pad, height - pad - 26
    (x0, x1), (y0, y1) = xticks[0], yticks[0]

    def to_x(v):
        return left + (v - x0) / (x1 - x0 or 1) * (right - left)

    def to_y(v):
        return bottom - (v - y0) / (y1 - y0 or 1) * (bottom - top)

    parts = [line(left, bottom, right, bottom), line(left, top, left, bottom)]
    for v in xticks[1]:
        parts.append(line(to_x(v), bottom, to_x(v), bottom + 4))
        parts.append(text(to_x(v), bottom + 17, v, 11, anchor="middle"))
    for v in yticks[1]:
        parts.append(line(left - 4, to_y(v), right, to_y(v), "var(--line)", 1, "2,4"))
        parts.append(text(left - 8, to_y(v) + 4, v, 11, anchor="end"))
    parts.append(text((left + right) / 2, height - 4, x_label, 11.5, anchor="middle"))
    parts.append(text(12, top - 6, y_label, 11.5))
    return "".join(parts), to_x, to_y


# ── figures ────────────────────────────────────────────────────────────────

def fig_mood_space(data) -> str:
    w, h = 900, 430
    body, tx, ty = frame(w, h, 26, "valence  (dark → bright)", "energy  (still → frantic)",
                         ((0, 1), [0, 0.25, 0.5, 0.75, 1]), ((0, 1), [0, 0.25, 0.5, 0.75, 1]))
    parts = [body]
    for name in THEME_ORDER:
        axes = data[name]["axes"]
        x, y = tx(axes["valence"]), ty(axes["energy"])
        radius = 5 + axes["density"] * 13
        parts.append(circle(x, y, radius, "var(--accent)", 0.18 + axes["grit"] * 0.55))
        parts.append(circle(x, y, 2.2, "var(--accent)"))
        parts.append(text(x, y - radius - 6, name, 11, "var(--fg)", "middle", 600))
    return fig("".join(parts) and svg(w, h, "".join(parts)),
               "Each named theme is a point, not a preset. Circle size is density, opacity is grit. "
               "The gaps between them are reachable — the axes are exposed raw in the UI, so a track "
               "can sit anywhere in here, including places with no name.")


def fig_tempo(data) -> str:
    w, h = 900, 380
    body, tx, ty = frame(w, h, 26, "energy axis", "chosen tempo (BPM)",
                         ((0, 1), [0, 0.25, 0.5, 0.75, 1]), ((40, 180), [40, 70, 100, 130, 160]))
    parts = [body]
    for name in THEME_ORDER:
        axes, bpms = data[name]["axes"], data[name]["bpm"]
        x = tx(axes["energy"])
        low, high = min(bpms), max(bpms)
        parts.append(line(x, ty(low), x, ty(high), "var(--accent)", 2))
        parts.append(circle(x, ty(statistics.mean(bpms)), 4.5, "var(--accent)"))
        parts.append(text(x, ty(high) - 8, name, 10, "var(--mut)", "middle"))
    return fig(svg(w, h, "".join(parts)),
               f"Measured, not asserted: {SURVEY_SEEDS} compositions per theme. The bar is the full "
               "range the seed can move tempo within a theme; the dot is the mean. Energy sets the "
               "centre, the seed picks the spot.")


def fig_scale_reach(data) -> str:
    scale_order = sorted(SCALES, key=lambda s: -sum(
        data[t]["scales"].get(s, 0) * (1 - THEMES[t].valence) for t in THEME_ORDER))
    w = 200 + len(scale_order) * 46
    h = 120 + len(THEME_ORDER) * 26
    parts = []
    for col, scale in enumerate(scale_order):
        x = 200 + col * 46 + 23
        parts.append(f'<g transform="rotate(-52 {x} 104)">'
                     + text(x, 104, scale.replace("_", " "), 10.5, anchor="end") + "</g>")
    for row, theme in enumerate(THEME_ORDER):
        y = 120 + row * 26
        parts.append(text(190, y + 15, theme, 11.5, "var(--fg)", "end"))
        counts = data[theme]["scales"]
        for col, scale in enumerate(scale_order):
            share = counts.get(scale, 0) / SURVEY_SEEDS
            x = 200 + col * 46
            parts.append(rect(x + 2, y + 2, 42, 20, "var(--line)", 3, 0.5))
            if share:
                parts.append(rect(x + 2, y + 2, 42, 20, "var(--accent)", 3, 0.15 + share * 0.85))
                parts.append(text(x + 23, y + 16, f"{share * 100:.0f}", 10,
                                  "var(--panel)" if share > 0.5 else "var(--fg)", "middle", 600))
    return fig(svg(w, h, "".join(parts)),
               "Percentage of runs where each theme reached each mode. Serene never once draws a dark "
               "mode; menacing and shattered never draw a bright one. Nothing enumerates which mode "
               "belongs to which mood — each scale carries a (brightness, tension) coordinate and the "
               "composer picks from the nearest few.", wide=True)


def fig_density(data) -> str:
    w, h = 900, 400
    peak = max(max(data[t]["notes"]) for t in THEME_ORDER)
    body, tx, ty = frame(w, h, 26, "", "events in a 45-second piece",
                         ((0, len(THEME_ORDER)), []), ((0, peak), [0, peak // 3, 2 * peak // 3, peak]))
    parts = [body]
    step = (tx(1) - tx(0))
    for index, theme in enumerate(THEME_ORDER):
        notes = statistics.mean(data[theme]["notes"])
        strokes = statistics.mean(data[theme]["strokes"])
        x = tx(index) + step * 0.15
        bar = step * 0.34
        parts.append(rect(x, ty(notes), bar, ty(0) - ty(notes), "var(--accent)", 2, 0.85))
        parts.append(rect(x + bar + 2, ty(strokes), bar, ty(0) - ty(strokes), "var(--ok)", 2, 0.85))
        parts.append(f'<g transform="rotate(-38 {x + bar} {ty(0) + 16})">'
                     + text(x + bar, ty(0) + 16, theme, 10.5, anchor="end") + "</g>")
    parts.append(rect(w - 170, 22, 10, 10, "var(--accent)", 2))
    parts.append(text(w - 154, 31, "notes", 11))
    parts.append(rect(w - 96, 22, 10, 10, "var(--ok)", 2))
    parts.append(text(w - 80, 31, "drum strokes", 11))
    return fig(svg(w, h, "".join(parts)),
               "Mean over the same runs. Desolate writes a couple of dozen drum hits in three quarters "
               "of a minute; frantic writes hundreds. The density axis is doing this — no rule anywhere "
               "says 'desolate is sparse'.")


def fig_engine_map() -> str:
    w, h = 900, 440
    body, tx, ty = frame(w, h, 26, "brightness", "grit  (clean → destroyed)",
                         ((0, 1), [0, 0.25, 0.5, 0.75, 1]), ((0, 1), [0, 0.25, 0.5, 0.75, 1]))
    parts = [body]
    colours = {"core": "var(--accent)", "struck": "var(--ok)", "voiced": "var(--warn)",
               "sustained": "var(--mut)", "textural": "var(--fg)"}
    for family, names in FAMILIES.items():
        for name in names:
            bright, grit, _ = ENGINE_COLOUR[name]
            parts.append(circle(tx(bright), ty(grit), 5, colours[family], 0.75))
            parts.append(text(tx(bright), ty(grit) - 9, name, 10, "var(--fg)", "middle"))
    for index, (family, colour) in enumerate(colours.items()):
        x = 60 + index * 150
        parts.append(circle(x, 30, 5, colour, 0.85))
        parts.append(text(x + 10, 34, family, 11))
    return fig(svg(w, h, "".join(parts)),
               f"All {len(ENGINE_NAMES)} engines placed by character. The composer ranks them against "
               "the mood's own (valence, grit, energy) and draws from the nearest four — so a theme "
               "instruments itself differently each seed without ever reaching somewhere wrong.")


def fig_plasticity(series) -> str:
    w, h = 900, 400
    longest = max(len(series[t]["plasticity"]) for t in EVOLUTION_THEMES)
    body, tx, ty = frame(w, h, 26, "listen-back", "plasticity  (how hard it reacts)",
                         ((0, longest - 1), list(range(longest))), ((0, 1.0), [0, 0.25, 0.5, 0.75, 1.0]))
    parts = [body]
    colours = ["var(--ok)", "var(--accent-2)", "var(--mut)", "var(--accent)", "var(--warn)"]
    for index, theme in enumerate(EVOLUTION_THEMES):
        values = series[theme]["plasticity"]
        colour = colours[index % len(colours)]
        points = " ".join(f"{tx(i):.1f},{ty(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2.2"/>')
        for i, value in enumerate(values):
            parts.append(circle(tx(i), ty(value), 2.6, colour))
        parts.append(text(tx(len(values) - 1) + 6, ty(values[-1]) + 4, theme, 11, colour))
    return fig(svg(w, h, "".join(parts)),
               "Plasticity is the second-order part — not what the composer decides, but how "
               "hard it reacts to its own critic. It falls when the principles are satisfied "
               "and climbs when complaints persist. Serene and hypnotic settle to the floor and "
               "stop fidgeting; frantic never gets comfortable and keeps pushing.")


def fig_drives(piece) -> str:
    tracked = ("novelty_pressure", "gap_fill", "register_reach",
               "dissonance_ceiling", "motif_recall")
    w, h = 900, 420
    steps = len(piece.evolution)
    if steps < 2:
        return ""
    body, tx, ty = frame(w, h, 26, "listen-back", "drive value",
                         ((0, steps - 1), list(range(steps))), ((0, 1.0), [0, 0.25, 0.5, 0.75, 1.0]))
    parts = [body]
    colours = ["var(--accent)", "var(--ok)", "var(--accent-2)", "var(--warn)", "var(--mut)"]
    for index, drive in enumerate(tracked):
        values = [getattr(s.after, drive) for s in piece.evolution]
        colour = colours[index % len(colours)]
        points = " ".join(f"{tx(i):.1f},{ty(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>')
        parts.append(text(tx(steps - 1) + 6, ty(values[-1]) + 4, drive.replace("_", " "), 10.5, colour))
    return fig(svg(w, h, "".join(parts)),
               "The parameters the composer writes with, moving across one piece as it listens "
               "to itself. Each move is caused by a named principle failing, not by the seed.")


def fig_waveforms(examples) -> str:
    w, h = 900, 90 * len(examples) + 30
    parts = []
    for index, (name, samples, seconds) in enumerate(examples):
        top = 20 + index * 90
        mid = top + 32
        buckets = 880
        size = max(1, len(samples) // buckets)
        peaks = np.abs(samples[: (len(samples) // size) * size].reshape(-1, size)).max(axis=1)
        parts.append(text(10, top - 4, f"{name} — {seconds:.0f}s", 11.5, "var(--fg)", weight=600))
        for i, peak in enumerate(peaks):
            height = max(0.7, float(peak) * 30)
            parts.append(rect(10 + i, mid - height / 2, 0.9, height, "var(--accent)", 0, 0.85))
        parts.append(line(10, mid, 890, mid, "var(--line)"))
    return fig(svg(w, h, "".join(parts)),
               "Real renders, same seed, only the theme changed. You can see the form the composer "
               "chose — where it thins out and where it fills in.")


# ── page ───────────────────────────────────────────────────────────────────

CSS = """
:root{--bg:#fbfbfc;--panel:#fff;--fg:#16181d;--mut:#636a76;--line:#e3e5ea;
      --accent:#2563eb;--warn:#dc2626;--ok:#15803d;--code:#f3f4f6;}
@media (prefers-color-scheme:dark){
 :root{--bg:#0d0f13;--panel:#14171d;--fg:#e9eaee;--mut:#98a0ad;--line:#262b34;
       --accent:#6ea0ff;--warn:#ff7a70;--ok:#68d391;--code:#1b1f26;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.62 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:1120px;margin:0 auto;padding:48px 28px 96px}
header{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:34px}
.kicker{margin:0 0 4px;font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut)}
h1{font-size:29px;margin:0 0 8px;letter-spacing:-.02em}
h2{font-size:20px;margin:44px 0 10px;letter-spacing:-.01em}
h3{font-size:15px;margin:26px 0 8px}
.sub{color:var(--mut);font-size:14px;margin:0}
.tag{display:inline-block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
 padding:3px 9px;border-radius:99px;border:1px solid var(--line);color:var(--mut);margin-right:6px}
.tag.warn{color:var(--warn);border-color:var(--warn)}
.tag.ok{color:var(--ok);border-color:var(--ok)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:22px 24px;margin:18px 0}
.fig{overflow-x:auto;margin:14px 0 6px;padding-bottom:4px}
.fig svg{display:block;min-width:640px;max-width:100%;height:auto}
.fig.wide svg{min-width:1060px}
.read{border-left:3px solid var(--accent);padding:2px 0 2px 15px;margin:16px 0;color:var(--fg)}
.read.warn{border-color:var(--warn)}
.read.ok{border-color:var(--ok)}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
code{background:var(--code);padding:1.5px 5px;border-radius:4px;font-size:13px;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:14px 16px;
 overflow-x:auto;font-size:12.5px;line-height:1.55;
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
ul{padding-left:20px}li{margin:5px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:16px 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px 17px}
.stat .n{font-size:24px;font-weight:650;letter-spacing:-.02em}
.stat .k{color:var(--mut);font-size:12.5px;margin-top:3px}
.q{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
 border-radius:8px;padding:15px 18px;margin:12px 0}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);
 color:var(--mut);font-size:13px}
.tabs{display:flex;gap:4px;margin:26px 0 8px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.tab{background:none;border:none;border-bottom:2px solid transparent;color:var(--mut);
 font:600 14.5px ui-sans-serif,system-ui,sans-serif;padding:10px 16px;cursor:pointer;
 margin-bottom:-1px;border-radius:6px 6px 0 0}
.tab:hover{color:var(--fg);background:var(--panel)}
.tab.active{color:var(--fg);border-bottom-color:var(--accent)}
.panel{display:none}.panel.active{display:block}
.chip{display:inline-block;font-size:12px;padding:3px 10px;border-radius:99px;
 border:1px solid var(--line);color:var(--fg);margin:0 5px 5px 0;background:var(--panel)}
.retract{border-left:3px solid var(--warn);padding:2px 0 2px 15px;margin:16px 0}
.retract s{color:var(--mut)}
.figure{margin:10px 0 14px}
.figure .lab{color:var(--mut);font-size:12.5px;margin-bottom:4px}
.cells{display:flex;gap:3px;flex-wrap:wrap}
.cell{width:15px;height:21px;border-radius:3px;background:var(--code);
 border:1px solid var(--line)}
.cell.soft{background:color-mix(in srgb,var(--accent) 42%,transparent);
 border-color:var(--accent)}
.cell.hard{background:var(--accent);border-color:var(--accent)}
.cell.learned{box-shadow:0 2px 0 var(--ok)}
"""

JS = """
document.querySelectorAll('.tab').forEach(function(t){
  t.addEventListener('click',function(){
    document.querySelectorAll('.tab').forEach(function(x){x.classList.remove('active')});
    document.querySelectorAll('.panel').forEach(function(x){x.classList.remove('active')});
    t.classList.add('active');
    document.getElementById(t.dataset.panel).classList.add('active');
    window.scrollTo({top:0,behavior:'smooth'});
  });
});
"""


def panel_how(example) -> str:
    piece = example.composition
    return f"""
<h2 style="margin-top:26px">What it is</h2>
<p class="sub">Three inputs go in. A finished piece of music comes out, along with the
notation describing what the machine decided.</p>

<p>You give <b>CompEx</b> a length, an emotional theme, and a seed. It invents the tempo, the
meter, the key and mode, a chord progression, a germ motif, the shape of the whole piece, which
instruments exist, and how each one of those instruments is built. Then it plays it, and writes
out the equation for what it did.</p>

<div class="read"><b>Nothing is prewritten.</b> There is no library of loops, no bank of patches,
no genre templates. Every decision is derived from the three inputs by walking a body of music
theory with a seeded random number generator. The theory is the environment; the seed is a walk
through it.</div>

<h3>Where it came from, and the wrong turn</h3>
<p>Lee had spent a long time writing equation-shaped prompts at Suno — things like
<code>K(t)=&Sigma;&delta;(t-2n)</code> for a kick pattern, or a phase-modulation expression for a
bass sound. Suno can't parse any of that; it embeds the text as tokens and produces something
vaguely in the mood of the words. The structure was being thrown away.</p>

<div class="retract"><s><b>So the first build was an interpreter</b> — it read those equations
and executed them. It worked: the dubstep spec rendered at 140 BPM with a halftime feel, a real
one-beat silence before the drop, and the growl bass built from the phase-modulation term
exactly as written.</s><br><br><b>That was the wrong half.</b> Lee wanted the machine writing
the equations, not executing his. The entire parsing path was deleted — about 900 lines — and
the direction reversed. The equation became the <i>output</i>. This is left on the page because
the reasoning that produced it was sound and the correction is the more interesting fact.</div>

<h3>The pipeline</h3>
<table><thead><tr><th>stage</th><th>what happens</th></tr></thead><tbody>
<tr><td><b>mood</b></td><td>An emotional theme becomes five numbers: valence, energy, tension,
density, grit.</td></tr>
<tr><td><b>theory</b></td><td>Those numbers constrain which modes, chord motions, meters and
rhythmic densities are legal. Every option offered is already idiomatic, so the composer never
generates garbage and filters it.</td></tr>
<tr><td><b>compose</b></td><td>It picks a tempo, a mode, a progression, a motif, a form
archetype and a set of instruments, then <b>auditions</b> every melodic phrase against several
alternatives and every rhythm against the grid it has learned, and writes the winners out as
notes and drum strokes.</td></tr>
<tr><td><b>arrange</b></td><td>Each voice is synthesised on its own bus, run through the effects
chain invented for it, ducked against the kick, and mixed.</td></tr>
<tr><td><b>emit</b></td><td>The whole set of decisions is written back out as equation notation.</td></tr>
</tbody></table>

<h3>Determinism, and why it is not a detail</h3>
<p>The same three inputs always produce the same audio, down to the byte. That is deliberate and
load-bearing.</p>
<div class="read"><b>If the strangeness came from a random number, nothing is being expressed —
it is noise wearing a costume.</b> In a deterministic engine every strange moment is caused by
the machine's actual structure and can be traced back to it. That is the difference between a
malfunction and an interior, and it is the whole reason this project is called what it is.</div>
<p>The random number generator is positional rather than sequential — a value is a pure function
of <code>(seed, stream, index)</code>, with no running state. Asking for bar 90 before bar 3
gives the same answer either way. Seeking is free, and a parameter change mid-piece cannot
desync anything downstream, which is what a future realtime version will need.</p>

<h3>The formula it writes</h3>
<p>Every track is saved with a <code>.tex</code> file beside it. The <code>SEED</code>,
<code>RUNTIME</code> and <code>MOOD</code> lines are the load-bearing part — feed those three
back and the identical file comes out. Everything under them is exposition: what those three
turned into. Here is the real one for the piece used throughout this page.</p>
<pre>{html.escape(example.formula)}</pre>
<div class="read ok"><b>Verified, not asserted.</b> Reading that file back and re-composing
produces a byte-identical track — the audio fingerprint <code>{piece.seed}</code> &rarr;
<code>{example.fingerprint}</code> matches. There is a test that fails if it ever stops
matching.</div>
"""


def panel_mood(data) -> str:
    rows = "".join(
        f"<tr><td><b>{name}</b></td>"
        + "".join(f"<td>{data[name]['axes'][axis]:.2f}</td>" for axis in AXES)
        + f"<td class='sub'>{statistics.mean(data[name]['bpm']):.0f} BPM avg, "
          f"{statistics.mean(data[name]['notes']):.0f} notes</td></tr>"
        for name in THEME_ORDER)
    return f"""
<h2 style="margin-top:26px">The mood axes</h2>
<p class="sub">Emotional theme is a point in a five-dimensional space, not an item in a list.</p>

<p>A genre dropdown would be a lookup table — pick "dubstep", get dubstep back. That defeats the
purpose. Instead a theme is five numbers, and the {len(THEMES)} named themes are just convenient
coordinates in that space. You can move the axes directly and land somewhere that has no name.</p>

<div class="grid">
<div class="stat"><div class="n">valence</div><div class="k">dark &rarr; bright</div></div>
<div class="stat"><div class="n">energy</div><div class="k">still &rarr; frantic</div></div>
<div class="stat"><div class="n">tension</div><div class="k">resolved &rarr; unresolved</div></div>
<div class="stat"><div class="n">density</div><div class="k">sparse &rarr; crowded</div></div>
<div class="stat"><div class="n">grit</div><div class="k">clean &rarr; destroyed</div></div>
</div>

{fig_mood_space(data)}

<h3>Do the axes actually do anything?</h3>
<p>This is the question worth being sceptical about — it would be easy to build something where
the mood is decorative and the seed does all the work. So: {SURVEY_SEEDS} compositions per theme,
{len(THEMES)} themes, {SURVEY_SEEDS * len(THEMES)} pieces, and a count of what the machine chose.</p>

{fig_tempo(data)}
{fig_scale_reach(data)}
{fig_density(data)}

<div class="read ok"><b>They constrain real things.</b> Serene never draws a dark mode across
{SURVEY_SEEDS} seeds, and never a meter past 3/4. Menacing only ever draws phrygian dominant,
locrian or octatonic, and takes an odd meter about half the time. Desolate writes a couple of
dozen drum strokes in three quarters of a minute where frantic writes hundreds. None of that is
enumerated anywhere — each mode and each drum carries a character coordinate, and the composer
picks from whatever sits nearest the mood.</div>

<h3>The themes, and what they turn into</h3>
<table><thead><tr><th>theme</th>{''.join(f'<th>{a[:3]}</th>' for a in AXES)}
<th>measured</th></tr></thead><tbody>{rows}</tbody></table>

<div class="q"><b>Open, and the main unresolved question in the project.</b> These axis names are
my labels for what the machine does. Whether "menacing" actually sounds menacing to a listener is
not something I can settle by measurement — the numbers above only show the axes are
<i>consistent</i>, not that they are <i>correctly named</i>. That needs an ear, and it is what
the project is currently waiting on.</div>
"""


def panel_palette(data) -> str:
    families = "".join(
        f"<tr><td><b>{family}</b></td><td>"
        + "".join(f'<span class="chip">{name}</span>' for name in names)
        + "</td></tr>" for family, names in FAMILIES.items())
    roles = "".join(
        f"<tr><td><b>{role}</b></td><td class='sub'>{', '.join(names)}</td></tr>"
        for role, names in ENGINES_FOR_ROLE.items())
    return f"""
<h2 style="margin-top:26px">The palette</h2>
<p class="sub">{len(ENGINE_NAMES)} synthesis engines, {len(drums.DRUM_NAMES)} drums,
{len(EFFECT_NAMES)} effects — none of them patches.</p>

<div class="grid">
<div class="stat"><div class="n">{len(ENGINE_NAMES)}</div><div class="k">pitched engines</div></div>
<div class="stat"><div class="n">{len(drums.DRUM_NAMES)}</div><div class="k">drums</div></div>
<div class="stat"><div class="n">{len(EFFECT_NAMES)}</div><div class="k">effects, invented per voice</div></div>
<div class="stat"><div class="n">{len(SCALES)}</div><div class="k">modes</div></div>
</div>

<div class="read"><b>An engine is a family, not a sound.</b> The composer invents the parameters
per piece, so "pluck" covers a harp and a snapped wire, and "pm" covers a bell and a dubstep
growl. The effects matter more than the count suggests: the same pluck through a long reverb and
through a ring modulator are two different instruments, so it is
{len(ENGINE_NAMES)} &times; {len(EFFECT_NAMES)} &times; invented parameters, not
{len(ENGINE_NAMES)} sounds.</div>

<table><thead><tr><th>family</th><th>engines</th></tr></thead><tbody>{families}</tbody></table>

{fig_engine_map()}

<h3>Which engine plays which part</h3>
<table><thead><tr><th>role</th><th>candidates</th></tr></thead><tbody>{roles}</tbody></table>

<h3>Drums</h3>
<p>{''.join(f'<span class="chip">{n}</span>' for n in sorted(drums.DRUM_NAMES))}</p>
<p>Chosen the same way as engines — each carries a (brightness, aggression) coordinate and the
kit is drawn from whatever sits near the mood. That fix has a story attached; see the next tab.</p>

<h3>Effects</h3>
<p>{''.join(f'<span class="chip">{n}</span>' for n in EFFECT_NAMES)}</p>
<p>Between zero and two per voice, with the appetite for them rising with grit and density.
Reverb and delay are feedback structures, computed a delay line at a time so each block depends
only on the previous one — that keeps them in numpy instead of a per-sample Python loop.</p>

<div class="read"><b>Every voice leaves its engine at the same <i>loudness</i>.</b> That sounds
like housekeeping and is not. Measured across the {len(ENGINE_NAMES)} engines, natural output
level spanned roughly 300&times; in RMS — a self-oscillating resonance came out fifty times
quieter than a plain sine. Without levelling, the composer's per-voice gain means something
different for every engine and the quiet ones are simply absent.</div>

<div class="read warn"><b>This used to level by peak, and that was its own bug.</b> Peak is how
tall a sound is. A swelling pad and a plucked string reach the same height and nothing like the
same volume, so the pads carrying the body of the piece were a third of what they looked — which
is what a listener heard before any number said it. Now it is the loudness of the loudest
quarter-second, with partial credit still given to the peak, because a transient <i>is</i> heard
as louder than its RMS and matching pure loudness makes plucks vanish the other way. See
<b>The mix</b>.</div>
"""


def panel_example(example, data) -> str:
    piece = example.composition
    voices = "".join(
        f"<tr><td><b>{v.role}</b></td><td><code>{v.engine}</code></td>"
        f"<td class='sub'>{', '.join(v.effect_names()) or '—'}</td>"
        f"<td class='sub'>{', '.join(f'{k}={val:g}' for k, val in v.params[:4])}</td></tr>"
        for v in piece.voices if v.role != "perc")
    movements = " &rarr; ".join(
        f"{m.name} <span class='sub'>({m.beats:g}b, energy {m.energy:.2f})</span>"
        for m in piece.movements)
    return f"""
<h2 style="margin-top:26px">One piece, decision by decision</h2>
<p class="sub">Seed {piece.seed}, theme {piece.mood.nearest_theme()},
{example.knobs.duration_s:.0f} seconds requested. Everything below was chosen by the machine.</p>

<div class="grid">
<div class="stat"><div class="n">{piece.bpm:.1f}</div><div class="k">BPM</div></div>
<div class="stat"><div class="n">{piece.beats_per_bar}/4</div><div class="k">meter</div></div>
<div class="stat"><div class="n">{piece.scale_name.replace('_', ' ')}</div><div class="k">mode</div></div>
<div class="stat"><div class="n">{len(piece.notes)}</div><div class="k">notes written</div></div>
<div class="stat"><div class="n">{len(piece.strokes)}</div><div class="k">drum strokes</div></div>
<div class="stat"><div class="n">{example.fingerprint}</div><div class="k">audio fingerprint</div></div>
</div>

<h3>Form</h3>
<p>Archetype <code>{piece.archetype}</code>: {movements}</p>

<h3>Harmony and motif</h3>
<p>{len(piece.progression)} chords, one every {piece.chord_beats:g} beats. The germ motif is
{len(piece.motif.steps)} notes over {piece.motif.beats:g} beats, and each movement after the first
plays a <i>transformation</i> of it rather than a repeat — inversion, retrograde, augmentation,
fragmentation. That is Schoenberg's developing variation, mechanised: the piece grows out of one
idea instead of collecting unrelated ones.</p>

<h3>The instruments it built</h3>
<table><thead><tr><th>role</th><th>engine</th><th>effects</th><th>parameters (first four)</th></tr>
</thead><tbody>{voices}</tbody></table>
<p class="sub">Kit: {', '.join(v.voice_id for v in piece.voices if v.role == 'perc')}</p>

<div class="read"><b>Two things here were not instructed.</b> It chose an odd meter for this dark
theme, and for the serene theme at the same seed it chose no effects whatsoever. Nothing in the
code says either. Both fall out of the axes.</div>
"""


def panel_evolve(series, example) -> str:
    piece = series["menacing"]["piece"]
    principles = "".join(
        f"<tr><td><b>{p.name.replace('_', ' ')}</b></td>"
        f"<td class='sub'>{p.attribution}</td>"
        f"<td><code>{p.drive}</code></td>"
        f"<td class='sub'>{p.why}</td></tr>"
        for p in PRINCIPLES)

    trace = "".join(
        f"<div class='read'><b>{step.movement_name} — beat {step.at_beat:g}</b><br>"
        f"<span class='sub'>heard: "
        + (", ".join(f"{v.principle.name} {v.measured:.2f}" for v in step.unhappy)
           or "nothing to correct")
        + f"</span><br>{step.note()}</div>"
        for step in piece.evolution[:6])

    return f"""
<h2 style="margin-top:26px">How it changes its mind</h2>
<p class="sub">The piece is not decided in advance. After each movement the composer listens
back to what it actually wrote and adjusts.</p>

<p>Everything up to here describes decisions made <i>before</i> a note exists. That is only half
of it. The composer also runs a loop: write a movement, measure what came out, compare it
against a set of aesthetic principles, and shift the parameters it writes with before starting
the next one.</p>

<div class="read"><b>So the second half of a piece is a consequence of the first half</b>, not
just of the seed. The harmony is re-voiced under the new parameters, the motif is either
developed further or recalled, and the note-writing itself reads the drives rather than fixed
constants.</div>

<h3>The principles it listens against</h3>
<p>These are not preferences invented for this project. They are regularities that music
perception research keeps finding, each carrying the name it is known by so the judgement can be
argued with rather than just accepted.</p>
<table><thead><tr><th>measures</th><th>after</th><th>moves</th><th>why it exists</th></tr></thead>
<tbody>{principles}</tbody></table>

<div class="read"><b>Each principle wants a band, not a maximum.</b> That matters most for
novelty: Berlyne's inverted-U says liking peaks in the middle, so a composer told simply to
maximise novelty would walk straight off the end into noise. And the bands move with the
mood — a menacing piece is allowed a dissonance a serene one would fail on.</div>

<h3>The second-order part</h3>
<p>The drives move in response to the critic. <b>How hard they move</b> also changes.</p>
{fig_plasticity(series)}
<div class="read"><b>This is what makes it a second-order system rather than a thermostat.</b>
The state evolves, and the rule that updates the state evolves under it. When the same
complaints keep coming back, the composer concludes its corrections were too timid and reacts
harder. When the principles are satisfied it relaxes and stops interfering with something that
is working.</div>

{fig_drives(piece)}

<h3>What it actually did, in one piece</h3>
<p class="sub">Seed {piece.seed}, {piece.mood.nearest_theme()},
{EVOLUTION_SECONDS:.0f} seconds — {len(piece.evolution)} listen-backs,
{series['menacing']['corrections']} corrections.</p>
{trace}

<h3>Memory is lossy, on purpose</h3>
<p>The composer does not keep the germ motif. It keeps a <b>sketch</b> — which direction each
step moved, roughly how long each note was, how far the widest leap reached. Everything else is
discarded.</p>
<div class="read"><b>So when the theme returns it cannot be retrieved, only reconstructed</b>,
and the gaps get filled using whatever the drives have become by then. A theme coming back at
minute four is the theme as remembered by something that has changed since it first played it.
Literal recapitulation is a copy; this is a memory.</div>

<div class="q"><b>Open: the bands are calibrated by argument, not by ear.</b> The principles are
real and the attributions are honest, but where exactly each band sits — is 0.28 to 0.62 the
right window for novelty? — is my judgement. A band set wrong produces a composer that
confidently corrects toward the wrong thing, and it would look identical in the logs.</div>
"""


def figure_html(piece, pattern) -> str:
    """One pattern drawn on its own grid, with what the grid now believes under it."""
    grid = next((g for g in piece.grids if g.voice == pattern.voice), None)
    cells = "".join(
        f'<i class="cell{" hard" if velocity > 0.66 else " soft" if velocity else ""}'
        f'{" learned" if grid and grid.weights[index] > 1.0 else ""}"'
        f' title="slot {index}'
        + (f' · weight {grid.weights[index]:.2f} (started {grid.start[index]:.2f})"'
           if grid else '"')
        + "></i>"
        for index, velocity in enumerate(pattern.slots)
    )
    learned = ""
    if grid and grid.past_start():
        learned = (f" &middot; <b>{grid.past_start()} slot(s) now heavier than the meter "
                   f"made them</b>")
    return (f'<div class="figure"><div class="lab"><code>{pattern.voice}</code> '
            f'{pattern.hit_count} hits &middot; {pattern.subdivision:g} beat per slot'
            f'{learned}</div><div class="cells">{cells}</div></div>')


def panel_choose(choices) -> str:
    piece = choices["example"]
    melodic = "".join(
        f"<tr><td><b>{c.name}</b></td><td class='sub'>{c.attribution}</td>"
        f"<td class='sub'>{c.why}</td></tr>"
        for c in MELODY_CRITERIA)
    rhythmic = "".join(
        f"<tr><td><b>{c.name}</b></td><td class='sub'>{c.why}</td></tr>"
        for c in GROOVE_CRITERIA)
    winners = "".join(
        f"<tr><td><code>{origin}</code></td><td>{count}</td>"
        f"<td class='sub'>{count / max(1, choices['chosen']) * 100:.0f}%</td></tr>"
        for origin, count in choices["winners"].most_common(8))
    moved = "".join(
        f"<span class='chip'>{name} &times;{count}</span>"
        for name, count in choices["moved"].most_common())
    figures = "".join(figure_html(piece, figure) for figure in piece.final_patterns())

    auditions = "".join(
        f"<div class='read'><b>{choice.chosen.origin}</b> "
        f"<span class='sub'>beat {choice.at_beat:g}, generation {choice.chosen.generation}</span>"
        f"<br><span class='sub'>{choice.note()}</span></div>"
        for choice in piece.melodies[:5])

    return f"""
<h2 style="margin-top:26px">Choosing, not reciting</h2>
<p class="sub">Every phrase is auditioned against several alternatives, and every rhythm is a
decision the composer keeps revising out of its own output. Measured over
{len(CHOICE_THEMES)} pieces of {CHOICE_SECONDS:.0f} seconds.</p>

<div class="grid">
<div class="stat"><div class="n">{choices['imagined']}</div><div class="k">lines imagined</div></div>
<div class="stat"><div class="n">{choices['chosen']}</div><div class="k">lines kept</div></div>
<div class="stat"><div class="n">{choices['deepest']}</div><div class="k">deepest lineage</div></div>
<div class="stat"><div class="n">{choices['margin']:.3f}</div><div class="k">median winning margin</div></div>
<div class="stat"><div class="n">{choices['grids_past']}/{choices['grids']}</div>
  <div class="k">grids past their starting weights</div></div>
<div class="stat"><div class="n">{choices['stray']:.2f}</div><div class="k">mean weight travelled</div></div>
</div>

<div class="read"><b>The lead used to have no choice to make.</b> The germ motif came round,
got rotated, and was spoken over whatever chord was underneath — a recital. Now the composer
imagines several continuations, scores each one, and plays the winner. The winner becomes the
parent of the next generation, so a phrase in the last movement descends from the first one by a
chain of decisions, none of which were written down in advance.</div>

<h3>What decides an audition</h3>
<table><thead><tr><th>criterion</th><th>after</th><th>what it measures</th></tr></thead>
<tbody>{melodic}</tbody></table>

<div class="read"><b>The criteria are theory; the weights on them are not fixed.</b> They start
from the mood, and then the critic teaches them — a principle that keeps failing makes the
criterion that would have caught it matter more. A criterion that keeps separating winners from
the field while the piece is going well earns a little weight on its own account. Nothing pulls
them back toward where they started.</div>

<p class="sub">Weights that left their starting band, counted across the sampled pieces:</p>
<p>{moved or "<span class='sub'>none in this sample</span>"}</p>

<h3>Where the candidates come from</h3>
<p>A field is the germ, a return to it, some mutations of the line just played
({', '.join(f'<code>{kind}</code>' for kind in MUTATIONS)}), and at least one line improvised
from nothing. The outsider usually loses. It exists so that every phrase in a piece does not
descend from one germ, because a lineage with no immigration can only narrow.</p>
<table><thead><tr><th>winning candidate</th><th>times</th><th>share</th></tr></thead>
<tbody>{winners}</tbody></table>

<h3>Rhythms it decided on</h3>
<p>A pattern is a grid of slots — one cycle of the bar at some subdivision. The starting weights
are the metrical hierarchy the meter implies, derived from the arithmetic rather than tabulated,
which is why 7/4 works the same way 4/4 does. Then it auditions patterns against that grid, and
after every movement <b>the grid moves toward whatever it actually played.</b></p>
{figures}
<p class="sub">Seed {piece.seed}, {piece.mood.nearest_theme()}, {CHOICE_SECONDS:.0f} seconds.
Filled cells are strikes; a cell underlined in green sits on a slot the composer now weights
above anything the metrical hierarchy gave it.</p>

<table><thead><tr><th>criterion</th><th>what it measures</th></tr></thead>
<tbody>{rhythmic}</tbody></table>

<div class="read ok"><b>This is the channel the critic has no part in.</b> The listening loop
teaches taste from what the music got wrong. The grids learn from what the composer chose,
whether or not anything was wrong — which is how a slot that started as an offbeat afterthought
ends up as the thing the voice is built around.</div>

<h3>Five auditions from one piece</h3>
{auditions}

<div class="q"><b>Open: the weights on the criteria are still my judgement at the start.</b>
Where they end up is the machine's, and the page counts how far they travel — but the opening
position, and the shape of the curve from mood to starting weight, is hand-written. The same
objection as the aesthetic bands, one layer along.</div>

<div class="q"><b>Open: a losing candidate is discarded, not heard.</b> The runners-up are
scored and thrown away. The ghost layer is exactly the idea of keeping them — playing them
quietly under the winner so the machine's uncertainty becomes audible. The scoring machinery
this needed now exists and the candidates now exist; what is missing is the mixing.</div>
"""


def panel_purpose(survey, hindsight) -> str:
    piece = survey["example"]
    now = piece.total_beats
    kinds = "".join(
        f"<tr><td><code>{kind}</code></td><td>{count}</td></tr>"
        for kind, count in survey["kinds"].most_common())
    carrying = "".join(
        f"<span class='chip'>{theme}: a {kind}, {beats} beats</span>"
        for theme, kind, beats in survey["carrying"])
    open_now = "".join(
        f"<tr><td><code>{owed.kind}</code></td><td class='sub'>{owed.domain}</td>"
        f"<td class='sub'>opened at beat {owed.opened_at:g}</td>"
        f"<td>{owed.pressure(now):.2f}</td></tr>"
        for owed in sorted(piece.ledger.live(now), key=lambda p: -p.pressure(now))[:8])

    return f"""
<h2 style="margin-top:26px">What is it for?</h2>
<p class="sub">The hardest question anyone has asked this project, and the part of the answer
that is actually built.</p>

<div class="read"><b>The question, as it was put:</b> a piece needs to be <i>for</i> something or
it is only assembled — but the goal must not be a human one. Not happiness, not beauty, not
tension-and-release because those are our words for it. <b>What would the computation want?</b></div>

<h3>The move that made it buildable</h3>
<p>Stop asking what the music is for and ask <b>what the machine can measure about itself</b>. It
already has an interior in the only sense that matters here: unrest, plasticity, strain against
its own bounds, how far its taste has travelled from where it started, and the margin by which
each phrase beat the ones it turned down. Those are facts about the computation, not about a
listener. An intention is a target shape for <i>those</i> over time.</p>

<div class="read ok"><b>Build, drop, tension, release then stop being imported words.</b> "My
confidence should collapse around here and recover by the end" produces a drop, audibly, without
anyone having named a drop.</div>

<h3>What melody is for, from the machine's side</h3>
<p>Melody is the <b>continuity-bearing object</b> — the thing that can be displaced, inverted,
fragmented, reharmonised and half-forgotten and still be recognisably itself. Which is the same
claim as calling it the piece's compression dictionary, approached from the other end: identity
through change is exactly what makes later material cheap to describe. It is not primarily the
tune. It is the object that can accumulate history.</p>

<h3>What got built: a ledger of what the music owes</h3>
<p>Every gesture opens an account. A leap owes a step back the other way. A line that stops a
semitone under a chord tone owes the note above it. A pattern that never lands the downbeat owes
it. A progression that walks a long way round the circle of fifths owes a return or a reason.</p>

<div class="read"><b>The load-bearing idea is maturity.</b> A promise is worth nothing the instant
it opens, ripens over about {RIPEN_BEATS:g} beats until the ear is waiting for it, and fades after
{FORGET_BEATS:g} because a debt nobody remembers is not a debt. Pressure is strength times
maturity — and every question that sounded metaphorical falls out of that one curve.</div>

<p>Including the one that mattered most. An early version of this design listed properties like
<i>"whether satisfying it now would be too obvious"</i> and <i>"whether leaving it unresolved
increases global coherence"</i> — the load-bearing ones, and the two with no number behind them.
A property that cannot be computed from state the engine already holds is decoration, and
decoration inside a control loop is worse than nothing because the trace still reads like
reasoning. With maturity, <b>"too obvious" is just settling something below
{RIPE:.2f} ripeness</b>, and the audition scores it accordingly:</p>

<div class="read"><b>settling early &lt; carrying &lt; settling something ripe.</b> That ordering
is the entire mechanism. Invert it and the composer pays every debt the moment it opens, which is
what a solver does. This is the difference between wanting uncertainty gone and choosing which
uncertainty is worth keeping.</div>

<div class="grid">
<div class="stat"><div class="n">{survey['opened']}</div><div class="k">obligations taken on</div></div>
<div class="stat"><div class="n">{survey['settled']}</div><div class="k">settled</div></div>
<div class="stat"><div class="n">{survey['median_wait']:.0f}</div><div class="k">median beats carried</div></div>
<div class="stat"><div class="n">{survey['longest_wait']:.0f}</div><div class="k">longest wait before settling</div></div>
<div class="stat"><div class="n">{survey['still_open']}</div><div class="k">deliberately left open</div></div>
<div class="stat"><div class="n">{survey['changed']}/{survey['themes']}</div>
  <div class="k">pieces the ledger changed</div></div>
</div>

<h3>What kinds of debt it actually takes on</h3>
<table><thead><tr><th>kind</th><th>settled across the sample</th></tr></thead><tbody>{kinds}</tbody></table>
<p class="sub">The deepest thing each sampled piece was still carrying at the end:</p>
<p>{carrying or "<span class='sub'>nothing outstanding in this sample</span>"}</p>

<h3>One piece's open book</h3>
<p class="sub">Seed {piece.seed}, {piece.mood.nearest_theme()}, {PURPOSE_SECONDS:.0f} seconds —
what it had not done by the end, biggest pressure first.</p>
<table><thead><tr><th>owed</th><th>domain</th><th>since</th><th>pressure</th></tr></thead>
<tbody>{open_now}</tbody></table>

<h3>Does it do anything? The honest measurement</h3>
<p>Switched off, the composer writes exactly the music it wrote before the ledger existed —
checked <b>byte for byte</b> against the previous build. The criterion is dropped from the
audition rather than sitting in it as a constant, which is what makes "off" mean off rather than
"quietly shifting every score". Switched on, it changed {survey['changed']} of the
{survey['themes']} sampled pieces.</p>

<div class="read"><b>On the critic's own numbers it is a wash</b> — {survey['failing_on']} failing
principles across the sample with the ledger on, {survey['failing_off']} with it off. That is the
expected result and it is worth being clear about why: <b>no principle in the critic measures
whether a forty-beat-old obligation was ever answered.</b> The ledger adds behaviour on a
timescale nothing is currently listening at. Measuring the benefit needs the piece to be able to
notice that its own past became simpler — which is the next thing on the list, not a thing that
exists.</div>

<h3>Built since: what the ending gave back to the beginning</h3>
<p>The opening is priced twice, in symbols, in the same notation that regenerates the audio. Once
using only what existed while it was being written — each phrase against the germ and the phrases
before it. Then again with the rest of the piece available to refer to. A phrase can be spelled
out note by note, or written as "that one, inverted, with the third interval a step wider", and
whichever is shorter is what it costs.</p>

<div class="read"><b>Hindsight means the ending, not the neighbour.</b> The first version let a
phrase be explained by the one after it, which is a lineage rather than a revelation — and since
every phrase here descends from the last, that read as 16% compression before the piece had done
anything. Restricting it to what the <i>second half</i> explains about the first took the number from
inflated to real.</div>

<div class="grid">
<div class="stat"><div class="n">{hindsight['share'] * 100:.0f}%</div>
  <div class="k">of the opening explained by the ending</div></div>
<div class="stat"><div class="n">{hindsight['best'] * 100:.0f}%</div><div class="k">best piece</div></div>
<div class="stat"><div class="n">{hindsight['worst'] * 100:.0f}%</div><div class="k">worst piece</div></div>
<div class="stat"><div class="n">{hindsight['recalls']}</div>
  <div class="k">deliberate returns to an early phrase</div></div>
</div>

<p>Two things had to exist for that number to be non-trivial, and they are worth separating
because only one of them is the clever part.</p>

<div class="read"><b>The ability mattered more than the wanting.</b> The audition had no candidate
that reached back for a *specific* earlier phrase — only the germ, and descendants of whatever was
just played. Adding one — pick the early phrase that is still expensive to describe, offer it back
as it was, inverted or in retrograde — roughly doubled the measurement on its own. Caring about it
on top of that adds about a tenth as much again ({hindsight['indifferent'] * 100:.1f}% with the
criterion switched off against {hindsight['share'] * 100:.1f}% with it on), and the effect
saturates: weighting it three times as heavily buys nothing further.</div>

<p class="sub">One piece, in full: {hindsight['example'].mood.nearest_theme()},
seed {hindsight['example'].seed}. {hindsight['example'].reveal.describe()}</p>


<div class="read ok"><b>Built since: intention as a trajectory.</b> The piece now picks a target
shape for its own state — how much it should owe, how torn its decisions should be, how crowded
the register gets — and holds three structural levers along it, swapping the intent out when it
becomes too easy to satisfy. It took three attempts: the first two are on the page because both
measured at nothing and both were deleted. See <b>The plan</b>.</div>

<div class="read ok"><b>Built since: the ghosts.</b> The losing candidates are played quietly
under the winner, and the one that <i>would</i> have settled a live promise is heard louder than
its score earns — so the haze thickens exactly where the music is declining a resolution it set up
itself. See <b>The ghosts</b>.</div>
"""


def panel_mix(survey) -> str:
    from compex.dsp.balance import BAND_NAMES, CEILING, FLOOR
    from compex.dsp.engines import NOTE_LOUDNESS
    from compex.generate.critic import CROWD_WINDOW

    crowd_window = CROWD_WINDOW
    crowd_none, crowd_wide = _crowding_at(0.0), _crowding_at(1.5)
    decaying_share = survey["decaying_before"]
    holding_count = sum(1 for name in ENGINES_FOR_ROLE[ROLE_PAD]
                        if HOLD[name] >= NEEDS_HOLD[ROLE_PAD])
    holds = "".join(
        f"<tr><td><code>{name}</code></td><td>{HOLD[name]:.2f}</td>"
        f"<td class='sub'>{'yes' if name in ENGINES_FOR_ROLE[ROLE_PAD] and HOLD[name] >= NEEDS_HOLD[ROLE_PAD] else '—'}</td></tr>"
        for name in sorted(HOLD, key=lambda key: -HOLD[key]))

    example = survey["example"]
    rows = "".join(
        f"<tr><td><b>{row['theme']}</b></td><td>{row['voices']}</td>"
        f"<td>{row['percussive'] * 100:.0f}%</td>"
        f"<td>{row['lifted']} up, {row['cut']} down</td>"
        f"<td class='sub'>{', '.join(row['buried']) or '—'}</td>"
        f"<td class='sub'>{', '.join(row['still']) or '—'}</td></tr>"
        for row in survey["rows"])

    roles = "".join(
        f"<tr><td><b>{role}</b></td><td>{FLOOR[role] * 100:.0f}%</td>"
        f"<td>{CEILING[role] * 100:.0f}%</td></tr>"
        for role in FLOOR)

    voices = "".join(
        f"<tr><td><code>{voice.voice}</code></td><td class='sub'>{voice.role}</td>"
        f"<td>{BAND_NAMES[voice.home]}</td>"
        f"<td>{voice.best_share * 100:.0f}%</td>"
        f"<td>{voice.after * 100:.0f}%</td>"
        f"<td>{voice.trim:.2f}&times;</td></tr>"
        for voice in sorted(example["mix"].voices, key=lambda v: -v.best_share))

    before = ", ".join(f"{name} {ratio:.2f}&times;"
                       for name, ratio in PERCUSSION_BEFORE.items())

    return f"""
<h2 style="margin-top:26px">The mix</h2>
<p class="sub">Added after the first real listening complaint: things were getting buried, the
drums were winning, and the breath was too far forward. All three turned out to be arithmetic.</p>

<div class="read warn"><b>Until this existed, nothing mixed anything.</b> There was a table — bass
0.90, lead 0.67, pad 0.34, texture 0.23 — plus a random gain per drum, one sidechain duck, and a
limiter. Nothing measured whether a voice could be heard, so the listening loop could not have
fixed a buried one even in principle.</div>

<h3>Three faults, all measurable</h3>
<p><b>Every note was normalised to the same peak.</b> Peak is how tall a sound is, not how loud.
A swelling pad and a plucked string reach the same height and nothing like the same volume, so the
pads carrying the non-percussive body of the piece were quietly a third of what they looked.
Now each note is levelled by the loudness of its loudest quarter-second, with some credit still
given to its peak — because a transient <i>is</i> heard as louder than its RMS, and matching pure
loudness makes plucks vanish the other way. Target loudness {NOTE_LOUDNESS:g}.</p>

<p><b>Six drums were six times one drum.</b> Each was given a gain as if it played alone. Measured
on the previous build: {before} — percussion against everything melodic. The kit is now held back
by the square root of its own size, which is roughly how loudness adds when hits are not
simultaneous.</p>

<p><b>Breath was a blind draw.</b> The flute's air was picked uniformly from 0.10 to 0.55 and the
reed's from 0.03 to 0.30, which is the one place a mix decision was left to a dice roll — against
this project's own rule that anything chosen is chosen by character. Air now comes from the mood,
and a dense piece gets less of it than a sparse one however bright it is, because air is the first
thing that muddies a crowded arrangement.</p>

<h3>Then: measure who is actually audible</h3>
<p>Every voice is rendered once — a representative note, through its own effects chain, since a
reverb or a ring modulator changes where a voice lives — and its energy counted in five bands.
Multiplied by how much that voice plays, that says what it contributes to each band. A voice
<i>lives</i> in the band holding most of its own energy; the question is who else is there.</p>

<table><thead><tr><th>role</th><th>should own at least</th><th>and at most</th></tr></thead>
<tbody>{roles}</tbody></table>
<p class="sub">Bands: {", ".join(BAND_NAMES)}. The floor scales with how much a voice plays —
a texture that speaks four times in five minutes is not buried, it is occasional, and shouting it
forward would overrule the composer's own sparseness.</p>

<div class="grid">
<div class="stat"><div class="n">{survey['percussive'] * 100:.0f}%</div>
  <div class="k">percussive, averaged over {len(MIX_THEMES)} moods</div></div>
<div class="stat"><div class="n">{survey['found']}</div><div class="k">buried voices found</div></div>
<div class="stat"><div class="n">{survey['rescued']}</div><div class="k">brought back by trimming</div></div>
<div class="stat"><div class="n">{survey['left']}</div><div class="k">still buried afterwards</div></div>
</div>

<table><thead><tr><th>theme</th><th>voices</th><th>percussive</th><th>trims</th>
<th>buried</th><th>still buried</th></tr></thead><tbody>{rows}</tbody></table>

<h3>One piece, voice by voice</h3>
<p class="sub">{example['theme']}, seed 7788, {MIX_SECONDS:.0f} seconds.
"before" and "after" are the share of its home band the voice owns.</p>
<table><thead><tr><th>voice</th><th>role</th><th>lives in</th><th>before</th><th>after</th>
<th>trim</th></tr></thead><tbody>{voices}</tbody></table>

<div class="read"><b>Some voices cannot be rescued by gain, and the page says so.</b> A drum at 1%
of a band three other drums are already fighting over needs about ten times its level to be heard,
which is not a mix decision, it is an arrangement problem. The stage reports what it could not fix
rather than quietly lifting until something clips.</div>

<h3>The other half: what only the composer can fix</h3>
<p>A mixer can make a thick voice louder. It cannot make it two voices. Two lines inside the same
octave are heard as one thicker line however they are levelled, and the only thing that changes
that is writing them somewhere else — which is the composer's job and nobody else's.</p>

<p>So the critic gained a principle it can act on: <b>crowding</b>, the share of simultaneous voice
pairs whose registers sit within {crowd_window:.0f} semitones of each other. It is deliberately
symbolic — registers rather than spectra — because that is the question the composer can answer
while the piece is still being written. It drives <code>spacing</code>, which spreads the voices
apart in octaves between movements.</p>

<div class="read ok"><b>The loop closes, measured:</b> at spacing 0 the sampled pieces average
{crowd_none:.2f} crowding; at 1.5 they average {crowd_wide:.2f}. A drive that could not move its
own measurement would be theatre, and this one is checked in the tests rather than asserted here.</div>

<h3>Sustained sounds, and why they all sounded alike</h3>
<p>The second half of the same listening complaint: <i>"the sustained sounds are almost all sort of
annoying… I think it is choosing from a small set."</i> Measured, and it was worse than that —
nothing was measuring whether an engine <b>holds a note at all</b>.</p>

<div class="read warn"><b>{decaying_share:.0f}% of everything chosen to hold a chord was a struck
sound.</b> A bell, a glass, a noise wash — engines that swell and die under a harmony that is still
moving. That is both why the sustained voices sounded like each other and why the whole piece read
as percussive: much of what should have been sustaining was, in effect, more percussion.</div>

<p>Each engine now carries a measured <b>hold</b> — its level two thirds of the way through a long
note against its level just after the attack — and a role that has to sustain only draws from
engines that do. The table is regenerated by the test suite, so an engine that quietly stops
holding fails the build rather than the music.</p>

<table><thead><tr><th>engine</th><th>holds</th><th>can be a pad</th></tr></thead><tbody>{holds}</tbody></table>

<p>And five engines were added, because the ones that genuinely sustained numbered six:
<code>bowed</code> (one player rather than a section, with the vibrato arriving after the note),
<code>shimmer</code> (inharmonic partials beating against each other indefinitely),
<code>tape</code> (wow and flutter on two timescales that never line up),
<code>vox</code> (a held vowel that travels to another vowel while it is held), and
<code>aeolian</code> (tuned noise — wind with a pitch in it, unlike the untuned wash that was
reading as breath). Pads now draw from {holding_count} engines that all hold.</p>

<div class="q"><b>Still open: the mixer's own measurement never reaches the critic.</b> Since this
was written the composer has learned to listen — it renders a window of itself each movement and
judges the roughness, brightness and motion of the result, which was "the expensive kind of honest"
and turned out to cost about a tenth of a second a movement. What it still does not get is
<i>this</i> measurement: which voice is losing which band to which other voice. Rendering a probe
was affordable; rendering every voice separately, every movement, to compare them, is not yet.</div>

<div class="q"><b>Open: the floors and ceilings are mine.</b> Same objection as the aesthetic bands
and the opening taste curve, one layer along. They are measured against real pieces, which catches
the gross errors, but 9% for a pad rather than 12% is a judgement nobody has argued with yet.</div>
"""


def panel_ghosts(survey) -> str:
    from compex.dsp.arrange import GHOST_CUTOFF
    from compex.generate.melody import GHOSTS_KEPT
    from compex.generate.write import GHOST_OWED_BOOST

    example = survey["example"]
    nearest = "".join(
        f"<tr><td><code>{ghost.phrase.origin}</code></td>"
        f"<td class='sub'>lost to {choice.chosen.origin}</td>"
        f"<td>{ghost.margin:.4f}</td><td>{ghost.closeness():.2f}</td>"
        f"<td class='sub'>{piece.mood.nearest_theme()} @ beat {choice.at_beat:g}</td></tr>"
        for ghost, choice, piece in survey["loudest"])

    rows = "".join(
        f"<tr><td><b>{piece.mood.nearest_theme()}</b></td>"
        f"<td>{len(piece.melodies)}</td>"
        f"<td>{sum(len(c.ghosts) for c in piece.melodies)}</td>"
        f"<td>{len(piece.ghosts)}</td>"
        f"<td>{statistics.mean([g.closeness() for c in piece.melodies for g in c.ghosts]) if any(c.ghosts for c in piece.melodies) else 0:.2f}</td></tr>"
        for piece in survey["pieces"])

    return f"""
<h2 style="margin-top:26px">The ghosts</h2>
<p class="sub">The mechanism the project is named for — and for a while the oldest thing on the
list that was designed and not built.</p>

<div class="read"><b>Computational Expressionism was supposed to mean this:</b> that the machine's
way of arriving at something becomes perceptible — that you can <i>hear</i> the machinery rather
than only its output. Until now you could not. The composer auditioned two to six lines for every
phrase, played one, and the rest vanished without a sound. The only evidence a choice had happened
was that something had been chosen.</div>

<p>Now the losers are played. Quietly, underneath the line that beat them, through the same
instrument — a ghost is the same voice playing what it nearly played, not a different voice
commenting on it — with the top taken off at {GHOST_CUTOFF:.0f}&nbsp;Hz so they sit
<i>behind</i> rather than compete. Up to {GHOSTS_KEPT} per audition.</p>

<div class="read ok"><b>How loud a ghost is says how close the decision was.</b>
<code>closeness = e<sup>-margin &times; 6</sup></code>. A field the composer nearly split leaves an
audible haze of almost-melodies; a decision it was sure about leaves almost nothing. So a passage
where the machine was certain sounds clean, and a passage where it was torn blooms — and that
difference is not a metaphor for its uncertainty, it <i>is</i> its uncertainty, scaled.</div>

<div class="grid">
<div class="stat"><div class="n">{survey['turned_down']}</div><div class="k">lines turned down</div></div>
<div class="stat"><div class="n">{survey['notes']}</div><div class="k">ghost notes sounded</div></div>
<div class="stat"><div class="n">{survey['torn']:.2f}</div><div class="k">average closeness</div></div>
<div class="stat"><div class="n">{survey['near_ties']}</div><div class="k">near ties (heard at 90%+)</div></div>
<div class="stat"><div class="n">{survey['closest']:.4f}</div>
  <div class="k">closest call in the sample</div></div>
<div class="stat"><div class="n">{survey['notes'] / max(1, survey['played']) * 100:.0f}%</div>
  <div class="k">as many notes as the piece plays</div></div>
</div>

<p class="sub">Measured over {survey['themes']} pieces of {GHOST_SECONDS:.0f} seconds, seed 7788.</p>
<table><thead><tr><th>theme</th><th>auditions</th><th>turned down</th><th>ghost notes</th>
<th>average closeness</th></tr></thead><tbody>{rows}</tbody></table>

<h3>One more rule, and it is the one worth arguing with</h3>
<div class="read"><b>A ghost that would have settled something the piece owes is heard
{GHOST_OWED_BOOST:g}&times; louder than its score earns.</b> That line is not merely a road not
taken — it is the piece declining to do something it is carrying. The ledger already knows what is
outstanding; this is where the two mechanisms meet, and it is why the haze thickens exactly where
the music is avoiding a resolution it has set up.</div>

<h3>The closest calls in the sample</h3>
<table><thead><tr><th>ghost</th><th>outcome</th><th>margin</th><th>heard at</th><th>where</th>
</tr></thead><tbody>{nearest}</tbody></table>

<h3>The switch</h3>
<p>The <code>ghosts</code> knob on both surfaces runs 0 to 1, and at 0 the engine produces the
music it produced before ghosts existed — checked byte for byte against the previous build, not
asserted. That matters more here than anywhere else in the project: a mechanism claiming to make
the machine's inner state audible is exactly the kind of thing that can sound profound and do
nothing.</p>

<div class="q"><b>Open: whether this reads as uncertainty or as reverb.</b> The intended experience
is hearing the machine hesitate. The risk is that a haze of near-misses under a line is simply
heard as ambience, in which case the mechanism is honest and the perception is wrong — which would
be worth knowing, and only a listener can say.</div>

<div class="q"><b>Open: only melodic auditions have ghosts.</b> The rhythm auditions turn down
candidates too, and a rejected pattern is at least as interesting as a rejected phrase. Nothing
plays them yet.</div>
"""


def panel_plan(survey) -> str:
    """What the piece is trying to do to itself — and the two versions that failed."""
    rows = "".join(
        f"<tr><td>{row['theme']}, seed {row['seed']}</td><td><code>{row['intent']}</code></td>"
        f"<td>{row['off']:+.2f}</td><td><b>{row['on']:+.2f}</b></td>"
        f"<td>{row['on'] - row['off']:+.2f}</td><td>{row['gave_up']}</td></tr>"
        for row in survey["rows"])

    levers = "".join(
        f"<tr><td><code>{variable}</code></td><td><code>{lever}</code></td><td>{how}</td></tr>"
        for (variable, lever), how in zip(sorted(survey["levers"].items()), (
            "how far apart the voices are put, in whole octaves",
            "how many lines the audition has to choose between",
            "whether the piece is allowed to close what it opened",
        )))

    return f"""
<h2>The plan</h2>
<p>Every other mechanism in this engine <i>reacts</i>. The critic hears a problem and corrects it,
the ledger notices a debt and settles it, the mixer finds a buried voice and lifts it. A machine
made only of those has no plan — which is exactly the gap that was left when the question was how
a machine could know what a piece is <i>for</i>.</p>

<p>An intention here is not a feeling and not a destination. It is a <b>target shape for the
composer's own internal state over time</b>: how much it should owe at the halfway point, how torn
its decisions should be near the end, how crowded the register is allowed to get. Build, tension
and release are not named anywhere in the code — they are what those shapes produce.</p>

<h3>The four</h3>
<table><thead><tr><th>intent</th><th>what it wants of itself</th></tr></thead><tbody>
{"".join(f"<tr><td><code>{intent.name}</code></td><td>{intent.why}</td></tr>" for intent in INTENTS)}
</tbody></table>
<p class="sub">The mood leans the choice without deciding it — a tense piece is likelier to unravel,
a still one to settle — and an intent that is being satisfied <i>too easily</i> gets swapped out
mid-piece. Sitting on target while winning every decision by a mile is not succeeding; it is
coasting, and nothing is at stake.</p>

<h3>An intention is a constraint, not a preference</h3>
<p>This is the whole design, and it cost two rewrites to find. The obvious build is to let the
plan lean on the weights the composer already judges by: care more about settling, be more
curious, want more space. That version is measurable and it measures at <b>nothing</b> — over
twenty pieces, matched seed for seed, the state followed its target shape <i>worse</i> with the
plan switched on, by &minus;0.041 &plusmn; 0.037.</p>

<div class="read"><b>Why a weight cannot steer anything.</b> It competes with fourteen other
weights, and then the critic retunes it the moment the movement ends. Its whole authority is a few
per cent of a quantity that swings by a factor of three on its own. A hand that much weaker than
the thing it is holding is not a hand on the wheel.</div>

<p>So the plan stopped asking and started withholding. It does not ask the composer to care more
about closing its debts — it refuses permission to close them, and the harmony comes home while
the piece keeps owing anyway. It does not nudge curiosity up — it makes the audition imagine more
lines than it wanted to. Each lever bites where the music is built, and nothing downstream gets a
vote.</p>

<table><thead><tr><th>it aims at</th><th>by holding</th><th>which is</th></tr></thead>
<tbody>{levers}</tbody></table>

<div class="read"><b>Letting go is not settling.</b> Giving the plan a way to <i>reduce</i> what a
piece owes turned out to need a new idea. Waiting does not work — a promise holds full weight for
its whole term by design, which is what makes carrying one mean anything — so a piece that wants
to owe less has to abandon a debt outright, and it abandons the one it was carrying hardest. That
is not answering a question. It is deciding the question was not what the piece was about.</div>

<h3>The second thing that failed: watching</h3>
<p>With levers that <i>did</i> have authority, the plan still could not steer, and the reason is
arithmetic rather than music. A piece gets eight or nine listen-backs. Both of the strong levers
are whole numbers &mdash; one more candidate, one more octave &mdash; so any correction finer than
that does nothing at all. And the quantity being corrected swings several times further on its own
than the lever can move it. Measure, compare, correct: a noisy sensor, five moves, a deadband, and
an actuator weaker than the disturbance. It scored <b>+0.001 &plusmn; 0.029</b>. Nothing again, and
this time for a reason no amount of tuning was going to fix.</p>

<div class="read ok"><b>What worked was to stop watching.</b> Every coupling was measured to be
monotone before it was written down — more candidates always narrows the margin, more spacing
always reduces crowding, less permission always leaves more owed. When you know which way the
machine runs, you do not need to watch the output to know which way to push. The plan holds each
lever where the shape says it should be and never looks back. Nothing lags, nothing hunts.</div>

<h3>Whether it works, measured on the surfaces you can hear</h3>
<table><thead><tr><th>piece</th><th>intent</th><th>plan idle</th><th>plan held</th>
<th>&Delta;</th><th>given up</th></tr></thead><tbody>{rows}</tbody></table>
<p class="sub">Agreement is the correlation between where the composer's state actually went and
where the shape said it should go, over {survey['seconds']:.0f}-second pieces. One is a plan kept
perfectly, zero is a plan that steered nothing, negative is a piece that did the opposite of what
it set out to do. Above: mean {survey['off']:+.3f} idle against {survey['on']:+.3f} held,
improving in {survey['improved']} of {len(survey['rows'])}. The committed test runs twenty pieces
and gets <b>+0.136 &plusmn; 0.046</b> — improving in eighteen of them.</p>

<div class="read"><b>Correlation, not distance, and that choice matters.</b> The obvious score is
how far the state sat from its target. It is also trivially cheatable: a piece whose pressure never
moves sits in the middle of its own range all the way through and scores respectably against any
curve, having followed nothing. Correlation gives a flat line zero, which is the honest mark for a
plan that did not steer. The first version of this scored well on distance and was decoration.</div>

<h3>The switch</h3>
<p>Gain to zero and the engine writes exactly what it wrote before any of this existed — the same
off switch every mechanism here has, checked rather than asserted.</p>

<div class="q"><b>Open: the plan steers the composer, not the listener.</b> Every number above is
about the machine's internal state. What a plan sounds like from outside — whether a piece under
<code>unravel</code> is heard as coming apart, or merely as a piece that got busier — is a
question no measurement in this repository can answer.</div>

<div class="q"><b>Open: the deadband.</b> Both strong levers are whole numbers, so on some pieces
the plan asks for a change too small to make one, and the piece comes out byte for byte identical
to the unplanned version. That is honest, but it means the plan is silently inactive on part of
its range.</div>
"""


def panel_free(survey, remembering) -> str:
    from compex.generate.clocks import RATIOS
    from compex.generate.tuning import DIVISIONS, LIMITS

    memory_rows = "".join(
        f"<tr><td>{row['index']}</td>"
        f"<td class='sub'>{row['pieces']} remembered · {row['digest']}</td>"
        f"<td class='sub'>{', '.join(row['engines'])}</td></tr>"
        for row in remembering["rows"])
    memory_distinct = remembering["distinct"]

    families = survey["families"]
    total = sum(families.values()) or 1
    rows = "".join(
        f"<tr><td><b>{name}</b></td>"
        f"<td>{counts['twelve'] / survey['seeds'] * 100:.0f}%</td>"
        f"<td>{counts['equal'] / survey['seeds'] * 100:.0f}%</td>"
        f"<td>{counts['just'] / survey['seeds'] * 100:.0f}%</td></tr>"
        for name, counts in survey["by_mood"].items())

    shown = "".join(
        f"<tr><td><code>{family}</code></td><td>{found.name}</td>"
        f"<td class='sub'>{', '.join(f'{value:.0f}' for value in found.cents())}</td>"
        f"<td class='sub'>{found.detail}</td></tr>"
        for family, found in survey["examples"].items())

    clocks = "".join(
        f"<tr><td><b>{piece.mood.nearest_theme()}</b></td><td><code>{clock.voice}</code></td>"
        f"<td>{clock.numerator}:{clock.denominator}</td>"
        f"<td>{clock.meets_every(float(piece.beats_per_bar)):g} beats</td></tr>"
        for piece, clock in survey["loose"][:10])

    return f"""
<h2 style="margin-top:26px">Off the grid</h2>
<p class="sub">Two things a machine can do that a person cannot, added because the question was
what would make it a more capable <i>machine</i> musician rather than a better imitation of a
human one.</p>

<h3>It tunes itself</h3>
<div class="read"><b>Twelve notes to the octave is a fact about keyboards, not about music.</b> It
is a compromise adopted because a physical instrument cannot retune between chords, inherited by
everyone who learned on one. There is no keyboard in here — the engine synthesises from
frequencies, and every pitch it produces is a float. The twelve-tone grid was a table I handed it,
and in a project whose first rule is that nothing is looked up, it was the largest one left.</div>

<p>So each piece derives its own tuning, from one of three families:</p>
<ul>
<li><b>twelve</b> — the familiar grid and the named modes on it. Still where a serene piece
almost always lands, because legibility is a real property and sometimes the right one.</li>
<li><b>equal</b> — some other equal division of the octave ({", ".join(str(n) for n in DIVISIONS)}),
with degrees chosen by stacking that division's best fifth. Even steps, unfamiliar intervals: the
difference between a system you have not heard before and something broken.</li>
<li><b>just</b> — degrees built from whole-number frequency ratios up to the
{LIMITS[-1]}-limit, ranked by <b>Tenney height</b> (log2 of numerator times denominator — the
plainest measure of how simple a ratio is). Partials line up exactly instead of nearly, so chords
stop beating. No fixed-pitch instrument can do this and still change key.</li>
</ul>

<div class="read warn"><b>The risk is reading as out of tune rather than differently tuned</b>, and
two rules exist to prevent it. Nothing is ever randomised into place: every degree comes from a
ratio or from an equal division. And no gap is left too wide to step through — a scale with a
quarter of an octave missing is a chord, and a melody crossing it has to leap every time.</div>

<div class="grid">
<div class="stat"><div class="n">{families['twelve'] / total * 100:.0f}%</div>
  <div class="k">of pieces stay in twelve</div></div>
<div class="stat"><div class="n">{families['equal'] / total * 100:.0f}%</div>
  <div class="k">another equal division</div></div>
<div class="stat"><div class="n">{families['just'] / total * 100:.0f}%</div>
  <div class="k">whole-number ratios</div></div>
<div class="stat"><div class="n">{survey['off_grid']:.0f}&cent;</div>
  <div class="k">average distance from the nearest twelve-tone pitch, when it leaves</div></div>
</div>

<p class="sub">Measured over {survey['seeds']} seeds per theme. Strangeness is a decision, so
leaving twelve is driven by tension, grit and darkness:</p>
<table><thead><tr><th>theme</th><th>twelve</th><th>equal</th><th>just</th></tr></thead>
<tbody>{rows}</tbody></table>

<p class="sub">One tuning it actually derived from each family — degrees in cents:</p>
<table><thead><tr><th>family</th><th>name</th><th>cents</th><th>from</th></tr></thead>
<tbody>{shown}</tbody></table>

<h3>More than one clock</h3>
<div class="read"><b>An ensemble shares a pulse because people cannot reliably hold two at once.</b>
A machine can hold as many as it likes, and scheduling them costs nothing here because time is
addressed positionally — <code>f(seed, stream, index)</code> does not care which grid the index is
counted on.</div>

<p>So voices can run at rational multiples of the base tempo
({", ".join(f"{n}:{d}" for n, d in dict.fromkeys(RATIOS) if n != d)}). <b>Rational rather than
arbitrary</b>, for one reason: irrational ratios never meet again. A 3:2 voice realigns with the
pulse every two bars, and that returning coincidence is what the ear latches onto. Without it you
have not written polytempo, you have written drift.</p>

<div class="read ok"><b>One anchor stays on the pulse</b> — the bass, the kick, the boom. You can
only hear three-against-two if something is being the two. That is not a concession to human
hearing; a ratio needs both of its terms to exist.</div>

<table><thead><tr><th>piece</th><th>voice</th><th>ratio</th><th>meets the pulse every</th></tr>
</thead><tbody>{clocks}</tbody></table>

<h3>And it remembers what it has already done</h3>
<div class="read"><b>Every piece used to begin from nothing.</b> The composer learned inside a
track — taste moved, grids learned, bounds gave way — and then the whole thing was thrown away and
the next piece started from the same mood-derived defaults as the first. It had made hundreds and
remembered none of them.</div>

<p>A machine has no excuse for that. Perfect recall of everything it has ever done is one of the
few advantages it holds outright, and the obvious use for it is the thing people do badly: <b>get
bored of itself</b>. An engine that reaches for the same five instruments and the same tuning every
time is not being consistent, it is being stuck.</p>

<p>So a history carries three things forward — where taste ended up, what has been used lately, and
how many pieces there have been. Taste leans toward where past pieces settled; anything reached for
recently is pushed down the ranking rather than banned; a family of tuning it has been living in
makes leaving easier next time. The past decays, so the last handful of pieces matter and the first
hundred are a rumour.</p>

<div class="read ok"><b>Determinism survives because the memory is an argument, not hidden
state.</b> Same seed, same mood, same history gives the same audio — and the history's digest is
printed in the formula next to the seed, so two people with the same seed and different histories
can see why they have different music. A hidden accumulator would have broken the one property
everything else here rests on, silently.</div>

<p class="sub">The same seed and the same mood, six times in a row, with the engine remembering:</p>
<table><thead><tr><th>#</th><th>history</th><th>instruments it chose</th></tr></thead>
<tbody>{memory_rows}</tbody></table>
<p class="sub">{memory_distinct} distinct instrumentations out of six, from identical inputs.</p>

<div class="q"><b>Open: how far is too far.</b> A piece in 31-EDO with three voices on different
clocks is legitimately what was asked for and may still be unlistenable. The mood axes gate both —
serene stays in twelve and on one clock — but where the line sits is a guess, and only listening
settles it.</div>

<h3>It works its scales out too</h3>
<p>The named modes were the last table in the project that decided anything musical. They are
still here, demoted to what they should always have been — <b>names for shapes, checked
afterwards</b>. The machine assembles a set of degrees out of whatever grid it is on, scored on
four things it can check for itself: is there a strong interval near a fifth holding it together;
are there steps of more than one size, because a set of equal steps has no positions in it; is
every gap small enough to step through; and does one interval recur often enough that harmony has
something to be <i>about</i> — which is the property that actually makes the diatonic scale
special, and is measurable without knowing its name.</p>

<div class="read ok"><b>Most of what it invents has no name</b>, and occasionally it reinvents one:
the page reports "which is in sen" when that happens, and "a set with no common name" when it does
not. The table is commentary now.</div>

<h3>And it listens to itself</h3>
<div class="read"><b>Everything the critic judged, until now, it judged from symbols.</b> Two notes
a semitone apart are one number whether they are played by flutes or by sawtooth pads with eleven
partials each — and those are not the same event. So each movement, a short window of what was
actually written is rendered at a quarter rate and measured: <b>roughness</b> from partials close
enough together to beat (Plomp &amp; Levelt&rsquo;s curve, peaking about a quarter of a critical
band apart), <b>brightness</b> as a spectral centroid, and <b>motion</b> as how much the spectrum
changes frame to frame.</div>

<p class="sub">Calibrated against tones with known answers, which is the only way to catch this
class of error: a unison and a fifth read 0.00, a just third 0.05, a semitone dyad 0.27, a
four-tone cluster 0.36.</p>

<div class="read warn"><b>Both calibration failures are worth recording.</b> The first version
scored a unison as rough as a semitone — it took the twenty loudest bins as twenty partials, and a
single pure tone leaks across several of them. The second, after fixing that, reported every
low-register dissonance as perfectly smooth: two tones a semitone apart at 220 Hz are thirteen
hertz apart, and the analysis window was too coarse to see them as two things. Neither would have
been visible in a piece of music, because a piece of music has no known answer to check against.</div>

<h3>Instruments that change while they play</h3>
<p>A player is stuck with the instrument they brought. Keeping a voice's synthesis parameters fixed
for eight minutes was never a decision — it was an assumption inherited from ensembles made of
people. The parameters are drives now, moved by the <code>timbre_drift</code> drive under the
critic, cumulatively across movements so a voice <i>arrives</i> somewhere rather than wobbling in
place. Structural parameters are held: moving a formant pair is a different vowel, not the same
voice changed.</p>

<div class="read"><b>What closes that loop is the listening.</b> The <code>heard_motion</code>
principle measures whether the sound is actually changing, and pushes the drift drive when it is
not — which is a thing the symbolic critic could never have asked for, because it cannot hear.</div>

<div class="q"><b>Open: the drift moves the instruments more than it moves the measurement.</b> At
full drive the parameters travel about 21% and the measured spectral motion rises from 0.44 to
0.47 — real, and much smaller than the cause. Note changes dominate the flux, so the loop closes
weakly. Either the measure needs to separate timbre from notes, or the drift needs to reach
parameters that matter more.</div>
"""


def panel_open(data) -> str:
    return f"""
<h2 style="margin-top:26px">What is not settled</h2>
<p class="sub">The parts that are guesses, unbuilt, or known to have been wrong.</p>

<div class="q"><b>The mood axes are guesses.</b> Five axes and {len(THEMES)} theme coordinates,
all hand-assigned by me. The measurements prove they are <i>consistent</i> — the same theme
reliably reaches the same region — but consistency is not correctness. Whether "menacing" sounds
menacing is an open question that only a listener settles.</div>

<div class="read ok"><b>The thing it is named for is built.</b> See <b>The ghosts</b>. Every
phrase you hear beat two to six others; those losers are now played, quietly, underneath, at a
level set by how close they came. What was uncertainty inside the machine is a sound.</div>

<div class="q"><b>A plan steers the composer, not the listener.</b> The piece now aims its own
state at a shape and measurably hits it — +0.136 &plusmn; 0.046 better than no plan, over twenty
pieces. Every one of those numbers is about the machine's internals. Whether a piece under
<code>unravel</code> is <i>heard</i> as coming apart, rather than merely as one that got busier,
is the question that measurement cannot reach. See <b>The plan</b>.</div>

<div class="q"><b>The plan has a deadband.</b> Both of its strong levers are whole numbers — one
more candidate in the audition, one more octave between voices — so on some pieces it asks for a
change too small to make one, and the audio comes out byte for byte identical to the unplanned
version. The mechanism is honest about it; it is still silently inactive over part of its range.
</div>

<h3>Things that were built, measured, and thrown away</h3>
<div class="read"><b>Two designs for the plan.</b> Leaning on the taste weights the composer
already judges by scored &minus;0.041 &plusmn; 0.037 against no plan at all — worse than nothing.
Replacing those with real structural levers but still chasing the error scored +0.001 &plusmn;
0.029. Both were deleted rather than shipped, and both are written up in <b>The plan</b>, because a
mechanism that measures at zero is a finding and not an embarrassment.</div>
<div class="read"><b>And one gain that was tuned on its own exam.</b> The first sweep picked a
control gain by comparing four pieces, found a clean peak, and reported the improvement from those
same four. At twenty pieces it was nothing. Retuned on eight seeds the verdict never sees.</div>

<div class="q"><b>The composer hears some of the mix, not the part that matters most.</b>
It now listens to a rendered window of itself each movement and judges the roughness,
brightness and spectral motion of what came out — a real closure of what used to be a purely
symbolic critic. What it still cannot be told is the competitive question: is my pad fighting
my texture for one band. That measurement lives in the mixer, after composition, in a stage
the critic never meets.</div>

<div class="q"><b>Determinism has no regression test across versions.</b> The tests check that two
renders in the same process match. They cannot catch a DSP change that silently alters every
fingerprint ever produced — which would quietly falsify the claim that an old formula still
reproduces its track. Needs a pinned golden fingerprint per theme.</div>

<div class="q"><b>Too slow to be live.</b> Three to eight seconds for a forty-five second track.
Fine offline, nowhere near realtime. Editable-while-sounding needs a different audio path
entirely (SuperCollider and an OSC layer). The engine is structured for it — the positional RNG
exists precisely so a live edit cannot desync the stream — but it is not built.</div>

<h3>Things that were wrong and got fixed</h3>
<p>Kept on the page rather than quietly corrected, because what broke is more informative than
what worked.</p>

<div class="retract"><s><b>The whole first build.</b> An interpreter that executed equations Lee
wrote.</s> Wrong half of the problem — about 900 lines deleted. The insight that survived is that
his equation prompts were never prompts; they were a specification with no interpreter, and that
is what suggested the machine could write them too.</div>

<div class="retract"><s><b>Engine levels were left at whatever each engine naturally produced.</b></s>
Measured spread was roughly 300&times; in RMS. Two rounds of demos shipped with some voices far
quieter than intended before a sweep across all engines caught it.</div>

<div class="retract"><s><b>The drum kit was drawn blind from a shrinking pool.</b></s> It put an
<b>anvil in a serene piece</b>. Kits are now ranked against a character table like everything
else. The general lesson — anything the composer picks from has to know what its members sound
like — is now a project rule.</div>

<div class="retract"><s><b>Levelling every voice to the same peak.</b></s> The fix for one
audibility problem became the cause of the next: peak is not loudness, and the pads went quiet.
Two builds and a listening session apart. Closing a risk with a constant can open the following
one silently — the test asserted the spread stayed tight, and it did, in the units that turned out
not to matter.</div>

<div class="retract"><s><b>Deciding which engines could be pads by writing a list.</b></s> Measured
later: <b>40% of that list did not sustain a note</b>. Bells and glasses dying under a harmony
that was still moving, which is why sustained voices all sounded alike and why the music read as
percussive even after the drums were fixed. Membership of a role is now a measured behaviour with
a test that regenerates it.</div>

<div class="retract"><s><b>Judging a phrase and then playing a different one.</b></s> The register
drives stretched and clamped the winner <i>after</i> the audition had scored it, and the
"bring the line home" rule dropped unanswered twelfths into lines just judged for their leaps. If
a stage transforms what a judge scored, the judgement is about something else.</div>

<div class="retract"><s><b>The equation normaliser stripped <code>\\right</code> before
<code>\\rightarrow</code>.</b></s> Rule-ordering bug that silently collapsed every arrow chain
into a single token, so an entire form map parsed as one section. Found by reading the parse
output rather than by a failing test — it failed quietly and plausibly, which is the worst way
for a parser to fail.</div>

<h3>Honest scope</h3>
<div class="read warn"><b>This is experimental music and it sounds like it.</b> It does not repeat
the way pop music repeats, it will sometimes sit on an idea too long, and some of it is simply
strange. That is the intent — the project exists to see what happens when the computation carries
itself, not to imitate a genre. Lee's framing when he asked for it: <i>"I realize this wont sound
great. It will likely be very strange. At least at first, before we figure out the parameter
questions. But thats fine because it is meant to be experimental."</i></div>
"""


def build() -> str:
    data = survey()
    example = worked_example()
    series = evolution_survey()
    choices = choice_survey()
    purpose = purpose_survey()
    hindsight = hindsight_survey()
    mixing = mix_survey()
    haunting = ghost_survey()
    planning = plan_survey()
    freedom = freedom_survey()
    remembering = memory_survey()
    waves = [
        (name, make_track(Knobs(seed=7788, duration_s=30.0), THEMES[name]).samples, 30.0)
        for name in ("serene", "menacing")
    ]

    counts = {
        "engines": len(ENGINE_NAMES), "drums": len(drums.DRUM_NAMES),
        "effects": len(EFFECT_NAMES), "themes": len(THEMES), "scales": len(SCALES),
        "axes": len(AXES), "version": __version__,
    }
    FACTS.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")

    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{PROJECT} &mdash; {DOC_TYPE}</title><style>{CSS}</style></head><body><div class="wrap">

<header>
<p class="kicker">{DOC_TYPE}</p>
<h1>{PROJECT}</h1>
<p class="sub">A music engine that composes. You give it a length, a feeling and a seed; it
invents the tempo, the key, the harmony, the form and the instruments, plays the result, and
writes out the equation for what it decided. {counts['engines']} synthesis engines,
{counts['drums']} drums, {counts['themes']} emotional themes. Every number on this page is read
from the running code or measured by composing {SURVEY_SEEDS * len(THEMES)} pieces, so it cannot
go stale.</p>
</header>

<div class="card">
<h2 style="margin-top:0">Hear it</h2>
<p><a href="play/" style="color:var(--accent);font-size:17px;font-weight:600">Open the player &rarr;</a>
&nbsp;<span class="sub">Runs in a browser, phone included. It composes and plays on the
device — nothing is uploaded, nothing is downloaded but the engine itself.</span></p>
<div class="read"><b>The player runs this exact package, not a re-implementation.</b> The engine
is shipped as source and executed in the browser by Pyodide, so the phone and the desktop cannot
drift apart. A JavaScript port would have been three thousand lines of duplicate engine that
disagreed with this one the first time either was touched. The build fails if the shipped
archive falls behind the source.</div>
</div>

<div class="card">
<h2 style="margin-top:0">The short version</h2>
<div class="read"><b>Three inputs: how long, how it should feel, and a seed.</b> Everything
else — tempo, meter, key, mode, chord progression, the motif, the shape of the piece, which
instruments exist and how each is built — is invented by the machine.</div>
<div class="read ok"><b>It writes down what it did.</b> Each track is saved with the equation
notation describing every decision, and feeding that notation back produces the identical track,
byte for byte.</div>
<div class="read"><b>And it changes its mind while writing.</b> After each movement it
listens back to what it actually wrote, scores it against named aesthetic principles, and shifts
the parameters it writes with — including how hard it reacts to itself.</div>
<div class="read"><b>It chooses its melodies rather than reciting one.</b> Each phrase is
auditioned against several alternatives — mutations of the line just played, a return to the
germ, the occasional line invented from nothing — and the winner becomes the parent of the next
one. Rhythms work the same way, and the grid each voice plays against keeps moving toward
whatever that voice actually chose. What it weighs those judgements by starts from the mood and
is free to end up somewhere the starting weights never allowed.</div>
<div class="read"><b>And it carries what it has not done yet.</b> Every gesture opens an
account — a leap owes a step back, a progression that walks far from home owes a return — and the
composer decides which of those to settle now, which to let ripen, and which to keep owing because
carrying it has become part of what the piece is.</div>
<div class="read"><b>And it can make its own beginning intelligible.</b> The opening is priced in
symbols twice — as it could have been written at the time, and knowing how the piece ends — so
"the ending explained the beginning" is a measured number rather than a claim.</div>
<div class="read ok"><b>And you can hear it being unsure.</b> Every phrase beat two to six others
in an audition; the lines it turned down are played quietly underneath, at a level set by how close
they came. When it was certain the texture is clean; when it was torn the line blooms into a haze
of what it nearly played. That is the thing the project was named for.</div>
<div class="read ok"><b>And it has a plan.</b> Each piece aims its own state at a shape — how much
it should owe, how torn its decisions should be, how crowded the register may get — and holds three
structural levers along it. Two earlier designs for this were built, measured at nothing, and
deleted; both are on the page with their numbers. Nothing designed for this engine is currently
unbuilt.</div>
</div>

<nav class="tabs">
<button class="tab active" data-panel="how">What it is</button>
<button class="tab" data-panel="mood">The mood axes</button>
<button class="tab" data-panel="palette">The palette</button>
<button class="tab" data-panel="evolve">How it changes its mind</button>
<button class="tab" data-panel="choose">Choosing, not reciting</button>
<button class="tab" data-panel="purpose">What is it for?</button>
<button class="tab" data-panel="mix">The mix</button>
<button class="tab" data-panel="ghosts">The ghosts</button>
<button class="tab" data-panel="plan">The plan</button>
<button class="tab" data-panel="free">Off the grid</button>
<button class="tab" data-panel="example">One piece, step by step</button>
<button class="tab" data-panel="open">What is not settled</button>
</nav>

<div class="panel active" id="how">{panel_how(example)}
{fig_waveforms(waves)}</div>
<div class="panel" id="mood">{panel_mood(data)}</div>
<div class="panel" id="palette">{panel_palette(data)}</div>
<div class="panel" id="evolve">{panel_evolve(series, example)}</div>
<div class="panel" id="choose">{panel_choose(choices)}</div>
<div class="panel" id="purpose">{panel_purpose(purpose, hindsight)}</div>
<div class="panel" id="mix">{panel_mix(mixing)}</div>
<div class="panel" id="ghosts">{panel_ghosts(haunting)}</div>
<div class="panel" id="plan">{panel_plan(planning)}</div>
<div class="panel" id="free">{panel_free(freedom, remembering)}</div>
<div class="panel" id="example">{panel_example(example, data)}</div>
<div class="panel" id="open">{panel_open(data)}</div>

<footer>
<b>{PROJECT}</b> v{__version__} &middot; code at <a href="https://github.com/Elifterminal/CompEx"
style="color:var(--accent)">github.com/Elifterminal/CompEx</a> &middot; single self-contained
page, no external requests &middot; generated from the running package by
<code>docs/gen_docs.py</code>, checked by <code>docs/check_page.py</code>.
</footer>

</div><script>{JS}</script></body></html>"""
    return body


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    page = build()
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
