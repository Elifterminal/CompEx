#!/usr/bin/env python3
"""Fail the build when the living page drifts from the code, or loses its name.

Two jobs:

1. **Identity.** A living page belongs to exactly one project and must say which,
   by name, in its own ``<h1>`` and ``<title>``. A generic noun identifies nothing.
2. **Drift.** Every count on the page is generated from the running package, so a
   stale page means the generator was skipped. This catches that.

A check you have never seen fail is not a check, so ``--self-test`` corrupts the
page in memory in each of the ways that should be caught and confirms every rule
actually fires.

    PYTHONPATH=src python3 docs/check_page.py [--self-test]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compex import __version__                       # noqa: E402
from compex.dsp import drums                         # noqa: E402
from compex.dsp.effects import EFFECT_NAMES          # noqa: E402
from compex.dsp.engines import ENGINE_NAMES          # noqa: E402
from compex.generate import THEMES                   # noqa: E402
from compex.generate.mood import AXES                # noqa: E402
from compex.generate.theory import SCALES            # noqa: E402

PAGE = ROOT / "docs" / "index.html"
FACTS = ROOT / "docs" / "facts.json"
PROJECT = "CompEx"

#: Titles that would fit any other project unchanged, so identify nothing.
GENERIC = {
    "asset log", "log", "notebook", "findings", "index", "study", "notes",
    "record", "project", "docs", "documentation", "report", "results", "readme",
    "overview", "summary", "page", "home",
}

#: Anything that would make the browser fetch from another host.
EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*["']https?://(?!github\.com/Elifterminal)"""
    r"""|<link[^>]+rel=["']?stylesheet"""
    r"""|url\(\s*["']?https?://""",
    re.I,
)


class CheckFailed(Exception):
    """A rule the page must satisfy did not hold."""


def live_counts() -> dict[str, int | str]:
    return {
        "engines": len(ENGINE_NAMES),
        "drums": len(drums.DRUM_NAMES),
        "effects": len(EFFECT_NAMES),
        "themes": len(THEMES),
        "scales": len(SCALES),
        "axes": len(AXES),
        "version": __version__,
    }


# ── rules ──────────────────────────────────────────────────────────────────

def check_h1_names_the_project(page: str) -> None:
    match = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S | re.I)
    if not match:
        raise CheckFailed("no <h1> at all — the page does not say what it is for")
    heading = re.sub(r"<[^>]+>", "", match.group(1)).strip()
    if not heading:
        raise CheckFailed("<h1> is empty")
    if heading.lower() in GENERIC:
        raise CheckFailed(
            f"<h1> is {heading!r}, a document type rather than a project. "
            "A restriction on using a name is not permission to have none."
        )
    if PROJECT.lower() not in heading.lower():
        raise CheckFailed(f"<h1> is {heading!r} and does not name {PROJECT}")


def check_title(page: str) -> None:
    match = re.search(r"<title>(.*?)</title>", page, re.S | re.I)
    if not match:
        raise CheckFailed("no <title>")
    title = match.group(1).strip()
    if PROJECT.lower() not in title.lower():
        raise CheckFailed(f"<title> is {title!r} and does not name {PROJECT}")
    if "—" not in title and "&mdash;" not in title:
        raise CheckFailed(f"<title> should read '<project> — <document type>', got {title!r}")


def check_counts_match_code(page: str) -> None:
    if not FACTS.is_file():
        raise CheckFailed(f"{FACTS.name} missing — regenerate the page")
    recorded = json.loads(FACTS.read_text(encoding="utf-8"))
    live = live_counts()
    for key, value in live.items():
        if recorded.get(key) != value:
            raise CheckFailed(
                f"{key}: page was generated with {recorded.get(key)!r}, code now has {value!r}. "
                "Rerun docs/gen_docs.py."
            )
    for key in ("engines", "drums", "effects", "themes"):
        if not re.search(rf"\b{live[key]}\b", page):
            raise CheckFailed(f"page never states the {key} count ({live[key]})")


def check_self_contained(page: str) -> None:
    hit = EXTERNAL.search(page)
    if hit:
        raise CheckFailed(f"page makes an external request: {hit.group(0)[:60]!r}")


def check_has_negatives(page: str) -> None:
    """The genre's defining feature: what failed stays on the page."""
    if "<div class=\"retract\">" not in page:
        raise CheckFailed("no retractions on the page — negative results are not optional here")
    if "<div class=\"q\">" not in page:
        raise CheckFailed("no open questions on the page")


RULES = (
    ("h1 names the project", check_h1_names_the_project),
    ("title names the project", check_title),
    ("counts match the live code", check_counts_match_code),
    ("no external requests", check_self_contained),
    ("negatives and open questions present", check_has_negatives),
)


# ── running ────────────────────────────────────────────────────────────────

def run(page: str) -> list[str]:
    failures = []
    for name, rule in RULES:
        try:
            rule(page)
        except CheckFailed as exc:
            failures.append(f"{name}: {exc}")
    return failures


def self_test(page: str) -> int:
    """Prove each rule fires. A check that has never failed is not a check."""
    mutations = [
        ("h1 replaced with a generic noun",
         lambda p: re.sub(r"<h1[^>]*>.*?</h1>", "<h1>Notebook</h1>", p, flags=re.S)),
        ("h1 removed",
         lambda p: re.sub(r"<h1[^>]*>.*?</h1>", "", p, flags=re.S)),
        ("title stripped of the project name",
         lambda p: re.sub(r"<title>.*?</title>", "<title>Findings</title>", p, flags=re.S)),
        ("engine count made stale",
         lambda p: p.replace(str(len(ENGINE_NAMES)), "999")),
        ("external stylesheet injected",
         lambda p: p.replace("</head>", '<link rel="stylesheet" href="https://cdn.example/x.css"></head>')),
        ("retractions removed",
         lambda p: p.replace('<div class="retract">', '<div class="plain">')),
    ]
    bad = 0
    for label, mutate in mutations:
        if run(mutate(page)):
            print(f"  caught: {label}")
        else:
            print(f"  MISSED: {label}  <-- a rule is not working")
            bad += 1
    return bad


def main(argv: list[str]) -> int:
    if not PAGE.is_file():
        print(f"no page at {PAGE} — run docs/gen_docs.py first")
        return 1
    page = PAGE.read_text(encoding="utf-8")

    if "--self-test" in argv:
        print("self-test — every rule should catch its own corruption:")
        missed = self_test(page)
        if missed:
            print(f"\n{missed} rule(s) did not fire")
            return 1
        print("\nall rules fire")

    failures = run(page)
    if failures:
        print("\npage check FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    counts = live_counts()
    print(f"page check OK — {PROJECT} v{counts['version']}, "
          f"{counts['engines']} engines / {counts['drums']} drums / "
          f"{counts['effects']} effects / {counts['themes']} themes, "
          f"{len(page) / 1024:.0f} KB, no external requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
