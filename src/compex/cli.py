"""Command line front end.

``compex make --mood menacing --duration 120 --format mp3``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compex import __version__
from compex.audio import encode
from compex.config import OUTPUT_DIR, KnobError, Knobs
from compex.delivery import DEFAULT_RECIPIENT, DeliveryError, send_render
from compex.generate.mood import AXES, Mood, MoodError, THEMES, theme_names
from compex.notation import read_formula
from compex.pipeline import make_track


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compex", description=__doc__)
    parser.add_argument("--version", action="version", version=f"compex {__version__}")
    subs = parser.add_subparsers(dest="command", required=True)

    make = subs.add_parser("make", help="invent and render a track")
    make.add_argument("--seed", type=int, default=Knobs.seed)
    make.add_argument("--duration", type=float, default=Knobs.duration_s, help="seconds")
    make.add_argument("--mood", default="melancholy",
                      help=f"emotional theme: {', '.join(theme_names())}")
    make.add_argument("--axis", action="append", default=[], metavar="NAME=VALUE",
                      help=f"override one mood axis ({', '.join(AXES)}); repeatable")
    make.add_argument("--format", default="wav", choices=encode.FORMATS)
    make.add_argument("--master", type=float, default=Knobs.master_gain)
    make.add_argument("-o", "--out", help="output path (extension set by --format)")
    make.add_argument("--email", nargs="?", const=DEFAULT_RECIPIENT,
                      help=f"email the result (default {DEFAULT_RECIPIENT})")
    make.add_argument("--stems", nargs="?", const="", metavar="DIR",
                      help="also write every voice as its own file, at the level it has "
                           "in the mix (default: a stems/ folder beside the track)")
    make.add_argument("-q", "--quiet", action="store_true")

    replay = subs.add_parser("replay", help="re-render a track from its formula file")
    replay.add_argument("formula", help="path to a .tex formula, or - for stdin")
    replay.add_argument("--format", default="wav", choices=encode.FORMATS)
    replay.add_argument("-o", "--out")

    subs.add_parser("themes", help="list the emotional themes")
    subs.add_parser("ui", help="start the local web UI")
    return parser


def _mood_from_args(args: argparse.Namespace) -> Mood:
    mood = Mood.from_theme(args.mood)
    if not args.axis:
        return mood

    overrides: dict[str, str] = {}
    for item in args.axis:
        if "=" not in item:
            raise SystemExit(f"--axis expects NAME=VALUE, got {item!r}")
        name, _, value = item.partition("=")
        overrides[name.strip()] = value.strip()

    checked = Mood.parse(overrides)  # validates axis names and ranges
    merged = mood.as_dict()
    merged.update({name: getattr(checked, name) for name in overrides})
    return Mood(**merged)


def _make(args: argparse.Namespace) -> int:
    try:
        knobs = Knobs.parse({
            "seed": args.seed,
            "duration_s": args.duration,
            "master_gain": args.master,
        })
        mood = _mood_from_args(args)
    except (KnobError, MoodError) as exc:
        raise SystemExit(f"bad input: {exc}") from exc

    progress = None if args.quiet else (lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"))
    result = make_track(knobs, mood, progress=progress)

    stem = f"{mood.nearest_theme()}_seed{knobs.seed}_{result.fingerprint}"
    destination = Path(args.out) if args.out else OUTPUT_DIR / stem
    path = result.save(destination, args.format)
    formula_path = result.save_formula(destination)

    stems: list[Path] = []
    if args.stems is not None:
        where = Path(args.stems) if args.stems else Path(destination).with_suffix("") / "stems"
        stems = result.save_stems(where, args.format)

    if not args.quiet:
        print(result.composition.summary())
        print(f"\nwrote {path}")
        print(f"      {formula_path}")
        for stem_path in stems:
            print(f"      {stem_path}")

    if args.email:
        try:
            print(send_render(path, args.email, body=result.formula))
        except DeliveryError as exc:
            raise SystemExit(f"email failed: {exc}") from exc
    return 0


def _replay(args: argparse.Namespace) -> int:
    text = sys.stdin.read() if args.formula == "-" else Path(args.formula).read_text("utf-8")
    formula = read_formula(text)
    knobs = Knobs(seed=formula.seed, duration_s=formula.runtime_s)
    result = make_track(knobs, formula.mood)

    stem = f"replay_seed{formula.seed}_{result.fingerprint}"
    path = result.save(Path(args.out) if args.out else OUTPUT_DIR / stem, args.format)
    print(f"wrote {path}")
    return 0


def _themes() -> int:
    width = max(len(name) for name in THEMES)
    for name in theme_names():
        print(f"  {name:<{width}}  {THEMES[name].describe()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "make":
        return _make(args)
    if args.command == "replay":
        return _replay(args)
    if args.command == "themes":
        return _themes()
    if args.command == "ui":
        from compex.ui.server import main as ui_main

        return ui_main([])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
