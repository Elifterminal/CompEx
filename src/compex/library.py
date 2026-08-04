"""Where finished work goes to live.

``out/`` is a scratch directory: everything lands in it flat, named by mood and
seed, and after a few dozen tracks it is a wall of files with the stems, the
formula and the audio all mixed together. That is fine for a thing that emits
one artifact. This engine emits three, and two of them are only useful if you
can find the third.

So the library is three parallel trees under ``~/Music/CompEx``, one per kind of
artifact, with **the same folder name in each**. Find a track in ``Tracks/`` and
you already know where its stems and its formula are — the name is the key, and
the three trees line up under it.

The name carries the date first, so a file browser sorts by when it was made,
which is how anyone actually looks for a piece. The fingerprint stays on the
end because it is the only part that means anything precise: two folders with
the same fingerprint hold the same recording, whatever else changed.

Nothing here affects the audio. A date in a directory name cannot reach the
composer, and the engine's determinism does not know this module exists.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from compex.config import STEMS_DIR, TRACKS_DIR, TRACK_META_DIR

#: Characters allowed in a track name. Everything else becomes an underscore —
#: a mood can be a set of raw axes with no theme name, and a track that cannot
#: be written because its name has a slash in it is a poor trade.
SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def track_name(theme: str, seed: int, fingerprint: str, when: date | None = None) -> str:
    """``2026-08-04_melancholy_seed249984309_0fb34466``."""
    stamp = (when or date.today()).isoformat()
    # Half the fingerprint. Eight hex characters is four billion, which is
    # plenty to tell one personal library's tracks apart, and the full digest
    # is in the formula and the report if anything ever needs to be exact.
    raw = f"{stamp}_{theme}_seed{seed}_{fingerprint[:8]}"
    return "".join(character if character in SAFE else "_" for character in raw)


def save(result, audio_format: str = "wav", stems: bool = True,
         report_json: dict | None = None, when: date | None = None) -> dict:
    """Write a finished render into the library. Returns what was written.

    ``stems`` is the expensive part — it re-renders the piece once per voice —
    so it is a choice rather than something that always happens.
    """
    name = track_name(result.mood.nearest_theme(), result.knobs.seed,
                      result.fingerprint, when)

    track_dir = TRACKS_DIR / name
    track_dir.mkdir(parents=True, exist_ok=True)
    audio = result.save(track_dir / name, audio_format)

    meta_dir = TRACK_META_DIR / name
    meta_dir.mkdir(parents=True, exist_ok=True)
    formula = result.save_formula(meta_dir / "formula")
    written = {"name": name, "audio": audio, "formula": formula, "stems": []}

    if report_json is not None:
        report_path = meta_dir / "report.json"
        report_path.write_text(json.dumps(report_json, indent=2, sort_keys=True,
                                          default=str) + "\n", encoding="utf-8")
        written["report"] = report_path

    if stems:
        written["stems"] = result.save_stems(STEMS_DIR / name, audio_format)

    return written


def relative(path: Path) -> str:
    """A path as the user would say it, for anything that has to print one."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)
