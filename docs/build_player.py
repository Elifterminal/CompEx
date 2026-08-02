#!/usr/bin/env python3
"""Package the engine for the browser.

The player runs the real package under Pyodide rather than a JavaScript port,
so it needs the source shipped alongside the page. This zips `src/compex` into
`docs/play/compex-src.zip` and records a hash of the sources it came from.

`check_page.py` recomputes that hash, so a stale zip fails the build. Without
it the phone would quietly keep playing an old engine while the desktop had
moved on — the exact silent drift a port was avoided to prevent.

    PYTHONPATH=src python3 docs/build_player.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "compex"
TARGET = ROOT / "docs" / "play" / "compex-src.zip"
STAMP = ROOT / "docs" / "play" / "engine.json"

#: Not needed in the browser, and it drags in an outbound-mail path nothing there uses.
SKIP_DIRS = {"__pycache__", "ui", "delivery"}


def sources() -> list[Path]:
    return sorted(
        path for path in SOURCE.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.relative_to(SOURCE).parts)
    )


def fingerprint(paths: list[Path]) -> str:
    """One hash over every source file that goes into the zip."""
    digest = hashlib.blake2b(digest_size=16)
    for path in paths:
        digest.update(str(path.relative_to(SOURCE)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build() -> dict:
    paths = sources()
    if not paths:
        raise SystemExit(f"no sources found under {SOURCE}")

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            # Fixed timestamps so an unchanged engine produces an identical zip.
            info = zipfile.ZipInfo(str(Path("compex") / path.relative_to(SOURCE)),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())

    sys.path.insert(0, str(ROOT / "src"))
    from compex import __version__  # noqa: E402  (needs the path above)

    stamp = {
        "version": __version__,
        "fingerprint": fingerprint(paths),
        "modules": len(paths),
        "bytes": TARGET.stat().st_size,
    }
    STAMP.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return stamp


def main() -> int:
    stamp = build()
    print(f"wrote {TARGET.relative_to(ROOT)} "
          f"({stamp['modules']} modules, {stamp['bytes'] / 1024:.0f} KB)")
    print(f"      fingerprint {stamp['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
