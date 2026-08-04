"""Local web UI for CompEx.

Standard-library HTTP server bound to loopback. A browser is the only UI
toolkit guaranteed to be on this box — there is no tkinter here — and it
keeps the app dependency-free for packaging later.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from compex import __version__, report
from compex.audio import encode
from compex import library
from compex.config import MEMORY_PATH, MUSIC_DIR, OUTPUT_DIR, KnobError, Knobs
from compex.delivery import DEFAULT_RECIPIENT, DeliveryError, send_render
from compex.generate.mood import AXES, THEMES, Mood, MoodError, theme_names
from compex.pipeline import make_track
from compex.remember import Memory

HOST = "127.0.0.1"
DEFAULT_PORT = 8733
PORT_ATTEMPTS = 20
MAX_BODY_BYTES = 2_000_000
WAVEFORM_POINTS = 900

STATIC_DIR = Path(__file__).resolve().parent
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".tex": "text/plain; charset=utf-8",
}


#: Every POST route this build serves, reported to the page so it can tell the
#: difference between "that failed" and "you are talking to an older process".
POST_ROUTES = frozenset({"make", "save", "email"})

#: The most recent render, so Save can file it without composing it again.
_LAST: dict = {}
_LAST_LOCK = threading.Lock()


class CompexHandler(BaseHTTPRequestHandler):
    server_version = f"compex/{__version__}"

    # -- routing ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        route = urlparse(self.path).path
        try:
            if route in {"/", "/index.html"}:
                self._send_static("index.html")
            elif route in {"/app.css", "/app.js", "/icon.svg"}:
                self._send_static(route.lstrip("/"))
            elif route == "/api/themes":
                self._send_json({
                    "themes": [{"name": name, **THEMES[name].as_dict()} for name in theme_names()],
                    "axes": list(AXES),
                    "formats": list(encode.FORMATS) if encode.available() else ["wav"],
                    # What this process can actually do. The page is served
                    # fresh from disk on every request but the Python is not
                    # reloaded, so an app left open across an edit ends up with
                    # a new front end talking to an old server — which showed
                    # up as a Save button that answered "not found". The UI
                    # compares this against what it needs and says so plainly.
                    "api": sorted(POST_ROUTES),
                    "version": __version__,
                })
            elif route.startswith("/api/file/"):
                self._send_file(unquote(route[len("/api/file/"):]))
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # never let one bad request kill the server
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            if route == "/api/make":
                self._send_json(self._make(payload))
            elif route == "/api/save":
                self._send_json(self._save(payload))
            elif route == "/api/email":
                self._send_json(self._email(payload))
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KnobError, MoodError, DeliveryError, encode.EncodeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": f"unexpected: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    # -- endpoints -------------------------------------------------------

    def _make(self, payload: dict) -> dict:
        raw_knobs = payload.get("knobs") or {}
        if not isinstance(raw_knobs, dict):
            raise ValueError("knobs must be an object")
        knobs = Knobs.parse(raw_knobs)

        raw_mood = payload.get("mood")
        if isinstance(raw_mood, dict) or isinstance(raw_mood, str) or raw_mood is None:
            mood = Mood.parse(raw_mood)
        else:
            raise ValueError("mood must be a theme name or an object of axes")

        audio_format = str(payload.get("format") or "wav").lower()
        if audio_format not in encode.FORMATS:
            raise ValueError(f"unknown format {audio_format!r}")

        remembered = _load_memory()
        result = make_track(knobs, mood, memory=remembered)
        _save_memory(result.memory)
        stem = f"{mood.nearest_theme()}_seed{knobs.seed}_{result.fingerprint}"
        path = result.save(OUTPUT_DIR / stem, audio_format)
        formula_path = result.save_formula(OUTPUT_DIR / stem)

        # Held so Save can file it without composing the whole thing again.
        # One slot, because this is a local single-user app and the only thing
        # anyone ever wants to save is the piece currently on screen.
        with _LAST_LOCK:
            _LAST["result"] = result
            _LAST["knobs"] = knobs

        return {
            "ok": True,
            "file": path.name,
            "url": f"/api/file/{path.name}",
            "formula_file": formula_path.name,
            "formula_url": f"/api/file/{formula_path.name}",
            "bytes": path.stat().st_size,
            "duration": round(result.duration_s, 2),
            "fingerprint": result.fingerprint,
            "formula": result.formula,
            "summary": result.composition.summary(),
            "theme": mood.nearest_theme(),
            "peaks": _waveform(result.samples),
            **report.everything(result, knobs.ghost_gain),
        }

    def _save(self, payload: dict) -> dict:
        """File the track on screen: audio, formula, report and every stem.

        One button rather than four, because the three artifacts are only
        useful together — a stem with no formula beside it is an orphan, and
        the whole point of the library layout is that one name finds all of it.
        """
        audio_format = str(payload.get("format") or "wav").lower()
        if audio_format not in encode.FORMATS:
            raise ValueError(f"unknown format {audio_format!r}")

        with _LAST_LOCK:
            result = _LAST.get("result")
            knobs = _LAST.get("knobs")
        if result is None:
            raise ValueError("nothing to save yet — make a track first")

        shelved = library.save(result, audio_format, stems=True,
                               report_json=report.everything(result, knobs.ghost_gain))
        return {
            "ok": True,
            "name": shelved["name"],
            "track": library.relative(shelved["audio"]),
            "formula": library.relative(shelved["formula"]),
            "report": library.relative(shelved["report"]),
            "stems": [library.relative(one) for one in shelved["stems"]],
            "root": library.relative(MUSIC_DIR),
        }

    def _email(self, payload: dict) -> dict:
        filename = payload.get("file", "")
        if not isinstance(filename, str) or not _SAFE_NAME.match(filename):
            raise ValueError("bad or missing file name")
        path = _resolve_output(filename)
        recipient = str(payload.get("to") or DEFAULT_RECIPIENT)
        note = str(payload.get("note") or "")
        subject = str(payload.get("subject") or "").strip() or None
        return {"ok": True, "message": send_render(path, recipient, subject=subject, body=note)}

    # -- plumbing --------------------------------------------------------

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body too large ({length} bytes)")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"),
                         "application/json; charset=utf-8", status)

    def _send_static(self, name: str) -> None:
        path = STATIC_DIR / name
        if not path.is_file():
            self._send_json({"error": f"missing asset {name}"}, HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(path.read_bytes(),
                         _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

    def _send_file(self, filename: str) -> None:
        if not _SAFE_NAME.match(filename):
            self._send_json({"error": "bad file name"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            path = _resolve_output(filename)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(
            path.read_bytes(),
            _CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
            extra={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def _send_bytes(self, body: bytes, content_type: str,
                    status: HTTPStatus = HTTPStatus.OK, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}")


def _resolve_output(filename: str) -> Path:
    """Resolve a name inside OUTPUT_DIR, refusing anything that escapes it."""
    root = OUTPUT_DIR.resolve()
    path = (root / filename).resolve()
    if root not in path.parents:
        raise ValueError("path escapes the output directory")
    if not path.is_file():
        raise ValueError(f"no such file: {filename}")
    return path


def _waveform(samples: np.ndarray, points: int = WAVEFORM_POINTS) -> list[float]:
    """Peak envelope, downsampled for the canvas."""
    if len(samples) == 0:
        return []
    bucket = max(1, len(samples) // points)
    usable = (len(samples) // bucket) * bucket
    folded = np.abs(samples[:usable]).reshape(-1, bucket).max(axis=1)
    return [round(float(value), 4) for value in folded]


def _load_memory() -> Memory:
    """What this machine remembers. A missing or broken file is simply amnesia."""
    try:
        return Memory.parse(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Memory()


def _save_memory(memory: Memory) -> None:
    try:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(memory.to_json() + "\n", encoding="utf-8")
    except OSError:
        pass    # a history that cannot be written is not worth failing a render over


def _bind(preferred: int) -> ThreadingHTTPServer:
    last: OSError | None = None
    for offset in range(PORT_ATTEMPTS):
        try:
            return ThreadingHTTPServer((HOST, preferred + offset), CompexHandler)
        except OSError as exc:
            last = exc
    raise SystemExit(f"could not bind a port from {preferred}: {last}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CompEx local UI")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    httpd = _bind(args.port)
    url = f"http://{HOST}:{httpd.server_address[1]}/"

    print(f"CompEx {__version__}")
    print(f"  UI      {url}")
    print(f"  tracks  {OUTPUT_DIR}")
    print("  ctrl-c to stop")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
