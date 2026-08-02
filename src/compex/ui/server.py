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
from compex.config import OUTPUT_DIR, KnobError, Knobs
from compex.delivery import DEFAULT_RECIPIENT, DeliveryError, send_render
from compex.generate.mood import AXES, THEMES, Mood, MoodError, theme_names
from compex.pipeline import make_track

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

        result = make_track(knobs, mood)
        stem = f"{mood.nearest_theme()}_seed{knobs.seed}_{result.fingerprint}"
        path = result.save(OUTPUT_DIR / stem, audio_format)
        formula_path = result.save_formula(OUTPUT_DIR / stem)

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
            "movements": report.movements(result.composition),
            "evolution": report.evolution(result.composition),
            "melody": report.melody(result.composition),
            "patterns": report.patterns(result.composition),
            "taste": report.taste(result.composition),
            "ledger": report.ledger(result.composition),
            "hindsight": report.hindsight(result.composition),
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
