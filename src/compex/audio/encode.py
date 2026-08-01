"""MP3 export via ffmpeg.

ffmpeg is already on this box and carries libmp3lame, so this costs no Python
dependency. If it ever is not there, the error says so plainly rather than
producing a zero-byte file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"
DEFAULT_BITRATE = "192k"
ENCODE_TIMEOUT_S = 180

FORMATS: tuple[str, ...] = ("wav", "mp3")


class EncodeError(RuntimeError):
    """Raised when transcoding fails or the encoder is unavailable."""


def available() -> bool:
    return shutil.which(FFMPEG) is not None


def to_mp3(wav_path: str | Path, bitrate: str = DEFAULT_BITRATE,
           destination: str | Path | None = None) -> Path:
    """Transcode a WAV to MP3 next to it (or to ``destination``). Returns the new path."""
    source = Path(wav_path).expanduser()
    if not source.is_file():
        raise EncodeError(f"no such wav: {source}")
    if not available():
        raise EncodeError("ffmpeg not found on PATH — cannot write MP3, use WAV instead")

    target = Path(destination).expanduser() if destination else source.with_suffix(".mp3")
    target.parent.mkdir(parents=True, exist_ok=True)

    command = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-codec:a", "libmp3lame", "-b:a", str(bitrate),
        str(target),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=ENCODE_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise EncodeError(f"ffmpeg timed out after {ENCODE_TIMEOUT_S}s") from exc
    except OSError as exc:
        raise EncodeError(f"could not run ffmpeg: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise EncodeError(f"ffmpeg failed: {detail[-1] if detail else result.returncode}")
    if not target.is_file() or target.stat().st_size == 0:
        raise EncodeError(f"ffmpeg produced no output at {target}")

    return target
