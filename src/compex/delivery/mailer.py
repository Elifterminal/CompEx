"""Deliver a render by email.

This box has no speakers, so "listen to it" means "send it somewhere that
has some". Rather than re-implementing SMTP, this loads the existing
``~/scripts/send_email.py`` and calls its ``send_email`` function.

Nothing here sends anything on its own — it only runs when the UI's send
button is pressed.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Callable

SENDER_SCRIPT = Path.home() / "scripts" / "send_email.py"
DEFAULT_RECIPIENT = "bobbykershii@gmail.com"
MAX_ATTACHMENT_BYTES = 24 * 1024 * 1024

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


class DeliveryError(RuntimeError):
    """Raised when a render cannot be sent — missing sender, bad address, oversized file."""


def _load_sender() -> Callable[..., object]:
    if not SENDER_SCRIPT.exists():
        raise DeliveryError(f"sender script not found at {SENDER_SCRIPT}")
    spec = importlib.util.spec_from_file_location("lee_send_email", SENDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise DeliveryError(f"could not load {SENDER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # the script imports config of its own
        raise DeliveryError(f"{SENDER_SCRIPT.name} failed to import: {exc}") from exc
    sender = getattr(module, "send_email", None)
    if not callable(sender):
        raise DeliveryError(f"{SENDER_SCRIPT.name} has no send_email() function")
    return sender


def send_render(
    wav_path: str | Path,
    recipient: str = DEFAULT_RECIPIENT,
    subject: str | None = None,
    body: str = "",
) -> str:
    """Email ``wav_path`` as an attachment. Returns a short confirmation string."""
    path = Path(wav_path).expanduser()
    if not path.is_file():
        raise DeliveryError(f"no such render: {path}")
    if not _EMAIL.match(recipient.strip()):
        raise DeliveryError(f"that does not look like an email address: {recipient!r}")

    size = path.stat().st_size
    if size > MAX_ATTACHMENT_BYTES:
        raise DeliveryError(
            f"{path.name} is {size / 1e6:.1f} MB, over the {MAX_ATTACHMENT_BYTES / 1e6:.0f} MB limit"
        )

    sender = _load_sender()
    line = subject or f"CompEx render — {path.stem}"
    try:
        sender(recipient.strip(), line, body or f"Attached: {path.name} ({size / 1e6:.1f} MB)",
               attachments=[str(path)])
    except Exception as exc:
        raise DeliveryError(f"send failed: {exc}") from exc

    return f"sent {path.name} ({size / 1e6:.1f} MB) to {recipient.strip()}"
