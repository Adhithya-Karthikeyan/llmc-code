"""Image attachment support for vision-capable local models.

Pure, dependency-free (stdlib ``base64`` + ``mimetypes`` + ``os`` only) and
network-free: an image is read from disk, size-capped, and base64-inlined into
the OpenAI/LM Studio vision content shape

    {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<b64>"}}

so it is only ever sent to the already loopback-pinned local server. Vision
models (gemma, qwen-vl, llava, nemotron-omni, ...) accept this; text-only models
reject it (surfaced gracefully by the caller).

``text_of`` lives here too: it extracts the plain text from a message ``content``
that may be EITHER a string (the common text-only turn) OR the multimodal list
form above, so token-estimation / memory / summarization never ingest a giant
base64 blob.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import shlex
import urllib.parse

# Hard cap on a single image's on-disk size. base64 images bloat the prompt and
# slow local decode, so reject anything larger with an actionable error. Backed
# by the optional ``config.max_image_bytes`` field (default == this value).
MAX_IMAGE_BYTES = 5_000_000

# Supported image types, keyed by lowercase extension. Extension-first detection
# is reliable for this fixed set; ``mimetypes`` is the fallback. Anything not in
# (or resolving to) an ``image/*`` type is rejected.
_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _mime_for(path: str) -> str:
    """Return the image MIME type for ``path`` or raise ValueError if unsupported."""
    ext = os.path.splitext(path)[1].lower()
    mime = _EXT_MIME.get(ext)
    if mime is None:
        guessed, _ = mimetypes.guess_type(path)
        if guessed and guessed.startswith("image/"):
            mime = guessed
    if mime is None:
        supported = ", ".join(sorted(_EXT_MIME))
        raise ValueError(
            f"not a supported image: {path} (supported: {supported})"
        )
    return mime


def encode_image_bytes(data: bytes, mime: str) -> dict:
    """Build the OpenAI vision ``image_url`` part from raw bytes + a MIME type.

    Pure and network-free; does NOT enforce the size cap (caller's job).
    """
    b64 = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def encode_image(path: str, *, max_bytes: int = MAX_IMAGE_BYTES) -> dict:
    """Read an image file and return its OpenAI vision ``image_url`` content part.

    Raises ValueError for a missing/non-file path, an unsupported (non-image)
    type, or a file larger than ``max_bytes``. The error text is actionable so
    the REPL can print it verbatim.
    """
    if not os.path.exists(path):
        raise ValueError(f"image not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"not a file: {path}")
    mime = _mime_for(path)
    size = os.path.getsize(path)
    if max_bytes and size > max_bytes:
        raise ValueError(
            f"image too large: {size / 1_000_000:.1f} MB > "
            f"{max_bytes / 1_000_000:.1f} MB cap; resize it"
        )
    with open(path, "rb") as fh:
        data = fh.read()
    return encode_image_bytes(data, mime)


def text_of(content) -> str:
    """Extract plain text from a message ``content`` of either shape.

    - string  -> returned as-is (the common text-only turn),
    - list    -> the ``text`` parts joined with newlines (image parts ignored),
                 so base64 image blobs never reach BM25/memory/token-estimation,
    - None / other -> "".
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(t for t in parts if t)
    return ""


# --------------------------------------------------------------------------- #
# Drag-and-drop path detection
#
# Dragging a file into a terminal pastes its PATH as literal text into the
# prompt. The exact spelling varies by terminal/OS, so these helpers normalize
# every common form into a plain path and conservatively decide whether that
# path points at a real, supported image file. "Conservative" is the whole game
# here: a token only becomes an attachment if the file actually EXISTS and is an
# image, so ordinary prose that merely mentions "foo.png" stays as text.
# --------------------------------------------------------------------------- #


def normalize_dropped_path(token: str) -> str | None:
    """Clean one dropped token into a candidate filesystem path (no existence check).

    Handles, in order: surrounding whitespace, one layer of matching single/
    double quotes, a leading ``file://`` URL (with ``%XX`` percent-decoding),
    backslash-escaped spaces (``\\ `` -> `` ``), and leading-``~`` expansion.
    Returns the cleaned path, or ``None`` if the token is empty/whitespace-only.
    """
    if token is None:
        return None
    s = token.strip()
    if not s:
        return None
    # Strip one layer of surrounding matching quotes ('...' or "...").
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    # file:// URL -> local path, percent-decoding %XX (e.g. %20 -> space).
    if s.startswith("file://"):
        s = urllib.parse.unquote(s[len("file://"):])
    # Unescape backslash-escaped spaces (macOS Terminal/iTerm drag form). A
    # no-op on tokens shlex already unescaped, so it is safe to always apply.
    s = s.replace("\\ ", " ")
    s = os.path.expanduser(s).strip()
    return s or None


def is_image_path(path: str | None) -> bool:
    """True only if ``path`` exists, is a regular file, and is a supported image.

    Extension/MIME support is delegated to ``_mime_for`` (the same logic the
    encoder uses), so the supported set never drifts between detection and encode.
    """
    if not path or not os.path.isfile(path):
        return False
    try:
        _mime_for(path)
    except ValueError:
        return False
    return True


def extract_image_paths(line: str) -> tuple[list[str], str]:
    """Split a raw input line into (existing-image paths, remaining text).

    Tokenizes with ``shlex.split`` (so quoted/escaped paths survive), falling
    back to a plain whitespace split when shlex raises on unbalanced quotes or a
    lone backslash. Each token is normalized; if it resolves to an EXISTING image
    file it is collected as a path, otherwise the original token stays as text.
    The remaining tokens are rejoined with single spaces (order preserved).
    """
    if not line or not line.strip():
        return [], line
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    paths: list[str] = []
    remaining: list[str] = []
    for tok in tokens:
        norm = normalize_dropped_path(tok)
        if norm is not None and is_image_path(norm):
            paths.append(norm)
        else:
            remaining.append(tok)
    return paths, " ".join(remaining)
