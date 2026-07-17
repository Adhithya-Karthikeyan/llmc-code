"""Per-project session persistence (Claude-Code-style: auto-save, opt-in resume).

The REPL auto-saves the running conversation for the current working directory
after every completed turn (so a crash never loses it) and on a clean exit.
Resuming is OPT-IN — a fresh launch stays light: the saved history is only loaded
back when the user passes ``--continue``/``-c`` or runs ``/resume``. A dim
startup hint mentions that a prior session exists.

Sessions are stored LOCALLY ONLY under ``~/.llm-cli/sessions`` and are NEVER sent
anywhere. The conversation may contain file contents the model read; that is fine
for a local tool, but the saved file stays on this machine.

stdlib only (json, hashlib, pathlib, datetime, os, tempfile) — no new deps. Every
disk operation is best-effort: a save NEVER raises, and a load returns ``None``
rather than crashing on a missing/corrupt/invalid file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Sessions live alongside the rest of the app's state under ~/.llm-cli.
SESSIONS_DIRNAME = "sessions"


def _sanitize(name: str) -> str:
    """Reduce a directory basename to a filesystem-safe slug ([A-Za-z0-9._-]).

    Any other character becomes '-'. An empty/whitespace-only result falls back
    to "workspace" so the id always has a readable leading component.
    """
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "-" for c in str(name))
    return cleaned or "workspace"


def session_id(cwd: str) -> str:
    """A STABLE slug identifying the workspace dir = "<basename>-<12 hex>".

    The hex is the first 12 chars of sha256(abspath(cwd)), so the SAME directory
    always maps to the SAME id and DIFFERENT directories map to different ids
    (even when their basenames collide). The basename is sanitized to
    [A-Za-z0-9._-] so it is safe to use directly as a filename component.
    """
    abspath = os.path.abspath(cwd)
    digest = hashlib.sha256(abspath.encode("utf-8")).hexdigest()[:12]
    return f"{_sanitize(os.path.basename(abspath))}-{digest}"


def sessions_dir() -> Path:
    """Directory holding per-project session files (~/.llm-cli/sessions).

    Created on demand by save_session; reads tolerate it not existing yet.
    """
    return Path.home() / ".llm-cli" / SESSIONS_DIRNAME


def _harden_state_dir() -> None:
    """Best-effort: restrict ~/.llm-cli to the owner (0700) so the per-project
    session/memory filenames aren't enumerable by other local users. Never raises
    (a missing dir / unsupported chmod is silently ignored)."""
    try:
        os.chmod(Path.home() / ".llm-cli", 0o700)
    except OSError:
        pass


def session_path(cwd: str) -> Path:
    """Path to the JSON session file for ``cwd``."""
    return sessions_dir() / f"{session_id(cwd)}.json"


def derive_title(messages: list[dict]) -> str:
    """The first user message, single-lined and truncated to ~60 chars.

    Falls back to "(no title)" when there is no user message (or it is empty),
    so a saved session always has a readable label for the startup hint.
    """
    for msg in messages or []:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        text = " ".join(content.split())
        if not text:
            continue
        return text[:60] + ("…" if len(text) > 60 else "")
    return "(no title)"


def save_session(cwd: str, messages: list[dict], model: str, title: str) -> None:
    """Atomically persist the conversation for ``cwd`` (best-effort, never raises).

    Writes a temp file in the sessions dir then ``os.replace``s it into place, so
    a reader never sees a half-written file. Skips writing entirely when there is
    nothing to remember — ``messages`` with <=1 entry is just the system prompt.
    All OSError is swallowed: a failed save must never crash the REPL.
    """
    if not isinstance(messages, list):
        return
    # Drop EPHEMERAL tagged messages (`_memory` retrieval blocks, `_nudge`
    # re-prompts) before persisting: they are regenerated every turn, so saving
    # them would reload a stale block as context on resume. Filter BEFORE the
    # <=1 skip check so a history that is only system + an ephemeral block is
    # correctly treated as "nothing to remember".
    messages = [
        m for m in messages
        if not (isinstance(m, dict) and (m.get("_memory") or m.get("_nudge")))
    ]
    if len(messages) <= 1:
        return  # only the system prompt (or empty) — nothing to remember
    payload = {
        "cwd": os.path.abspath(cwd),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "title": title,
        "messages": messages,
    }
    try:
        directory = sessions_dir()
        directory.mkdir(parents=True, exist_ok=True)
        _harden_state_dir()  # restrict ~/.llm-cli to the owner (best-effort)
        # Temp file in the SAME dir so os.replace is atomic (same filesystem).
        fd, tmp = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, str(session_path(cwd)))
        except OSError:
            # Clean up the orphan temp file; ignore if it is already gone.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        return  # best-effort: disk full / permissions / etc. — silently skip


def load_session(cwd: str) -> dict | None:
    """Read and parse the saved session for ``cwd``; ``None`` if unavailable.

    Returns ``None`` (never raises) when the file is missing, unreadable, not
    valid JSON, not an object, or lacks a usable ``messages`` list.
    """
    try:
        raw = session_path(cwd).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        return None
    # Drop any non-dict element so a corrupt/hand-edited file (e.g. a stray
    # scalar in the list) can never crash a downstream consumer that does
    # msg.get(...). A list left empty after filtering is still a valid (if
    # trivial) "no real history" result.
    data["messages"] = [m for m in messages if isinstance(m, dict)]
    return data


def session_meta(cwd: str) -> dict | None:
    """Lightweight session summary WITHOUT the big ``messages`` list.

    Returns ``{"updated_at", "title", "message_count", "model"}`` for the startup
    hint, or ``None`` when there is no usable saved session. The message list is
    intentionally dropped so the hint never pulls a whole conversation into memory.
    """
    data = load_session(cwd)
    if data is None:
        return None
    return {
        "updated_at": data.get("updated_at", ""),
        "title": data.get("title", "") or "(no title)",
        "message_count": len(data.get("messages", [])),
        "model": data.get("model", ""),
    }


def clear_session(cwd: str) -> None:
    """Delete the saved session file for ``cwd`` (best-effort, never raises)."""
    try:
        session_path(cwd).unlink()
    except OSError:
        return  # already gone / unreadable — nothing to do


def relative_time(iso: str) -> str:
    """A short human "5m ago / 3h ago / 2d ago" from an ISO-8601 timestamp.

    Compared against ``datetime.now(timezone.utc)``. Tolerant of bad input: an
    unparsable/empty/future timestamp returns "just now" rather than raising.
    """
    try:
        when = datetime.fromisoformat(str(iso))
    except (ValueError, TypeError):
        return "just now"
    # Treat a naive timestamp as UTC so the subtraction never raises on mixed
    # aware/naive datetimes (older saves, hand-edited files).
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - when).total_seconds()
    if seconds < 45:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"
