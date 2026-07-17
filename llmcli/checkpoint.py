"""File-snapshot safety net for reversible edits (git-free ``/undo``).

Before a mutating tool (``write_file`` / ``edit_file``) touches the disk, a
caller records a CHECKPOINT: the CURRENT on-disk bytes of each affected path
(or a "did-not-exist" marker for a brand-new file). A later ``/undo`` restores
the most recent checkpoint — putting the old bytes back, or deleting a file
that did not exist before — so a bad edit is always one step reversible without
any version-control system.

Storage is LOCAL ONLY, under a per-project directory keyed the SAME way
:mod:`llmcli.session` keys its per-project session file (``session_id``), so a
checkpoint set never leaks between unrelated projects:

    ~/.llm-cli/checkpoints/<project-id>/
        index.json        # ordered metadata (oldest -> newest)
        blobs/<ref>.blob  # a byte-for-byte copy of each snapshotted file

The public functions accept an OPTIONAL ``session`` token. When given, the
checkpoints are stored/read under a per-session subdirectory so a fresh REPL
session only ever sees (and can ``/undo``) its OWN checkpoints:

    ~/.llm-cli/checkpoints/<project-id>/sessions/<session>/
        index.json
        blobs/<ref>.blob

When ``session`` is ``None`` the layout above (no ``sessions/`` level) is used
unchanged — existing callers keep byte-identical behavior.

History is bounded to the last :data:`MAX_CHECKPOINTS` checkpoints; the oldest
is evicted (index entry + its blobs) when the bound is exceeded.

stdlib only (json, os, time, tempfile, pathlib) — no new deps, no git. Every
disk write is atomic (temp file in the same dir + ``os.replace``) so a crash
mid-write never leaves a half-written index or blob. Confinement is enforced on
BOTH snapshot and restore: a path that resolves outside ``root`` is refused, so
an undo can never write outside the project tree.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .session import session_id

# Checkpoints live alongside the rest of the app's state under ~/.llm-cli.
CHECKPOINTS_DIRNAME = "checkpoints"
# Bounded history: keep at most this many checkpoints per project; evict oldest.
MAX_CHECKPOINTS = 50


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def checkpoints_dir() -> Path:
    """Root directory holding every project's checkpoints (~/.llm-cli/checkpoints).

    Created on demand by :func:`snapshot`; reads tolerate it not existing yet.
    """
    return Path.home() / ".llm-cli" / CHECKPOINTS_DIRNAME


def _session_subdir(session: str | None) -> str | None:
    """Sanitized single path component for ``session`` (``None`` -> ``None``).

    The token is filtered to filesystem-safe characters so a caller-supplied
    session can never escape the checkpoints tree (no ``..``, ``/`` or absolute
    paths); ``None`` is passed through so the legacy (session-less) layout is
    selected unchanged.
    """
    if session is None:
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(session))
    return safe.strip(".") or "session"


def project_dir(root: str, session: str | None = None) -> Path:
    """Directory holding the checkpoints for the project rooted at ``root``.

    The per-project component is :func:`llmcli.session.session_id` of ``root`` —
    the exact scheme :mod:`llmcli.session` uses for its per-project session file
    — so the SAME project always maps to the SAME directory and different
    projects never collide.

    When ``session`` is given, a ``sessions/<session>`` subdirectory is appended
    so each REPL session's checkpoints are isolated. When ``session`` is
    ``None`` the legacy path (``<checkpoints>/<project-id>``) is returned
    unchanged for back-compat.
    """
    base = checkpoints_dir() / session_id(str(root))
    sub = _session_subdir(session)
    if sub is not None:
        return base / "sessions" / sub
    return base


def _blobs_dir(root: str, session: str | None = None) -> Path:
    """Directory holding the stored byte-copies for ``root``'s checkpoints."""
    return project_dir(root, session) / "blobs"


def _index_path(root: str, session: str | None = None) -> Path:
    """Path to the JSON metadata index for ``root``'s checkpoints."""
    return project_dir(root, session) / "index.json"


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _harden_state_dir() -> None:
    """Best-effort: restrict ~/.llm-cli to the owner (0700). Never raises.

    Mirrors :func:`llmcli.session._harden_state_dir` so snapshotted file bytes
    aren't world-readable via a lax parent dir.
    """
    try:
        os.chmod(Path.home() / ".llm-cli", 0o700)
    except OSError:
        pass


def _abs_root(root: str) -> str:
    """The confinement anchor for ``root``: its fully resolved absolute path.

    ``realpath`` resolves symlinks in existing components so a symlinked project
    dir can't be used to smuggle a restore outside the real tree.
    """
    return os.path.realpath(str(root))


def _resolve_within(path: str, abs_root: str) -> tuple[str, str]:
    """Resolve ``path`` against ``abs_root`` and require it to stay inside it.

    ``path`` may be absolute or relative to ``abs_root``. Returns
    ``(relative_path, absolute_path)`` where ``relative_path`` is what gets
    stored in the index (portable + re-verified on restore). Raises
    :class:`ValueError` when the target escapes ``abs_root`` (``..`` traversal,
    an absolute path elsewhere, a symlink pointing out), equals the root itself,
    or is an existing non-regular file (e.g. a directory) — only regular files
    are snapshotted.
    """
    target = os.path.realpath(os.path.join(abs_root, str(path)))
    try:
        inside = os.path.commonpath([abs_root, target]) == abs_root
    except ValueError:
        inside = False  # different drives / mixed abs+rel — treat as outside
    if not inside or target == abs_root:
        raise ValueError(f"path {path!r} resolves outside root {abs_root!r}")
    if os.path.isdir(target):
        raise ValueError(f"path {path!r} is a directory; only files are supported")
    rel = os.path.relpath(target, abs_root)
    return rel, target


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` (temp file in same dir + replace).

    Creates parent directories on demand. Cleans up the temp file on failure.
    Raises OSError on a genuine disk error (callers decide whether to swallow).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, str(path))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_index(root: str, session: str | None = None) -> list[dict]:
    """Read the metadata index for ``root``; ``[]`` on missing/corrupt/invalid.

    Never raises. Drops any element that is not a checkpoint-shaped dict (a
    ``str`` id and a ``list`` of file entries) so a hand-edited/corrupt file can
    never crash a consumer downstream.
    """
    try:
        raw = _index_path(root, session).read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    records = data.get("checkpoints") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return []
    return [
        r for r in records
        if isinstance(r, dict)
        and isinstance(r.get("id"), str)
        and isinstance(r.get("files"), list)
    ]


def _save_index(root: str, records: list[dict], session: str | None = None) -> bool:
    """Atomically persist the ordered checkpoint list for ``root`` (never raises).

    Returns ``True`` on success and ``False`` if the write could not be made
    (disk full / permissions), so a caller can react — e.g. clean up blobs it
    just wrote — instead of leaving an index/blob mismatch on disk.
    """
    payload = {"checkpoints": records}
    try:
        directory = project_dir(root, session)
        directory.mkdir(parents=True, exist_ok=True)
        _harden_state_dir()
        _atomic_write_bytes(
            _index_path(root, session),
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        return True
    except OSError:
        return False  # best-effort: disk full / permissions — signal failure


def _delete_blobs(root: str, record: dict, session: str | None = None) -> None:
    """Best-effort remove every blob referenced by ``record`` (never raises)."""
    blobs = _blobs_dir(root, session)
    for entry in record.get("files", []):
        ref = entry.get("blob") if isinstance(entry, dict) else None
        if not ref:
            continue
        try:
            (blobs / str(ref)).unlink()
        except OSError:
            pass


def _new_id(ts: float) -> str:
    """A sortable, collision-resistant checkpoint id: ``ck-<ms>-<6 hex>``."""
    return f"ck-{int(ts * 1000)}-{os.urandom(3).hex()}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def snapshot(
    paths,
    *,
    root: str,
    label: str = "",
    timestamp: float | None = None,
    session: str | None = None,
) -> str:
    """Record the CURRENT on-disk state of ``paths`` and return a checkpoint id.

    For each path (absolute, or relative to ``root``):
      - if a regular file exists, its exact bytes are copied to a blob and the
        entry is marked ``existed=True`` (binary-safe — raw bytes, no decode);
      - if it does not exist, the entry is marked ``existed=False`` so a later
        :func:`undo` can DELETE the file the caller is about to create.

    ``label`` is a free-form note (e.g. the tool + reason). ``timestamp``
    defaults to :func:`time.time`; pass one in for deterministic tests.
    ``session`` (optional) isolates this checkpoint under a per-session
    subdirectory so only the SAME session's :func:`undo`/:func:`list_checkpoints`
    can see it; ``None`` keeps the legacy shared layout. The new checkpoint
    becomes the most-recent; when the history exceeds :data:`MAX_CHECKPOINTS`
    the oldest checkpoint (and its blobs) is evicted.

    Raises :class:`ValueError` if any path resolves outside ``root`` or is a
    directory — nothing is written in that case, so a bad path can never leave a
    partial checkpoint that a later undo would act on.
    """
    ts = time.time() if timestamp is None else float(timestamp)
    abs_root = _abs_root(root)
    ck_id = _new_id(ts)

    # Resolve+validate ALL paths first (fail fast, before writing any blob).
    resolved: list[tuple[str, str]] = [_resolve_within(p, abs_root) for p in paths]

    blobs = _blobs_dir(root, session)
    entries: list[dict] = []
    for i, (rel, target) in enumerate(resolved):
        if os.path.isfile(target):
            ref = f"{ck_id}-{i}.blob"
            _atomic_write_bytes(blobs / ref, Path(target).read_bytes())
            entries.append({"path": rel, "existed": True, "blob": ref})
        else:
            entries.append({"path": rel, "existed": False, "blob": None})

    record = {"id": ck_id, "timestamp": ts, "label": str(label), "files": entries}
    index = _load_index(root, session)
    index.append(record)
    while len(index) > MAX_CHECKPOINTS:
        _delete_blobs(root, index.pop(0), session)  # evict oldest: entry + blobs

    # If the index can't be persisted, the blobs we just wrote for THIS
    # checkpoint would be orphaned (the checkpoint is not recorded). Best-effort
    # remove them so no half-recorded checkpoint is left behind.
    try:
        saved = _save_index(root, index, session)
    except Exception:  # a monkeypatched/failing save must not surface here
        saved = False
    if not saved:
        _delete_blobs(root, record, session)
    return ck_id


def undo(root: str, *, session: str | None = None) -> dict:
    """Restore the MOST RECENT checkpoint for ``root`` and pop it off the stack.

    ``session`` (optional) scopes the undo to a single REPL session's
    checkpoints; ``None`` uses the legacy shared layout. A fresh session with no
    checkpoints of its own correctly reports "nothing to undo".

    For each file in the checkpoint:
      - ``existed=True``  -> the stored old bytes are written back (atomically);
      - ``existed=False`` -> the file is deleted (it did not exist before).

    Confinement is re-verified per path, so a tampered index can never make undo
    write or delete outside ``root``. Returns a summary::

        {
          "undone": bool,       # True iff a checkpoint was restored
          "id": str | None,     # the checkpoint id that was undone
          "label": str,
          "restored": [rel],    # files whose old bytes were put back
          "deleted": [rel],     # newly-created files that were removed
          "errors": [str],      # per-file problems (missing blob, bad path, ...)
          "message": str,       # human-readable one-liner
        }

    When there is nothing to undo, ``undone`` is ``False`` and ``message`` is an
    actionable "nothing to undo" note (never raises).
    """
    index = _load_index(root, session)
    if not index:
        return {
            "undone": False,
            "id": None,
            "label": "",
            "restored": [],
            "deleted": [],
            "errors": [],
            "message": "Nothing to undo — no checkpoints recorded for this project.",
        }

    abs_root = _abs_root(root)
    record = index[-1]
    restored: list[str] = []
    deleted: list[str] = []
    errors: list[str] = []

    for entry in record.get("files", []):
        rel = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(rel, str):
            errors.append(f"skipped malformed entry {entry!r}")
            continue
        try:
            _, target = _resolve_within(rel, abs_root)  # re-verify confinement
        except ValueError:
            errors.append(f"skipped out-of-root path {rel!r}")
            continue
        if entry.get("existed"):
            ref = entry.get("blob")
            try:
                data = (_blobs_dir(root, session) / str(ref)).read_bytes()
            except OSError:
                errors.append(f"missing stored bytes for {rel!r}")
                continue
            try:
                _atomic_write_bytes(Path(target), data)
                restored.append(rel)
            except OSError as exc:
                errors.append(f"could not restore {rel!r}: {exc}")
        else:
            try:
                os.unlink(target)
                deleted.append(rel)
            except FileNotFoundError:
                pass  # already absent = the intended post-undo state
            except OSError as exc:
                errors.append(f"could not delete {rel!r}: {exc}")

    # Pop the undone checkpoint and drop its blobs only AFTER restoring.
    index.pop()
    _save_index(root, index, session)
    _delete_blobs(root, record, session)

    parts = []
    if restored:
        parts.append(f"restored {len(restored)}")
    if deleted:
        parts.append(f"removed {len(deleted)}")
    if errors:
        parts.append(f"{len(errors)} error(s)")
    detail = ", ".join(parts) if parts else "no file changes"
    label = record.get("label") or ""
    suffix = f" ({label})" if label else ""
    return {
        "undone": True,
        "id": record.get("id"),
        "label": label,
        "restored": restored,
        "deleted": deleted,
        "errors": errors,
        "message": f"Undid checkpoint {record.get('id')}{suffix}: {detail}.",
    }


def list_checkpoints(root: str, *, session: str | None = None) -> list[dict]:
    """Return the recorded checkpoints for ``root``, NEWEST first.

    Each item is a light copy safe to display — ``{"id", "timestamp", "label",
    "files": [rel, ...], "count"}`` — without exposing internal blob refs. Empty
    list when there are none (never raises). ``session`` (optional) scopes the
    listing to a single REPL session; ``None`` uses the legacy shared layout.
    """
    out: list[dict] = []
    for record in reversed(_load_index(root, session)):
        files = [
            e.get("path") for e in record.get("files", [])
            if isinstance(e, dict) and isinstance(e.get("path"), str)
        ]
        out.append({
            "id": record.get("id"),
            "timestamp": record.get("timestamp"),
            "label": record.get("label", ""),
            "files": files,
            "count": len(files),
        })
    return out


def clear(root: str, *, session: str | None = None) -> None:
    """Delete ALL checkpoints (index + blobs) for ``root`` (best-effort).

    Removes the per-project directory (or, when ``session`` is given, only that
    session's subdirectory). Never raises: a missing or partially-removable
    directory is silently tolerated.
    """
    directory = project_dir(root, session)
    # Manual bottom-up removal keeps it dependency-free and best-effort per node.
    try:
        for base, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                try:
                    os.unlink(os.path.join(base, name))
                except OSError:
                    pass
            for name in dirs:
                try:
                    os.rmdir(os.path.join(base, name))
                except OSError:
                    pass
        os.rmdir(directory)
    except OSError:
        return  # already gone / not fully removable — nothing more to do
