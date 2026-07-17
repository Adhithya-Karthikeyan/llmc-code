"""Resolve user-typed ``@``-mentions in a prompt into injected context.

The user can pull files, directories, URLs, or the working-tree diff into the
model's context BEFORE a turn is sent, so the model does not spend slow
tool round-trips fetching things the user already knows it needs.

Supported mention forms (each must sit at a word boundary — start of string or
after whitespace — so an email address like ``a@b.com`` never misfires):

- ``@<relpath>``   inject that file's contents (workspace-relative, size-capped).
- ``@<dir>/``      inject a listing of that directory (ignore rules respected).
- ``@url:<http…>`` / bare ``@https://…``  fetch+inject via an injected callable.
- ``@diff``        inject the working-tree diff via an injected callable.

Design notes
------------
* This module is dependency-injected: ``web_fetch`` / ``git_diff`` / ``read_file``
  are passed in as callables by the caller, so ``mentions`` has NO import cycle
  with ``providers`` (network/model) and is trivially testable with fakes.
* Ignore/skip rules are NOT duplicated: the file walk reuses
  ``tools._iter_source_files`` (prunes ``.git``/venvs/build dirs) and
  ``tools._is_skip_path`` (drops minified/binary/lockfile assets). ``tools`` is
  stdlib-only and never imports this module, so the ``from .tools import`` below
  is one-directional (no cycle).
* Nothing here mutates the user's text: mentions are *left in place* in the
  prompt the user sees; only the returned context blocks are new.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

from .tools import _is_skip_path, _iter_source_files, _looks_binary

__all__ = ["project_files", "expand_mentions", "render_blocks"]

# A "source" file this large is treated as generated/data bloat and excluded
# from the completer list AND from ``@file`` injection-by-default budgeting. The
# repo_map/read_file helpers cap context similarly; this is the list-time gate.
_MAX_LIST_FILE_BYTES = 1_000_000

# Cap the number of entries a single ``@dir/`` listing injects, so mentioning a
# large tree cannot flood the preamble.
_MAX_DIR_ENTRIES = 200

# Grab a mention token: an ``@`` at a word boundary (start-of-string or after
# whitespace — never mid-token, so ``a@b.com`` is skipped) followed by a run of
# non-whitespace characters (the raw ref). Classification happens afterwards.
_MENTION_RE = re.compile(r"(?:^|(?<=\s))@(\S+)")

# Trailing punctuation stripped from a raw ref (sentence punctuation / closing
# brackets / quotes). ``/`` is deliberately NOT included so ``@dir/`` keeps its
# trailing slash marker.
_TRAILING_PUNCT = ".,;:!?)]}'\"`>"


def _cap(text: str, max_bytes: int) -> str:
    """Byte-cap ``text`` with a truncation marker (UTF-8 safe).

    Mirrors ``tools._truncate`` semantics but is local so this module has no
    dependency on that private helper's signature.
    """
    if max_bytes <= 0:
        return ""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return text
    return data[:max_bytes].decode("utf-8", errors="replace") + "\n...[truncated]"


def _rel_within(root: Path, ref: str) -> Optional[Path]:
    """Resolve workspace-relative ``ref`` under ``root``; return the resolved
    absolute path only if it stays inside ``root`` (equal or a descendant).

    Returns ``None`` for absolute paths, ``..`` escapes, or symlink targets that
    resolve outside ``root`` — the caller turns that into a refusal notice.
    """
    ref = ref.strip()
    if not ref:
        return None
    candidate = Path(ref)
    if candidate.is_absolute():
        return None
    try:
        root_res = root.expanduser().resolve()
        resolved = (root_res / candidate).resolve()
    except (OSError, RuntimeError):
        return None
    if resolved == root_res or root_res in resolved.parents:
        return resolved
    return None


def project_files(root, *, limit: int = 2000) -> list[str]:
    """Workspace-relative paths of source files under ``root``.

    Respects the SAME skip/ignore rules the ``repo_map``/``code_index`` walks use
    (no ``.git``, virtualenvs, build artifacts, minified/binary assets) and drops
    files larger than ``_MAX_LIST_FILE_BYTES`` (generated data bloat). Results are
    path-sorted and capped at ``limit``. This feeds the REPL's fuzzy ``@``-file
    completer, so it is intentionally cheap and never raises on walk errors.
    """
    root_path = Path(root).expanduser()
    try:
        root_res = root_path.resolve()
    except (OSError, RuntimeError):
        return []
    out: list[str] = []
    for p in _iter_source_files(root_path):
        if _is_skip_path(p):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > _MAX_LIST_FILE_BYTES:
            continue
        if _looks_binary(p):
            continue
        rel = os.path.relpath(str(p), str(root_res)).replace(os.sep, "/")
        out.append(rel)
        if len(out) >= limit:
            break
    out.sort()
    return out


def _classify(ref: str) -> tuple[str, str]:
    """Map a raw ref to ``(kind, value)``.

    kinds: ``diff`` (value unused), ``url`` (value=url), ``dir`` (value=relpath
    with trailing slash), ``file`` (value=relpath).
    """
    if ref == "diff":
        return "diff", ""
    if ref.startswith("url:"):
        return "url", ref[len("url:"):]
    if ref.startswith(("http://", "https://")):
        return "url", ref
    if ref.endswith("/"):
        return "dir", ref
    return "file", ref


def _file_block(root: Path, ref: str, max_file_bytes: int,
                read_file: Optional[Callable[[str], str]]) -> dict:
    """Build a ``file`` context block (or a notice block on failure)."""
    resolved = _rel_within(root, ref)
    if resolved is None:
        return {"kind": "notice", "ref": ref,
                "content": f"[skipped @{ref}: outside workspace or invalid path]"}
    if read_file is not None:
        # Caller-supplied reader (e.g. the hardened tools reader / a fake). It is
        # responsible for its own IO; we only size-cap what it returns.
        try:
            content = read_file(ref)
        except Exception as exc:  # never let an injected callable raise through
            return {"kind": "notice", "ref": ref,
                    "content": f"[could not read @{ref}: {exc}]"}
        return {"kind": "file", "ref": ref,
                "content": _cap(str(content), max_file_bytes)}
    if not resolved.exists() or not resolved.is_file():
        return {"kind": "notice", "ref": ref,
                "content": f"[not found: @{ref}]"}
    if _looks_binary(resolved):
        return {"kind": "notice", "ref": ref,
                "content": f"[skipped @{ref}: looks binary]"}
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"kind": "notice", "ref": ref,
                "content": f"[could not read @{ref}: {exc}]"}
    return {"kind": "file", "ref": ref, "content": _cap(text, max_file_bytes)}


def _dir_block(root: Path, ref: str) -> dict:
    """Build a ``dir`` listing block (or a notice block on failure)."""
    resolved = _rel_within(root, ref)
    if resolved is None:
        return {"kind": "notice", "ref": ref,
                "content": f"[skipped @{ref}: outside workspace or invalid path]"}
    if not resolved.exists() or not resolved.is_dir():
        return {"kind": "notice", "ref": ref,
                "content": f"[not a directory: @{ref}]"}
    try:
        root_res = root.expanduser().resolve()
    except (OSError, RuntimeError):
        root_res = root
    entries: list[str] = []
    truncated = False
    for p in _iter_source_files(resolved):
        if _is_skip_path(p):
            continue
        rel = os.path.relpath(str(p), str(root_res)).replace(os.sep, "/")
        entries.append(rel)
        if len(entries) >= _MAX_DIR_ENTRIES:
            truncated = True
            break
    entries.sort()
    body = "\n".join(entries) if entries else "(no source files)"
    if truncated:
        body += f"\n...[listing truncated at {_MAX_DIR_ENTRIES} entries]"
    return {"kind": "dir", "ref": ref, "content": body}


def _url_block(url: str, ref: str, max_file_bytes: int,
               web_fetch: Optional[Callable[[str], str]]) -> dict:
    """Build a ``url`` block using the injected fetcher (or a notice)."""
    url = url.strip()
    if not url:
        return {"kind": "notice", "ref": ref, "content": "[empty url]"}
    if web_fetch is None:
        return {"kind": "notice", "ref": ref,
                "content": f"[cannot fetch @{ref}: no web_fetch available]"}
    try:
        content = web_fetch(url)
    except Exception as exc:
        return {"kind": "notice", "ref": ref,
                "content": f"[could not fetch {url}: {exc}]"}
    return {"kind": "url", "ref": url, "content": _cap(str(content), max_file_bytes)}


def _diff_block(ref: str, max_file_bytes: int,
                git_diff: Optional[Callable[[], str]]) -> dict:
    """Build a ``diff`` block using the injected git_diff callable (or a notice)."""
    if git_diff is None:
        return {"kind": "notice", "ref": ref,
                "content": "[cannot inject @diff: no git_diff available]"}
    try:
        content = git_diff()
    except Exception as exc:
        return {"kind": "notice", "ref": ref,
                "content": f"[could not get diff: {exc}]"}
    return {"kind": "diff", "ref": "diff", "content": _cap(str(content), max_file_bytes)}


def expand_mentions(
    text: str,
    root,
    *,
    web_fetch: Optional[Callable[[str], str]] = None,
    git_diff: Optional[Callable[[], str]] = None,
    read_file: Optional[Callable[[str], str]] = None,
    max_file_bytes: int = 16000,
) -> tuple[str, list[dict]]:
    """Scan ``text`` for ``@``-mentions and resolve each into a context block.

    Returns ``(text, blocks)`` where ``text`` is returned UNCHANGED (the mention
    tokens stay in what the user sees) and ``blocks`` is a list of dicts shaped
    ``{"kind", "ref", "content"}``. ``content`` is already size-capped with
    truncation markers. A mention that cannot be resolved (missing file, path
    outside ``root``, no fetcher available) becomes a ``kind="notice"`` block
    rather than raising. An ``@`` that matches no pattern is left alone.

    Dependency-injected callables (all optional):
      * ``web_fetch(url) -> str`` — used for ``@url:``/bare-``@http`` mentions.
      * ``git_diff() -> str``     — used for ``@diff``.
      * ``read_file(relpath) -> str`` — override the default on-disk file reader.
    """
    if not isinstance(text, str) or "@" not in text:
        return text, []
    root_path = Path(root)
    blocks: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for m in _MENTION_RE.finditer(text):
        raw = m.group(1)
        # Strip trailing sentence punctuation (but keep a trailing ``/``).
        ref = raw.rstrip(_TRAILING_PUNCT)
        if not ref:
            continue
        kind, value = _classify(ref)
        dedup_key = (kind, value or ref)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        if kind == "diff":
            blocks.append(_diff_block(ref, max_file_bytes, git_diff))
        elif kind == "url":
            blocks.append(_url_block(value, ref, max_file_bytes, web_fetch))
        elif kind == "dir":
            blocks.append(_dir_block(root_path, ref.rstrip("/") or "."))
        else:  # file
            blocks.append(_file_block(root_path, ref, max_file_bytes, read_file))
    return text, blocks


def render_blocks(blocks: list[dict]) -> str:
    """Format context ``blocks`` into one injectable string.

    Produces a clearly delimited preamble the caller can prepend as a user-role
    message, e.g.::

        # Attached context
        ## @file: path/to/x.py
        ```
        <contents>
        ```

    Returns an empty string for no blocks.
    """
    if not blocks:
        return ""
    parts: list[str] = ["# Attached context"]
    for b in blocks:
        kind = b.get("kind", "notice")
        ref = b.get("ref", "")
        content = b.get("content", "")
        if kind == "notice":
            parts.append(f"## notice: {ref}\n{content}")
        else:
            parts.append(f"## @{kind}: {ref}\n```\n{content}\n```")
    return "\n\n".join(parts)
