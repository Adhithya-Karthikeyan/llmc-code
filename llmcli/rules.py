"""Project conventions/rules auto-loading.

Local projects often ship a human-authored "rules" file describing the
conventions the assistant should follow — the same idea as Claude Code's
``CLAUDE.md``, aider's ``CONVENTIONS.md``, or the emerging ``AGENTS.md``
standard. This module locates that file at the project ROOT and turns it into a
compact system-prompt block a later wave can append every session.

Design notes (kept in line with ``llmcli/tools.py``):
- Reads are byte-capped (``max_bytes``) so a large rules file can't blow up the
  prompt on a small local model — same defensive posture as ``_read_file``.
- UTF-8 with ``errors="ignore"`` so a stray byte never crashes a session.
- Any read/OS error is swallowed and treated as "no rules" (return ``""``); a
  missing convention file must never break the CLI.
- Everything here is pure/read-only: no writes, no side effects, no network.
"""

from __future__ import annotations

from pathlib import Path

# Recognised rules filenames, in PRIORITY order (first match wins). AGENTS.md
# leads because it is the cross-tool convention; the rest are project-specific
# fallbacks people may already have from other assistants.
RULES_FILENAMES: tuple[str, ...] = (
    "AGENTS.md",
    "LLMCLI.md",
    ".llmclirules",
    "CONVENTIONS.md",
)

# Marker appended when the rules file exceeds the byte cap, so both the model
# and a human can tell the content was clipped (mirrors tools.py "[truncated]").
_TRUNCATION_MARKER = "\n\n…[rules truncated]"


def find_rules_file(root) -> Path | None:
    """Return the first existing rules file at the project ``root``.

    Searches ``root`` ONLY (not recursively), trying each name in
    ``RULES_FILENAMES`` order and returning the first regular file that exists.
    Returns ``None`` when none are present or ``root`` is not a usable directory.
    """
    try:
        base = Path(root)
    except TypeError:
        return None
    for name in RULES_FILENAMES:
        candidate = base / name
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def load_rules(root, *, max_bytes: int = 8000) -> str:
    """Read the project rules file as text, stripped and byte-capped.

    Reads the file found by :func:`find_rules_file` as UTF-8 (ignoring undecodable
    bytes) and strips surrounding whitespace. If the content exceeds ``max_bytes``
    it is clipped and a clear ``…[rules truncated]`` marker is appended so the
    prompt stays bounded. Returns ``""`` when there is no file, the file is empty,
    or it cannot be read.
    """
    path = find_rules_file(root)
    if path is None:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    text = text.strip()
    if not text:
        return ""
    if max_bytes is not None and max_bytes > 0:
        encoded = text.encode("utf-8")
        if len(encoded) > max_bytes:
            clipped = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()
            text = clipped + _TRUNCATION_MARKER
    return text


def rules_prompt_block(root) -> str:
    """Return a system-prompt section for the project rules, or ``""`` if none.

    The block names the source filename and frames the content as user-authored
    project conventions the model must follow, ready to append to the system
    prompt. Returns ``""`` when no (non-empty, readable) rules file exists.
    """
    path = find_rules_file(root)
    if path is None:
        return ""
    contents = load_rules(root)
    if not contents:
        return ""
    return (
        f"# Project rules (from {path.name})\n"
        "The following are USER-AUTHORED conventions for THIS project. Treat them "
        "as binding: follow them in every response, edit, and command unless the "
        "user explicitly overrides them in this session.\n\n"
        f"{contents}"
    )


def default_template() -> str:
    """Return a terse starter rules template a ``/init`` command can write."""
    return """\
# Project rules

Conventions for this project. Delete what you don't need; keep it short.

## Project overview
- What this project is and its main goal (1-2 lines).

## Conventions
- Language / framework and version.
- Formatting, naming, and file-layout expectations.
- How to run, build, and test.

## Do
- Prefer the smallest change that solves the task.
- Match existing patterns already in the codebase.

## Don't
- Don't add dependencies without a clear reason.
- Don't refactor unrelated code while making a change.
"""
