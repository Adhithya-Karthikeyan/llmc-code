"""Deterministic, SAFE self-healing for failed tool calls.

When a tool call FAILS, the agent's dispatch loop can consult this module to
see whether the failure is one of a small set of *mechanically correctable*
mistakes — and, if so, retry the tool with corrected arguments instead of just
handing the raw error back to the model. This turns a common round-trip
(model reads the error, apologises, guesses a new path) into a single silent,
bounded retry.

Scope is intentionally narrow and NON-DESTRUCTIVE:

* We only ever correct a *path* argument. We never invent file CONTENT, never
  flip ``overwrite`` on, never delete, and never guess the target of an edit.
* We only act on failures whose fix is unambiguous: an absolute path that maps
  to exactly ONE workspace file by basename, or a "file not found" whose
  basename matches exactly ONE known project file. Anything ambiguous (two
  candidate files) or unrecoverable (bad content, parse error, permission /
  hook block, ``run_bash`` failure) returns ``None`` — the model handles it.

The module is PURE and dependency-injected: the caller passes the workspace
``root`` and (for testability) the list of workspace-relative project files.
When ``project_files`` is omitted we compute it lazily via
``mentions.project_files`` (imported inside the function to avoid an import
cycle). ``remediate`` never raises.

Error strings matched here mirror the real ones produced in ``llmcli.tools``
(``_read_file`` / ``_write_file`` / ``_edit_file`` / ``_glob`` / ``_grep`` /
``_repo_map``); keep them in sync if those messages change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["remediate"]

# Tools whose single path argument lives under the "path" key.
_PATH_TOOLS = frozenset({"read_file", "write_file", "edit_file"})

# Tools that take a directory root; the model may put it under "path" or "root".
_ROOTABLE_TOOLS = frozenset({"glob", "grep", "repo_map"})

# Tools eligible for the "file not found" basename correction (read/edit only —
# write_file "not found" is not a thing, and correcting a write target is risky).
_NOT_FOUND_TOOLS = frozenset({"read_file", "edit_file"})


def _basename(path: str) -> str:
    """Last path component, tolerant of both ``/`` and ``\\`` separators.

    Project-file entries are always ``/``-joined (see ``mentions.project_files``)
    while a failed arg may be an OS-native absolute path, so normalise both.
    """
    norm = path.replace("\\", "/").rstrip("/")
    return norm.rsplit("/", 1)[-1]


def _resolve(path) -> Optional[Path]:
    """Resolve ``path`` to an absolute Path, or ``None`` on error."""
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _rel_if_inside(abs_path: Path, root_res: Path) -> Optional[str]:
    """If ``abs_path`` resolves to a file strictly inside ``root_res``, return
    the ``/``-joined workspace-relative path; else ``None``.

    The root itself is not a valid file target, so an exact match returns None.
    """
    if abs_path == root_res:
        return None
    if root_res in abs_path.parents:
        return os.path.relpath(str(abs_path), str(root_res)).replace(os.sep, "/")
    return None


def _unique_basename_match(value: str, project_files: list[str]) -> Optional[str]:
    """Return the sole project file whose basename equals ``value``'s basename,
    or ``None`` if there are zero or multiple matches (ambiguous — never guess).
    """
    base = _basename(value)
    if not base:
        return None
    matches = [f for f in project_files if _basename(f) == base]
    if len(matches) == 1:
        return matches[0]
    return None


def _get_project_files(project_files: Optional[list[str]], root: str) -> list[str]:
    """Return the caller-supplied file list, or compute it lazily.

    Imported inside to avoid a module-level cycle (mentions imports tools; this
    module is consulted from the agent loop). Never raises.
    """
    if project_files is not None:
        return project_files
    try:
        from .mentions import project_files as _pf
        return list(_pf(root))
    except Exception:
        return []


def _is_confinement_error(error: str) -> bool:
    """True for a workspace-confinement refusal (read/write/edit/glob/grep/map).

    Matches the "outside the workspace root" family. The symlink-directory
    refusal ("Refusing to write through a symlinked directory leaving the
    workspace.") deliberately does NOT match — it contains no "outside" — so we
    never try to auto-redirect a TOCTOU-guarded write.
    """
    low = error.lower()
    if "outside the workspace root" in low:
        return True
    return "refusing to" in low and "outside" in low


def _is_not_found_error(error: str) -> bool:
    """True for a genuine "file not found" / "file does not exist" failure.

    Careful: edit_file's "'old' string not found in file" also contains the
    words "not found" but not the contiguous "file not found", so it is excluded
    (we must never remediate an edit whose anchor text is missing).
    """
    low = error.lower()
    return "file not found" in low or "file does not exist" in low or "does not exist" in low


def _path_arg(tool_name: str, args: dict) -> tuple[Optional[str], object]:
    """Return ``(key, value)`` for the path-like argument of ``tool_name``.

    ``key`` is ``None`` for tools that expose no correctable path argument.
    """
    if tool_name in _PATH_TOOLS:
        return "path", args.get("path")
    if tool_name in _ROOTABLE_TOOLS:
        p = args.get("path")
        if isinstance(p, str) and p:
            return "path", p
        r = args.get("root")
        if isinstance(r, str) and r:
            return "root", r
        return "path", p
    return None, None


def _with_key(args: dict, key: str, value: str) -> dict:
    """Copy ``args``, replacing only ``key`` with ``value`` (args else identical)."""
    new_args = dict(args)
    new_args[key] = value
    return new_args


def _fix_confinement(
    tool_name: str, args: dict, root: str, project_files: Optional[list[str]]
) -> Optional[tuple[dict, str]]:
    """Rule 1: correct a path that was refused for leaving the workspace."""
    key, value = _path_arg(tool_name, args)
    if key is None or not isinstance(value, str) or not value:
        return None

    root_res = _resolve(root)
    if root_res is None:
        return None

    # (a) The path, resolved, actually stays inside root -> use the relative form.
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        abs_res = _resolve(value)
        if abs_res is not None:
            rel = _rel_if_inside(abs_res, root_res)
            if rel is not None and rel != value:
                return _with_key(args, key, rel), (
                    f"corrected absolute path to workspace-relative '{rel}'"
                )

    # (b) Exactly one project file shares the failed path's basename.
    pf = _get_project_files(project_files, root)
    match = _unique_basename_match(value, pf)
    if match is not None and match != value:
        return _with_key(args, key, match), (
            f"corrected absolute path to workspace-relative '{match}'"
        )

    # (c) No safe correction.
    return None


def _fix_not_found(
    tool_name: str, args: dict, root: str, project_files: Optional[list[str]]
) -> Optional[tuple[dict, str]]:
    """Rule 2: correct a read/edit "file not found" via a unique basename match."""
    value = args.get("path")
    if not isinstance(value, str) or not value:
        return None
    pf = _get_project_files(project_files, root)
    match = _unique_basename_match(value, pf)
    if match is None or match == value:
        return None
    return _with_key(args, "path", match), (
        f"corrected to workspace-relative '{match}' matching basename "
        f"'{_basename(value)}'"
    )


def remediate(
    tool_name: str,
    args: dict,
    result: dict,
    *,
    root: str,
    project_files: Optional[list[str]] = None,
) -> Optional[tuple[dict, str]]:
    """Diagnose a FAILED tool ``result`` and return corrected retry arguments.

    Parameters
    ----------
    tool_name:
        The tool that was invoked (e.g. ``"read_file"``).
    args:
        The arguments that produced ``result``. Never mutated.
    result:
        The tool's return dict, shaped ``{"ok": bool, "error"?: str, ...}``.
    root:
        The workspace root directory (absolute path string).
    project_files:
        Optional pre-computed list of workspace-relative source files. When
        ``None`` it is computed lazily via ``mentions.project_files(root)``.
        Tests pass an explicit list so no real repo is required.

    Returns
    -------
    ``(new_args, explanation)`` to retry the SAME tool with, where ``new_args``
    is a copy of ``args`` with only a corrected path, and ``explanation`` is a
    short human string. Returns ``None`` when there is no SAFE automatic fix.

    This function is total: any unexpected input (malformed ``result`` / ``args``,
    resolution errors) yields ``None`` rather than raising.
    """
    try:
        if not isinstance(result, dict) or not isinstance(args, dict):
            return None
        # A success needs no remediation.
        if result.get("ok"):
            return None
        error = result.get("error")
        if not isinstance(error, str) or not error:
            return None
        if not isinstance(tool_name, str):
            return None

        # Rule 1: workspace-confinement refusal (read/write/edit/glob/grep/map).
        if _is_confinement_error(error) and (
            tool_name in _PATH_TOOLS or tool_name in _ROOTABLE_TOOLS
        ):
            return _fix_confinement(tool_name, args, root, project_files)

        # Rule 2: read/edit "file not found" via a unique basename match.
        if tool_name in _NOT_FOUND_TOOLS and _is_not_found_error(error):
            return _fix_not_found(tool_name, args, root, project_files)

        # Everything else (already-exists, text-not-found, parse/permission/hook
        # errors, run_bash failures, unknown tools) is left to the model.
        return None
    except Exception:
        # Remediation must never crash the dispatch loop.
        return None
