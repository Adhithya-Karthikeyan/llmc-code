"""Pure git helpers via subprocess — safe, typed, and degrade-gracefully.

This module is a THIN, self-contained wrapper around the ``git`` CLI. Later
waves build /diff, /commit, auto-commit, and a dirty-tree warning on top of it.
It intentionally does NOT generate commit messages — callers pass one in.

Subprocess discipline (mirrors tools.py ``_run_bash``):
  - ALWAYS an argv list, NEVER ``shell=True`` — no shell metacharacter risk.
  - A short per-call timeout so a wedged git cannot hang the caller.
  - stdout/stderr captured; output is byte-capped where it can be large.

Graceful degradation is the core contract: NOTHING in this module raises to the
caller. If git is missing, the cwd is not a repo, git errors out, or it times
out, the boolean helpers return ``False``, the string helpers return ``""`` (or
``None`` where documented), and ``commit_all`` returns ``{"ok": False,
"error": ...}``. Every git invocation funnels through ``_run_git`` which
converts every failure mode into a structured result instead of an exception.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

# Short timeout: local git operations are near-instant; anything slower is a
# wedged process (lock contention, network remote hang) we refuse to wait on.
_GIT_TIMEOUT = 15

# Byte ceilings so a huge diff / status cannot balloon caller context. Diffs are
# the big one; status is capped smaller. Both append an explicit marker on trim
# so the caller (and the model) know the text is incomplete.
_DIFF_MAX_BYTES = 20_000
_STATUS_MAX_BYTES = 4_000
_TRUNC_MARKER = "\n... [truncated]"


def _run_git(root: str, argv: list[str]) -> dict:
    """Run ``git <argv>`` in ``root`` and return a structured result.

    Returns a dict with keys:
      - ``ok``:   bool — True only when git ran AND exited 0.
      - ``code``: int  — the git exit code (or -1 when git never ran).
      - ``out``:  str  — captured stdout (empty on failure).
      - ``err``:  str  — captured stderr, or a human error when git never ran.

    NEVER raises: a missing binary, timeout, or OSError all map to ``ok=False``.
    """
    if shutil.which("git") is None:
        return {"ok": False, "code": -1, "out": "", "err": "git is not installed"}
    # Build a minimal env like _run_bash: git is invoked programmatically here,
    # so drop the model/user shell's secrets. Infra vars only. We also pin
    # GIT_TERMINAL_PROMPT=0 so git can NEVER block on an interactive credential
    # or passphrase prompt (which would defeat the timeout).
    env = {
        k: os.environ[k]
        for k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE")
        if k in os.environ
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = subprocess.run(
            ["git", *argv],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "code": -1, "out": "", "err": f"git timed out after {_GIT_TIMEOUT}s"}
    except OSError as exc:
        return {"ok": False, "code": -1, "out": "", "err": f"failed to run git: {exc}"}
    return {
        "ok": proc.returncode == 0,
        "code": proc.returncode,
        "out": proc.stdout or "",
        "err": (proc.stderr or "").strip(),
    }


def _cap(text: str, limit: int) -> str:
    """Byte-cap ``text`` to ``limit`` bytes, appending a marker if trimmed."""
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= limit:
        return text
    # Cut on a byte boundary then drop any partial trailing multibyte char.
    return encoded[:limit].decode("utf-8", "ignore") + _TRUNC_MARKER


def git_available() -> bool:
    """True when a ``git`` executable is on PATH."""
    return shutil.which("git") is not None


def is_repo(root: str) -> bool:
    """True when ``root`` is inside a git working tree.

    False when git is absent, ``root`` is not a repo, or anything errors.
    """
    res = _run_git(root, ["rev-parse", "--is-inside-work-tree"])
    return res["ok"] and res["out"].strip() == "true"


def is_dirty(root: str) -> bool:
    """True when the working tree has uncommitted or untracked changes.

    False when clean, not a repo, or on any error (fail safe: a false negative
    is a warning we skip, never a crash).
    """
    if not is_repo(root):
        return False
    res = _run_git(root, ["status", "--porcelain"])
    if not res["ok"]:
        return False
    return bool(res["out"].strip())


def short_status(root: str) -> str:
    """Return the porcelain status summary, byte-capped.

    Empty string when clean, not a repo, or on error.
    """
    if not is_repo(root):
        return ""
    res = _run_git(root, ["status", "--porcelain"])
    if not res["ok"]:
        return ""
    return _cap(res["out"].strip(), _STATUS_MAX_BYTES)


def current_branch(root: str) -> Optional[str]:
    """Return the current branch name, or None.

    None when not a repo, on error, or in a detached-HEAD state (where git
    reports the branch as ``HEAD``).
    """
    if not is_repo(root):
        return None
    res = _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    if not res["ok"]:
        return None
    name = res["out"].strip()
    if not name or name == "HEAD":
        return None
    return name


def diff(root: str, path: Optional[str] = None, staged: bool = False) -> str:
    """Return a unified diff as text, byte-capped to ~20KB.

    - ``path``:   restrict the diff to a single file/dir (optional).
    - ``staged``: when True, diff the index vs HEAD (``--cached``); otherwise
      diff the working tree vs the index.

    Empty string when not a repo, no changes, or on any error.
    """
    if not is_repo(root):
        return ""
    argv = ["diff"]
    if staged:
        argv.append("--cached")
    if path:
        # ``--`` terminates options so a path that looks like a flag is safe.
        argv.extend(["--", path])
    res = _run_git(root, argv)
    if not res["ok"]:
        return ""
    return _cap(res["out"], _DIFF_MAX_BYTES)


def commit_all(root: str, message: str) -> dict:
    """Stage all tracked+untracked changes and commit with ``message``.

    Returns ``{"ok": True, "commit_hash": <sha>}`` on success, or
    ``{"ok": False, "error": <reason>}`` when git is absent, ``root`` is not a
    repo, ``message`` is empty, there is nothing to commit, or git errors.
    Never raises.
    """
    if not isinstance(message, str) or not message.strip():
        return {"ok": False, "error": "commit message is required"}
    if not git_available():
        return {"ok": False, "error": "git is not installed"}
    if not is_repo(root):
        return {"ok": False, "error": "not a git repository"}
    if not is_dirty(root):
        return {"ok": False, "error": "nothing to commit"}

    # Stage everything (new, modified, deleted). `git add -A` covers untracked
    # + tracked + removals across the whole tree.
    staged = _run_git(root, ["add", "-A"])
    if not staged["ok"]:
        return {"ok": False, "error": staged["err"] or "git add failed"}

    committed = _run_git(root, ["commit", "-m", message])
    if not committed["ok"]:
        # e.g. a pre-commit hook rejected it, or a race emptied the index.
        return {"ok": False, "error": committed["err"] or committed["out"].strip() or "git commit failed"}

    head = _run_git(root, ["rev-parse", "HEAD"])
    if not head["ok"]:
        # The commit landed but we could not read the hash — still a success.
        return {"ok": True, "commit_hash": ""}
    return {"ok": True, "commit_hash": head["out"].strip()}


def last_commit(root: str) -> Optional[dict]:
    """Return ``{"hash": <sha>, "subject": <first line>}`` for HEAD, or None.

    None when not a repo, when the repo has no commits yet, or on any error.
    """
    if not is_repo(root):
        return None
    # %H = full hash, %s = subject; NUL-separated so a subject can hold anything.
    res = _run_git(root, ["log", "-1", "--pretty=format:%H%x00%s"])
    if not res["ok"] or not res["out"].strip():
        return None
    parts = res["out"].split("\x00", 1)
    commit_hash = parts[0].strip()
    subject = parts[1].strip() if len(parts) > 1 else ""
    if not commit_hash:
        return None
    return {"hash": commit_hash, "subject": subject}
