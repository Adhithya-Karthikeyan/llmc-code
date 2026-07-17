"""User-configurable lifecycle hooks — deterministic guardrails around tool use.

This module lets the operator wire small shell scripts into the agent loop
*independently of the model*, so guardrails are enforced deterministically
rather than relying on the LLM to behave. A later wave calls these functions
from the agent loop; this module is intentionally self-contained and does not
import or edit any other part of the package.

Config source
-------------
``~/.llm-cli/hooks.json`` (same location convention as ``mcp.json``). It is
OPT-IN: a missing or malformed file yields ``{}`` and the CLI behaves exactly
as before. Shape::

    {
      "PreToolUse":  [{"match": "write_file|edit_file", "command": "<shell>", "timeout": 10}],
      "PostToolUse": [{"match": "",                      "command": "<shell>", "timeout": 10}],
      "Stop":        [{"command": "<shell>"}]
    }

* ``match`` is a regex matched (``re.search``) against the tool name. Omitted
  or empty means "all tools". A hook whose ``match`` is not a compilable regex
  is dropped at load time.
* ``command`` is required and must be a non-empty string; an entry without one
  is dropped.
* ``timeout`` is per-hook seconds (default :data:`DEFAULT_HOOK_TIMEOUT`).

Contracts
---------
* PreToolUse is a *veto gate*: if ANY matching hook exits non-zero, the tool
  call is blocked. A timeout is treated as a block too (fail-closed — a
  guardrail that cannot finish must not silently allow the action).
* PostToolUse is fire-and-forget (e.g. auto-lint): a non-zero exit is recorded
  in the returned info dict but never blocks.
* Stop runs end-of-turn hooks; failures are recorded, never raised.

Security note
-------------
The MODEL-controlled tool arguments are passed to your hook as DATA only — via
the ``LLMC_TOOL_NAME``/``LLMC_TOOL_ARGS`` environment variables and on stdin —
never as part of the command line. Your hook ``command`` string comes solely
from your own ``hooks.json``. Treat ``$LLMC_TOOL_ARGS`` as UNTRUSTED input:
never ``eval``/``source`` it or splice it into a shell command, or you would
re-introduce an injection the framework deliberately prevents.

Every entry point wraps its own errors — nothing here raises to the caller.
The subprocess conventions (``/bin/bash`` preferred with ``/bin/sh`` fallback,
``shell=False`` argv, own process group, hard timeout + group kill, captured
and byte-capped output) mirror ``tools.py``'s ``_run_bash``.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time as _time
from pathlib import Path
from typing import Any

# Same directory convention as mcp.py's MCP_CONFIG_PATH.
HOOKS_CONFIG_PATH = Path.home() / ".llm-cli" / "hooks.json"

# The lifecycle events a hook can bind to. Anything else in the config is
# ignored so an unknown/typo'd key can never accidentally fire.
HOOK_EVENTS = ("PreToolUse", "PostToolUse", "Stop")

# Per-hook wall-clock budget when the entry omits "timeout". Kept small: a hook
# is a guardrail, not a long task.
DEFAULT_HOOK_TIMEOUT = 10

# Byte ceiling on each captured stream (stdout/stderr) per hook. Mirrors the
# spirit of tools.py's output caps so a chatty hook can't flood the caller /
# model context. Applied to the text we hand back as the block reason / info.
_MAX_HOOK_OUTPUT = 8_000

# Prefer /bin/bash (matches the recent run_bash fix — /bin/sh on macOS is bash
# in POSIX mode and mangles `echo -n`), fall back to /bin/sh on minimal images.
_SHELL_BIN = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"


def _clip(text: str, limit: int = _MAX_HOOK_OUTPUT) -> str:
    """Byte-cap a captured stream so a runaway hook can't flood the caller."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[...truncated at {limit} chars]"


def _coerce_timeout(value: Any) -> int:
    """Return a positive int timeout, coercing bad/bool values to the default.

    ``bool`` is a subclass of ``int`` so it is excluded explicitly (mirrors the
    run_bash timeout-coercion guard).
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return DEFAULT_HOOK_TIMEOUT
    return value


def load_hooks(path: Path | None = None) -> dict[str, list[dict]]:
    """Load and validate ``~/.llm-cli/hooks.json`` -> normalized hook config.

    Returns a dict keyed by the events in :data:`HOOK_EVENTS`, each mapping to a
    list of ``{"match": str, "command": str, "timeout": int}`` entries. A
    missing file, unreadable file, bad JSON, or wrong top-level shape all yield
    ``{}`` (hooks are opt-in). Malformed individual entries are dropped:

    * entry is not a dict, or has no non-empty string ``command`` -> dropped
    * ``match`` is present but not a compilable regex -> dropped
    * ``match`` missing/empty -> normalized to ``""`` (matches all tools)
    """
    p = HOOKS_CONFIG_PATH if path is None else Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}

    out: dict[str, list[dict]] = {}
    for event in HOOK_EVENTS:
        raw = data.get(event)
        if not isinstance(raw, list):
            continue
        entries: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            command = item.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            match = item.get("match")
            if match is None:
                match = ""
            if not isinstance(match, str):
                continue
            # A guardrail with an uncompilable regex is a config error; drop it
            # rather than crash (or silently treat it as match-all) later.
            if match:
                try:
                    re.compile(match)
                except re.error:
                    continue
            entries.append(
                {
                    "match": match,
                    "command": command,
                    "timeout": _coerce_timeout(item.get("timeout")),
                }
            )
        if entries:
            out[event] = entries
    return out


def _matches(match: str, tool_name: str) -> bool:
    """True if ``tool_name`` matches the entry's regex (empty == match all)."""
    if not match:
        return True
    try:
        return re.search(match, tool_name or "") is not None
    except re.error:
        # Should not happen (validated at load) but never raise from the loop.
        return False


def _run_hook(
    command: str,
    timeout: int,
    *,
    cwd: str | os.PathLike | None,
    env: dict | None,
    extra_env: dict[str, str],
    stdin_text: str,
) -> dict:
    """Run one hook command; never raises. Returns a captured-result dict.

    Keys: ``ok`` (started + exit 0), ``exit_code`` (int|None), ``stdout``,
    ``stderr``, ``timed_out`` (bool), ``error`` (str, on launch failure).

    Uses ``shell=False`` with an argv list and its own process group so a
    timeout can SIGKILL the whole group (grandchildren included), exactly like
    run_bash.
    """
    # Hooks are the operator's own scripts, so (unlike the model-driven
    # run_bash shell) they inherit the real environment by default; callers /
    # tests can override via `env`. The LLMC_* context vars are always layered
    # on top so a hook can read the tool name/args.
    base_env = dict(os.environ if env is None else env)
    base_env.update(extra_env)

    argv = [_SHELL_BIN, "-c", command]
    try:
        proc = subprocess.Popen(
            argv,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
            env=base_env,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "error": f"failed to launch hook: {exc}",
        }

    timed_out = False
    try:
        stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        # SIGKILL the whole process group so detached grandchildren die too.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            stdout, stderr = "", ""
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "exit_code": proc.returncode,
            "stdout": "",
            "stderr": f"hook communication error: {exc}",
            "timed_out": False,
            "error": str(exc),
        }

    exit_code = proc.returncode
    return {
        "ok": (not timed_out) and exit_code == 0,
        "exit_code": exit_code,
        "stdout": _clip(stdout or ""),
        "stderr": _clip(stderr or ""),
        "timed_out": timed_out,
        "error": "",
    }


def _tool_env(tool_name: str, args: Any, result: Any = None) -> dict[str, str]:
    """Build the LLMC_* env vars a hook receives (never raises)."""
    try:
        args_json = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        args_json = str(args)
    env = {
        "LLMC_TOOL_NAME": str(tool_name),
        "LLMC_TOOL_ARGS": _clip(args_json),
    }
    if result is not None:
        try:
            result_json = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            result_json = str(result)
        env["LLMC_TOOL_RESULT"] = _clip(result_json)
    return env


def run_pre_tool(
    hooks: dict,
    tool_name: str,
    args: Any,
    *,
    cwd: str | os.PathLike | None = None,
    env: dict | None = None,
) -> dict:
    """Run matching PreToolUse hooks as a veto gate before a tool executes.

    The tool name and JSON-encoded args are exposed to each hook via the
    ``LLMC_TOOL_NAME`` / ``LLMC_TOOL_ARGS`` env vars and also piped on stdin.

    Returns:
    * ``{"decision": "allow"}`` if no PreToolUse hook matches or all matching
      hooks exit 0.
    * ``{"decision": "block", "reason": <str>}`` on the FIRST matching hook that
      exits non-zero (reason = its stderr, else stdout, else a generic note).
    * A timeout is a block too (fail-closed): a guardrail that cannot finish
      must not silently allow the action.

    Never raises — any internal error is reported as an allow decision with an
    ``error`` note so a broken hook config cannot wedge the agent loop, EXCEPT a
    hook that runs and vetoes, which blocks as designed.
    """
    entries = hooks.get("PreToolUse") if isinstance(hooks, dict) else None
    if not isinstance(entries, list) or not entries:
        return {"decision": "allow"}

    extra_env = _tool_env(tool_name, args)
    stdin_text = extra_env["LLMC_TOOL_ARGS"]

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _matches(entry.get("match", ""), tool_name):
            continue
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            continue
        res = _run_hook(
            command,
            _coerce_timeout(entry.get("timeout")),
            cwd=cwd,
            env=env,
            extra_env=extra_env,
            stdin_text=stdin_text,
        )
        if res.get("timed_out"):
            reason = (
                res.get("stderr")
                or res.get("stdout")
                or f"PreToolUse hook timed out after "
                f"{_coerce_timeout(entry.get('timeout'))}s"
            )
            return {"decision": "block", "reason": reason.strip()}
        if res.get("error"):
            # The hook could not even launch. Fail-closed like a timeout: a
            # guardrail that never ran must not silently allow the action.
            return {"decision": "block", "reason": res["error"].strip()}
        if not res.get("ok"):
            reason = (
                res.get("stderr")
                or res.get("stdout")
                or f"PreToolUse hook denied {tool_name} "
                f"(exit {res.get('exit_code')})"
            )
            return {"decision": "block", "reason": reason.strip()}

    return {"decision": "allow"}


def run_post_tool(
    hooks: dict,
    tool_name: str,
    args: Any,
    result: Any,
    *,
    cwd: str | os.PathLike | None = None,
    env: dict | None = None,
) -> dict:
    """Fire-and-forget PostToolUse hooks (e.g. auto-lint) after a tool ran.

    Never blocks and never raises. A non-zero exit / timeout is recorded in the
    returned info dict but does not affect control flow. Returns::

        {"ran": <int>, "results": [{"exit_code","timed_out","stdout","stderr"}...]}
    """
    info: dict[str, Any] = {"ran": 0, "results": []}
    entries = hooks.get("PostToolUse") if isinstance(hooks, dict) else None
    if not isinstance(entries, list) or not entries:
        return info

    extra_env = _tool_env(tool_name, args, result)
    stdin_text = extra_env["LLMC_TOOL_ARGS"]

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _matches(entry.get("match", ""), tool_name):
            continue
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            continue
        res = _run_hook(
            command,
            _coerce_timeout(entry.get("timeout")),
            cwd=cwd,
            env=env,
            extra_env=extra_env,
            stdin_text=stdin_text,
        )
        info["ran"] += 1
        info["results"].append(
            {
                "exit_code": res.get("exit_code"),
                "timed_out": res.get("timed_out", False),
                "stdout": res.get("stdout", ""),
                "stderr": res.get("stderr", ""),
                "error": res.get("error", ""),
            }
        )
    return info


def run_stop(
    hooks: dict,
    *,
    cwd: str | os.PathLike | None = None,
    env: dict | None = None,
) -> dict:
    """Run end-of-turn Stop hooks. Fire-and-forget; never blocks or raises.

    Returns the same shape as :func:`run_post_tool`. Stop hooks are not tied to
    a tool, so no LLMC_TOOL_* env vars are set.
    """
    info: dict[str, Any] = {"ran": 0, "results": []}
    entries = hooks.get("Stop") if isinstance(hooks, dict) else None
    if not isinstance(entries, list) or not entries:
        return info

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        command = entry.get("command")
        if not isinstance(command, str) or not command:
            continue
        res = _run_hook(
            command,
            _coerce_timeout(entry.get("timeout")),
            cwd=cwd,
            env=env,
            extra_env={},
            stdin_text="",
        )
        info["ran"] += 1
        info["results"].append(
            {
                "exit_code": res.get("exit_code"),
                "timed_out": res.get("timed_out", False),
                "stdout": res.get("stdout", ""),
                "stderr": res.get("stderr", ""),
                "error": res.get("error", ""),
            }
        )
    return info
