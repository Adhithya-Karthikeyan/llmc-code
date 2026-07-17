"""Validate the tools on a temp dir + their result shapes."""

from __future__ import annotations

from llmcli.tools import (
    FULL,
    READ_ONLY,
    REGISTRY,
    _MAX_CAPTURE_BYTES,
    _MAX_OUTPUT,
    get_tool,
    openai_schema,
    tool_subset,
)


def _call(name: str, args: dict) -> dict:
    return get_tool(name).fn(args)


def test_registry_has_all_tools():
    for name in FULL:
        assert name in REGISTRY
    assert set(FULL) == {
        "read_file", "write_file", "edit_file", "run_bash", "glob", "grep",
        "repo_map", "web_fetch",
    }
    assert READ_ONLY == ["read_file", "glob", "grep", "repo_map", "web_fetch"]


def test_confirmation_flags():
    for name in ("write_file", "edit_file", "run_bash"):
        assert get_tool(name).requires_confirmation is True
    for name in ("read_file", "glob", "grep", "repo_map", "web_fetch"):
        assert get_tool(name).requires_confirmation is False


def test_write_then_read(tmp_workspace):
    res = _call("write_file", {"path": "a.txt", "content": "hello\n"})
    assert res["ok"] is True
    assert res["result"]["bytes_written"] == 6

    res = _call("read_file", {"path": "a.txt"})
    assert res["ok"] is True
    assert res["result"] == "hello\n"


def test_read_missing_file(tmp_workspace):
    res = _call("read_file", {"path": "nope.txt"})
    assert res["ok"] is False
    assert "error" in res


def test_edit_file_unique(tmp_workspace):
    _call("write_file", {"path": "b.txt", "content": "foo bar baz"})
    res = _call("edit_file", {"path": "b.txt", "old": "bar", "new": "QUX"})
    assert res["ok"] is True
    assert res["result"]["replacements"] == 1
    assert _call("read_file", {"path": "b.txt"})["result"] == "foo QUX baz"


def test_edit_file_non_unique_errors(tmp_workspace):
    _call("write_file", {"path": "c.txt", "content": "x x x"})
    res = _call("edit_file", {"path": "c.txt", "old": "x", "new": "y"})
    assert res["ok"] is False
    assert "unique" in res["error"].lower()


def test_edit_file_absent_old_errors(tmp_workspace):
    _call("write_file", {"path": "d.txt", "content": "abc"})
    res = _call("edit_file", {"path": "d.txt", "old": "zzz", "new": "y"})
    assert res["ok"] is False


def test_glob(tmp_workspace):
    _call("write_file", {"path": "pkg/one.py", "content": "1"})
    _call("write_file", {"path": "pkg/two.py", "content": "2"})
    _call("write_file", {"path": "pkg/readme.md", "content": "m"})
    res = _call("glob", {"pattern": "**/*.py"})
    assert res["ok"] is True
    matches = res["result"]["matches"]
    assert any(m.endswith("one.py") for m in matches)
    assert any(m.endswith("two.py") for m in matches)
    assert not any(m.endswith(".md") for m in matches)


def test_grep(tmp_workspace):
    _call("write_file", {"path": "src.txt", "content": "alpha\nbeta TARGET\ngamma\n"})
    res = _call("grep", {"pattern": "TARGET", "path": "."})
    assert res["ok"] is True
    matches = res["result"]["matches"]
    assert len(matches) == 1
    assert matches[0]["line_number"] == 2
    assert "TARGET" in matches[0]["line"]


def test_grep_line_preview_capped(tmp_workspace):
    """PERF-2: each match's line preview is capped (~160 chars), not 500."""
    from llmcli.tools import _MAX_GREP_LINE

    _call("write_file", {"path": "long.txt", "content": "NEEDLE " + "x" * 1000 + "\n"})
    res = _call("grep", {"pattern": "NEEDLE"})
    assert res["ok"] is True
    line = res["result"]["matches"][0]["line"]
    assert len(line) == _MAX_GREP_LINE


def test_grep_payload_byte_capped(tmp_workspace):
    """PERF-2: the grep payload obeys the _MAX_OUTPUT byte budget (it bypasses
    _truncate), trimming matches and flagging truncated when it would overflow."""
    import json as _json

    from llmcli.tools import _MAX_GREP_LINE, _MAX_OUTPUT

    # Many matching lines, each near the preview cap => serialized size >> budget.
    body = "".join(f"NEEDLE {'y' * (_MAX_GREP_LINE - 8)}\n" for _ in range(400))
    _call("write_file", {"path": "big.txt", "content": body})
    res = _call("grep", {"pattern": "NEEDLE"})
    assert res["ok"] is True
    assert res["result"]["truncated"] is True
    payload = _json.dumps(res["result"]["matches"]).encode("utf-8")
    assert len(payload) <= _MAX_OUTPUT


def test_grep_finds_nested_recursive(tmp_workspace):
    """TOOLS-3: the os.walk-pruned grep still recurses with the default **/* and
    honors a **/*.ext glob, while never descending ignored dirs."""
    _call("write_file", {"path": "a/b/deep.py", "content": "FIND_ME\n"})
    _call("write_file", {"path": "a/b/deep.md", "content": "FIND_ME\n"})
    _call("write_file", {"path": ".venv/lib/skip.py", "content": "FIND_ME\n"})
    res = _call("grep", {"pattern": "FIND_ME", "glob": "**/*.py"})
    files = [m["file"] for m in res["result"]["matches"]]
    assert any(f.endswith("a/b/deep.py") for f in files)
    assert not any(f.endswith(".md") for f in files)        # glob filtered
    assert not any(".venv" in f for f in files)             # pruned subtree


def test_glob_recursive_matches_root_and_nested(tmp_workspace):
    """TOOLS-3: '**/*.py' matches both top-level and nested files (pathlib-like)."""
    _call("write_file", {"path": "top.py", "content": "1"})
    _call("write_file", {"path": "pkg/sub/inner.py", "content": "2"})
    matches = _call("glob", {"pattern": "**/*.py"})["result"]["matches"]
    assert any(m.endswith("top.py") for m in matches)
    assert any(m.endswith("inner.py") for m in matches)


def test_run_bash_tail_truncation_keeps_failure_summary(tmp_workspace):
    """ACBUILD-2: large output is truncated head+tail so the FAILURE summary at
    the tail survives (head-only truncation would discard it)."""
    cmd = (
        "python3 -c \""
        "import sys;"
        "print('HEAD_MARKER');"
        "sys.stdout.write('x'*200000);"
        "print('\\nFAILED: 1 test failed')\""
    )
    res = _call("run_bash", {"command": cmd, "timeout": 30})
    out = res["result"]["stdout"]
    assert "FAILED: 1 test failed" in out      # tail diagnostics preserved
    assert "...[middle truncated]..." in out   # head+tail marker
    assert "HEAD_MARKER" in out                # head preserved too


def test_run_bash_byte_ceiling_kills_runaway_output(tmp_workspace):
    """A runaway command producing > _MAX_CAPTURE_BYTES is SIGKILLed and the
    result is flagged truncated, instead of buffering megabytes in memory.
    The 200KB tail-truncation test above stays UNDER the ceiling so it still
    completes normally; this one blows past it on purpose."""
    # ~2 MB of 'y' — well over the 256KB capture ceiling. Writes to a pipe that
    # blocks once we stop reading, so the process won't exit on its own; the
    # loop must SIGKILL it once the ceiling is hit.
    cmd = "python3 -c \"import sys; sys.stdout.write('y'*2_000_000); sys.stdout.flush()\""
    res = _call("run_bash", {"command": cmd, "timeout": 30})
    assert res["ok"] is False
    assert res["result"]["truncated"] is True
    # The display output is bounded by _truncate_tail (_MAX_OUTPUT//2 per stream),
    # so the returned stdout is far smaller than the 2MB produced.
    out = res["result"]["stdout"]
    assert len(out) <= _MAX_OUTPUT + 64


def test_run_bash_small_output_unchanged(tmp_workspace):
    """Small-output commands behave identically to the old communicate() path:
    no truncated flag, exit code reported, output captured fully."""
    res = _call("run_bash", {"command": "echo hello"})
    assert res["ok"] is True
    assert res["result"]["exit_code"] == 0
    assert "hello" in res["result"]["stdout"]
    assert "truncated" not in res["result"]


def test_run_bash_success(tmp_workspace):
    res = _call("run_bash", {"command": "echo hi"})
    assert res["ok"] is True
    assert res["result"]["exit_code"] == 0
    assert "hi" in res["result"]["stdout"]


def test_run_bash_nonzero_exit(tmp_workspace):
    res = _call("run_bash", {"command": "exit 3"})
    assert res["ok"] is False
    assert res["result"]["exit_code"] == 3


def test_read_file_confined_to_workspace(tmp_workspace):
    # A path resolving outside the workspace root must be refused, even for reads.
    res = _call("read_file", {"path": "/etc/hosts"})
    assert res["ok"] is False
    assert "outside the workspace" in res["error"]


def test_glob_confined_to_workspace(tmp_workspace):
    res = _call("glob", {"pattern": "*", "path": "/etc"})
    assert res["ok"] is False
    assert "outside the workspace" in res["error"]


def test_grep_confined_to_workspace(tmp_workspace):
    res = _call("grep", {"pattern": "x", "path": "/etc"})
    assert res["ok"] is False
    assert "outside the workspace" in res["error"]


def test_glob_prunes_vendor_dirs(tmp_workspace):
    _call("write_file", {"path": ".git/config", "content": "x"})
    _call("write_file", {"path": ".venv/lib/m.py", "content": "x"})
    _call("write_file", {"path": "src/real.py", "content": "x"})
    res = _call("glob", {"pattern": "**/*"})
    matches = res["result"]["matches"]
    assert any(m.endswith("real.py") for m in matches)
    assert not any(".git" in m or ".venv" in m for m in matches)
    assert res["result"]["truncated"] is False


def test_grep_prunes_vendor_dirs_and_flags_truncation(tmp_workspace):
    _call("write_file", {"path": ".git/x.txt", "content": "needle\n"})
    _call("write_file", {"path": "code.txt", "content": "needle here\n"})
    res = _call("grep", {"pattern": "needle"})
    files = [m["file"] for m in res["result"]["matches"]]
    assert any(f.endswith("code.txt") for f in files)
    assert not any(".git" in f for f in files)
    assert res["result"]["truncated"] is False


def test_read_file_extensionless_fallback(tmp_workspace):
    _call("write_file", {"path": "README.md", "content": "# Hi\n"})
    res = _call("read_file", {"path": "README"})
    assert res["ok"] is True
    assert "# Hi" in res["result"]


def test_run_bash_strips_provider_secrets(tmp_workspace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "lm-secret-456")
    res = _call("run_bash", {"command": "echo key=$OPENAI_API_KEY lm=$LMSTUDIO_API_KEY"})
    assert res["ok"] is True
    out = res["result"]["stdout"]
    assert "sk-secret-123" not in out
    assert "lm-secret-456" not in out


def test_run_bash_env_is_allowlist_not_denylist(tmp_workspace, monkeypatch):
    """finding #16: the child env is default-DENY. Any secret NOT on the
    allowlist (AWS/GitHub/Anthropic/arbitrary) must be absent, while PATH (an
    allowlisted infra var) flows through."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-secret")
    monkeypatch.setenv("MY_RANDOM_SECRET", "random-secret")
    res = _call(
        "run_bash",
        {"command": "echo a=$AWS_SECRET_ACCESS_KEY g=$GITHUB_TOKEN "
                    "t=$ANTHROPIC_API_KEY r=$MY_RANDOM_SECRET p=${PATH:+set}"},
    )
    assert res["ok"] is True
    out = res["result"]["stdout"]
    assert "aws-secret" not in out
    assert "ghp-secret" not in out
    assert "ant-secret" not in out
    assert "random-secret" not in out
    # PATH (allowlisted) is still present so commands resolve.
    assert "p=set" in out


def test_run_bash_bool_timeout_coerced(tmp_workspace):
    """finding #24: bool is an int subclass; timeout=True must NOT become a 1s
    timeout — it is coerced to the default and noted."""
    res = _call("run_bash", {"command": "echo hi", "timeout": True})
    assert res["ok"] is True
    assert res["result"].get("note", "").startswith("invalid timeout")


def test_run_bash_pipefail_propagates_pipeline_failure(tmp_workspace):
    """finding #21: under the sandbox, `set -o pipefail` makes a failing stage of
    a pipeline propagate instead of being masked by a later success. Only runs
    where sandbox-exec exists (the private-mode wrapper that adds pipefail)."""
    import shutil as _shutil

    import llmcli.tools as t

    if _shutil.which("sandbox-exec") is None:
        import pytest

        pytest.skip("sandbox-exec unavailable; pipefail is only added in the sandbox path")
    if t._PRIVATE is not True:
        import pytest

        pytest.skip("pipefail is only prepended in private mode")
    # `false | true` exits 0 without pipefail; with pipefail it exits non-zero.
    res = _call("run_bash", {"command": "false | true"})
    assert res["ok"] is False
    assert res["result"]["exit_code"] != 0


def test_run_bash_pipefail_propagates_under_network_mode(tmp_workspace, monkeypatch):
    """finding #4: pipefail is applied in --allow-network mode too, so a failing
    pipeline stage propagates identically to private mode (no longer masked)."""
    import llmcli.tools as t

    monkeypatch.setattr(t, "_PRIVATE", False)
    res = _call("run_bash", {"command": "false | true"})
    assert res["ok"] is False
    assert res["result"]["exit_code"] != 0


def test_run_bash_sigpipe_from_head_is_success(tmp_workspace):
    """A producer SIGPIPE-killed by a consumer closing the pipe early (the
    `cmd | head` idiom) exits 141 under pipefail. That is benign and MUST report
    success, otherwise every `... | head` security/audit command shows as failed.
    `yes` is infinite, so `head -1` reliably SIGPIPEs it."""
    res = _call("run_bash", {"command": "yes 2>/dev/null | head -1"})
    assert res["ok"] is True
    assert res["result"]["exit_code"] == 141
    assert "SIGPIPE" in res["result"].get("note", "")
    assert res["result"]["stdout"].strip() == "y"


def test_run_bash_sigpipe_does_not_mask_real_failure(tmp_workspace):
    """The SIGPIPE allowance is narrow: a genuine pipeline failure carries its
    OWN exit code (here 1 from `false`), never 141, so it still reports failure
    — finding #21's masking guard stays intact."""
    res = _call("run_bash", {"command": "false | true"})
    assert res["ok"] is False
    assert res["result"]["exit_code"] == 1


def test_run_bash_standalone_exit_141_is_not_masked(tmp_workspace):
    """The SIGPIPE allowance is scoped to commands containing a REAL pipe. A
    standalone command (or one using only `||`) that exits 141 is NOT a pipe
    SIGPIPE and must still report failure."""
    res = _call("run_bash", {"command": "exit 141"})
    assert res["ok"] is False
    assert res["result"]["exit_code"] == 141
    # `||` is the logical-or operator, not a pipe — must not trigger the allowance.
    res2 = _call("run_bash", {"command": "false || exit 141"})
    assert res2["ok"] is False
    assert res2["result"]["exit_code"] == 141


def test_run_bash_timeout_returns_partial_output(tmp_workspace):
    res = _call("run_bash", {"command": "echo early; sleep 5", "timeout": 1})
    assert res["ok"] is False
    assert res["result"]["timed_out"] is True
    assert "early" in res["result"]["stdout"]


def test_openai_schema_subset_and_all():
    full = openai_schema(None)
    assert len(full) == len(REGISTRY)
    assert all(t["type"] == "function" for t in full)

    sub = openai_schema(READ_ONLY)
    names = {t["function"]["name"] for t in sub}
    assert names == set(READ_ONLY)


def test_tool_subset():
    sub = tool_subset(READ_ONLY)
    assert set(sub) == set(READ_ONLY)
