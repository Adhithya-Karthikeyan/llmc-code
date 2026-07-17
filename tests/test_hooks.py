"""Tests for llmcli.hooks — user-configurable lifecycle hooks.

Hooks are tiny inline shell commands run around tool use. Tests inject the
config dict directly (via load_hooks with a tmp_path file, or by building the
dict), so no real ~/.llm-cli/hooks.json is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmcli import hooks  # noqa: E402


# --------------------------------------------------------------------------- #
# load_hooks
# --------------------------------------------------------------------------- #
def _write(path: Path, obj) -> Path:
    path.write_text(
        obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8"
    )
    return path


def test_load_hooks_missing_file_returns_empty(tmp_path):
    assert hooks.load_hooks(tmp_path / "nope.json") == {}


def test_load_hooks_malformed_json_returns_empty(tmp_path):
    p = _write(tmp_path / "hooks.json", "{ this is not json ]")
    assert hooks.load_hooks(p) == {}


def test_load_hooks_non_dict_top_level_returns_empty(tmp_path):
    p = _write(tmp_path / "hooks.json", ["not", "a", "dict"])
    assert hooks.load_hooks(p) == {}


def test_load_hooks_valid_shape(tmp_path):
    cfg = {
        "PreToolUse": [
            {"match": "write_file|edit_file", "command": "exit 0", "timeout": 5}
        ],
        "PostToolUse": [{"command": "true"}],
        "Stop": [{"command": "echo bye"}],
    }
    p = _write(tmp_path / "hooks.json", cfg)
    loaded = hooks.load_hooks(p)
    assert loaded["PreToolUse"][0]["match"] == "write_file|edit_file"
    assert loaded["PreToolUse"][0]["command"] == "exit 0"
    assert loaded["PreToolUse"][0]["timeout"] == 5
    # Missing match normalizes to "" (match all); missing timeout -> default.
    assert loaded["PostToolUse"][0]["match"] == ""
    assert loaded["PostToolUse"][0]["timeout"] == hooks.DEFAULT_HOOK_TIMEOUT
    assert loaded["Stop"][0]["command"] == "echo bye"


def test_load_hooks_drops_malformed_entries(tmp_path):
    cfg = {
        "PreToolUse": [
            {"command": ""},  # empty command -> dropped
            {"match": "ok"},  # no command -> dropped
            "not a dict",  # -> dropped
            {"match": "[unclosed", "command": "true"},  # bad regex -> dropped
            {"command": "true"},  # valid, kept
        ],
        "Unknown": [{"command": "true"}],  # unknown event ignored
    }
    p = _write(tmp_path / "hooks.json", cfg)
    loaded = hooks.load_hooks(p)
    assert len(loaded["PreToolUse"]) == 1
    assert loaded["PreToolUse"][0]["command"] == "true"
    assert "Unknown" not in loaded


def test_load_hooks_bad_timeout_coerced(tmp_path):
    cfg = {"Stop": [{"command": "true", "timeout": True},
                    {"command": "true", "timeout": -3},
                    {"command": "true", "timeout": "x"}]}
    p = _write(tmp_path / "hooks.json", cfg)
    loaded = hooks.load_hooks(p)
    for e in loaded["Stop"]:
        assert e["timeout"] == hooks.DEFAULT_HOOK_TIMEOUT


# --------------------------------------------------------------------------- #
# run_pre_tool
# --------------------------------------------------------------------------- #
def test_pre_tool_no_hooks_allows(tmp_path):
    assert hooks.run_pre_tool({}, "write_file", {}, cwd=str(tmp_path)) == {
        "decision": "allow"
    }


def test_pre_tool_exit_zero_allows(tmp_path):
    cfg = {"PreToolUse": [{"match": "", "command": "exit 0"}]}
    out = hooks.run_pre_tool(cfg, "write_file", {"path": "x"}, cwd=str(tmp_path))
    assert out == {"decision": "allow"}


def test_pre_tool_exit_nonzero_blocks_with_reason(tmp_path):
    cfg = {
        "PreToolUse": [
            {"match": "", "command": "echo 'blocked: no writes' >&2; exit 1"}
        ]
    }
    out = hooks.run_pre_tool(cfg, "write_file", {}, cwd=str(tmp_path))
    assert out["decision"] == "block"
    assert "blocked: no writes" in out["reason"]


def test_pre_tool_reason_falls_back_to_stdout(tmp_path):
    # Non-zero exit with only stdout -> reason comes from stdout.
    cfg = {"PreToolUse": [{"match": "", "command": "echo denied-on-stdout; exit 2"}]}
    out = hooks.run_pre_tool(cfg, "write_file", {}, cwd=str(tmp_path))
    assert out["decision"] == "block"
    assert "denied-on-stdout" in out["reason"]


def test_pre_tool_regex_filters_by_tool_name(tmp_path):
    # Hook only matches write_file|edit_file and always blocks.
    cfg = {
        "PreToolUse": [
            {"match": "write_file|edit_file", "command": "exit 1"}
        ]
    }
    # Non-matching tool -> allowed (hook never runs).
    assert hooks.run_pre_tool(cfg, "read_file", {}, cwd=str(tmp_path)) == {
        "decision": "allow"
    }
    # Matching tool -> blocked.
    assert (
        hooks.run_pre_tool(cfg, "edit_file", {}, cwd=str(tmp_path))["decision"]
        == "block"
    )


def test_pre_tool_hook_sees_tool_name_env(tmp_path):
    # Hook exits non-zero only when LLMC_TOOL_NAME is the expected value, and
    # emits the value it saw so we can assert it was passed correctly.
    cfg = {
        "PreToolUse": [
            {
                "match": "",
                "command": 'echo "saw=$LLMC_TOOL_NAME" >&2; '
                'test "$LLMC_TOOL_NAME" = "write_file" && exit 1 || exit 0',
            }
        ]
    }
    out = hooks.run_pre_tool(cfg, "write_file", {}, cwd=str(tmp_path))
    assert out["decision"] == "block"
    assert "saw=write_file" in out["reason"]


def test_pre_tool_hook_sees_args_env_and_stdin(tmp_path):
    # LLMC_TOOL_ARGS carries the JSON args; also piped on stdin. Block if the
    # env var contains the secret path so we know it was delivered.
    cfg = {
        "PreToolUse": [
            {
                "match": "",
                "command": 'echo "$LLMC_TOOL_ARGS" >&2; '
                'case "$LLMC_TOOL_ARGS" in *"/etc/passwd"*) exit 1;; esac; exit 0',
            }
        ]
    }
    out = hooks.run_pre_tool(
        cfg, "write_file", {"path": "/etc/passwd"}, cwd=str(tmp_path)
    )
    assert out["decision"] == "block"
    assert "/etc/passwd" in out["reason"]


def test_pre_tool_stdin_receives_args(tmp_path):
    cfg = {
        "PreToolUse": [
            {"match": "", "command": "grep -q needle && exit 1 || exit 0"}
        ]
    }
    out = hooks.run_pre_tool(
        cfg, "write_file", {"key": "needle"}, cwd=str(tmp_path)
    )
    assert out["decision"] == "block"


def test_pre_tool_timeout_blocks_fail_closed(tmp_path):
    cfg = {"PreToolUse": [{"match": "", "command": "sleep 5", "timeout": 1}]}
    out = hooks.run_pre_tool(cfg, "write_file", {}, cwd=str(tmp_path))
    assert out["decision"] == "block"
    assert "timed out" in out["reason"].lower()


def test_pre_tool_env_injection_overrides(tmp_path):
    # Custom env is honored; LLMC_* is layered on top of it.
    cfg = {
        "PreToolUse": [
            {"match": "", "command": 'test "$MY_FLAG" = "on" && exit 1 || exit 0'}
        ]
    }
    out = hooks.run_pre_tool(
        cfg, "write_file", {}, cwd=str(tmp_path), env={"MY_FLAG": "on"}
    )
    assert out["decision"] == "block"


def test_pre_tool_first_matching_block_short_circuits(tmp_path):
    # Two matching hooks; the first blocks, second (would allow) never matters.
    marker = tmp_path / "second_ran"
    cfg = {
        "PreToolUse": [
            {"match": "", "command": "exit 1"},
            {"match": "", "command": f"touch {marker}; exit 0"},
        ]
    }
    out = hooks.run_pre_tool(cfg, "write_file", {}, cwd=str(tmp_path))
    assert out["decision"] == "block"
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# run_post_tool
# --------------------------------------------------------------------------- #
def test_post_tool_nonzero_does_not_block(tmp_path):
    cfg = {"PostToolUse": [{"match": "", "command": "echo oops >&2; exit 3"}]}
    info = hooks.run_post_tool(
        cfg, "write_file", {"path": "x"}, {"ok": True}, cwd=str(tmp_path)
    )
    assert info["ran"] == 1
    assert info["results"][0]["exit_code"] == 3
    assert "oops" in info["results"][0]["stderr"]


def test_post_tool_no_hooks(tmp_path):
    assert hooks.run_post_tool({}, "write_file", {}, {}, cwd=str(tmp_path)) == {
        "ran": 0,
        "results": [],
    }


def test_post_tool_regex_filters(tmp_path):
    cfg = {"PostToolUse": [{"match": "read_file", "command": "true"}]}
    info = hooks.run_post_tool(cfg, "write_file", {}, {}, cwd=str(tmp_path))
    assert info["ran"] == 0


def test_post_tool_sees_result_env(tmp_path):
    out_file = tmp_path / "captured"
    cfg = {
        "PostToolUse": [
            {"match": "", "command": f'printf "%s" "$LLMC_TOOL_RESULT" > {out_file}'}
        ]
    }
    hooks.run_post_tool(
        cfg, "write_file", {"path": "x"}, {"bytes_written": 42}, cwd=str(tmp_path)
    )
    assert "42" in out_file.read_text()


# --------------------------------------------------------------------------- #
# run_stop
# --------------------------------------------------------------------------- #
def test_run_stop_runs_hooks(tmp_path):
    marker = tmp_path / "stopped"
    cfg = {"Stop": [{"command": f"touch {marker}"}]}
    info = hooks.run_stop(cfg, cwd=str(tmp_path))
    assert info["ran"] == 1
    assert marker.exists()


def test_run_stop_no_hooks(tmp_path):
    assert hooks.run_stop({}, cwd=str(tmp_path)) == {"ran": 0, "results": []}


def test_run_stop_nonzero_recorded_not_raised(tmp_path):
    cfg = {"Stop": [{"command": "exit 7"}]}
    info = hooks.run_stop(cfg, cwd=str(tmp_path))
    assert info["results"][0]["exit_code"] == 7


# --------------------------------------------------------------------------- #
# end-to-end via load_hooks + run_pre_tool
# --------------------------------------------------------------------------- #
def test_end_to_end_load_then_block(tmp_path):
    cfg = {
        "PreToolUse": [
            {
                "match": "write_file",
                "command": "echo 'writes disabled' >&2; exit 1",
            }
        ]
    }
    p = _write(tmp_path / "hooks.json", cfg)
    loaded = hooks.load_hooks(p)
    out = hooks.run_pre_tool(loaded, "write_file", {"path": "a"}, cwd=str(tmp_path))
    assert out["decision"] == "block"
    assert "writes disabled" in out["reason"]
    # A different tool is unaffected.
    assert hooks.run_pre_tool(loaded, "read_file", {}, cwd=str(tmp_path)) == {
        "decision": "allow"
    }


def test_malformed_config_yields_allow(tmp_path):
    # Broken file -> load_hooks -> {} -> run_pre_tool allows.
    p = _write(tmp_path / "hooks.json", "totally broken {{{")
    loaded = hooks.load_hooks(p)
    assert loaded == {}
    assert hooks.run_pre_tool(loaded, "write_file", {}, cwd=str(tmp_path)) == {
        "decision": "allow"
    }
