"""Focused regression tests for the perf fixes in agent.py (Fixes 2/3/4).

Fix 1 (truncation-retry cap latch) is NOT tested here: it is already addressed
upstream — the gentle token cap is bypassed on tool-capable turns in
``llmcli/providers.py`` (``gentle_for_this_call = self.gentle_mode and not
tools``), so a write_file/edit_file tool call is never truncated by the gentle
cap and there is nothing to latch. Documented here for the record.
"""

from __future__ import annotations

import json

from llmcli.agent import (
    Agent,
    _BLOATING_TOOL_NAMES,
    _BLOATING_TOOL_PATTERNS,
    _is_bloating_tool,
    _msg_chars,
)
from llmcli.providers import MockProvider


def _agent() -> Agent:
    return Agent(
        provider=MockProvider(scenario="plain"),
        system_prompt="sys",
        tool_names=[],
    )


# --------------------------------------------------------------------------- #
# Fix 2 — incremental token estimation equals full recompute
# --------------------------------------------------------------------------- #
def test_estimate_tokens_cached_equals_static_recompute():
    """The incremental cache returns the SAME value as the full recompute."""
    a = _agent()
    a.messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "please read README.md and summarize"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "README.md"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file",
         "content": "x" * 4000},
    ]
    # Dirty the cache, then query — must match the static method exactly.
    a._invalidate_token_est()
    cached = a._estimate_tokens_cached()
    static = Agent._estimate_tokens(a.messages)
    assert cached == static

    # Append a new message; the delta path must keep the cache in sync with a
    # fresh full recompute (the O(n^2) path we're replacing).
    a.messages.append({"role": "user", "content": "thanks!"})
    a._account_appended_msg_token_est()
    assert a._estimate_tokens_cached() == Agent._estimate_tokens(a.messages)

    # A second append too (the common multi-iteration tool-react pattern).
    a.messages.append({"role": "assistant", "content": "you're welcome"})
    a._account_appended_msg_token_est()
    assert a._estimate_tokens_cached() == Agent._estimate_tokens(a.messages)


def test_estimate_tokens_cached_invalidates_on_trim_marker():
    """After _invalidate_token_est the next query recomputes from scratch."""
    a = _agent()
    a.messages = [{"role": "user", "content": "hello"}]
    a._invalidate_token_est()
    assert a._running_token_chars is None
    val = a._estimate_tokens_cached()
    assert a._running_token_chars is not None  # rearmed
    assert val == Agent._estimate_tokens(a.messages)


def test_msg_chars_matches_estimate_tokens_per_message_formula():
    """_msg_chars(m) is exactly the per-message contribution to _estimate_tokens."""
    m = {
        "role": "assistant",
        "content": "abc",
        "tool_calls": [{"id": "1", "type": "function",
                        "function": {"name": "t", "arguments": "{}"}}],
    }
    # _estimate_tokens sums _msg_chars over messages then // 4; for a single
    # message the char total is _msg_chars(m), so tokens == _msg_chars(m) // 4.
    assert Agent._estimate_tokens([m]) == _msg_chars(m) // 4


# --------------------------------------------------------------------------- #
# Fix 3 — summarizer caps tool-call arguments
# --------------------------------------------------------------------------- #
def test_serialize_for_summary_caps_tool_call_arguments():
    """A 50KB write_file content in tool_call arguments is capped to ~500 chars."""
    huge = "Z" * 50_000
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": "big.txt", "content": huge}
                        ),
                    },
                }
            ],
        }
    ]
    out = Agent._serialize_for_summary(msgs)
    # The tool name survives (the gist).
    assert "write_file" in out
    # The 50KB blob does NOT go through verbatim into the summarizer prompt.
    assert huge not in out
    # The arguments string in the transcript is capped at ~500 chars (the same
    # cap style applied to tool results).
    assert huge[:600] not in out


def test_serialize_for_summary_short_arguments_pass_through():
    """Short arguments are not truncated (mirror the tool-result cap behavior)."""
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "README.md"}),
                    },
                }
            ],
        }
    ]
    out = Agent._serialize_for_summary(msgs)
    assert "README.md" in out


# --------------------------------------------------------------------------- #
# Fix 4 — MCP read tools are bloating
# --------------------------------------------------------------------------- #
def test_bloating_patterns_set_exists():
    """_BLOATING_TOOL_PATTERNS carries read-oriented MCP SUFFIX substrings.

    Note: matching is suffix-only (after the last ``__``), so ``"mem"`` is
    intentionally NOT a pattern — it would false-match a server named
    ``kyp-mem`` and flag that server's write/delete tools as bloating.
    """
    assert isinstance(_BLOATING_TOOL_PATTERNS, frozenset)
    for p in ("search", "context", "project", "session"):
        assert p in _BLOATING_TOOL_PATTERNS
    assert "mem" not in _BLOATING_TOOL_PATTERNS


def test_bloating_names_set_unchanged():
    """The built-in exact-match set is NOT modified (existing invariant holds)."""
    assert _BLOATING_TOOL_NAMES == frozenset(
        {"read_file", "grep", "repo_map", "code_search", "glob"}
    )


def test_is_bloating_tool_matches_builtins():
    assert _is_bloating_tool("read_file") is True
    assert _is_bloating_tool("grep") is True
    assert _is_bloating_tool("write_file") is False
    assert _is_bloating_tool("run_bash") is False


def test_is_bloating_tool_matches_mcp_read_patterns():
    """MCP read-tool names (mcp__*mem* etc.) are bloating."""
    assert _is_bloating_tool("mcp__kyp-mem__kyp_search") is True
    assert _is_bloating_tool("mcp__kyp-mem__kyp_project_context") is True
    assert _is_bloating_tool("mcp__myserver__search_index") is True
    assert _is_bloating_tool("mcp__ctx__get_context") is True
    assert _is_bloating_tool("mcp__proj__project_list") is True
    assert _is_bloating_tool("mcp__sess__session_recent") is True


def test_is_bloating_tool_does_not_match_mcp_write_tools():
    """MCP tools WITHOUT a read-pattern substring in their SUFFIX are not bloating.

    Regression: a server NAMED ``kyp-mem`` must not cause that server's
    write/delete tools (whose suffixes are ``kyp_write``/``kyp_delete``) to be
    flagged as bloating — only the suffix is matched, not the server name.
    """
    assert _is_bloating_tool("mcp__myserver__write_note") is False
    assert _is_bloating_tool("mcp__myserver__delete") is False
    assert _is_bloating_tool("mcp__myserver__ping") is False
    # The real false-positive the suffix-only match fixes:
    assert _is_bloating_tool("mcp__kyp-mem__kyp_write") is False
    assert _is_bloating_tool("mcp__kyp-mem__kyp_delete") is False
    assert _is_bloating_tool("mcp__kyp-mem__kyp_objective_set") is False
    # ...while read tools on that same server are still flagged:
    assert _is_bloating_tool("mcp__kyp-mem__kyp_project_context") is True
    assert _is_bloating_tool("mcp__kyp-mem__kyp_session_search") is True


def test_account_tool_read_counts_mcp_read_tool():
    """A heavy MCP memory/search read trips the read-budget nudge."""
    a = _agent()
    big = "x" * (a.read_nudge_bytes if hasattr(a, "read_nudge_bytes") else 32_000)
    out = a._account_tool_read("mcp__kyp-mem__kyp_search", big)
    assert "[context-budget]" in out
    assert a._read_nudge_fired is True