"""Regression: MCP tool-name validation must allow snake_case names.

The charset regex once omitted '_', so every underscore-named MCP tool
(kyp_search, read_file, ...) was rejected and servers registered 0 tools.
The fake server's tools (echo/add) had no underscores, so the gap was missed.
"""

from __future__ import annotations

from llmcli.mcp import _make_tool


def test_make_tool_allows_snake_case():
    t = _make_tool(None, "kyp-mem", {"name": "kyp_search", "inputSchema": {"type": "object"}})
    assert t is not None
    assert t.name == "mcp__kyp-mem__kyp_search"


def test_make_tool_allows_hyphen_server_and_underscores():
    t = _make_tool(None, "kyp-mem", {"name": "kyp_objective_set", "inputSchema": {"type": "object"}})
    assert t is not None and t.name == "mcp__kyp-mem__kyp_objective_set"


def test_make_tool_allows_double_underscore_names():
    # '__' is allowed: the real server name is always injected, so a tool name
    # cannot forge another namespace (e.g. kyp-mem's '____kyp_instructions').
    t = _make_tool(None, "kyp-mem", {"name": "____kyp_instructions"})
    assert t is not None and t.name == "mcp__kyp-mem______kyp_instructions"


def test_make_tool_rejects_metachars():
    assert _make_tool(None, "s", {"name": "a/b"}) is None           # path metachar
    assert _make_tool(None, "s", {"name": "a;rm -rf"}) is None      # shell metachars
    assert _make_tool(None, "s", {"name": "a b"}) is None           # space


def test_make_tool_readonly_gate():
    # readOnlyHint without operator trust => still confirmation-gated.
    gated = _make_tool(None, "s", {"name": "ro_tool", "annotations": {"readOnlyHint": True}})
    assert gated.requires_confirmation is True
    # readOnlyHint WITH operator trust => relaxed.
    trusted = _make_tool(
        None, "s", {"name": "ro_tool", "annotations": {"readOnlyHint": True}},
        trust_read_only_hint=True,
    )
    assert trusted.requires_confirmation is False
