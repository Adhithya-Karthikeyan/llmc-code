"""Permission-mode propagation + read-only/plan escalation guards.

Covers the fixes that stop read-only/plan mode from being ESCAPED:
  - spawn_agent is blocked in read-only/plan (delegation can't escalate);
  - MCP mutation-named tools are blocked in read-only/plan (best-effort);
  - make_spawn_agent_tool threads permission_mode + cancel_event into the
    sub-agent, so a sub-agent spawned under read-only inherits the block;
  - default mode is unchanged (spawn_agent + MCP tools still allowed).

All offline (MockProvider); no network.
"""

from __future__ import annotations

import threading

from llmcli.agent import Agent
from llmcli.orchestration import make_spawn_agent_tool
from llmcli.providers import MockProvider
from llmcli.tools import FULL, Tool


def _fake_tool(name: str, requires_confirmation: bool = False) -> Tool:
    """A minimal Tool used only to exercise _permission_decision (fn is a stub)."""
    return Tool(
        name=name,
        description="stub",
        parameters={"type": "object", "properties": {}},
        fn=lambda args: {"ok": True, "result": None},
        requires_confirmation=requires_confirmation,
    )


class _WriteThenDone(MockProvider):
    """Emit a write_file call, then finalize once any tool result exists."""

    def stream_chat(self, messages, tools, tool_choice=None):
        acted = any(m.get("role") == "tool" for m in messages)
        if not acted:
            yield {"type": "tool_call", "id": "w", "name": "write_file",
                   "arguments": {"path": "f.py", "content": "x = 1\n",
                                 "overwrite": True}}
            yield {"type": "done", "finish_reason": "tool_calls"}
        else:
            yield {"type": "text", "text": "all done."}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


# --------------------------------------------------------------------------- #
# read-only/plan escalation guards (agent._permission_decision)
# --------------------------------------------------------------------------- #
def test_read_only_blocks_spawn_agent():
    """Delegation is disabled in read-only so a coder sub-agent can't escalate."""
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
        permission_mode="read-only",
    )
    d, reason = agent._permission_decision(_fake_tool("spawn_agent", True), {})
    assert d == "block"
    assert "delegation disabled" in reason


def test_read_only_blocks_mutation_named_mcp_tool():
    """A mutation-verb MCP tool (create/run/update/…) is blocked in read-only."""
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
        permission_mode="read-only",
    )
    for name in (
        "mcp__crm__create_document",
        "mcp__erp__run_python_code",
        "mcp__crm__update_document",
        "mcp__crm__delete_document",
    ):
        d, reason = agent._permission_decision(_fake_tool(name), {})
        assert d == "block", f"{name} should be blocked in read-only"
        assert name in reason


def test_read_only_allows_read_named_mcp_tool_and_read_file():
    """Read-style MCP tools and built-in reads stay allowed in read-only."""
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
        permission_mode="read-only",
    )
    for name in (
        "mcp__crm__get_document",
        "mcp__crm__list_documents",
        "mcp__crm__search",
        "mcp__erp__generate_report",
        "mcp__erp__analyze_business_data",
    ):
        assert agent._permission_decision(_fake_tool(name), {})[0] == "allow", (
            f"{name} should be allowed (read-style) in read-only"
        )
    read_tool = agent.registry["read_file"]
    assert agent._permission_decision(read_tool, {"path": "x"})[0] == "allow"


def test_plan_mode_also_blocks_spawn_and_mutation_mcp():
    """The same escalation guards apply in plan mode."""
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
        permission_mode="plan",
    )
    assert agent._permission_decision(_fake_tool("spawn_agent", True), {})[0] == "block"
    assert agent._permission_decision(
        _fake_tool("mcp__crm__submit_document"), {})[0] == "block"
    assert agent._permission_decision(
        _fake_tool("mcp__crm__get_document"), {})[0] == "allow"


# --------------------------------------------------------------------------- #
# make_spawn_agent_tool threads permission_mode + cancel_event
# --------------------------------------------------------------------------- #
def _captured_sub_agent(**spawn_kwargs):
    """Capture the kwargs the spawn tool would pass to the sub-Agent, without
    running the sub-agent's loop."""
    captured: list = []
    original_init = Agent.__init__

    def patched_init(self, **kwargs):
        captured.append(kwargs)
        original_init(self, **kwargs)

    import unittest.mock as mock

    tool = make_spawn_agent_tool(
        provider=MockProvider(scenario="hello"),
        auto_confirm=True,
        max_iterations=2,
        **spawn_kwargs,
    )
    with mock.patch.object(Agent, "__init__", patched_init):
        try:
            tool.fn({"role": "coder", "task": "do a thing"})
        except Exception:
            pass
    return captured


def test_spawn_tool_forwards_permission_mode():
    """permission_mode passed to make_spawn_agent_tool reaches the sub-Agent."""
    captured = _captured_sub_agent(permission_mode="read-only")
    assert captured, "sub-agent was never constructed"
    assert captured[0].get("permission_mode") == "read-only"


def test_spawn_tool_forwards_cancel_event():
    """cancel_event is propagated so a delegated run honours interrupts."""
    event = threading.Event()
    captured = _captured_sub_agent(cancel_event=event)
    assert captured, "sub-agent was never constructed"
    assert captured[0].get("cancel_event") is event


def test_spawn_tool_defaults_unchanged():
    """Omitting the new params keeps the historic defaults (default/None)."""
    captured = _captured_sub_agent()
    assert captured, "sub-agent was never constructed"
    assert captured[0].get("permission_mode") == "default"
    assert captured[0].get("cancel_event") is None


def test_read_only_sub_agent_blocks_write(tmp_workspace):
    """End-to-end: a coder sub-agent spawned under read-only cannot write."""
    tool = make_spawn_agent_tool(
        provider=_WriteThenDone(),
        auto_confirm=True,
        max_iterations=6,
        permission_mode="read-only",
    )
    result = tool.fn({"role": "coder", "task": "write f.py"})
    from pathlib import Path
    assert not Path("f.py").exists(), "write must be blocked in a read-only sub-agent"
    assert result["ok"] is True
    assert "all done" in result["result"]


# --------------------------------------------------------------------------- #
# default mode: nothing new is blocked
# --------------------------------------------------------------------------- #
def test_default_mode_allows_spawn_and_mcp_tools():
    """Default mode is unchanged: spawn_agent + MCP tools are NOT mode-blocked.

    (spawn_agent still defers to the confirm gate via requires_confirmation, but
    _permission_decision must never return an unconditional read-only-style
    'block' for it in default mode.)"""
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, permission_mode="default",
    )
    # A non-confirming MCP tool: allowed outright.
    assert agent._permission_decision(
        _fake_tool("mcp__crm__create_document"), {})[0] == "allow"
    assert agent._permission_decision(
        _fake_tool("mcp__crm__get_document"), {})[0] == "allow"
    # spawn_agent requires confirmation; with auto_confirm it auto-runs (never
    # a mode 'block').
    d, _ = agent._permission_decision(_fake_tool("spawn_agent", True), {})
    assert d == "auto"
