"""Tests for make_spawn_agent_tool context-guard threading (ORCH-3 fix)."""

from __future__ import annotations

from llmcli.orchestration import make_spawn_agent_tool
from llmcli.providers import MockProvider


def _captured_agent(context_budget: int, context_ceiling: int,
                    context_adaptive: bool = True):
    """Build a spawn tool with the given context params and capture the sub-Agent
    that would be constructed, without actually running it."""
    provider = MockProvider(scenario="hello")
    captured: list = []

    import llmcli.orchestration as _mod
    from llmcli.agent import Agent

    original_init = Agent.__init__

    def patched_init(self, **kwargs):
        captured.append(kwargs)
        original_init(self, **kwargs)

    import unittest.mock as mock

    tool = make_spawn_agent_tool(
        provider=provider,
        auto_confirm=True,
        max_iterations=2,
        context_budget=context_budget,
        context_ceiling=context_ceiling,
        context_adaptive=context_adaptive,
    )

    # Patch Agent.__init__ to intercept the sub-agent construction.
    with mock.patch.object(Agent, "__init__", patched_init):
        # _spawn will fail to run (MockProvider is fine but task is irrelevant);
        # we only need the Agent to be constructed to capture kwargs.
        try:
            tool.fn({"role": "explorer", "task": "list files"})
        except Exception:
            pass

    return captured


def test_spawn_agent_threads_context_ceiling():
    """Spawned sub-agents must receive a non-zero context_ceiling so
    _maybe_auto_compact is active (ORCH-3 fix)."""
    captured = _captured_agent(context_budget=10_000, context_ceiling=50_000)
    assert captured, "Agent.__init__ was never called — spawn did not construct a sub-agent"
    kwargs = captured[0]
    assert kwargs.get("context_ceiling") == 50_000, (
        f"Expected context_ceiling=50_000, got {kwargs.get('context_ceiling')}"
    )


def test_spawn_agent_threads_context_budget():
    """context_budget is also forwarded so the adaptive per-turn budget works."""
    captured = _captured_agent(context_budget=8_000, context_ceiling=40_000)
    assert captured
    assert captured[0].get("context_budget") == 8_000


def test_spawn_agent_zero_defaults_preserved():
    """Default (0, 0) still produces a sub-agent with context_ceiling=0 —
    backward-compatible with callers that don't pass context params yet."""
    captured = _captured_agent(context_budget=0, context_ceiling=0)
    assert captured
    assert captured[0].get("context_ceiling") == 0


def test_spawn_agent_threads_context_adaptive_false():
    """A user who disabled adaptive budgeting (/context off) must have it
    forwarded to spawned sub-agents, not silently reset to the Agent default."""
    captured = _captured_agent(
        context_budget=10_000, context_ceiling=50_000, context_adaptive=False
    )
    assert captured
    assert captured[0].get("context_adaptive") is False


def test_spawn_agent_threads_context_adaptive_true():
    """True propagates too (the explicit/default adaptive-on case)."""
    captured = _captured_agent(
        context_budget=10_000, context_ceiling=50_000, context_adaptive=True
    )
    assert captured
    assert captured[0].get("context_adaptive") is True


def _captured_agent_with_code_search():
    """Build a spawn tool WITH an injected code_search tool and capture the
    sub-Agent kwargs without running it."""
    from llmcli.agent import Agent
    from llmcli.code_index import make_code_search_tool

    provider = MockProvider(scenario="hello")
    code_search = make_code_search_tool(provider=provider, workspace=".")
    captured: list = []
    original_init = Agent.__init__

    def patched_init(self, **kwargs):
        captured.append(kwargs)
        original_init(self, **kwargs)

    import unittest.mock as mock

    tool = make_spawn_agent_tool(
        provider=provider, auto_confirm=True, max_iterations=2,
        code_search_tool=code_search,
    )
    with mock.patch.object(Agent, "__init__", patched_init):
        try:
            tool.fn({"role": "explorer", "task": "find code"})
        except Exception:
            pass
    return captured


def test_spawn_agent_threads_code_search_into_subagent():
    """A spawned sub-agent must see code_search in both its tool_names and its
    registry (it is injected, not in the global REGISTRY)."""
    captured = _captured_agent_with_code_search()
    assert captured, "Agent.__init__ was never called"
    kwargs = captured[0]
    assert "code_search" in kwargs.get("tool_names", [])
    registry = kwargs.get("registry") or {}
    assert "code_search" in registry
    # Sub-agents never get spawn_agent (no recursion); code_search is fine to share.
    assert "spawn_agent" not in registry


def test_spawn_agent_without_code_search_uses_default_registry():
    """Back-compat: no code_search tool -> sub-agent registry stays None (falls
    back to the global REGISTRY), unchanged from before."""
    captured = _captured_agent(context_budget=0, context_ceiling=0)
    assert captured
    assert captured[0].get("registry") is None
