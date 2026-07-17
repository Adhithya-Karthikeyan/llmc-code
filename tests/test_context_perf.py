"""Per-turn PERF regression tests (no observable-behavior change).

Two hot-path fixes are exercised here:

1. TOKEN-CACHE INVALIDATION GUARD — the per-turn stale-tool-output trim must
   invalidate the incremental token-estimate cache ONLY when it actually rewrote
   bytes. On a steady-state turn (everything already trimmed) it mutates nothing,
   so the cache must be reused instead of being force-recomputed O(n) every turn.

2. TOOL-SCHEMA MEMOIZATION — the tool schema is identical unless the active tool
   NAMES change, so it must be built once and reused across turns, and rebuilt
   only when the tool-name set changes.
"""

from __future__ import annotations

from llmcli.agent import Agent, _STALE_TOOL_RESULT_CAP
from llmcli.providers import MockProvider


def _agent(tool_names=None) -> Agent:
    return Agent(
        provider=MockProvider(scenario="plain"),
        system_prompt="sys",
        tool_names=list(tool_names or []),
    )


# --------------------------------------------------------------------------- #
# Fix 1 — token-cache invalidation guard
# --------------------------------------------------------------------------- #
def test_trim_sets_mutation_flag_true_only_when_bytes_change():
    """The reliable mutation flag reflects real rewrites, not the lossy return."""
    ag = _agent()
    big = "y" * (_STALE_TOOL_RESULT_CAP * 4)
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "tool", "content": big},
        {"role": "user", "content": "b"},
    ]
    # First pass rewrites the oversized stale tool result.
    ag._trim_stale_tool_outputs()
    assert ag._last_trim_mutated is True
    # Second pass finds it already carries the trim marker -> nothing to change.
    ag._trim_stale_tool_outputs()
    assert ag._last_trim_mutated is False


def test_trim_invalidates_token_cache_only_once(monkeypatch):
    """Calling the trim path twice with no NEW trimmable content invalidates the
    token cache exactly once (the second, no-op trim must reuse the cache)."""
    ag = _agent()
    big = "z" * (_STALE_TOOL_RESULT_CAP * 4)
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "tool", "content": big},
        {"role": "user", "content": "b"},
    ]

    calls = {"n": 0}
    real_invalidate = ag._invalidate_token_est

    def _counting_invalidate():
        calls["n"] += 1
        real_invalidate()

    monkeypatch.setattr(ag, "_invalidate_token_est", _counting_invalidate)

    # Mirror the per-turn caller: trim, then invalidate ONLY when bytes changed.
    def _turn():
        ag._trim_stale_tool_outputs()
        if ag._last_trim_mutated:
            ag._invalidate_token_est()

    # Turn 1: real trim happens -> invalidate once.
    _turn()
    assert calls["n"] == 1
    # Arm the cache (simulate _maybe_auto_compact reading the estimate).
    first_est = ag._estimate_tokens_cached()
    assert ag._running_token_chars is not None

    # Turn 2: nothing new to trim -> must NOT invalidate again.
    _turn()
    assert calls["n"] == 1
    # The armed cache is still intact and reused (no forced O(n) recompute).
    assert ag._running_token_chars is not None
    assert ag._estimate_tokens_cached() == first_est


def test_trim_no_mutation_when_nothing_trimmable():
    """A history with only small tool output never mutates or flags a change."""
    ag = _agent()
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "tool", "content": "short"},
        {"role": "user", "content": "b"},
    ]
    ag._trim_stale_tool_outputs()
    assert ag._last_trim_mutated is False


# --------------------------------------------------------------------------- #
# Fix 2 — tool-schema memoization
# --------------------------------------------------------------------------- #
def test_tools_payload_built_once_and_reused(monkeypatch):
    """_tools_payload is built once and reused across turns when names are fixed."""
    import llmcli.agent as agent_mod

    ag = _agent(tool_names=["read_file", "write_file"])

    build_calls = {"n": 0}
    real_schema = agent_mod.tools_mod.openai_schema

    def _counting_schema(names=None, registry=None):
        build_calls["n"] += 1
        return real_schema(names, registry=registry)

    monkeypatch.setattr(agent_mod.tools_mod, "openai_schema", _counting_schema)

    first = ag._tools_payload()
    assert first is not None
    assert build_calls["n"] == 1
    # Repeated turns reuse the SAME cached object; no re-serialization.
    second = ag._tools_payload()
    third = ag._tools_payload()
    assert second is first
    assert third is first
    assert build_calls["n"] == 1


def test_tools_payload_rebuilds_when_tool_names_change(monkeypatch):
    """Changing the active tool-name set rebuilds the schema (and stays correct)."""
    import llmcli.agent as agent_mod

    ag = _agent(tool_names=["read_file"])

    build_calls = {"n": 0}
    real_schema = agent_mod.tools_mod.openai_schema

    def _counting_schema(names=None, registry=None):
        build_calls["n"] += 1
        return real_schema(names, registry=registry)

    monkeypatch.setattr(agent_mod.tools_mod, "openai_schema", _counting_schema)

    payload_a = ag._tools_payload()
    assert build_calls["n"] == 1
    names_a = {t["function"]["name"] for t in payload_a}
    assert names_a == {"read_file"}

    # A new tool-name set -> rebuild, and the new schema reflects it.
    ag.tool_names = ["read_file", "write_file"]
    payload_b = ag._tools_payload()
    assert build_calls["n"] == 2
    assert payload_b is not payload_a
    names_b = {t["function"]["name"] for t in payload_b}
    assert names_b == {"read_file", "write_file"}


def test_tools_payload_none_when_no_tools():
    """No tool names -> None, and no schema build attempted."""
    ag = _agent(tool_names=[])
    assert ag._tools_payload() is None
