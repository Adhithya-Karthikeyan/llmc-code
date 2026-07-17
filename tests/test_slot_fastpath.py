"""Tests for two perf fixes:

1. KV-cache slot isolation for spawned sub-agents: an orchestrator LocalProvider
   pins slot 0, but a sub-agent must get a clone that does NOT pin slot 0 so its
   requests do not evict the orchestrator's warm slot-0 KV prefix.
2. extract_text_tool_calls fast-path: a plain answer that merely contains a
   ```python fence (NO high-confidence sigil) must return None WITHOUT running the
   expensive fence/inline/parser sweeps, while every real sigil format still parses.
"""

from __future__ import annotations

import llmcli.providers as providers
from llmcli.providers import LocalProvider, extract_text_tool_calls


# --------------------------------------------------------------------------
# Fix 1: sub-agent KV-cache slot isolation
# --------------------------------------------------------------------------

def _orchestrator_provider() -> LocalProvider:
    return LocalProvider(
        model="m",
        base_url="http://127.0.0.1:1234/v1",
        api_key="x",
        id_slot=0,
    )


def test_orchestrator_provider_pins_slot_0():
    """The orchestrator provider keeps id_slot=0 (warm prefix)."""
    orch = _orchestrator_provider()
    assert orch.id_slot == 0


def test_with_id_slot_none_produces_unpinned_subagent_provider():
    """A sub-agent clone reports id_slot=None so it does not collide on slot 0,
    while the orchestrator's own provider still uses slot 0."""
    orch = _orchestrator_provider()
    sub = orch.with_id_slot(None)

    assert sub.id_slot is None
    assert orch.id_slot == 0  # unchanged: orchestrator keeps its warm slot
    assert sub is not orch
    # Config is shared/copied so the sub-agent talks to the same server.
    assert sub.model == orch.model
    assert sub.base_url == orch.base_url


def test_with_id_slot_none_omits_id_slot_from_request():
    """When id_slot is None the provider must NOT send an id_slot key, so a
    sub-agent request cannot evict the orchestrator's slot-0 cache."""
    sub = _orchestrator_provider().with_id_slot(None)
    captured: dict = {}

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return iter(())  # empty stream

    sub._client = _FakeClient()
    list(sub.stream_chat([], None))

    extra_body = captured.get("extra_body", {})
    assert "id_slot" not in extra_body


def test_pinned_provider_sends_id_slot_0():
    """Sanity: the orchestrator (slot 0) DOES send id_slot=0."""
    orch = _orchestrator_provider()
    captured: dict = {}

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return iter(())

    orch._client = _FakeClient()
    list(orch.stream_chat([], None))

    assert captured.get("extra_body", {}).get("id_slot") == 0


def test_orchestration_uses_with_id_slot_when_available(monkeypatch):
    """make_spawn_agent_tool must hand the spawned sub-agent a provider produced
    via with_id_slot(None), not the orchestrator's slot-0 provider itself."""
    import llmcli.orchestration as orch_mod

    orch = _orchestrator_provider()

    seen: dict = {}

    class _FakeAgent:
        def __init__(self, provider=None, **kwargs):
            seen["provider"] = provider

        def run(self, task):
            return "done"

    monkeypatch.setattr(orch_mod, "Agent", _FakeAgent)

    spawn = orch_mod.make_spawn_agent_tool(orch, auto_confirm=True)
    result = spawn.fn({"role": "explorer", "task": "look around"})

    assert result["ok"] is True
    sub_provider = seen["provider"]
    assert sub_provider.id_slot is None       # sub-agent unpinned
    assert sub_provider is not orch           # a distinct clone
    assert orch.id_slot == 0                   # orchestrator untouched


def test_orchestration_falls_back_when_provider_lacks_with_id_slot(monkeypatch):
    """A provider without with_id_slot (e.g. MockProvider) is shared unchanged."""
    import llmcli.orchestration as orch_mod
    from llmcli.providers import MockProvider

    mock = MockProvider(scenario="hello")
    assert not hasattr(mock, "with_id_slot")

    seen: dict = {}

    class _FakeAgent:
        def __init__(self, provider=None, **kwargs):
            seen["provider"] = provider

        def run(self, task):
            return "done"

    monkeypatch.setattr(orch_mod, "Agent", _FakeAgent)

    spawn = orch_mod.make_spawn_agent_tool(mock, auto_confirm=True)
    result = spawn.fn({"role": "explorer", "task": "hi"})

    assert result["ok"] is True
    assert seen["provider"] is mock  # shared unchanged, no crash


# --------------------------------------------------------------------------
# Fix 2: extract_text_tool_calls fast-path (bare fence, no sigil)
# --------------------------------------------------------------------------

def test_plain_python_fence_answer_returns_no_calls():
    """A normal answer containing a ```python fence (NO tool-call sigil) yields
    no tool calls."""
    text = (
        "Sure, here is an example:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "That defines a simple function."
    )
    assert extract_text_tool_calls(text) is None


def test_bare_fence_skips_heavy_parse(monkeypatch):
    """A bare ```-fenced answer with no sigil must short-circuit BEFORE the
    expensive fence/inline sweeps and the four parser passes run."""
    calls_made: list[str] = []

    def _spy(name, fn):
        def wrapper(*a, **k):
            calls_made.append(name)
            return fn(*a, **k)
        return wrapper

    # Spy on the expensive helpers the fast-path is supposed to skip.
    monkeypatch.setattr(
        providers, "_outer_fence_spans",
        _spy("_outer_fence_spans", providers._outer_fence_spans),
    )
    monkeypatch.setattr(
        providers, "_inline_code_spans",
        _spy("_inline_code_spans", providers._inline_code_spans),
    )
    monkeypatch.setattr(
        providers, "_parse_qwen",
        _spy("_parse_qwen", providers._parse_qwen),
    )
    monkeypatch.setattr(
        providers, "_parse_llama",
        _spy("_parse_llama", providers._parse_llama),
    )

    text = "Here is code:\n```js\nconsole.log(1)\n```\nDone."
    assert extract_text_tool_calls(text) is None
    assert calls_made == []  # none of the heavy helpers ran


# --------------------------------------------------------------------------
# Fix 2 (guard): every real sigil format STILL parses correctly
# --------------------------------------------------------------------------

def test_real_deepseek_call_still_parses():
    ds_begin = providers._DS_CALL_BEGIN
    ds_end = providers._DS_CALL_END
    ds_sep = providers._DS_SEP
    text = (
        f"{ds_begin}function{ds_sep}read_file\n"
        "```json\n"
        '{"path": "main.py"}\n'
        "```"
        f"{ds_end}"
    )
    calls = extract_text_tool_calls(text)
    assert calls and calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["path"] == "main.py"


def test_real_qwen_call_still_parses():
    text = '<tool_call>{"name": "glob", "arguments": {"pattern": "*.py"}}</tool_call>'
    calls = extract_text_tool_calls(text)
    assert calls and calls[0]["name"] == "glob"
    assert calls[0]["arguments"]["pattern"] == "*.py"


def test_real_qwen_xml_function_call_still_parses():
    text = '<function=read_file>{"path": "main.py"}</function>'
    calls = extract_text_tool_calls(text)
    assert calls and calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["path"] == "main.py"


def test_real_mistral_call_still_parses():
    text = '[TOOL_CALLS] [{"name": "grep", "arguments": {"pattern": "foo"}}]'
    calls = extract_text_tool_calls(text)
    assert calls and calls[0]["name"] == "grep"
    assert calls[0]["arguments"]["pattern"] == "foo"


def test_real_llama_python_tag_call_still_parses():
    tag = providers._LLAMA_PY_TAG
    text = tag + '{"name": "read_file", "parameters": {"path": "a.py"}}'
    calls = extract_text_tool_calls(text)
    assert calls and calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["path"] == "a.py"
