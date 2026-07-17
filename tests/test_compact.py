"""Tests for Agent.compact: history reduction, no-op on failure, short-circuit."""

from __future__ import annotations

import pytest

from llmcli.agent import Agent
from llmcli.providers import MockProvider
from llmcli.tools import FULL


def _summarizing_provider(summary_text: str):
    """A provider whose stream_chat yields a fixed summary then done.

    Used to make compact() deterministic without a network call.
    """

    class _P(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": summary_text}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}

    return _P()


def test_compact_reduces_message_count_and_length():
    agent = Agent(
        provider=_summarizing_provider("- did X\n- touched a.py"),
        system_prompt="SYS",
        tool_names=FULL,
    )
    # Build a long history: system + several user/assistant turns.
    agent.messages += [
        {"role": "user", "content": "first task " * 50},
        {"role": "assistant", "content": "long reply " * 50},
        {"role": "user", "content": "second task " * 50},
        {"role": "assistant", "content": "another long reply " * 50},
        {"role": "user", "content": "third task " * 50},
        {"role": "assistant", "content": "final long reply " * 50},
    ]
    n_before = len(agent.messages)
    before, after = agent.compact()

    # Fewer messages and fewer estimated tokens than before.
    assert len(agent.messages) < n_before
    assert after < before
    # The system prompt is preserved as message[0].
    assert agent.messages[0] == {"role": "system", "content": "SYS"}
    # A summary system note is injected.
    assert any(
        m["role"] == "system" and "Summary of earlier conversation" in str(m["content"])
        for m in agent.messages
    )
    # The last 1-2 user turns are kept (keep-from = 2nd-to-last user message).
    tail_contents = " ".join(str(m.get("content") or "") for m in agent.messages[2:])
    assert "second task" in tail_contents
    assert "third task" in tail_contents
    # ...and the earlier first turn was summarized away (not in the kept tail).
    assert "first task" not in tail_contents


def test_compact_raising_provider_becomes_runtimeerror():
    """A provider that RAISES (not yields an error event) -> uniform RuntimeError."""

    class _Raise(MockProvider):
        def stream_chat(self, messages, tools):
            raise ConnectionError("network down")
            yield  # pragma: no cover - make this a generator

    agent = Agent(provider=_Raise(), system_prompt="SYS", tool_names=FULL)
    agent.messages += [
        {"role": "user", "content": "a " * 40},
        {"role": "assistant", "content": "b " * 40},
        {"role": "user", "content": "c " * 40},
        {"role": "assistant", "content": "d " * 40},
        {"role": "user", "content": "e " * 40},
        {"role": "assistant", "content": "f " * 40},
    ]
    snapshot = list(agent.messages)
    with pytest.raises(RuntimeError):
        agent.compact()
    assert agent.messages == snapshot  # history untouched


def test_compact_two_user_turns_never_grows():
    """Regression: exactly 2 user turns must never GROW history/tokens.

    With 2 user messages keep_from points at the FIRST user turn, so the range
    to summarize is empty. compact() must short-circuit (no provider call, no
    empty summary note injected) and be a strict no-op, not add a message.
    """
    calls = {"n": 0}

    class _Counting(MockProvider):
        def stream_chat(self, messages, tools):
            calls["n"] += 1
            yield {"type": "text", "text": "should not be called"}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}

    agent = Agent(provider=_Counting(), system_prompt="SYS", tool_names=FULL)
    agent.messages += [
        {"role": "user", "content": "first task " * 50},
        {"role": "assistant", "content": "first reply " * 50},
        {"role": "user", "content": "second task " * 50},
        {"role": "assistant", "content": "second reply " * 50},
    ]  # system + 2 user turns
    n_before = len(agent.messages)
    before, after = agent.compact()

    # Strict no-op: never grows, provider never invoked.
    assert len(agent.messages) <= n_before
    assert after <= before
    assert calls["n"] == 0


def test_compact_short_circuits_when_history_small():
    agent = Agent(provider=_summarizing_provider("x"), system_prompt="SYS", tool_names=FULL)
    agent.messages += [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]  # system + 2 = 3 messages -> short-circuit
    before, after = agent.compact()
    assert before == after
    assert len(agent.messages) == 3  # untouched


def test_compact_noop_on_provider_failure_keeps_history():
    class _Boom(MockProvider):
        def stream_chat(self, messages, tools):
            # Empty summary -> compact() must raise so caller no-ops.
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 0}

    agent = Agent(provider=_Boom(), system_prompt="SYS", tool_names=FULL)
    # 3 user turns so there is a non-empty range to summarize (the provider is
    # actually reached; with only 2 turns compact() short-circuits as a no-op).
    agent.messages += [
        {"role": "user", "content": "a " * 40},
        {"role": "assistant", "content": "b " * 40},
        {"role": "user", "content": "c " * 40},
        {"role": "assistant", "content": "d " * 40},
        {"role": "user", "content": "e " * 40},
        {"role": "assistant", "content": "f " * 40},
    ]
    snapshot = list(agent.messages)
    with pytest.raises(RuntimeError):
        agent.compact()
    # History must be UNCHANGED (built locally, assigned only on success).
    assert agent.messages == snapshot


def test_compact_includes_tool_calls_in_transcript():
    """finding #33: an assistant tool_calls message (content=None) must still be
    serialized into the summarizer transcript, so engineering history isn't lost.
    The provider here ECHOES the transcript it receives as the summary."""

    class _Echo(MockProvider):
        def stream_chat(self, messages, tools):
            # messages[1] is the user transcript built by compact().
            transcript = str(messages[1]["content"])
            yield {"type": "text", "text": transcript}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}

    agent = Agent(provider=_Echo(), system_prompt="SYS", tool_names=FULL)
    agent.messages += [
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "run_bash", "arguments": '{"command": "pytest"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "run_bash", "content": "ok"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "reply3"},
    ]
    agent.compact()
    summary_note = next(
        m["content"] for m in agent.messages
        if m["role"] == "system" and "Summary of earlier conversation" in str(m["content"])
    )
    # The serialized tool_calls (name + args) made it into the summarized text.
    assert "run_bash" in summary_note
    assert "pytest" in summary_note


def test_compact_noop_on_provider_error_event():
    class _Err(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "[provider error: nope]"}
            yield {"type": "done", "finish_reason": "error", "output_tokens": None}

    agent = Agent(provider=_Err(), system_prompt="SYS", tool_names=FULL)
    # 3 user turns so the provider is actually reached (see note above).
    agent.messages += [
        {"role": "user", "content": "a " * 40},
        {"role": "assistant", "content": "b " * 40},
        {"role": "user", "content": "c " * 40},
        {"role": "assistant", "content": "d " * 40},
        {"role": "user", "content": "e " * 40},
        {"role": "assistant", "content": "f " * 40},
    ]
    snapshot = list(agent.messages)
    with pytest.raises(RuntimeError):
        agent.compact()
    assert agent.messages == snapshot
