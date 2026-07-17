"""Agent-loop tests for the four quality features:

  Feature 2: constrained-decode retry on a malformed tool call.
  Feature 3: auto-verify after edit.
  Feature 4: reviewer gate on writes.

All driven by the offline MockProvider (no network).
"""

from __future__ import annotations

from pathlib import Path

import llmcli.tools as tools_mod
from llmcli.agent import Agent
from llmcli.providers import MockProvider
from llmcli.tools import FULL


# ===========================================================================
# Feature 2: constrained-decode retry
# ===========================================================================

class _MalformedThenForced(MockProvider):
    """Malformed tool call on a normal request; a CLEAN one when forced with
    tool_choice='required'. After a tool result exists, it finalizes."""

    def __init__(self):
        super().__init__()
        self.tool_choices: list = []

    def stream_chat(self, messages, tools, tool_choice=None):
        self.tool_choices.append(tool_choice)
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if not has_tool_result:
            if tool_choice == "required":
                yield {"type": "tool_call", "id": "x", "name": "glob",
                       "arguments": {"pattern": "*"}}
            else:
                yield {"type": "tool_call", "id": "x", "name": "glob",
                       "arguments": {}, "_parse_error": "Expecting value"}
            yield {"type": "done", "finish_reason": "tool_calls"}
        else:
            yield {"type": "text", "text": "done."}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


def test_constrained_retry_recovers_malformed_call(tmp_workspace):
    provider = _MalformedThenForced()
    agent = Agent(
        provider=provider, system_prompt="s", tool_names=FULL,
        auto_confirm=True, constrained_retry=True, max_iterations=6,
    )
    out = agent.run("go")

    # Exactly ONE forced retry happened (tool_choice='required' sent once).
    assert provider.tool_choices.count("required") == 1
    # The retry recovered: glob executed and the loop finished cleanly.
    assert "done" in out
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    # No corrective "Could not parse" message — the clean retry replaced it.
    assert not any("Could not parse" in m["content"] for m in tool_msgs)


class _AlwaysMalformed(MockProvider):
    """Malformed tool call on EVERY request, even when forced -> the retry can't
    help, so the loop must fall back to the corrective-text behavior."""

    def __init__(self):
        super().__init__()
        self.tool_choices: list = []

    def stream_chat(self, messages, tools, tool_choice=None):
        self.tool_choices.append(tool_choice)
        yield {"type": "tool_call", "id": "x", "name": "glob",
               "arguments": {}, "_parse_error": "Expecting value"}
        yield {"type": "done", "finish_reason": "tool_calls"}


def test_constrained_retry_falls_back_when_retry_also_fails(tmp_workspace):
    provider = _AlwaysMalformed()
    agent = Agent(
        provider=provider, system_prompt="s", tool_names=FULL,
        auto_confirm=True, constrained_retry=True, max_iterations=12,
    )
    out = agent.run("go")

    # The forced retry was attempted once per parse-failure round (3 rounds before
    # the circuit-breaker aborts) — bounded to ONE retry per failure, never a loop.
    assert provider.tool_choices.count("required") == 3
    # Fallback still works: the corrective "Could not parse" message was fed back.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("Could not parse" in m["content"] for m in tool_msgs)
    # And the existing circuit-breaker still aborts after 3 parse-only rounds.
    assert "invalid tool-call JSON" in out


def test_no_retry_when_constrained_retry_disabled(tmp_workspace):
    provider = _AlwaysMalformed()
    agent = Agent(
        provider=provider, system_prompt="s", tool_names=FULL,
        auto_confirm=True, constrained_retry=False, max_iterations=12,
    )
    agent.run("go")
    # Disabled -> the forced retry is NEVER attempted.
    assert provider.tool_choices.count("required") == 0


# ===========================================================================
# Feature 3: auto-verify after edit
# ===========================================================================

class _WriteThenDone(MockProvider):
    """Writes a file (no run_bash), then finalizes once an assistant tool_calls
    message exists in history."""

    def stream_chat(self, messages, tools, tool_choice=None):
        wrote = any(
            m.get("role") == "assistant" and m.get("tool_calls") for m in messages
        )
        if not wrote:
            yield {"type": "tool_call", "id": "w", "name": "write_file",
                   "arguments": {"path": "f.py", "content": "x = 1\n",
                                 "overwrite": True}}
            yield {"type": "done", "finish_reason": "tool_calls"}
        else:
            yield {"type": "text", "text": "all done."}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


def test_auto_verify_runs_once_and_feeds_back(tmp_workspace):
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, verify_cmd="echo VERIFY_RAN_OK", max_iterations=8,
    )
    out = agent.run("write a file")

    assert Path("f.py").exists()
    # The verify command's output was fed back exactly once as an ephemeral msg.
    fed = [m for m in agent.messages
           if m.get("_nudge") and "auto-ran your verification" in str(m.get("content"))]
    assert len(fed) == 1
    assert "VERIFY_RAN_OK" in fed[0]["content"]
    assert agent._auto_verified is True
    # The build-nudge prose is NOT used when a verify_cmd auto-runs.
    assert not any("never ran the project's tests" in str(m.get("content"))
                   for m in agent.messages)
    assert "all done" in out


def test_no_verify_cmd_keeps_prose_build_nudge(tmp_workspace):
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, verify_cmd="", max_iterations=8,
    )
    agent.run("write a file")
    # Old behavior unchanged: the prose build-nudge fires, no auto-run feedback.
    assert any("never ran the project's tests" in str(m.get("content"))
               for m in agent.messages)
    assert not any("auto-ran your verification" in str(m.get("content"))
                   for m in agent.messages)
    assert agent._auto_verified is False


# ===========================================================================
# Feature 4: reviewer gate on writes
# ===========================================================================

def _spawn_recorder():
    """A stub spawn_agent Tool that records its calls and returns a finding."""
    calls: list = []

    def _fn(args):
        calls.append(dict(args))
        return {"ok": True, "result": "Looks good. No issues found."}

    tool = tools_mod.Tool(
        name="spawn_agent",
        description="stub",
        parameters={"type": "object", "properties": {}},
        fn=_fn,
        requires_confirmation=True,
    )
    return tool, calls


def test_reviewer_gate_triggers_once_on_code_write(tmp_workspace):
    spawn, calls = _spawn_recorder()
    registry = dict(tools_mod.REGISTRY)
    registry["spawn_agent"] = spawn
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s",
        tool_names=FULL + ["spawn_agent"], auto_confirm=True,
        registry=registry, review_writes=True, max_iterations=10,
    )
    out = agent.run("write a file")

    assert Path("f.py").exists()
    # Exactly ONE reviewer pass, on the reviewer role, naming the changed file.
    assert len(calls) == 1
    assert calls[0]["role"] == "reviewer"
    assert "f.py" in calls[0]["task"]
    # The findings were fed back into the loop before finalizing.
    assert any("A code reviewer examined your changes" in str(m.get("content"))
               for m in agent.messages)
    assert agent._reviewed_this_turn is True
    assert "all done" in out


def test_reviewer_gate_disabled_by_flag(tmp_workspace):
    spawn, calls = _spawn_recorder()
    registry = dict(tools_mod.REGISTRY)
    registry["spawn_agent"] = spawn
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s",
        tool_names=FULL + ["spawn_agent"], auto_confirm=True,
        registry=registry, review_writes=False, max_iterations=10,
    )
    agent.run("write a file")
    assert calls == []  # review_writes=False -> no reviewer spawned


def test_subagent_without_spawn_tool_never_reviews(tmp_workspace):
    # A sub-agent's registry has no spawn_agent, so the gate can never fire even
    # with review_writes=True -> no recursion.
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, review_writes=True, max_iterations=8,
    )
    agent.run("write a file")
    assert Path("f.py").exists()
    assert agent._reviewed_this_turn is False
