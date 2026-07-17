"""Integration tests for the 5 wired agent-loop features:

  1. Permission modes (read-only/plan, auto-edit, full-auto, default).
  3. Checkpoint-before-write.
  4. Pre-tool hook veto gate.
  5. Cooperative interrupt via cancel_event.
  6. Model-callable todo_write tool.

All driven by the offline MockProvider (no network). Every test also implicitly
asserts the behavior-preserving defaults: an Agent built without the new knobs
never touches checkpoints/hooks/cancel and keeps the historic confirm gate.
"""

from __future__ import annotations

import threading
from pathlib import Path

import llmcli.checkpoint as checkpoint_mod
import llmcli.tools as tools_mod
from llmcli.agent import Agent
from llmcli.providers import MockProvider
from llmcli.tools import FULL


# --------------------------------------------------------------------------- #
# Scripted providers
# --------------------------------------------------------------------------- #
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


class _EditThenBash(MockProvider):
    """Emit an edit_file + run_bash batch, then finalize after tool results."""

    def stream_chat(self, messages, tools, tool_choice=None):
        acted = any(m.get("role") == "tool" for m in messages)
        if not acted:
            yield {"type": "tool_call", "id": "e", "name": "edit_file",
                   "arguments": {"path": "f.py", "old": "x = 1", "new": "x = 2"}}
            yield {"type": "tool_call", "id": "b", "name": "run_bash",
                   "arguments": {"command": "echo hi"}}
            yield {"type": "done", "finish_reason": "tool_calls"}
        else:
            yield {"type": "text", "text": "edited."}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


class _CancelMidStream(MockProvider):
    """Yield partial text, set the cancel event, then keep streaming. The agent
    must stop after the event fires and finalize the partial answer."""

    def __init__(self, event):
        super().__init__()
        self.event = event
        self.calls = 0

    def stream_chat(self, messages, tools, tool_choice=None):
        self.calls += 1
        yield {"type": "text", "text": "partial answer"}
        self.event.set()  # cancel now, BEFORE the stream completes
        yield {"type": "text", "text": " (should be dropped)"}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


class _ConfirmSpy:
    """A confirm_fn stub that records which tools it was asked to confirm."""

    def __init__(self, answer: bool = True):
        self.calls: list[str] = []
        self.answer = answer

    def __call__(self, tool, args) -> bool:
        self.calls.append(tool.name)
        return self.answer


# --------------------------------------------------------------------------- #
# Feature 1 + 2: permission modes
# --------------------------------------------------------------------------- #
def test_read_only_mode_blocks_write_without_executing(tmp_workspace):
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, permission_mode="read-only", max_iterations=6,
    )
    out = agent.run("write a file")

    # The write NEVER executed — no file on disk.
    assert not Path("f.py").exists()
    # The tool result carries the block error (same {"ok":False,...} shape).
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs, "the blocked call still appends a tool result"
    assert "read-only mode" in tool_msgs[0]["content"]
    assert "disabled" in tool_msgs[0]["content"]
    # The loop still finished on the model's follow-up text.
    assert "all done" in out


def test_plan_mode_blocks_run_bash_but_allows_reads(tmp_workspace):
    from llmcli.agent import Agent as _A

    agent = _A(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
        permission_mode="plan",
    )
    # A read tool is allowed; a mutating tool is blocked.
    read_tool = agent.registry["read_file"]
    bash_tool = agent.registry["run_bash"]
    assert agent._permission_decision(read_tool, {"path": "x"})[0] == "allow"
    d, reason = agent._permission_decision(bash_tool, {"command": "ls"})
    assert d == "block"
    assert "plan mode" in reason and "run_bash" in reason


def test_auto_edit_skips_confirm_for_edit_but_confirms_run_bash(tmp_workspace):
    Path("f.py").write_text("x = 1\n")
    spy = _ConfirmSpy(answer=True)
    agent = Agent(
        provider=_EditThenBash(), system_prompt="s", tool_names=FULL,
        auto_confirm=False, confirm_fn=spy, permission_mode="auto-edit",
        max_iterations=6,
    )
    out = agent.run("edit then test")

    # edit_file was auto-approved (no confirm); run_bash STILL required a confirm.
    assert spy.calls == ["run_bash"]
    assert "edited" in out


def test_full_auto_executes_without_any_confirm(tmp_workspace):
    spy = _ConfirmSpy(answer=False)  # would DECLINE if ever asked
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=False, confirm_fn=spy, permission_mode="full-auto",
        max_iterations=6,
    )
    out = agent.run("write a file")

    # Confirm was never consulted, yet the write executed.
    assert spy.calls == []
    assert Path("f.py").exists()
    assert "all done" in out


def test_default_mode_declined_write_is_unchanged(tmp_workspace):
    spy = _ConfirmSpy(answer=False)
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=False, confirm_fn=spy, max_iterations=6,
    )
    agent.run("write a file")

    # Historic behavior: confirm_fn was consulted, returned False -> declined,
    # nothing written, and the SAME declined error is fed back.
    assert spy.calls == ["write_file"]
    assert not Path("f.py").exists()
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("User declined this tool call." in m["content"] for m in tool_msgs)


def test_default_mode_auto_confirm_still_executes(tmp_workspace):
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, max_iterations=6,
    )
    out = agent.run("write a file")
    assert Path("f.py").exists()
    assert "all done" in out


# --------------------------------------------------------------------------- #
# Feature 3: checkpoint-before-write
# --------------------------------------------------------------------------- #
def test_checkpoint_snapshot_called_before_write_when_enabled(tmp_workspace, monkeypatch):
    calls: list[dict] = []

    def _spy(paths, *, root, label="", session=None):
        calls.append({
            "paths": list(paths), "root": root, "label": label, "session": session,
        })
        return "ck-test"

    monkeypatch.setattr(checkpoint_mod, "snapshot", _spy)
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, checkpoints_enabled=True,
        workspace_root=str(tmp_workspace), max_iterations=6,
    )
    agent.run("write a file")

    assert Path("f.py").exists()
    assert len(calls) == 1
    assert calls[0]["paths"] == ["f.py"]
    assert calls[0]["label"] == "write_file"


def test_checkpoint_not_called_when_disabled(tmp_workspace, monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        checkpoint_mod, "snapshot",
        lambda *a, **k: calls.append((a, k)) or "ck",
    )
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, checkpoints_enabled=False, max_iterations=6,
    )
    agent.run("write a file")

    assert Path("f.py").exists()
    assert calls == []


# --------------------------------------------------------------------------- #
# Feature 4: pre-tool hook veto gate
# --------------------------------------------------------------------------- #
def test_blocking_pre_tool_hook_prevents_execution(tmp_workspace):
    hooks = {
        "PreToolUse": [
            {"match": "write_file", "command": "echo nope >&2; exit 1", "timeout": 5}
        ]
    }
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, hooks=hooks, workspace_root=str(tmp_workspace),
        max_iterations=6,
    )
    out = agent.run("write a file")

    # The hook vetoed -> the write never ran and the block error was fed back.
    assert not Path("f.py").exists()
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("Blocked by pre-tool hook" in m["content"] for m in tool_msgs)
    assert "all done" in out


def test_no_hooks_is_a_noop(tmp_workspace):
    agent = Agent(
        provider=_WriteThenDone(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, max_iterations=6,
    )
    agent.run("write a file")
    assert Path("f.py").exists()
    assert agent.hooks is None


# --------------------------------------------------------------------------- #
# Feature 5: cooperative interrupt via cancel_event
# --------------------------------------------------------------------------- #
def test_cancel_event_mid_stream_finalizes_cleanly(tmp_workspace):
    event = threading.Event()
    provider = _CancelMidStream(event)
    agent = Agent(
        provider=provider, system_prompt="s", tool_names=FULL,
        auto_confirm=True, cancel_event=event, max_iterations=8,
    )
    out = agent.run("do a long thing")

    # Exactly one stream ran, then the loop stopped.
    assert provider.calls == 1
    assert "[interrupted]" in out
    assert "partial answer" in out
    # The dropped post-cancel chunk never made it into the answer.
    assert "should be dropped" not in out
    # History is well-formed: last message is a plain assistant turn, no dangling
    # tool_calls awaiting results.
    last = agent.messages[-1]
    assert last["role"] == "assistant"
    assert "tool_calls" not in last


def test_cancel_event_before_loop_returns_interrupted(tmp_workspace):
    event = threading.Event()
    event.set()  # already cancelled before the first iteration
    provider = _CancelMidStream(event)
    agent = Agent(
        provider=provider, system_prompt="s", tool_names=FULL,
        auto_confirm=True, cancel_event=event, max_iterations=8,
    )
    out = agent.run("go")
    # The provider was never even called; we bailed at the top of the loop.
    assert provider.calls == 0
    assert out == "[interrupted]"


# --------------------------------------------------------------------------- #
# Feature 6: model-callable todo_write
# --------------------------------------------------------------------------- #
def test_todo_write_updates_state_and_appears_in_schema(tmp_workspace):
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
        todos_enabled=True,
    )
    # In the schema built from the (per-instance) registry.
    names = {t["function"]["name"] for t in agent._tools_payload()}
    assert "todo_write" in names

    tool = agent.registry["todo_write"]
    res = tool.fn({"items": [
        {"text": "design", "status": "done"},
        {"text": "build", "status": "in_progress"},
        "verify",  # plain string tolerated -> pending
    ]})
    assert res == {"ok": True, "result": {"count": 3}}
    assert agent._todos == [
        {"text": "design", "status": "done"},
        {"text": "build", "status": "in_progress"},
        {"text": "verify", "status": "pending"},
    ]
    # A bad status is coerced to pending; a bare list is also accepted.
    res2 = tool.fn([{"text": "x", "status": "bogus"}])
    assert res2["result"]["count"] == 1
    assert agent._todos == [{"text": "x", "status": "pending"}]


def test_todo_write_absent_and_registry_unmutated_by_default(tmp_workspace):
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=FULL,
    )
    names = {t["function"]["name"] for t in agent._tools_payload()}
    assert "todo_write" not in names
    # The default agent shares the global registry object (never copied/mutated).
    assert agent.registry is tools_mod.REGISTRY
    assert "todo_write" not in tools_mod.REGISTRY
