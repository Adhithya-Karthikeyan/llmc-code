"""Core agent-loop tests, all driven by the offline MockProvider."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from llmcode.agent import Agent
from llmcode.providers import MockProvider
from llmcode.tools import FULL, get_tool


def test_agent_executes_tool_and_feeds_result(tmp_workspace):
    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
    )
    final = agent.run("create hello.py that prints hi and run it")

    # The mock 'hello' scenario writes hello.py then runs it.
    assert Path("hello.py").exists()
    assert "print('hi')" in Path("hello.py").read_text()
    assert "Done" in final

    # The conversation must contain a tool result message fed back to the model.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) >= 2
    # Each tool message must carry a matching tool_call_id.
    for m in tool_msgs:
        assert m.get("tool_call_id")
        assert m.get("name")


def test_assistant_tool_call_message_shape(tmp_workspace):
    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
    )
    agent.run("hello")
    assistant_with_calls = [
        m for m in agent.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert assistant_with_calls
    tc = assistant_with_calls[0]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"]
    assert isinstance(tc["function"]["arguments"], str)  # JSON-serialized


def test_confirmation_declined_blocks_tool(tmp_workspace):
    calls = {"n": 0}

    def deny(tool, args):
        calls["n"] += 1
        return False

    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=False,
        confirm_fn=deny,
    )
    agent.run("hello")

    # write_file was gated and declined -> file must NOT exist.
    assert not Path("hello.py").exists()
    assert calls["n"] >= 1

    # The declined result must be fed back so the model can adapt.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("declined" in m["content"].lower() for m in tool_msgs)


def test_auto_confirm_runs(tmp_workspace):
    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
        confirm_fn=lambda t, a: (_ for _ in ()).throw(AssertionError("should not prompt")),
    )
    agent.run("hello")
    assert Path("hello.py").exists()


def test_max_iterations_guard(tmp_workspace):
    # A provider that always requests a tool call -> loop must terminate.
    class LoopProvider(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "tool_call", "id": "x", "name": "glob",
                   "arguments": {"pattern": "*"}}
            yield {"type": "done", "finish_reason": "tool_calls"}

    agent = Agent(
        provider=LoopProvider(),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
        max_iterations=3,
    )
    final = agent.run("loop forever")
    assert "max" in final.lower()


def test_duplicate_tool_call_loop_guard_trips(tmp_workspace):
    """A model stuck re-issuing the SAME tool call must break early (before the
    iteration cap), surfacing the duplicate-loop message — not spin to the cap."""
    class SameCallForever(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "tool_call", "id": "x", "name": "glob",
                   "arguments": {"pattern": "*.py"}}
            yield {"type": "done", "finish_reason": "tool_calls"}

    agent = Agent(
        provider=SameCallForever(),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
        max_iterations=50,
    )
    final = agent.run("loop the same call forever")
    # The dup-loop message, NOT the iteration-cap message.
    assert "repeated" in final.lower() or "identical" in final.lower()
    assert "max" not in final.lower()


def test_progressing_calls_do_not_trip_dup_guard(tmp_workspace):
    """A model whose call ARGS change each round (e.g. read_file advancing its
    offset) is making progress and must NOT trip the duplicate-loop guard."""
    class ProgressingRead(MockProvider):
        def __init__(self):
            super().__init__()
            self.round = 0

        def stream_chat(self, messages, tools):
            self.round += 1
            yield {"type": "tool_call", "id": "x", "name": "glob",
                   "arguments": {"pattern": f"*.py", "path": f"dir{self.round}"}}
            yield {"type": "done", "finish_reason": "tool_calls"}

    agent = Agent(
        provider=ProgressingRead(),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
        max_iterations=8,
    )
    final = agent.run("read the whole file")
    # Progressing args reset the count -> it runs to the cap, NOT the dup break.
    assert "max" in final.lower()
    assert "repeated" not in final.lower() and "identical" not in final.lower()


def test_last_turn_details_populated_with_full_detail(tmp_workspace):
    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
    )
    agent.run("hello")

    # The 'hello' scenario runs write_file then run_bash -> 2 detail records,
    # each carrying the FULL args and the FULL result (not truncated).
    names = [r["name"] for r in agent.last_turn_details]
    assert names == ["write_file", "run_bash"]
    write_rec = agent.last_turn_details[0]
    assert write_rec["args"]["path"] == "hello.py"
    assert write_rec["args"]["content"] == "print('hi')\n"  # full, untruncated
    assert write_rec["ok"] is True
    assert isinstance(write_rec["result"], dict)
    assert "elapsed" in write_rec


def test_last_turn_details_reset_each_run(tmp_workspace):
    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
    )
    agent.run("hello")
    assert len(agent.last_turn_details) == 2
    # A second run resets the buffer at the start (mock 'hello' is stateless and
    # re-runs the same 2-tool script for the new user turn).
    agent.run("hello again")
    assert len(agent.last_turn_details) == 2  # reset, not accumulated to 4


def test_render_details_runs_without_error(tmp_workspace, capsys):
    from rich.console import Console

    console = Console(markup=False)
    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
        console=console,  # use the captured console so render_details output is captured
    )
    agent.run("hello")
    capsys.readouterr()  # discard run() output
    agent.render_details(console)  # must not raise
    out = capsys.readouterr().out
    # Ctrl+O reveal prints the ⏺/⎿ tree: friendly display names and short summaries.
    # "Write(hello.py)" is the head line; "Wrote" is the result summary line.
    assert "Write(hello.py)" in out
    assert "⏺" in out and "⎿" in out
    # Full detail is derivable: the path appears in the head line.
    assert "hello.py" in out
    # run_bash result summary contains "hi" (stdout first line) or "exit".
    assert "Bash(" in out

    # Empty buffer path also renders cleanly.
    fresh = Agent(provider=MockProvider(), system_prompt="s", tool_names=FULL)
    fresh.render_details(console)
    out2 = capsys.readouterr().out
    assert "no tool activity" in out2


def test_footer_uses_gen_elapsed_when_present(tmp_workspace, capsys):
    """The done event's gen_elapsed must drive the tok/s, not the wall clock."""
    from rich.console import Console

    class GenElapsedProvider(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "hi there"}
            # 2 tokens over exactly 0.5s -> 4.0 tok/s regardless of wall time.
            yield {
                "type": "done", "finish_reason": "stop",
                "output_tokens": 2, "gen_elapsed": 0.5,
            }

    agent = Agent(
        provider=GenElapsedProvider(), system_prompt="s", tool_names=[],
        console=Console(markup=False),
    )
    agent.run("go")
    assert "4.0 tok/s" in capsys.readouterr().out


def test_empty_reasoning_only_turn_returns_notice(tmp_workspace):
    """A 'done' with no text and no tool_calls must surface a visible notice."""
    class EmptyProvider(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 0}

    agent = Agent(provider=EmptyProvider(), system_prompt="s", tool_names=[])
    out = agent.run("hi")
    assert "no answer produced" in out


def test_length_finish_reason_returns_text(tmp_workspace, capsys):
    from rich.console import Console

    class LenProvider(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "partial"}
            yield {"type": "done", "finish_reason": "length", "output_tokens": 1}

    agent = Agent(
        provider=LenProvider(), system_prompt="s", tool_names=[],
        console=Console(markup=False),
    )
    out = agent.run("hi")
    # finding #6: the truncation marker is baked into the RETURNED text (so
    # one-shot/spawn_agent callers see it), not only printed to the console.
    assert out == "partial\n[output truncated at token limit]"
    assert "truncated" in capsys.readouterr().out


def test_repeated_parse_errors_abort_early(tmp_workspace):
    """3 consecutive malformed-JSON tool turns must abort, not burn the budget."""
    class BadJSONProvider(MockProvider):
        def stream_chat(self, messages, tools):
            yield {
                "type": "tool_call", "id": "x", "name": "glob",
                "arguments": {}, "_parse_error": "Expecting value",
            }
            yield {"type": "done", "finish_reason": "tool_calls"}

    agent = Agent(
        provider=BadJSONProvider(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, max_iterations=12,
    )
    out = agent.run("loop bad json")
    # Aborted after 3 attempts, not after consuming all 12 iterations.
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    assert "invalid tool-call JSON" in out


def test_subagent_tool_trees_prefixed_and_no_ctrl_o_hint(tmp_workspace, capsys):
    from rich.console import Console

    class TwoToolProvider(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "a", "name": "glob",
                       "arguments": {"pattern": "*"}}
                yield {"type": "tool_call", "id": "b", "name": "glob",
                       "arguments": {"pattern": "*.py"}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "done."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    # Sub-agent context: non-empty line_prefix is carried on the collapsed summary
    # line, and there is no Ctrl+O false affordance on sub-agents.
    sub = Agent(
        provider=TwoToolProvider(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, max_iterations=2, console=Console(markup=False),
        line_prefix="  ↳ ",
    )
    sub.run("two tools")
    out = capsys.readouterr().out
    # Sub-agent run() prints the compact "  ↳ ◆ 2 tools · ✓2" counts line with NO
    # ctrl-o hint and NO auto-expanded tree (both are orchestrator-only).
    assert "↳ ◆" in out
    assert "2 tools" in out
    assert "✓2" in out
    assert "Ctrl+O" not in out and "ctrl-o" not in out
    assert "⎿" not in out  # sub-agent never auto-expands into the parent buffer


def test_subagent_render_answer_false_does_not_double_print(capsys):
    """A spawned sub-agent (render_answer=False) RETURNS its answer to the
    orchestrator but does NOT print it, so a delegated turn shows the answer once
    (the orchestrator renders it), not twice. Top-level default still prints."""
    from rich.console import Console

    phrase = "Hello from the mock provider"

    sub = Agent(
        provider=MockProvider(scenario="plain"), system_prompt="s", tool_names=[],
        console=Console(markup=False), line_prefix="  ↳ ", render_answer=False,
    )
    ret = sub.run("hi")
    out = capsys.readouterr().out
    assert phrase in ret          # still returned to the orchestrator
    assert phrase not in out      # but NOT printed by the sub-agent (no dup)

    top = Agent(
        provider=MockProvider(scenario="plain"), system_prompt="s", tool_names=[],
        console=Console(markup=False),
    )
    ret2 = top.run("hi")
    out2 = capsys.readouterr().out
    assert phrase in ret2 and phrase in out2  # top-level renders as before


def test_max_iterations_one_executes_but_no_react(tmp_workspace):
    """max_iterations=1 runs the tool but the model never sees the result."""
    agent = Agent(
        provider=MockProvider(scenario="hello"), system_prompt="s",
        tool_names=FULL, auto_confirm=True, max_iterations=1,
    )
    out = agent.run("hello")
    # write_file ran (side effect happened) but the loop stopped with no final.
    assert Path("hello.py").exists()
    assert "stopped" in out.lower()


def test_unknown_tool_call_feeds_error(tmp_workspace):
    """finding #20: a tool_call for a nonexistent tool -> {'ok':False,'error':
    'Unknown tool: ...'} fed back, loop continues to a final answer."""
    class _UnknownThenText(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "does_not_exist",
                       "arguments": {}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "recovered."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    agent = Agent(provider=_UnknownThenText(), system_prompt="s",
                  tool_names=FULL, auto_confirm=True)
    out = agent.run("go")
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("Unknown tool" in m["content"] for m in tool_msgs)
    assert "recovered" in out


def test_unknown_tool_suggests_close_match(tmp_workspace):
    """A near-miss tool name yields a 'did you mean' suggestion of the closest
    real registered tool(s) so a weak model can self-correct."""
    class _TypoThenText(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                # "read_fil" is a near-miss for the real "read_file" tool.
                yield {"type": "tool_call", "id": "x", "name": "read_fil",
                       "arguments": {}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "ok."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    agent = Agent(provider=_TypoThenText(), system_prompt="s",
                  tool_names=FULL, auto_confirm=True)
    agent.run("go")
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    err = next(m["content"] for m in tool_msgs if "Unknown tool" in m["content"])
    assert "did you mean" in err
    assert "read_file" in err


def test_unknown_tool_no_suggestion_when_nothing_close(tmp_workspace):
    """A wildly different tool name has no near match -> plain 'Unknown tool'
    with no 'did you mean' clause."""
    class _GibberishThenText(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "zzzqqqxyzzy",
                       "arguments": {}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "ok."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    agent = Agent(provider=_GibberishThenText(), system_prompt="s",
                  tool_names=FULL, auto_confirm=True)
    agent.run("go")
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    err = next(m["content"] for m in tool_msgs if "Unknown tool" in m["content"])
    assert "did you mean" not in err


def test_tool_exception_is_caught(tmp_workspace, monkeypatch):
    """finding #20: a tool fn that RAISES is caught -> {'ok':False,'error':type}
    fed back, loop never dies."""
    import llmcode.tools as tools_mod

    boom = tools_mod.Tool(
        name="glob",  # reuse a known name so it's in the schema/registry
        description="x",
        parameters={"type": "object", "properties": {}},
        fn=lambda args: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    monkeypatch.setitem(tools_mod.REGISTRY, "glob", boom)

    class _CallThenText(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "glob",
                       "arguments": {"pattern": "*"}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "ok done."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    agent = Agent(provider=_CallThenText(), system_prompt="s",
                  tool_names=FULL, auto_confirm=True)
    out = agent.run("go")
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("RuntimeError" in m["content"] for m in tool_msgs)
    assert "done" in out


def test_narration_and_tool_call_same_turn(tmp_workspace):
    """finding #20: a turn with BOTH narration text AND a (non-fence) tool_call
    stores the narration as assistant content and still executes the tool."""
    class _NarrateThenCall(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "text", "text": "thinking..."}
                yield {"type": "tool_call", "id": "x", "name": "glob",
                       "arguments": {"pattern": "*"}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "final."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    agent = Agent(provider=_NarrateThenCall(), system_prompt="s",
                  tool_names=FULL, auto_confirm=True)
    agent.run("go")
    assistant_with_calls = [
        m for m in agent.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    assert assistant_with_calls[0]["content"] == "thinking..."


def test_mixed_good_and_bad_call_resets_parse_breaker(tmp_workspace):
    """finding #23: a turn that lands one GOOD call plus one malformed-JSON call
    must NOT trip the parse-error circuit breaker (it's making progress)."""
    class _MixedForever(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "tool_call", "id": "good", "name": "glob",
                   "arguments": {"pattern": "*"}}
            yield {"type": "tool_call", "id": "bad", "name": "glob",
                   "arguments": {}, "_parse_error": "Expecting value"}
            yield {"type": "done", "finish_reason": "tool_calls"}

    agent = Agent(provider=_MixedForever(), system_prompt="s",
                  tool_names=FULL, auto_confirm=True, max_iterations=5)
    out = agent.run("go")
    # It is NOT aborted by the parse breaker; it runs to the iteration limit.
    assert "invalid tool-call JSON" not in out
    assert "max" in out.lower()


def test_auto_compact_triggers_when_over_soft_limit(tmp_workspace):
    """findings #4/#26: when history exceeds context_soft_limit, run() compacts
    BEFORE the provider call so the prefix stays bounded."""
    class _SummarizeThenAnswer(MockProvider):
        def stream_chat(self, messages, tools):
            # If this is the summarizer call (system prompt is the SUMMARIZER),
            # emit a short summary. Otherwise give a final answer.
            if "Summarize this engineering session" in str(messages[0]["content"]):
                yield {"type": "text", "text": "- compact summary"}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}
            else:
                yield {"type": "text", "text": "answer."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    agent = Agent(provider=_SummarizeThenAnswer(), system_prompt="SYS",
                  tool_names=[], context_soft_limit=100)
    # Pre-load a big history (well over ~100 est tok = ~400 chars).
    agent.messages += [
        {"role": "user", "content": "x " * 500},
        {"role": "assistant", "content": "y " * 500},
        {"role": "user", "content": "z " * 500},
        {"role": "assistant", "content": "w " * 500},
    ]
    n_before = len(agent.messages)
    agent.run("now answer")
    # The auto-compact replaced earlier turns with a summary note.
    assert any(
        m["role"] == "system" and "Summary of earlier conversation" in str(m["content"])
        for m in agent.messages
    )
    assert len(agent.messages) < n_before + 2  # not just append-only growth


def test_compact_reduces_giant_single_turn(tmp_workspace):
    """findings #1/#3: a SINGLE user turn whose own tool rounds overflow the
    window must still compact. Strict user-boundary anchoring left this a no-op;
    the in-tail fallback summarizes older completed rounds, preserving the user
    message + last rounds + tool_calls/tool pairing, and actually shrinks the
    estimate."""
    class _Summarizer(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "- summary of early rounds"}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 4}

    agent = Agent(provider=_Summarizer(), system_prompt="SYS", tool_names=[],
                  context_soft_limit=100)
    # One user turn, then MANY completed tool rounds with large results.
    agent.messages += [{"role": "user", "content": "read the whole repo"}]
    for i in range(10):
        agent.messages += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": f'{{"path": "f{i}.py"}}'}},
            ]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "X " * 300},
        ]
    # Exactly one user turn.
    assert sum(1 for m in agent.messages if m["role"] == "user") == 1
    before, after = agent.compact()
    assert after < before  # the no-op is fixed: it actually shrinks
    # The sole user message survives.
    assert any(m["role"] == "user" and m["content"] == "read the whole repo"
               for m in agent.messages)
    # An in-turn summary note was injected.
    assert any(m["role"] == "system" and "earlier tool activity" in str(m["content"])
               for m in agent.messages)
    # Pairing invariant: every kept tool message has a preceding assistant
    # tool_calls with a matching id.
    open_ids: set[str] = set()
    for m in agent.messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            open_ids.update(tc["id"] for tc in m["tool_calls"])
        if m.get("role") == "tool":
            assert m["tool_call_id"] in open_ids


def test_compact_aggressive_collapses_recent_rounds(tmp_workspace):
    """Regression for the '34233 -> 33759' near-no-op: a SINGLE user turn whose
    bulk sits in the RECENT tool rounds. With keep_rounds=1 (aggressive) the
    2nd-to-last round is now summarized away, so the shrink is meaningful — not
    the ~1% no-op the old keep_rounds=2 produced."""
    class _Summarizer(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "- summary of early rounds"}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 4}

    agent = Agent(provider=_Summarizer(), system_prompt="SYS", tool_names=[],
                  context_soft_limit=100)
    agent.messages += [{"role": "user", "content": "read the whole repo"}]
    for i in range(5):
        agent.messages += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": f'{{"path": "f{i}.py"}}'}},
            ]},
            # Each tool result is a LARGE block (the bulk).
            {"role": "tool", "tool_call_id": f"c{i}", "content": "X " * 300},
        ]
    before, after = agent.compact()
    # Aggressive: must shrink by a meaningful margin (>= 10%), not ~1%.
    assert after <= before * 0.90
    # The 2nd-to-last round's tool result is summarized away (not kept verbatim).
    kept_tool_ids = {
        m["tool_call_id"] for m in agent.messages if m.get("role") == "tool"
    }
    assert f"c3" not in kept_tool_ids  # 2nd-to-last round collapsed
    # The last round (c4) is kept verbatim.
    assert "c4" in kept_tool_ids


def test_auto_compact_disabled_when_limit_zero(tmp_workspace):
    """context_soft_limit=0 (default) never auto-compacts."""
    class _Answer(MockProvider):
        def stream_chat(self, messages, tools):
            assert "Summarize this engineering session" not in str(messages[0]["content"])
            yield {"type": "text", "text": "answer."}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    agent = Agent(provider=_Answer(), system_prompt="SYS", tool_names=[],
                  context_soft_limit=0)
    agent.messages += [{"role": "user", "content": "x " * 5000}]
    agent.run("go")  # must not raise / must not summarize


def test_spawn_agent_forwards_confirm_fn(tmp_workspace, monkeypatch):
    """finding #1: a spawned sub-agent must use the SAME confirm_fn the
    orchestrator was given, never builtin input(). We make input() raise, give
    spawn a confirm_fn that approves, and have the spawned coder call a gated
    tool — it must execute via the forwarded confirm."""
    from llmcode.orchestration import make_spawn_agent_tool

    def _boom(*a, **k):
        raise AssertionError("builtin input() must not be used by a sub-agent")

    monkeypatch.setattr(builtins, "input", _boom)

    seen = {"n": 0}

    def confirm(tool, args):
        seen["n"] += 1
        return True  # approve the gated write_file

    spawn = make_spawn_agent_tool(
        provider=MockProvider(scenario="hello"),
        console=None,
        auto_confirm=False,  # force the gate so confirm_fn is consulted
        max_iterations=4,
        private=False,  # so 'coder' keeps the full tool set (incl. run_bash)
        confirm_fn=confirm,
    )
    res = spawn.fn({"role": "coder", "task": "create hello.py and run it"})
    assert res["ok"] is True
    # The forwarded confirm was used (not input()) and the file was written.
    assert seen["n"] >= 1
    assert Path("hello.py").exists()


def test_repl_wires_non_input_confirm_fn(tmp_workspace, monkeypatch):
    """The REPL path must inject a prompt_toolkit confirm_fn, never builtin input().

    We monkeypatch builtins.input to RAISE, build a fake session whose .prompt
    returns 'y', wire it through make_ptk_confirm, and run a gated tool. The tool
    must execute via the injected confirm (proving builtin input() is unused).
    """
    from llmcode.repl import make_ptk_confirm

    def _boom(*a, **k):
        raise AssertionError("builtin input() must not be used in the REPL path")

    monkeypatch.setattr(builtins, "input", _boom)

    class FakeSession:
        def __init__(self):
            self.asked = []

        def prompt(self, text, placeholder=None, **kwargs):
            self.asked.append(text)
            return "y"

    session = FakeSession()
    confirm_fn = make_ptk_confirm(session)

    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=False,  # force the gate -> confirm_fn is consulted
        confirm_fn=confirm_fn,
    )
    agent.run("hello")

    # The injected confirm was used (not input()), and write_file executed.
    assert session.asked  # session.prompt was called for the gated tool
    assert Path("hello.py").exists()


def test_run_with_tools_prints_summary_and_auto_expands(tmp_workspace, capsys):
    """run() prints the ◆ counts one-liner AND auto-expands the ⏺/⎿ tree inline
    for a modest all-green batch (≤5 tools) — the work is never hidden behind
    Ctrl+O. Mid-turn narration is still suppressed (the tree prints once, at the
    end, from the activity summary)."""
    from rich.console import Console

    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
        console=Console(markup=False),
    )
    agent.run("hello")
    out = capsys.readouterr().out

    # ◆ counts one-liner present (2 successes, dim ctrl-o hint).
    assert "◆" in out
    assert "2 tools" in out
    assert "✓2" in out
    assert "ctrl-o" in out

    # The 2-tool batch auto-expands: the ⏺/⎿ tree is present inline.
    assert "⏺ Write(hello.py)" in out
    assert "⎿" in out


def test_render_details_reproduces_per_tool_tree(tmp_workspace, capsys):
    """render_details() on the same agent reproduces the full ⏺/⎿ tree that
    run() collapsed, so no detail is lost — just deferred to Ctrl+O."""
    from rich.console import Console

    console = Console(markup=False)
    agent = Agent(
        provider=MockProvider(scenario="hello"),
        system_prompt="sys",
        tool_names=FULL,
        auto_confirm=True,
        console=console,
    )
    agent.run("hello")
    capsys.readouterr()  # discard run() output

    agent.render_details(console)
    details = capsys.readouterr().out

    # Full ⏺/⎿ tree is present for each tool.
    assert "⏺ Write(hello.py)" in details
    assert "⎿" in details           # result summary connector
    assert "⏺ Bash(" in details
    # Successful call: no ✗ glyph on the bash result line.
    bash_result_lines = [ln for ln in details.splitlines() if "⎿" in ln and "Bash" not in ln]
    assert any("✗" not in ln for ln in bash_result_lines)
