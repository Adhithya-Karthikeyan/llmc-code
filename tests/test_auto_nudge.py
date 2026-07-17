"""Auto-nudge: when a turn ends with no answer (reasoning-only), the agent
re-prompts the model ONCE to write its answer instead of giving up immediately.
"""

from __future__ import annotations

from llmcli.agent import Agent

_EMPTY = [{"type": "done", "finish_reason": "stop", "output_tokens": 0}]  # no text, no tools
_ANSWER = [
    {"type": "text", "text": "The answer is 42."},
    {"type": "done", "finish_reason": "stop", "output_tokens": 5},
]


class _Prov:
    model = "m"

    def __init__(self, scripts):
        self.scripts = scripts
        self.n = 0
        self.seen = []  # messages passed to each call

    def stream_chat(self, messages, tools):
        self.seen.append(list(messages))
        script = self.scripts[min(self.n, len(self.scripts) - 1)]
        self.n += 1
        yield from script


def _agent(prov):
    return Agent(prov, "sys", [], console=None, auto_confirm=True, max_iterations=10)


def test_nudge_recovers_answer_after_empty_turn():
    p = _Prov([_EMPTY, _ANSWER])  # empty first, then (after the nudge) a real answer
    out = _agent(p).run("explain X")
    assert out == "The answer is 42."        # recovered, not the empty sentinel
    assert p.n == 2                            # original + 1 nudge
    # the answer-forcing nudge user message was injected before the 2nd call
    assert any(
        m.get("role") == "user" and "final answer" in str(m.get("content", "")).lower()
        for m in p.seen[1]
    )


def test_nudge_gives_up_after_two_retries():
    p = _Prov([_EMPTY, _EMPTY, _EMPTY, _EMPTY])  # always empty
    out = _agent(p).run("explain X")
    assert "no answer produced" in out         # sentinel after the two nudges
    assert p.n == 3                             # original + exactly 2 nudges, then stop


def test_normal_answer_is_never_nudged():
    p = _Prov([_ANSWER])
    out = _agent(p).run("explain X")
    assert out == "The answer is 42."
    assert p.n == 1                             # no nudge needed


# --------------------------------------------------------------------------- #
# Recovery generation: empty-turn nudge runs UNCAPPED + thinking-off, restored
# --------------------------------------------------------------------------- #

class _GentleProv:
    """Provider exposing gentle_mode/effort (like LocalProvider) that RECORDS
    their values at each stream_chat call, to prove the empty-turn recovery
    generation bypasses the gentle cap and disables thinking — and restores both.
    """

    model = "m"

    def __init__(self, scripts):
        self.scripts = scripts
        self.n = 0
        self.gentle_mode = True
        self.effort = "high"
        self.seen_gentle = []
        self.seen_effort = []

    def stream_chat(self, messages, tools):
        self.seen_gentle.append(self.gentle_mode)
        self.seen_effort.append(self.effort)
        script = self.scripts[min(self.n, len(self.scripts) - 1)]
        self.n += 1
        yield from script


def test_recovery_generation_bypasses_gentle_cap_and_disables_thinking():
    p = _GentleProv([_EMPTY, _ANSWER])  # empty first -> recovery shot -> answer
    out = _agent(p).run("explain X")
    assert out == "The answer is 42."
    # call 0 = normal (gentle on, effort untouched); call 1 = the recovery shot.
    assert p.seen_gentle == [True, False]       # recovery bypassed the gentle cap
    assert p.seen_effort == ["high", "off"]     # recovery disabled thinking
    # State is RESTORED after the recovery call so later normal turns stay capped.
    assert p.gentle_mode is True
    assert p.effort == "high"


def test_recovery_composes_with_build_nudge_no_loop_or_double_render(tmp_workspace):
    # write -> empty turn (-> recovery nudge) -> answer (-> build nudge) -> answer.
    # The empty-turn recovery and the build-verify gate compose: the loop
    # terminates with ONE clean answer (no sentinel, no infinite re-nudge).
    p = _Prov([_WRITE, _EMPTY, _ANSWER])
    agent = _agent_tools(p, max_iterations=12)
    out = agent.run("add a feature")
    assert out == "The answer is 42."           # single clean answer
    assert "no answer produced" not in out      # recovery beat the sentinel
    empty_nudges = [
        m for m in agent.messages
        if m.get("_nudge") and "final answer" in str(m.get("content", "")).lower()
    ]
    assert len(empty_nudges) == 1               # exactly one recovery nudge
    assert len(_build_nudges(agent)) == 1       # exactly one build-verify nudge
    assert p.n == 4                             # write, empty, recovery-answer, answer


# --------------------------------------------------------------------------- #
# ACBUILD-1: a write/edit without a subsequent run_bash earns ONE build nudge
# --------------------------------------------------------------------------- #

# Tool-call scripts for the build-verify nudge. write_file/run_bash are gated, so
# the agents below use auto_confirm=True; write_file writes into the tmp_workspace.
_WRITE = [
    {"type": "tool_call", "id": "w1", "name": "write_file",
     "arguments": {"path": "feat.py", "content": "print('x')\n"}},
    {"type": "done", "finish_reason": "tool_calls"},
]
_BASH = [
    {"type": "tool_call", "id": "b1", "name": "run_bash",
     "arguments": {"command": "echo hi"}},
    {"type": "done", "finish_reason": "tool_calls"},
]


def _agent_tools(prov, tool_names=("write_file", "edit_file", "run_bash"), **kw):
    kw.setdefault("max_iterations", 10)
    return Agent(prov, "sys", list(tool_names), console=None, auto_confirm=True, **kw)


def _build_nudges(agent):
    return [
        m for m in agent.messages
        if m.get("_nudge") and "Run the tests now" in str(m.get("content", ""))
    ]


def test_build_nudge_fires_once_when_writes_unvalidated(tmp_workspace):
    # write_file, then keep answering WITHOUT ever running bash.
    p = _Prov([_WRITE, _ANSWER])
    agent = _agent_tools(p)
    out = agent.run("add a feature")
    assert out == "The answer is 42."          # accepted after the single nudge
    assert len(_build_nudges(agent)) == 1      # exactly one verify nudge fired
    # write -> answer(=>nudge) -> answer(=>accept) == 3 provider calls.
    assert p.n == 3


def test_build_nudge_skipped_after_run_bash(tmp_workspace):
    # write_file then run_bash then answer -> the bash run cleared the flag.
    p = _Prov([_WRITE, _BASH, _ANSWER])
    agent = _agent_tools(p)
    out = agent.run("add and test a feature")
    assert out == "The answer is 42."
    assert _build_nudges(agent) == []          # run_bash cleared it -> no nudge
    assert p.n == 3                            # write, bash, answer (no nudge turn)


def test_build_nudge_never_on_pure_question(tmp_workspace):
    # No write happened -> the flag is never armed -> no nudge, even with run_bash.
    p = _Prov([_ANSWER])
    agent = _agent_tools(p)
    out = agent.run("what is 6 times 7")
    assert out == "The answer is 42."
    assert _build_nudges(agent) == []
    assert p.n == 1


def test_build_nudge_caps_at_one_no_loop(tmp_workspace):
    # Even though the model writes and then keeps answering WITHOUT running tests,
    # the nudge fires at most once and the loop terminates well under the iteration
    # cap (no infinite re-nudge).
    p = _Prov([_WRITE, _ANSWER])
    agent = _agent_tools(p, max_iterations=12)
    out = agent.run("add a feature")
    assert out == "The answer is 42."          # terminated with the real answer
    assert "max" not in out.lower()            # not the iteration-limit sentinel
    assert len(_build_nudges(agent)) == 1      # capped at one per write-batch


def test_build_nudge_skipped_without_run_bash_tool(tmp_workspace):
    # A read-only/doc agent WITHOUT run_bash never gets the nudge — nothing to run.
    p = _Prov([_WRITE, _ANSWER])
    agent = _agent_tools(p, tool_names=("write_file", "edit_file"))
    out = agent.run("add a feature")
    assert out == "The answer is 42."
    assert _build_nudges(agent) == []
    assert p.n == 2                            # write, answer (accepted, no nudge)
