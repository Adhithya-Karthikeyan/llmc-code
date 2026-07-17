"""Truncation recovery: when the gentle output cap (default 1024 tokens) cuts a
generation off mid-stream (finish_reason=="length"), the agent retries the SAME
request ONCE with the cap LIFTED so the full answer / tool call is produced.

This fixes two user-visible bugs:
  1. A long answer ending in "[output truncated at token limit]".
  2. A write_file/edit_file whose file content lives in the (now truncated)
     tool-call JSON arguments — the model "made the change" but the write never
     landed, because the call was cut off.
"""

from __future__ import annotations

from pathlib import Path

from llmcli.agent import Agent


class _GentleProv:
    """Scripted provider exposing gentle_mode/effort (like LocalProvider) and
    accepting the optional tool_choice kwarg the truncation retry forwards.
    Records gentle_mode at each call so a test can prove the retry lifted it."""

    model = "m"

    def __init__(self, scripts):
        self.scripts = scripts
        self.n = 0
        self.gentle_mode = True
        self.effort = "high"
        self.seen_gentle = []
        self.seen_tool_choice = []

    def stream_chat(self, messages, tools, tool_choice=None):
        self.seen_gentle.append(self.gentle_mode)
        self.seen_tool_choice.append(tool_choice)
        script = self.scripts[min(self.n, len(self.scripts) - 1)]
        self.n += 1
        yield from script


def _agent(prov, **kw):
    kw.setdefault("max_iterations", 10)
    return Agent(prov, "sys", list(kw.pop("tools", [])), console=None,
                 auto_confirm=True, **kw)


# --- truncated final answer -------------------------------------------------- #

_TRUNC_ANSWER = [
    {"type": "text", "text": "The answer begins but is cut off"},
    {"type": "done", "finish_reason": "length", "output_tokens": 1024},
]
_FULL_ANSWER = [
    {"type": "text", "text": "The complete answer is 42."},
    {"type": "done", "finish_reason": "stop", "output_tokens": 7},
]


def test_truncated_answer_is_recovered_uncapped():
    p = _GentleProv([_TRUNC_ANSWER, _FULL_ANSWER])
    out = _agent(p).run("explain X at length")
    # The full answer replaces the truncated one — NO truncation marker.
    assert out == "The complete answer is 42."
    assert "truncated at token limit" not in out
    assert p.n == 2                                  # original + 1 uncapped retry
    assert p.seen_gentle == [True, False]            # retry lifted the gentle cap


def test_truncation_marker_still_shows_if_retry_also_truncates():
    # A REAL context-window limit: even uncapped it truncates again. We retry
    # exactly once, then surface the marker rather than looping forever.
    p = _GentleProv([_TRUNC_ANSWER, _TRUNC_ANSWER])
    out = _agent(p).run("explain X at length")
    assert "truncated at token limit" in out
    assert p.n == 2                                  # original + exactly one retry


_RETRY_ERROR = [
    {"type": "text", "text": "[provider error: boom]"},
    {"type": "done", "finish_reason": "error"},
]


def test_retry_error_does_not_clobber_partial_answer():
    # If the uncapped retry hits a transient provider error, the original
    # partial answer (+ its truncation marker) must survive — NOT be replaced
    # by the "[provider error]" sentinel.
    p = _GentleProv([_TRUNC_ANSWER, _RETRY_ERROR])
    out = _agent(p).run("explain X at length")
    assert "The answer begins but is cut off" in out
    assert "truncated at token limit" in out
    assert "provider error" not in out
    assert p.n == 2


def test_non_gentle_truncation_is_not_retried():
    # gentle_mode off => the cap isn't the (fixable) cause, so no wasteful retry;
    # the marker fires immediately.
    p = _GentleProv([_TRUNC_ANSWER])
    p.gentle_mode = False
    out = _agent(p).run("explain X")
    assert "truncated at token limit" in out
    assert p.n == 1                                  # no retry attempted


# --- truncated write_file (the "did it but didn't write it" bug) ------------- #

_TRUNC_WRITE = [
    # Simulates a write_file whose huge content blew the cap: the generation
    # ends with finish_reason "length". (The scripted args are complete; the
    # finish_reason is what drives recovery.)
    {"type": "tool_call", "id": "w0", "name": "write_file",
     "arguments": {"path": "out.py", "content": "# truncated\n"}},
    {"type": "done", "finish_reason": "length"},
]


def test_truncated_unparseable_write_recovers_in_a_single_retry(tmp_workspace):
    # The REALISTIC truncated write: the cap cut the tool-call JSON mid-string,
    # so the args are a PARSE ERROR *and* finish_reason=="length". The constrained
    # -decode retry must lift the cap (recovery) and fix it in ONE retry — NOT
    # waste a capped retry first and then a second uncapped one (the old 3-gen
    # path). Proven by p.n == 3 (orig + 1 retry + answer) and gentle lifted once.
    full = "x = 1\n" * 80
    trunc_parse_err = [
        {"type": "tool_call", "id": "w0", "name": "write_file",
         "arguments": {}, "_parse_error": "Unterminated string in JSON"},
        {"type": "done", "finish_reason": "length"},
    ]
    full_write = [
        {"type": "tool_call", "id": "w1", "name": "write_file",
         "arguments": {"path": "out.py", "content": full}},
        {"type": "done", "finish_reason": "tool_calls"},
    ]
    p = _GentleProv([trunc_parse_err, full_write, _FULL_ANSWER])
    out = _agent(p, tools=["write_file"], constrained_retry=True).run("write out.py")
    assert out == "The complete answer is 42."
    assert (Path(tmp_workspace) / "out.py").read_text() == full
    assert p.n == 3                                  # exactly one retry, not two
    assert p.seen_gentle == [True, False, True]      # the single retry was uncapped
    assert p.seen_tool_choice[1] == "required"


def test_truncated_write_is_recovered_and_actually_written(tmp_workspace):
    full = "def main():\n    return 42\n" * 50
    full_write = [
        {"type": "tool_call", "id": "w1", "name": "write_file",
         "arguments": {"path": "out.py", "content": full}},
        {"type": "done", "finish_reason": "tool_calls"},
    ]
    p = _GentleProv([_TRUNC_WRITE, full_write, _FULL_ANSWER])
    out = _agent(p, tools=["write_file"]).run("write out.py")
    assert out == "The complete answer is 42."
    # The COMPLETE content landed on disk, not the truncated stub.
    written = Path(tmp_workspace) / "out.py"
    assert written.read_text() == full
    # The retry forced a tool call (tool_choice="required") with the cap lifted.
    assert p.seen_tool_choice[1] == "required"
    assert p.seen_gentle[1] is False
