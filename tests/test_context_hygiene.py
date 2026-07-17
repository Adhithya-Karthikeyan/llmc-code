"""Context hygiene: stale tool output + written-file content from COMPLETED
turns is trimmed so a tiny follow-up doesn't drag tens of thousands of stale
tokens (the "what happened? -> 20-minute 55k-token prefill" bug).
"""

from __future__ import annotations

import json

from llmcli.agent import (
    Agent, classify_request, _STALE_TOOL_RESULT_CAP, _STALE_ARG_FIELD_CAP,
    _MIN_TURN_BUDGET,
)
from llmcli.providers import MockProvider


def _agent():
    return Agent(MockProvider(), "sys", [], console=None)


# ----- per-message intent classification ------------------------------------ #

def test_classify_trivial_messages():
    for t in ("what happened?", "why?", "thanks", "summarize", "what did you do?", "ok"):
        assert classify_request(t) == "trivial", t


def test_classify_followup_messages():
    for t in ("run that again?", "is it done?", "what about them?"):
        assert classify_request(t) == "followup", t


def test_classify_task_messages():
    for t in ("add a cache to utils.py", "fix the failing test",
              "write config.py", "rename the helper function"):
        assert classify_request(t) == "task", t


def test_classify_broad_messages():
    assert classify_request("audit the whole project") == "broad"
    assert classify_request("refactor everything across the codebase") == "broad"
    assert classify_request("x" * 1100) == "broad"


def test_debug_question_is_not_trivial():
    # An investigation that names real things benefits from context -> task,
    # NOT trivial (which would halve the budget and risk re-fetching).
    assert classify_request("why is the login broken?") == "task"
    assert classify_request("explain the auth flow") == "task"
    # but a pure meta question stays trivial
    assert classify_request("what happened?") == "trivial"
    assert classify_request("what did you do?") == "trivial"


def test_trim_marker_is_collision_proof():
    # A real tool output that merely CONTAINS the word "trimmed" must STILL be
    # trimmed (the old plain-"trimmed" sentinel let such content escape).
    ag = _agent()
    big = ("the test suite reported 3 trimmed and removed " * 200)  # contains 'trimmed'
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "tool", "content": big},
        {"role": "user", "content": "b"},
    ]
    saved = ag._trim_stale_tool_outputs()
    assert saved > 0                                  # was NOT skipped
    assert len(ag.messages[2]["content"]) < len(big)
    # and it is still idempotent on the trimmed result
    assert ag._trim_stale_tool_outputs() == 0


def test_classify_biases_up_when_ambiguous():
    # An ambiguous, non-conversational phrase defaults to task (more context),
    # never silently to trivial — under-loading a real task is the costly mistake.
    assert classify_request("the parser in providers and the agent loop") == "task"


def test_trivial_message_gets_small_budget_task_gets_more():
    ag = _agent()
    ag.context_budget = 12000
    ag.context_adaptive = True
    trivial = ag._compute_turn_budget("what happened?")
    task = ag._compute_turn_budget("add retry logic to providers.py")
    broad = ag._compute_turn_budget("audit the whole project")
    assert trivial <= 6000                  # trivial: tiny, pull-on-demand
    assert trivial >= _MIN_TURN_BUDGET      # but never below the floor
    assert task > trivial                   # a concrete task gets more
    assert broad >= task                    # broad gets the most


def test_old_tool_result_is_trimmed_current_turn_kept_full():
    big = "X" * 5000
    ag = _agent()
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": big},                 # OLD turn -> should trim
        {"role": "assistant", "content": "did it"},
        {"role": "user", "content": "what happened?"},     # CURRENT turn
        {"role": "assistant", "tool_calls": [
            {"id": "2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": big},                 # CURRENT turn -> keep full
    ]
    ag._trim_stale_tool_outputs()
    assert len(ag.messages[3]["content"]) < len(big)       # old tool result trimmed
    assert "trimmed" in ag.messages[3]["content"]
    assert ag.messages[7]["content"] == big                # current turn untouched


def test_small_tool_results_untouched():
    ag = _agent()
    small = "ok, 3 files"
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "tool", "content": small},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "b"},
    ]
    ag._trim_stale_tool_outputs()
    assert ag.messages[2]["content"] == small              # under cap -> unchanged


def test_old_write_file_content_arg_is_trimmed_and_stays_valid_json():
    body = "def f():\n    return 1\n" * 200
    ag = _agent()
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "write it"},
        {"role": "assistant", "tool_calls": [{
            "id": "w1", "type": "function",
            "function": {"name": "write_file",
                         "arguments": json.dumps({"path": "a.py", "content": body})}}]},
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    ag._trim_stale_tool_outputs()
    args = ag.messages[2]["tool_calls"][0]["function"]["arguments"]
    parsed = json.loads(args)                              # MUST stay valid JSON
    assert parsed["path"] == "a.py"                        # path preserved
    assert len(parsed["content"]) < len(body)             # content trimmed
    assert "trimmed" in parsed["content"]


def test_message_count_and_pairing_preserved():
    ag = _agent()
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "Y" * 9000},
        {"role": "user", "content": "b"},
    ]
    n_before = len(ag.messages)
    roles_before = [m.get("role") for m in ag.messages]
    ag._trim_stale_tool_outputs()
    assert len(ag.messages) == n_before                    # no messages dropped
    assert [m.get("role") for m in ag.messages] == roles_before


def test_no_user_turn_is_a_noop():
    ag = _agent()
    ag.messages = [{"role": "system", "content": "s"}, {"role": "tool", "content": "Z" * 9000}]
    assert ag._trim_stale_tool_outputs() == 0
    assert len(ag.messages[1]["content"]) == 9000


def test_memory_blocks_are_not_trimmed():
    ag = _agent()
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "user", "_memory": True, "content": "M" * 9000},  # ephemeral, skip
        {"role": "user", "content": "b"},
    ]
    ag._trim_stale_tool_outputs()
    assert len(ag.messages[2]["content"]) == 9000          # _memory left alone


def test_trim_is_idempotent_across_turns():
    # Runs every turn; a second pass must NOT re-trim (which would eat the note).
    ag = _agent()
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "tool", "content": "Q" * 9000},
        {"role": "user", "content": "b"},
    ]
    ag._trim_stale_tool_outputs()
    once = ag.messages[2]["content"]
    second = ag._trim_stale_tool_outputs()   # second pass
    assert ag.messages[2]["content"] == once  # unchanged
    assert second == 0                        # nothing more saved


class _RecordingProv:
    """Records the messages of the FIRST request so a test can assert what extras
    (memory recall / trivial hint) were appended."""
    model = "m"

    def __init__(self):
        self.first_messages = None

    def stream_chat(self, messages, tools=None, tool_choice=None):
        if self.first_messages is None:
            self.first_messages = list(messages)
        yield {"type": "text", "text": "hello!"}
        yield {"type": "done", "finish_reason": "stop"}


class _SpyMemory:
    def __init__(self):
        self.records = [object()]      # non-empty so the gate is reachable
        self.retrieve_called = False

    def retrieve(self, *a, **k):
        self.retrieve_called = True
        return []


def _hint_in(messages):
    return any("brief conversational message" in str(m.get("content", "")) for m in messages)


def _mem_agent(prov, mem):
    return Agent(prov, "sys", [], console=None, auto_confirm=True,
                 memory=mem, memory_enabled=True, recall_mode="auto", max_iterations=2)


def test_trivial_message_skips_memory_retrieval_and_adds_hint():
    prov, mem = _RecordingProv(), _SpyMemory()
    _mem_agent(prov, mem).run("hi")
    assert mem.retrieve_called is False          # no retrieval (no embedding call)
    assert _hint_in(prov.first_messages)         # direct-answer hint injected


def test_task_message_does_retrieve_and_no_hint():
    prov, mem = _RecordingProv(), _SpyMemory()
    _mem_agent(prov, mem).run("add retry logic to providers.py")
    assert mem.retrieve_called is True           # a real task still recalls memory
    assert not _hint_in(prov.first_messages)     # no trivial hint on a task
    ag = _agent()
    ag.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "tool", "content": "W" * 8000},
        {"role": "user", "content": "b"},
    ]
    saved = ag._trim_stale_tool_outputs()
    assert saved > 1000                                    # ~ (8000-500)/4
