"""Per-turn read-budget guard tests (CHANGE 1).

The guard tracks how many bytes the context-bloating tools pull into context
within a SINGLE user turn and appends a ONE-TIME "stop reading" nudge once the
cumulative crosses ``_TURN_READ_NUDGE_BYTES``. It resets at the start of every
new turn. These tests drive ``Agent._account_tool_read`` directly (it carries
all the logic) plus one run()-level reset check via the offline MockProvider.
"""

from __future__ import annotations

from llmcli.agent import _BLOATING_TOOL_NAMES, _TURN_READ_NUDGE_BYTES, Agent
from llmcli.providers import MockProvider

_NUDGE_MARK = "[context-budget]"


def _agent() -> Agent:
    # No tools, no console: a bare agent is enough to exercise the accounting
    # helper and the per-turn reset in run().
    return Agent(provider=MockProvider(scenario="plain"), system_prompt="sys",
                 tool_names=[])


def test_under_threshold_no_nudge():
    """Cumulative reads below the budget never append the nudge."""
    agent = _agent()
    chunk = "x" * 1000
    total = 0
    while total + len(chunk) < _TURN_READ_NUDGE_BYTES:
        out = agent._account_tool_read("read_file", chunk)
        assert out == chunk  # passes through unchanged
        assert _NUDGE_MARK not in out
        total += len(chunk)
    assert agent._read_nudge_fired is False


def test_crossing_threshold_nudges_exactly_once():
    """Crossing the budget appends the nudge once; later reads do NOT re-nudge."""
    agent = _agent()
    # One chunk at/over the threshold trips the guard on the first call.
    big = "x" * _TURN_READ_NUDGE_BYTES
    out = agent._account_tool_read("grep", big)
    assert _NUDGE_MARK in out
    assert out.startswith(big)  # original text preserved, nudge appended after
    assert agent._read_nudge_fired is True

    # A subsequent bloating read in the SAME turn must not fire a second nudge.
    out2 = agent._account_tool_read("read_file", "y" * 5000)
    assert _NUDGE_MARK not in out2
    assert out2 == "y" * 5000


def test_non_bloating_tools_not_counted():
    """Side-effecting/non-read tools never accrue the read budget or nudge."""
    agent = _agent()
    out = agent._account_tool_read("write_file", "z" * (_TURN_READ_NUDGE_BYTES * 2))
    assert _NUDGE_MARK not in out
    assert out == "z" * (_TURN_READ_NUDGE_BYTES * 2)
    assert agent._read_bytes == 0
    assert agent._read_nudge_fired is False


def test_only_known_bloating_tools_in_set():
    """The bloating-tool set is exactly the five context-pulling read tools."""
    assert _BLOATING_TOOL_NAMES == frozenset(
        {"read_file", "grep", "repo_map", "code_search", "glob"}
    )


def test_config_read_nudge_default_is_32k():
    """The new config field defaults to 32_000 bytes (Feature 3)."""
    from llmcli.config import Config

    assert Config().read_nudge_bytes == 32_000


def test_config_read_nudge_round_trips(tmp_path):
    """A persisted custom threshold loads back; a bad value keeps the default."""
    import json

    from llmcli.config import Config, load_config, save_config

    p = tmp_path / "config.json"
    save_config(Config(read_nudge_bytes=5_000), path=p)
    assert load_config(path=p).read_nudge_bytes == 5_000
    # Non-positive / non-int values are rejected -> safe default kept.
    p.write_text(json.dumps({"read_nudge_bytes": 0}), encoding="utf-8")
    assert load_config(path=p).read_nudge_bytes == 32_000
    p.write_text(json.dumps({"read_nudge_bytes": "big"}), encoding="utf-8")
    assert load_config(path=p).read_nudge_bytes == 32_000


def test_agent_honors_custom_read_nudge_bytes():
    """A low custom threshold makes a small read cross the budget and nudge."""
    agent = Agent(
        provider=MockProvider(scenario="plain"), system_prompt="sys",
        tool_names=[], read_nudge_bytes=2_000,
    )
    assert agent.read_nudge_bytes == 2_000
    # A 2KB read is well under the 32KB default but AT the custom threshold.
    out = agent._account_tool_read("read_file", "x" * 2_000)
    assert _NUDGE_MARK in out
    assert agent._read_nudge_fired is True


def test_agent_non_positive_read_nudge_falls_back_to_default():
    """A bad threshold is coerced to the module default (guard never disabled)."""
    agent = Agent(
        provider=MockProvider(scenario="plain"), system_prompt="sys",
        tool_names=[], read_nudge_bytes=0,
    )
    assert agent.read_nudge_bytes == _TURN_READ_NUDGE_BYTES


def test_budget_resets_on_new_turn():
    """A fresh user turn (run) zeroes the tally and re-arms the nudge latch."""
    agent = _agent()
    # Simulate a turn that already crossed the budget and fired the nudge.
    agent._account_tool_read("read_file", "x" * _TURN_READ_NUDGE_BYTES)
    assert agent._read_nudge_fired is True
    assert agent._read_bytes >= _TURN_READ_NUDGE_BYTES

    # A new turn must reset both, so the nudge can fire again next turn.
    agent.run("hello")
    assert agent._read_bytes == 0
    assert agent._read_nudge_fired is False
