"""Regression tests for the gentle sub-agent spawn cool-down.

The wait fires ONLY when:
  - gentle_mode is on, AND
  - is_terminal is True, AND
  - gentle_spawn_gap_seconds > 0, AND
  - it is the 2nd+ spawn (spawn_count >= 1) — the first spawn never waits.

Tests monkeypatch ``llmcli.orchestration.time.sleep`` so no real sleep happens.
"""

from __future__ import annotations

import unittest.mock as mock

from llmcli.orchestration import make_spawn_agent_tool
from llmcli.providers import MockProvider


def _make_tool(gentle_mode, gap, is_terminal, sleeps):
    """Build a spawn tool whose time.sleep is spied into ``sleeps``."""
    provider = MockProvider(scenario="plain")
    tool = make_spawn_agent_tool(
        provider=provider,
        auto_confirm=True,
        max_iterations=2,
        gentle_mode=gentle_mode,
        gentle_spawn_gap_seconds=gap,
        is_terminal=is_terminal,
    )
    return tool


def _spawn_once(tool):
    """Invoke spawn_agent once and return its result dict."""
    return tool.fn({"role": "explorer", "task": "report"})


def test_second_spawn_sleeps_when_gentle_on_and_terminal():
    sleeps = []
    with mock.patch("llmcli.orchestration.time.sleep", lambda s: sleeps.append(s)):
        tool = _make_tool(gentle_mode=True, gap=0.01, is_terminal=True, sleeps=sleeps)
        # 1st spawn — no sleep.
        r1 = _spawn_once(tool)
        assert r1["ok"] is True
        assert sleeps == [], "first spawn must not sleep"
        # 2nd spawn — sleeps the gap.
        r2 = _spawn_once(tool)
        assert r2["ok"] is True
        assert sleeps == [0.01], f"second spawn must sleep the gap; got {sleeps}"


def test_first_spawn_never_sleeps():
    sleeps = []
    with mock.patch("llmcli.orchestration.time.sleep", lambda s: sleeps.append(s)):
        tool = _make_tool(gentle_mode=True, gap=0.01, is_terminal=True, sleeps=sleeps)
        _spawn_once(tool)
        assert sleeps == [], "first spawn must never sleep"


def test_no_sleep_when_gentle_mode_off():
    sleeps = []
    with mock.patch("llmcli.orchestration.time.sleep", lambda s: sleeps.append(s)):
        tool = _make_tool(gentle_mode=False, gap=0.01, is_terminal=True, sleeps=sleeps)
        _spawn_once(tool)
        _spawn_once(tool)
        assert sleeps == [], "gentle off => no sleep even on 2nd spawn"


def test_no_sleep_when_not_terminal():
    sleeps = []
    with mock.patch("llmcli.orchestration.time.sleep", lambda s: sleeps.append(s)):
        tool = _make_tool(gentle_mode=True, gap=0.01, is_terminal=False, sleeps=sleeps)
        _spawn_once(tool)
        _spawn_once(tool)
        assert sleeps == [], "non-terminal => no sleep even on 2nd spawn"


def test_no_sleep_when_gap_zero():
    sleeps = []
    with mock.patch("llmcli.orchestration.time.sleep", lambda s: sleeps.append(s)):
        tool = _make_tool(gentle_mode=True, gap=0.0, is_terminal=True, sleeps=sleeps)
        _spawn_once(tool)
        _spawn_once(tool)
        assert sleeps == [], "gap == 0 => no sleep"


def test_third_spawn_also_sleeps():
    """Every spawn from the 2nd on waits — the counter increments each time."""
    sleeps = []
    with mock.patch("llmcli.orchestration.time.sleep", lambda s: sleeps.append(s)):
        tool = _make_tool(gentle_mode=True, gap=0.01, is_terminal=True, sleeps=sleeps)
        _spawn_once(tool)
        _spawn_once(tool)
        _spawn_once(tool)
        assert sleeps == [0.01, 0.01], f"3rd spawn must also wait; got {sleeps}"