"""Tests for MANDATORY tool self-healing + thermal cooldown.

All offline: scripted providers (no network) + a tmp cwd + monkeypatched clock/
sleep so nothing ever blocks on the real wall clock.

Two features under test:
  1. Self-heal: a FAILED tool call is auto-corrected + retried by the harness
     (via llmcli.remediation) BEFORE the model sees the error. Bounded by
     auto_fix_max_attempts, non-destructive, and OFF by default.
  2. Cooldown: a long run breaks mid-way to let the GPU cool (via llmcli.cooldown).
     OFF by default so tests never pause.

The default Agent construction (both knobs off) must be behaviour-identical.
"""

from __future__ import annotations

import time

import pytest

import llmcli.cooldown as cooldown
import llmcli.remediation as remediation
import llmcli.repl as r
from llmcli.agent import Agent
from llmcli.config import Config, load_config, save_config
from llmcli.providers import MockProvider, Provider
from llmcli.tools import FULL


# --------------------------------------------------------------------------- #
# Scripted providers
# --------------------------------------------------------------------------- #
class OneToolThenDone(Provider):
    """Emit ONE scripted tool call on the first turn, then finish with text."""

    name = "onetool"
    model = "test"

    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0

    def stream_chat(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call",
                "id": "t1",
                "name": self.tool_name,
                "arguments": self.arguments,
            }
            yield {"type": "done", "finish_reason": "tool_calls"}
        else:
            yield {"type": "text", "text": "ok"}
            yield {"type": "done", "finish_reason": "stop"}


class ClockAdvancingProvider(Provider):
    """Advance a shared fake clock on each turn; emit one tool call then finish.

    The clock jump per turn (``step``) lets a test push the run across a cooldown
    interval so exactly one mid-run break fires.
    """

    name = "clockadv"
    model = "test"

    def __init__(self, clock: dict, step: float):
        self.clock = clock
        self.step = step
        self.calls = 0

    def stream_chat(self, messages, tools):
        self.calls += 1
        self.clock["t"] += self.step
        if self.calls == 1:
            yield {
                "type": "tool_call",
                "id": "c1",
                "name": "glob",
                "arguments": {"pattern": "*.py"},
            }
            yield {"type": "done", "finish_reason": "tool_calls"}
        else:
            yield {"type": "text", "text": "done"}
            yield {"type": "done", "finish_reason": "stop"}


# --------------------------------------------------------------------------- #
# Feature 1: tool self-healing
# --------------------------------------------------------------------------- #
def test_autofix_corrects_out_of_workspace_read(tmp_workspace):
    """An out-of-workspace read whose basename uniquely matches a real file is
    auto-corrected + retried, and the read SUCCEEDS at the relative path."""
    (tmp_workspace / "heal_target_unique.py").write_text("X = 1\n", encoding="utf-8")
    outside = str(tmp_workspace.parent / "elsewhere" / "heal_target_unique.py")
    provider = OneToolThenDone("read_file", {"path": outside})
    agent = Agent(
        provider=provider,
        system_prompt="s",
        tool_names=FULL,
        auto_confirm=True,
        auto_fix_tools=True,
    )
    agent.run("read the file")

    detail = agent.last_turn_details[0]
    assert detail["name"] == "read_file"
    assert detail["args"]["path"] == "heal_target_unique.py"  # corrected relative
    assert detail["result"]["ok"] is True
    assert agent._autofixes == 1


def test_autofix_corrects_out_of_workspace_write(tmp_workspace):
    """An out-of-workspace write is auto-corrected + retried; the file is written
    at the corrected relative path (content lands on disk)."""
    target = tmp_workspace / "heal_write_unique.py"
    target.write_text("old\n", encoding="utf-8")
    outside = str(tmp_workspace.parent / "elsewhere" / "heal_write_unique.py")
    provider = OneToolThenDone(
        "write_file", {"path": outside, "content": "new data\n", "overwrite": True}
    )
    agent = Agent(
        provider=provider,
        system_prompt="s",
        tool_names=FULL,
        auto_confirm=True,
        auto_fix_tools=True,
    )
    agent.run("write the file")

    detail = agent.last_turn_details[0]
    assert detail["result"]["ok"] is True
    assert detail["args"]["path"] == "heal_write_unique.py"
    assert target.read_text(encoding="utf-8") == "new data\n"
    assert agent._autofixes == 1


def test_no_autofix_when_disabled(tmp_workspace):
    """With auto_fix_tools=False (the default) the same failure is NOT healed —
    it fails exactly as before and the model sees the error."""
    (tmp_workspace / "heal_target_unique.py").write_text("X = 1\n", encoding="utf-8")
    outside = str(tmp_workspace.parent / "elsewhere" / "heal_target_unique.py")
    provider = OneToolThenDone("read_file", {"path": outside})
    agent = Agent(
        provider=provider,
        system_prompt="s",
        tool_names=FULL,
        auto_confirm=True,
        auto_fix_tools=False,
    )
    agent.run("read the file")

    detail = agent.last_turn_details[0]
    assert detail["args"]["path"] == outside  # unchanged
    assert detail["result"]["ok"] is not True
    assert agent._autofixes == 0


def test_autofix_no_safe_fix_falls_through(tmp_workspace):
    """A failure with NO safe correction (no unique basename match) falls through
    to the model unchanged even with auto-fix on."""
    (tmp_workspace / "unrelated.py").write_text("Y = 2\n", encoding="utf-8")
    outside = str(tmp_workspace.parent / "elsewhere" / "no_match_zzz.py")
    provider = OneToolThenDone("read_file", {"path": outside})
    agent = Agent(
        provider=provider,
        system_prompt="s",
        tool_names=FULL,
        auto_confirm=True,
        auto_fix_tools=True,
    )
    agent.run("read the file")

    detail = agent.last_turn_details[0]
    assert detail["result"]["ok"] is not True
    assert detail["args"]["path"] == outside
    assert agent._autofixes == 0


def test_autofix_is_bounded_by_max_attempts(tmp_workspace, monkeypatch):
    """Even if the remediator keeps returning a (never-working) fix, the retry
    loop never exceeds auto_fix_max_attempts."""
    calls = {"n": 0}

    def fake_remediate(name, args, result, *, root, project_files=None):
        calls["n"] += 1
        return (dict(args), "noop retry")  # a fix that never actually fixes

    monkeypatch.setattr(remediation, "remediate", fake_remediate)
    # A tool that always fails: a relative (in-workspace) but missing file.
    provider = OneToolThenDone("read_file", {"path": "missing_zzz.py"})
    agent = Agent(
        provider=provider,
        system_prompt="s",
        tool_names=FULL,
        auto_confirm=True,
        auto_fix_tools=True,
        auto_fix_max_attempts=3,
    )
    agent.run("read the file")

    assert calls["n"] == 3  # exactly auto_fix_max_attempts, never more
    assert agent.last_turn_details[0]["result"]["ok"] is not True
    # _autofixes counts SUCCESSES: a heal that never actually fixes yields 0.
    assert agent._autofixes == 0


def test_auto_fix_max_attempts_clamped():
    """The constructor clamps auto_fix_max_attempts into 1..5."""
    assert Agent(MockProvider(), "s", FULL, auto_fix_max_attempts=99).auto_fix_max_attempts == 5
    assert Agent(MockProvider(), "s", FULL, auto_fix_max_attempts=0).auto_fix_max_attempts == 1


# --------------------------------------------------------------------------- #
# Feature 2: thermal cooldown (agent loop)
# --------------------------------------------------------------------------- #
def test_cooldown_fires_exactly_one_break(tmp_workspace, monkeypatch):
    """An agent iteration crossing the interval triggers exactly ONE paused
    break; the clock + sleep are injected so nothing blocks on the real clock."""
    clock = {"t": 1000.0}
    slept: list[float] = []
    real_maybe_pause = cooldown.maybe_pause

    def rec(*, now=None, sleep=None, notify=None):
        # Inject the fake clock + a recording fake sleep (the agent passes only
        # notify), delegating to the REAL interval logic.
        return real_maybe_pause(
            now=clock["t"], sleep=lambda s: slept.append(s), notify=notify
        )

    monkeypatch.setattr(cooldown, "maybe_pause", rec)
    cooldown.configure(enabled=True, interval_seconds=10.0, duration_seconds=1.0)
    cooldown.reset(now=clock["t"])  # deterministic baseline

    provider = ClockAdvancingProvider(clock, step=20.0)  # each turn crosses 10s
    agent = Agent(
        provider=provider,
        system_prompt="s",
        tool_names=FULL,
        auto_confirm=True,
        cooldown_enabled=True,
    )
    agent.run("do work")

    assert slept == [1.0]  # exactly one break, for the configured duration


def test_default_agent_never_pauses(tmp_workspace, monkeypatch):
    """The default agent (cooldown_enabled=False) never calls the pacer."""
    clock = {"t": 1000.0}
    slept: list[float] = []
    real_maybe_pause = cooldown.maybe_pause

    def rec(*, now=None, sleep=None, notify=None):
        return real_maybe_pause(
            now=clock["t"], sleep=lambda s: slept.append(s), notify=notify
        )

    monkeypatch.setattr(cooldown, "maybe_pause", rec)
    cooldown.configure(enabled=True, interval_seconds=10.0, duration_seconds=1.0)
    cooldown.reset(now=clock["t"])

    provider = ClockAdvancingProvider(clock, step=20.0)
    agent = Agent(
        provider=provider,
        system_prompt="s",
        tool_names=FULL,
        auto_confirm=True,
        # cooldown_enabled defaults to False
    )
    agent.run("do work")

    assert slept == []


# --------------------------------------------------------------------------- #
# /cooldown command + idle reset (REPL wiring)
# --------------------------------------------------------------------------- #
class _FakeMCP:
    def __init__(self, *a, **k):
        self.configs = {}

    def registry(self):
        return {}

    def is_running(self):
        return False

    def start_all(self, **k):
        pass

    def shutdown_all(self):
        pass

    def status(self):
        return []


@pytest.fixture
def repl(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    disk = Config(provider="mock", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", model="m")
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_cooldown_command_toggles_and_persists(repl, monkeypatch, capsys):
    saved = {}
    monkeypatch.setattr(r, "save_config", lambda cfg, *a, **k: saved.setdefault("cfg", cfg))
    reconfigured = {}
    monkeypatch.setattr(
        cooldown, "configure",
        lambda **k: reconfigured.update(k),
    )
    assert repl.config.cooldown_enabled is True
    assert repl._dispatch_slash("/cooldown off") is True
    assert repl.config.cooldown_enabled is False
    assert "cfg" in saved  # persisted via _persist_config -> save_config
    assert reconfigured.get("enabled") is False  # pacer reconfigured


def test_cooldown_on_from_disabled_rebuilds_live_agent(monkeypatch, tmp_path):
    """A session started with cooldown OFF must, after `/cooldown on`, hold a
    LIVE agent whose cooldown_enabled is True: on/off is captured at build time,
    so the command has to rebuild the agent for the loop-top pause gate to see
    the new state (regression guard for --no-cooldown never turning on)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    disk = Config(provider="mock", model="m", cooldown_enabled=False)
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    monkeypatch.setattr(cooldown, "configure", lambda **k: None)
    cfg = Config(provider="mock", model="m", cooldown_enabled=False)
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    assert repl.agent.cooldown_enabled is False  # built with cooldown OFF
    assert repl._dispatch_slash("/cooldown on") is True
    assert repl.config.cooldown_enabled is True
    assert repl.agent.cooldown_enabled is True  # live agent rebuilt with ON


def test_cooldown_command_status_no_arg(repl, capsys):
    assert repl._dispatch_slash("/cooldown") is True
    out = capsys.readouterr().out
    assert "cooldown:" in out
    assert "GPU" in out  # honest note is printed


def test_cooldown_command_sets_interval_and_duration(repl, monkeypatch):
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(cooldown, "configure", lambda **k: None)
    assert repl._dispatch_slash("/cooldown interval 300") is True
    assert repl.config.cooldown_interval_seconds == 300.0
    assert repl._dispatch_slash("/cooldown duration 30") is True
    assert repl.config.cooldown_duration_seconds == 30.0


def test_cooldown_idle_reset_resets_pacer(repl, monkeypatch):
    """A long idle gap at the prompt resets the pacer so idle time doesn't
    accumulate toward a break."""
    reset_calls = {"n": 0}
    monkeypatch.setattr(cooldown, "reset", lambda *a, **k: reset_calls.__setitem__("n", reset_calls["n"] + 1))
    monkeypatch.setattr(repl.agent, "run", lambda *a, **k: "")
    repl.config.cooldown_enabled = True
    repl.config.cooldown_duration_seconds = 60.0
    # Idle far longer than the break duration → reset expected.
    repl._last_gen_end = time.monotonic() - 1000.0
    repl._submit("hello")
    assert reset_calls["n"] == 1


def test_cooldown_first_turn_no_reset(repl, monkeypatch):
    """The first turn (_last_gen_end == 0.0) must NOT reset the pacer."""
    reset_calls = {"n": 0}
    monkeypatch.setattr(cooldown, "reset", lambda *a, **k: reset_calls.__setitem__("n", reset_calls["n"] + 1))
    monkeypatch.setattr(repl.agent, "run", lambda *a, **k: "")
    repl.config.cooldown_enabled = True
    repl._last_gen_end = 0.0  # no prior generation
    repl._submit("hello")
    assert reset_calls["n"] == 0


# --------------------------------------------------------------------------- #
# Config round-trip + default construction
# --------------------------------------------------------------------------- #
def test_config_defaults_on():
    cfg = Config()
    assert cfg.auto_fix_tools is True
    assert cfg.auto_fix_max_attempts == 2
    assert cfg.cooldown_enabled is True
    assert cfg.cooldown_interval_seconds == 600.0
    assert cfg.cooldown_duration_seconds == 60.0


def test_config_round_trips(tmp_path):
    p = tmp_path / "config.json"
    save_config(
        Config(
            auto_fix_tools=False,
            auto_fix_max_attempts=4,
            cooldown_enabled=False,
            cooldown_interval_seconds=120.0,
            cooldown_duration_seconds=15.0,
        ),
        path=p,
    )
    cfg = load_config(path=p)
    assert cfg.auto_fix_tools is False
    assert cfg.auto_fix_max_attempts == 4
    assert cfg.cooldown_enabled is False
    assert cfg.cooldown_interval_seconds == 120.0
    assert cfg.cooldown_duration_seconds == 15.0


def test_config_validation_rejects_bad_values(tmp_path):
    import json

    p = tmp_path / "config.json"
    # out-of-range attempts, bool interval, negative duration → safe defaults kept.
    p.write_text(
        json.dumps(
            {
                "auto_fix_max_attempts": 99,
                "cooldown_interval_seconds": True,
                "cooldown_duration_seconds": -5,
                "auto_fix_tools": "nope",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.auto_fix_max_attempts == 2  # 99 rejected
    assert cfg.cooldown_interval_seconds == 600.0  # bool rejected
    assert cfg.cooldown_duration_seconds == 60.0  # negative rejected
    assert cfg.auto_fix_tools is True  # non-bool rejected


def test_default_agent_construction_is_off():
    """A bare Agent keeps the self-heal + cooldown knobs OFF (behaviour-identical)."""
    agent = Agent(MockProvider(), "s", FULL)
    assert agent.auto_fix_tools is False
    assert agent.cooldown_enabled is False
    assert agent._autofixes == 0
    assert agent.auto_fix_max_attempts == 2
