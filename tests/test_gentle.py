"""Gentle-mode tests: config defaults/guards, effective max-tokens min() rule,
the /gentle slash command (parse + persist + known-command registration), the
pure pacing helper math, and proof that pacing is TTY-gated so the suite never
actually sleeps.

Honesty contract: gentle mode lowers AVERAGE GPU load/heat (shorter bursts +
spaced turns); it does NOT cap GPU %. The user-facing note asserts this.
"""

from __future__ import annotations

import json

import pytest

import llmcli.repl as r
from llmcli.config import Config, load_config, save_config
from llmcli.providers import MockProvider, effective_max_tokens


# ---------------------------------------------------------------------------
# CONFIG defaults + load guards
# ---------------------------------------------------------------------------

def test_gentle_defaults():
    cfg = Config()
    assert cfg.gentle_mode is True
    assert cfg.gentle_max_tokens == 4096
    assert cfg.gentle_gap_seconds == 2.0


def test_gentle_mode_round_trips_and_rejects_nonbool(tmp_path):
    p = tmp_path / "config.json"
    save_config(Config(gentle_mode=False), path=p)
    assert load_config(path=p).gentle_mode is False
    # non-bool ignored -> safe default (True) kept.
    p.write_text(json.dumps({"gentle_mode": "nope"}), encoding="utf-8")
    assert load_config(path=p).gentle_mode is True


def test_gentle_max_tokens_guard(tmp_path):
    p = tmp_path / "config.json"
    # valid positive int round-trips.
    p.write_text(json.dumps({"gentle_max_tokens": 256}), encoding="utf-8")
    assert load_config(path=p).gentle_max_tokens == 256
    # rejects non-positive, bool, and wrong type -> default 4096.
    for bad in (0, -5, True, "512", 3.5):
        p.write_text(json.dumps({"gentle_max_tokens": bad}), encoding="utf-8")
        assert load_config(path=p).gentle_max_tokens == 4096


def test_gentle_gap_seconds_guard(tmp_path):
    p = tmp_path / "config.json"
    # accepts an int or float >= 0 (including 0).
    p.write_text(json.dumps({"gentle_gap_seconds": 0}), encoding="utf-8")
    assert load_config(path=p).gentle_gap_seconds == 0.0
    p.write_text(json.dumps({"gentle_gap_seconds": 3.5}), encoding="utf-8")
    assert load_config(path=p).gentle_gap_seconds == 3.5
    # rejects negative, bool, wrong type -> default 2.0.
    for bad in (-1, True, "2.0", None):
        p.write_text(json.dumps({"gentle_gap_seconds": bad}), encoding="utf-8")
        assert load_config(path=p).gentle_gap_seconds == 2.0


# ---------------------------------------------------------------------------
# EFFECTIVE max-tokens = min(existing, gentle) when on; unchanged when off
# ---------------------------------------------------------------------------

def test_effective_cap_off_is_passthrough():
    # Gentle off: the base cap is returned exactly (None stays None).
    assert effective_max_tokens(None, False, 1024) is None
    assert effective_max_tokens(4096, False, 1024) == 4096
    assert effective_max_tokens(256, False, 1024) == 256


def test_effective_cap_on_takes_min():
    # Unset/0 base -> gentle cap applies (lowers an uncapped generation).
    assert effective_max_tokens(None, True, 1024) == 1024
    assert effective_max_tokens(0, True, 1024) == 1024
    # Larger existing cap -> lowered to gentle.
    assert effective_max_tokens(4096, True, 1024) == 1024
    # Smaller existing cap -> gentle NEVER raises it; keeps the smaller one.
    assert effective_max_tokens(256, True, 1024) == 256


def test_effective_cap_on_with_unset_gentle_is_passthrough():
    # A non-positive gentle cap is treated as "no gentle cap" -> base unchanged.
    assert effective_max_tokens(4096, True, 0) == 4096
    assert effective_max_tokens(None, True, 0) is None


def test_local_provider_carries_gentle_attrs():
    prov = r.build_provider(
        "local", "m", "http://127.0.0.1:1234/v1",
        gentle_mode=True, gentle_max_tokens=512, max_output_tokens=4096,
    )
    assert prov.gentle_mode is True
    assert prov.gentle_max_tokens == 512
    # The provider would send min(4096, 512) = 512 to the server.
    assert effective_max_tokens(
        prov.max_output_tokens, prov.gentle_mode, prov.gentle_max_tokens
    ) == 512


def _capture_create_kwargs(prov, tools):
    """Drive prov.stream_chat and return the kwargs passed to create()."""
    captured = {}

    class _Stream:
        def __iter__(self):
            return iter([])

        def close(self):
            pass

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Stream()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    prov._client = _Client()
    list(prov.stream_chat([{"role": "user", "content": "hi"}], tools))
    return captured


def test_gentle_cap_bypassed_when_tools_present():
    """The gentle token cap must NOT be sent on a TOOL-capable turn — a tool call
    carries the whole file content in its args, so a 1024 cap truncates the write.
    Pure-chat turns keep the cap. This is the fix for write_file truncation."""
    from llmcli.providers import LocalProvider

    prov = LocalProvider(model="m", base_url="http://127.0.0.1:1234/v1", api_key="k",
                         gentle_mode=True, gentle_max_tokens=1024, max_output_tokens=None)
    tools = [{"type": "function", "function": {"name": "write_file", "parameters": {}}}]
    # Tools present -> no gentle cap (config cap is None -> no max_tokens at all).
    with_tools = _capture_create_kwargs(prov, tools)
    assert "max_tokens" not in with_tools
    # No tools (pure chat) -> the gentle 1024 cap IS applied.
    no_tools = _capture_create_kwargs(prov, None)
    assert no_tools.get("max_tokens") == 1024


# ---------------------------------------------------------------------------
# PURE pacing helper math (no sleeping)
# ---------------------------------------------------------------------------

def test_gentle_wait_within_gap_returns_positive():
    # now - last_end = 0.5s, gap 2.0s -> 1.5s remaining.
    assert r.gentle_wait(2.0, 10.5, 10.0) == pytest.approx(1.5)


def test_gentle_wait_after_gap_returns_zero():
    # User spent 5s typing, gap 2.0s -> no wait.
    assert r.gentle_wait(2.0, 15.0, 10.0) == 0.0


def test_gentle_wait_zero_gap_is_zero():
    # gap <= 0 (gentle effectively off for pacing) -> never wait.
    assert r.gentle_wait(0.0, 10.0, 10.0) == 0.0
    assert r.gentle_wait(-1.0, 10.0, 10.0) == 0.0


def test_gentle_wait_first_turn_no_wait_when_gap_elapsed():
    # last_end=0.0 (no prior gen) and a large now -> elapsed >> gap -> 0.
    assert r.gentle_wait(2.0, 1000.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# /gentle slash command: registration, status, on/off/tokens/gap, persistence
# ---------------------------------------------------------------------------

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
def repl(monkeypatch):
    saved = {}
    disk = Config(provider="mock", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)

    def _save(cfg, *a, **k):
        saved["cfg"] = cfg

    monkeypatch.setattr(r, "save_config", _save)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)
    repl._saved = saved  # expose persisted config for assertions
    return repl


def test_gentle_is_a_known_command():
    assert "/gentle" in r._KNOWN_COMMANDS


def test_gentle_status_is_honest(repl, capsys):
    assert repl._dispatch_slash("/gentle") is True
    out = capsys.readouterr().out
    assert "gentle: on" in out
    assert "4096" in out  # token cap (default)
    assert "2.0s" in out  # gap
    # honesty contract (console may wrap the line, so normalize whitespace).
    assert "does NOT cap GPU %" in " ".join(out.split())


def test_gentle_off_then_on_persists(repl):
    assert repl._dispatch_slash("/gentle off") is True
    assert repl.config.gentle_mode is False
    assert repl._saved["cfg"].gentle_mode is False
    assert repl._dispatch_slash("/gentle on") is True
    assert repl.config.gentle_mode is True
    assert repl._saved["cfg"].gentle_mode is True


def test_gentle_tokens_sets_and_validates(repl, capsys):
    assert repl._dispatch_slash("/gentle tokens 512") is True
    assert repl.config.gentle_max_tokens == 512
    assert repl._saved["cfg"].gentle_max_tokens == 512
    # reject non-positive + non-int (config unchanged).
    repl._dispatch_slash("/gentle tokens 0")
    assert repl.config.gentle_max_tokens == 512
    repl._dispatch_slash("/gentle tokens abc")
    assert repl.config.gentle_max_tokens == 512


def test_gentle_gap_sets_and_validates(repl):
    assert repl._dispatch_slash("/gentle gap 5") is True
    assert repl.config.gentle_gap_seconds == 5.0
    assert repl._saved["cfg"].gentle_gap_seconds == 5.0
    # reject negative + non-number (config unchanged).
    repl._dispatch_slash("/gentle gap -1")
    assert repl.config.gentle_gap_seconds == 5.0
    repl._dispatch_slash("/gentle gap nope")
    assert repl.config.gentle_gap_seconds == 5.0


def test_gentle_provider_reflects_token_change(repl):
    # /gentle tokens rebuilds the provider; mock has no cap surface, but switch to
    # a local provider via config to verify the rebuilt provider carries the cap.
    repl.config.provider = "local"
    repl.config.base_url = "http://127.0.0.1:1234/v1"
    repl._dispatch_slash("/gentle tokens 333")
    assert repl.provider.gentle_max_tokens == 333


# ---------------------------------------------------------------------------
# Pacing is TTY-gated: the suite must NEVER actually sleep
# ---------------------------------------------------------------------------

def test_submit_does_not_sleep_when_not_a_tty(repl, monkeypatch):
    slept = []
    monkeypatch.setattr(r.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(repl.agent, "run", lambda *a, **k: None)
    monkeypatch.setattr(repl, "_save_session", lambda *a, **k: None)
    # console.is_terminal is False under pytest capture -> gate blocks the sleep.
    assert getattr(repl.console, "is_terminal", False) is False
    repl._last_gen_end = r.time.monotonic()  # within the gap
    repl._submit("hi")
    assert slept == []  # never slept


def test_submit_sleeps_on_tty_within_gap(repl, monkeypatch):
    slept = []
    monkeypatch.setattr(r.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(repl.agent, "run", lambda *a, **k: None)
    monkeypatch.setattr(repl, "_save_session", lambda *a, **k: None)
    # Force a TTY + a controlled clock so the cool-down math is deterministic.
    monkeypatch.setattr(type(repl.console), "is_terminal", property(lambda self: True))
    clock = iter([100.0, 101.0])  # gentle_wait's now=100.0, then last_end update
    monkeypatch.setattr(r.time, "monotonic", lambda: next(clock))
    repl.config.gentle_gap_seconds = 2.0
    repl._last_gen_end = 99.0  # 1s ago -> 1s remaining wait
    repl._submit("hi")
    assert slept and slept[0] == pytest.approx(1.0)


def test_submit_no_sleep_when_gentle_off(repl, monkeypatch):
    slept = []
    monkeypatch.setattr(r.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(repl.agent, "run", lambda *a, **k: None)
    monkeypatch.setattr(repl, "_save_session", lambda *a, **k: None)
    monkeypatch.setattr(type(repl.console), "is_terminal", property(lambda self: True))
    repl.config.gentle_mode = False  # OFF -> reproduces today's behavior (no pacing)
    repl._last_gen_end = r.time.monotonic()
    repl._submit("hi")
    assert slept == []
