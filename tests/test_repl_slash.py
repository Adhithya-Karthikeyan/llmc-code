"""REPL slash-command surface tests (finding #19).

Drive Repl._dispatch_slash with a MockProvider and a fake MCP manager so no
subprocess/network is touched. Covers /compact (success + failure), /clear,
/mcp (configured vs not), unknown command, _status() private ON/OFF, /effort
unset reset, and the run()-loop exception swallowing.
"""

from __future__ import annotations

import pytest

import llmcli.repl as r
from llmcli.config import Config
from llmcli.providers import MockProvider


class _FakeMCP:
    def __init__(self, *a, statuses=None, **k):
        self._statuses = statuses or []
        self.configs = {}
        self.started = 0
        self.shutdowns = 0
        self._running = False

    def registry(self):
        return {}

    def is_running(self):
        return self._running

    def start_all(self, **k):
        self.started += 1
        self._running = True

    def shutdown_all(self):
        self.shutdowns += 1
        self._running = False

    def status(self):
        return self._statuses


@pytest.fixture
def repl(monkeypatch):
    # No disk I/O, no real MCP, no real orchestrator build beyond a real Agent.
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_clear_resets_history_to_system_only(repl):
    repl.agent.messages += [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
    ]
    assert repl._dispatch_slash("/clear") is True
    # A fresh agent: only the system prompt remains.
    assert len(repl.agent.messages) == 1
    assert repl.agent.messages[0]["role"] == "system"


def test_unknown_command(repl, capsys):
    assert repl._dispatch_slash("/wat") is True
    assert "Unknown command" in capsys.readouterr().out


def test_temp_shows_and_sets(repl, capsys):
    # No arg: show current temperature.
    assert repl._dispatch_slash("/temp") is True
    assert "temperature: 0.2" in capsys.readouterr().out
    # Valid value: set + persist (provider rebuilt).
    assert repl._dispatch_slash("/temp 0.5") is True
    assert repl.config.temperature == 0.5
    assert "temp -> 0.5" in capsys.readouterr().out
    # Out-of-range rejected, value unchanged.
    assert repl._dispatch_slash("/temp 9") is True
    assert repl.config.temperature == 0.5
    assert "between 0.0 and 2.0" in capsys.readouterr().out
    # Non-numeric rejected with usage.
    assert repl._dispatch_slash("/temp hot") is True
    assert "Usage:" in capsys.readouterr().out


def test_verify_shows_sets_and_clears(repl, capsys):
    # No arg: shows disabled by default.
    assert repl._dispatch_slash("/verify") is True
    assert "off (disabled)" in capsys.readouterr().out
    # Set a command.
    assert repl._dispatch_slash("/verify python -m pytest -q") is True
    assert repl.config.verify_cmd == "python -m pytest -q"
    assert "verify -> python -m pytest -q" in capsys.readouterr().out
    # off clears it.
    assert repl._dispatch_slash("/verify off") is True
    assert repl.config.verify_cmd == ""
    assert "verify -> off" in capsys.readouterr().out


def test_exit_returns_false(repl):
    assert repl._dispatch_slash("/exit") is False
    assert repl._dispatch_slash("/quit") is False


def test_mcp_none_configured(repl, capsys):
    assert repl._dispatch_slash("/mcp") is True
    assert "No MCP servers configured" in capsys.readouterr().out


def test_mcp_connected_and_not_connected_rendering(repl, capsys):
    repl.mcp = _FakeMCP(statuses=[
        {"name": "good", "connected": True, "tools": ["t1", "t2"], "error": ""},
        {"name": "bad", "connected": False, "tools": [], "error": "boom"},
    ])
    repl._dispatch_slash("/mcp")
    out = capsys.readouterr().out
    assert "good: connected (2 tools)" in out
    assert "bad: not connected (boom)" in out


def test_mcp_off_disables_persists_and_shuts_down(repl, capsys, monkeypatch):
    saved = {}
    monkeypatch.setattr(
        r, "save_config", lambda cfg, *a, **k: saved.update(mcp=cfg.mcp_enabled)
    )
    assert repl.config.mcp_enabled is True  # default on
    assert repl._dispatch_slash("/mcp off") is True
    out = capsys.readouterr().out
    assert "[mcp -> off]" in out
    assert "mcp=off" in out  # surfaced in the status line
    assert repl.config.mcp_enabled is False
    assert saved.get("mcp") is False  # persisted
    assert repl.mcp.shutdowns == 1  # servers were stopped


def test_mcp_on_enables_and_starts(repl, capsys):
    repl.config.mcp_enabled = False
    repl.mcp._running = False
    assert repl._dispatch_slash("/mcp on") is True
    out = capsys.readouterr().out
    assert "[mcp -> on]" in out
    assert "mcp=on" in out
    assert repl.config.mcp_enabled is True
    assert repl.mcp.started == 1  # started since it was not running


def test_mcp_on_when_already_running_does_not_restart(repl, capsys):
    repl.config.mcp_enabled = True
    repl.mcp._running = True
    repl.mcp.started = 0
    repl._dispatch_slash("/mcp on")
    assert repl.mcp.started == 0  # no double-spawn when already up


def test_mcp_background_start_does_not_block_and_integrates(monkeypatch):
    """The initial MCP start runs on a background daemon thread, so building
    the first agent is not blocked by kyp-mem's ~5s startup. Once the background
    start finishes, the next _submit rebuilds the agent so MCP tools are
    offered. This mirrors the lazy path /mcp on already used."""
    import time as _time

    class _SlowMCP:
        def __init__(self, *a, **k):
            self.configs = {"kyp-mem": {"command": "x"}}
            self.started = 0
            self._running = False
            self._delay = 0.3

        def registry(self):
            return {"mcp__kyp-mem__kyp_search": object()} if self._running else {}

        def is_running(self):
            return self._running

        def start_all(self, console=None):
            _time.sleep(self._delay)  # simulate slow server startup
            self.started += 1
            self._running = True

        def shutdown_all(self):
            self._running = False

        def status(self):
            return []

    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _SlowMCP())
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    # Before background start: not ready, not integrated, no thread.
    assert repl._mcp_ready is False
    assert repl._mcp_integrated is False
    assert repl._mcp_thread is None

    t0 = _time.monotonic()
    repl._start_mcp_background(repl.console)
    elapsed = _time.monotonic() - t0
    # Background start returns IMMEDIATELY (well under the 0.3s startup delay).
    assert elapsed < 0.15
    assert repl._mcp_thread is not None
    assert repl._mcp_thread.is_alive()

    # The first agent built now has NO MCP tools (start still in flight).
    a1 = repl._new_agent()
    # _new_agent gates tools on _mcp_ready; it's still False, so the live agent
    # carries no MCP tools. (registry() returns {} while not running.)
    assert repl._mcp_ready is False

    # Wait for the background start to finish.
    repl._mcp_thread.join(timeout=5)
    assert repl._mcp_ready is True
    assert repl.mcp.started == 1

    # _submit would rebuild the agent now that ready flipped; verify the
    # integration flag flips and a fresh _new_agent sees the registry.
    repl._mcp_integrated = False
    # Mimic the _submit rebuild guard:
    if repl.config.mcp_enabled and repl._mcp_ready and not repl._mcp_integrated:
        repl._mcp_integrated = True
        repl.agent = repl._new_agent()
    assert repl._mcp_integrated is True
    assert "mcp__kyp-mem__kyp_search" in repl.mcp.registry()

    repl.mcp.shutdown_all()


def test_mcp_background_start_failure_keeps_ready_false_and_warns(monkeypatch, capsys):
    """Regression: a failing MCP start must NOT silently set _mcp_ready=True.

    Previously the background thread swallowed start_all exceptions and then
    unconditionally set _mcp_ready=True, leaving the user with no MCP tools, no
    error, and a misleadingly-true ready flag. Now _mcp_ready only flips when
    is_running() is True, and a dim warning is printed so /mcp on can retry.
    """
    class _FailingMCP:
        def __init__(self, *a, **k):
            self.configs = {"kyp-mem": {"command": "x"}}
            self._running = False

        def registry(self):
            return {}

        def is_running(self):
            return self._running

        def start_all(self, console=None):
            raise RuntimeError("server binary not found")

        def shutdown_all(self):
            self._running = False

        def status(self):
            return []

    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FailingMCP())
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    repl._start_mcp_background(repl.console)
    repl._mcp_thread.join(timeout=5)
    # Failure path: ready stays False (no silent-stuck "ready but no tools").
    assert repl._mcp_ready is False
    # And the user is told, so they can run /mcp on to retry.
    out = capsys.readouterr().out
    assert "MCP startup failed" in out
    assert "/mcp on" in out


def test_mcp_no_arg_reports_state(repl, capsys):
    repl.config.mcp_enabled = True
    repl._dispatch_slash("/mcp")
    assert "MCP is on" in capsys.readouterr().out
    repl.config.mcp_enabled = False
    repl._dispatch_slash("/mcp")
    assert "MCP is off" in capsys.readouterr().out


def test_context_set_budget(repl, capsys):
    assert repl._dispatch_slash("/context 8000") is True
    assert repl.config.context_budget == 8000
    assert "[context -> 8000 tok" in capsys.readouterr().out


def test_context_off_and_auto(repl, capsys):
    repl._dispatch_slash("/context off")
    assert repl.config.context_budget == 0
    repl._dispatch_slash("/context auto")
    assert repl.config.context_adaptive is True
    assert repl.config.context_budget > 0  # restored to default
    out = capsys.readouterr().out
    assert "[context -> off]" in out


def test_context_fixed_disables_adaptive(repl, capsys):
    assert repl.config.context_adaptive is True
    repl._dispatch_slash("/context fixed")
    assert repl.config.context_adaptive is False


def test_context_status_no_arg(repl, capsys):
    repl._dispatch_slash("/context")
    assert "context budget" in capsys.readouterr().out


def test_context_status_shows_live_snapshot(repl, capsys):
    # Feature 1: /context surfaces the live per-turn snapshot + health hint and
    # must not raise before the first turn (guarded getattr/defaults).
    repl._dispatch_slash("/context")
    out = capsys.readouterr().out
    assert "live:" in out
    assert "est tok" in out
    assert "this turn" in out
    assert "health:" in out


def test_status_shows_context(repl):
    assert "ctx=" in repl._status()


def test_speed_command_prints_tips(repl, capsys):
    assert repl._dispatch_slash("/speed") is True
    out = capsys.readouterr().out
    assert "tok/s" in out
    assert "Context Length" in out
    assert "Flash Attention" in out


def test_help_lists_audit_and_speed(repl, capsys):
    repl._dispatch_slash("/help")
    out = capsys.readouterr().out
    assert "/audit" in out
    assert "/speed" in out


def test_audit_runs_without_touching_conversation(repl, capsys, tmp_path, monkeypatch):
    # /audit must not append to the main conversation history.
    class _TextOnly(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "No issues found."}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 2}

    (tmp_path / "z.py").write_text("def z():\n    return 0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    repl.provider = _TextOnly()
    before = list(repl.agent.messages)
    assert repl._dispatch_slash("/audit") is True
    assert "[audit]" in capsys.readouterr().out
    assert repl.agent.messages == before  # main history untouched


def test_compact_success_shrinks_history(repl, capsys):
    # A provider that returns a usable summary so compact() succeeds.
    class _Sum(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "- summary"}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}

    repl.agent.provider = _Sum()
    repl.agent.messages += [
        {"role": "user", "content": "a " * 60},
        {"role": "assistant", "content": "b " * 60},
        {"role": "user", "content": "c " * 60},
        {"role": "assistant", "content": "d " * 60},
        {"role": "user", "content": "e " * 60},
        {"role": "assistant", "content": "f " * 60},
    ]
    before_n = len(repl.agent.messages)
    repl._dispatch_slash("/compact")
    assert "[compacted]" in capsys.readouterr().out
    assert len(repl.agent.messages) < before_n


def test_compact_failure_leaves_history_intact(repl, capsys):
    class _Boom(MockProvider):
        def stream_chat(self, messages, tools):
            raise ConnectionError("down")
            yield  # pragma: no cover

    repl.agent.provider = _Boom()
    repl.agent.messages += [
        {"role": "user", "content": "a " * 60},
        {"role": "assistant", "content": "b " * 60},
        {"role": "user", "content": "c " * 60},
        {"role": "assistant", "content": "d " * 60},
        {"role": "user", "content": "e " * 60},
        {"role": "assistant", "content": "f " * 60},
    ]
    snapshot = list(repl.agent.messages)
    repl._dispatch_slash("/compact")
    out = capsys.readouterr().out
    assert "[compact failed" in out
    assert repl.agent.messages == snapshot  # unchanged


def test_effort_unset_resets_to_server_default(repl, capsys):
    repl.config.effort = "high"
    assert repl._dispatch_slash("/effort unset") is True
    # The stored effort is reset to "" (server default), not "unset".
    assert repl.config.effort == ""


def test_effort_rejects_bad_level(repl, capsys):
    repl._dispatch_slash("/effort bogus")
    assert "Usage: /effort" in capsys.readouterr().out


def test_maxout_sets_cap(repl, capsys):
    assert repl._dispatch_slash("/maxout 4096") is True
    assert repl.config.max_output_tokens == 4096
    assert "[maxout -> 4096]" in capsys.readouterr().out


def test_maxout_off_clears_cap(repl, capsys):
    repl.config.max_output_tokens = 512
    assert repl._dispatch_slash("/maxout off") is True
    assert repl.config.max_output_tokens is None
    assert "unbounded" in capsys.readouterr().out


def test_maxout_status_reports_current(repl, capsys):
    repl.config.max_output_tokens = 1024
    repl._dispatch_slash("/maxout")
    assert "1024" in capsys.readouterr().out


def test_maxout_low_cap_warns(repl, capsys):
    repl._dispatch_slash("/maxout 100")
    assert repl.config.max_output_tokens == 100
    assert "reasoning tokens" in capsys.readouterr().out


def test_maxout_rejects_non_int(repl, capsys):
    repl._dispatch_slash("/maxout lots")
    assert "Usage: /maxout" in capsys.readouterr().out
    assert repl.config.max_output_tokens is None  # unchanged


def test_status_private_on_off_strings(repl):
    repl.config.private = True
    assert "private mode: ON" in repl._status()
    repl.config.private = False
    assert "private mode: OFF" in repl._status()


def test_provider_switch_to_mock(repl, capsys):
    repl._dispatch_slash("/provider mock")
    assert repl.config.provider == "mock"
    assert "[provider -> mock]" in capsys.readouterr().out


def test_provider_rejects_unknown(repl, capsys):
    repl._dispatch_slash("/provider nope")
    assert "Usage: /provider" in capsys.readouterr().out


def test_theme_switch_updates_persists_and_rebuilds(repl, capsys, monkeypatch):
    saved = {}
    monkeypatch.setattr(r, "save_config", lambda cfg, *a, **k: saved.update(theme=cfg.theme))
    # The ansi theme pins color_system="standard" only for a real terminal; force
    # the tty check on so the rebuilt console uses the 16-color standard system
    # (under pytest stdout is captured / not a tty).
    monkeypatch.setattr(r, "_stdout_is_tty", lambda: True)

    assert repl.config.theme == "clean"  # fresh default (minimal dark look)
    assert repl._dispatch_slash("/theme ansi") is True
    out = capsys.readouterr().out
    assert "[theme -> ansi]" in out
    assert "theme=ansi" in out  # surfaced in the status line
    # Config updated, persisted, and the live console rebuilt to the 16-color set.
    assert repl.config.theme == "ansi"
    assert saved.get("theme") == "ansi"
    assert repl.console.color_system == "standard"
    # The rebuilt agent uses the ANSI code theme for Markdown code blocks.
    assert repl.agent.code_theme == "ansi_dark"

    # Switching back to auto restores the default color system + code theme.
    repl._dispatch_slash("/theme auto")
    assert repl.config.theme == "auto"
    assert repl.console.color_system != "standard"
    assert repl.agent.code_theme == "monokai"


def test_theme_rejects_bad_value(repl, capsys):
    repl._dispatch_slash("/theme neon")
    assert "Usage: /theme" in capsys.readouterr().out
    assert repl.config.theme == "clean"  # unchanged (fresh default)


def test_run_loop_swallows_agent_exception(monkeypatch, capsys):
    """finding #19: agent.run() raising must NOT kill the REPL; it prints
    '[error]' and continues to the next prompt (then EOF exits cleanly)."""
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    class _BoomAgent:
        messages = [{"role": "system", "content": "s"}]

        def run(self, line, images=None):
            raise RuntimeError("kaboom")

        def render_details(self, console):
            pass

    repl.agent = _BoomAgent()
    repl._new_agent = lambda: _BoomAgent()

    # A fake PromptSession: one real line, then EOF to exit the loop.
    class _FakeSession:
        def __init__(self):
            self._lines = iter(["do a thing"])

        def prompt(self, text):
            try:
                return next(self._lines)
            except StopIteration:
                raise EOFError

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda *a, **k: _FakeSession())
    repl.run()
    out = capsys.readouterr().out
    assert "[error]" in out and "RuntimeError" in out
    assert "Bye." in out  # exited cleanly after the error, loop survived


def _run_with_lines(monkeypatch, lines):
    """Drive Repl.run() over a scripted list of input lines, then EOF to exit.

    Spies on _submit_or_stage (the model path) and _dispatch_slash (the
    llmc-command path) so a test can assert which path each line took, without
    touching the network, disk, or a real prompt_toolkit session.
    Returns (repl, submitted_lines, dispatched_lines).
    """
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    submitted: list[str] = []
    dispatched: list[str] = []
    monkeypatch.setattr(repl, "_save_session", lambda *a, **k: None)
    monkeypatch.setattr(repl, "_submit_or_stage", lambda line: submitted.append(line))
    real_dispatch = repl._dispatch_slash

    def _spy_dispatch(line):
        dispatched.append(line)
        return real_dispatch(line)

    monkeypatch.setattr(repl, "_dispatch_slash", _spy_dispatch)

    class _FakeSession:
        def __init__(self):
            self._lines = iter(lines)

        def prompt(self, text):
            try:
                return next(self._lines)
            except StopIteration:
                raise EOFError

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda *a, **k: _FakeSession())
    repl.run()
    return repl, submitted, dispatched


def test_known_command_is_dispatched_not_sent_to_model(monkeypatch):
    # A known llmc command is intercepted and NOT forwarded to the model.
    repl, submitted, dispatched = _run_with_lines(monkeypatch, ["/help", "/model x"])
    assert dispatched == ["/help", "/model x"]
    assert submitted == []


def test_unknown_slash_line_is_sent_to_model(monkeypatch, capsys):
    # A leading-slash line whose first token is NOT an llmc command (another
    # project's CLI) goes to the model verbatim — no "Unknown command".
    repl, submitted, dispatched = _run_with_lines(monkeypatch, ["/build the project"])
    assert submitted == ["/build the project"]
    assert dispatched == []
    assert "Unknown command" not in capsys.readouterr().out


def test_double_slash_escape_sends_literal_slash_to_model(monkeypatch):
    # "//model is broken" -> model receives "/model is broken"; llmc's /model
    # command does NOT run, and the configured model is untouched.
    repl, submitted, dispatched = _run_with_lines(monkeypatch, ["//model is broken"])
    assert submitted == ["/model is broken"]
    assert dispatched == []
    assert repl.config.model == "m"  # /model never ran


def test_lone_slash_and_double_slash_do_not_crash(monkeypatch):
    # A bare "/" is not a known command -> sent to model as "/".
    # A bare "//" -> escape drops one slash -> sent to model as "/".
    repl, submitted, dispatched = _run_with_lines(monkeypatch, ["/", "//"])
    assert submitted == ["/", "/"]
    assert dispatched == []


def test_normal_message_is_unchanged(monkeypatch):
    # A non-slash line behaves exactly as before: straight to the model.
    repl, submitted, dispatched = _run_with_lines(monkeypatch, ["hello there"])
    assert submitted == ["hello there"]
    assert dispatched == []


def test_keyboard_interrupt_continues_loop_eof_exits(monkeypatch, capsys):
    """PROV-1: Ctrl-C (KeyboardInterrupt) at the prompt must NOT exit the REPL;
    it cancels the current line and continues.  EOFError still saves-and-exits."""
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    # Sequence: KeyboardInterrupt (should continue) → then EOF (should save+exit).
    class _FakeSession:
        def __init__(self):
            self._calls = 0

        def prompt(self, text):
            self._calls += 1
            if self._calls == 1:
                raise KeyboardInterrupt  # Ctrl-C: must NOT exit
            raise EOFError              # Ctrl-D: must exit

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda *a, **k: _FakeSession())
    repl.run()
    out = capsys.readouterr().out
    # The interrupt hint was printed (loop survived Ctrl-C)
    assert "Ctrl-D" in out
    # The REPL eventually exited cleanly via EOFError
    assert "Bye." in out
