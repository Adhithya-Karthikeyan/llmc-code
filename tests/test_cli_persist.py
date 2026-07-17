"""CLI flag persistence: session-only by default, persisted only with --save.

Regression guard for the bug where a throwaway run like
``llmc --provider mock -p ...`` silently overwrote the user's saved default
provider/model in ~/.llm-cli/config.json.
"""

from __future__ import annotations

import pytest

import llmcli.__main__ as m
import llmcli.tools as _tools_mod
from llmcli.config import Config


@pytest.fixture(autouse=True)
def _restore_private(monkeypatch):
    """main() calls set_private(), which mutates a process-wide flag. Pin it via
    monkeypatch so it is restored after each test and never leaks into others."""
    monkeypatch.setattr(_tools_mod, "_PRIVATE", _tools_mod._PRIVATE)


def _patch(monkeypatch):
    """Stub load_config/save_config/run_once/build_provider; record saves."""
    saved: list[Config] = []
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Config(provider="local", model="qwen/qwen3.6-35b-a3b"))
    monkeypatch.setattr(m, "save_config", lambda cfg, *a, **k: saved.append(cfg))
    # Avoid constructing a real provider or running a turn.
    monkeypatch.setattr(m, "build_provider", lambda *a, **k: object())
    monkeypatch.setattr(m, "run_once", lambda *a, **k: "")
    return saved


def test_oneshot_flags_do_not_persist(monkeypatch):
    saved = _patch(monkeypatch)
    rc = m.main(["--provider", "mock", "--model", "qwen", "--max-iterations", "14", "-p", "hi"])
    assert rc == 0
    assert saved == []  # nothing written: saved default is untouched


def test_save_flag_persists(monkeypatch):
    saved = _patch(monkeypatch)
    rc = m.main(["--provider", "mock", "--model", "qwen", "--save", "-p", "hi"])
    assert rc == 0
    assert len(saved) == 1
    assert saved[0].provider == "mock"
    assert saved[0].model == "qwen"


def test_no_flags_no_persist(monkeypatch):
    saved = _patch(monkeypatch)
    rc = m.main(["-p", "hi"])
    assert rc == 0
    assert saved == []


def test_private_base_url_flag_refused(monkeypatch):
    """A non-loopback --base-url is refused up front when --private is opted into."""
    _patch(monkeypatch)
    rc = m.main(["--private", "--base-url", "http://example.com/v1", "-p", "hi"])
    assert rc == 2


def test_default_permits_non_loopback_base_url(monkeypatch):
    """The new default (network-on) lets a non-loopback base_url through, no flag."""
    _patch(monkeypatch)
    rc = m.main(["--base-url", "http://example.com/v1", "-p", "hi"])
    assert rc == 0


def test_allow_network_permits_non_loopback_base_url(monkeypatch):
    """--allow-network (now a no-op alias) still lets a non-loopback base_url through."""
    _patch(monkeypatch)
    rc = m.main(["--allow-network", "--base-url", "http://example.com/v1", "-p", "hi"])
    assert rc == 0


def test_allow_network_persisted_with_save(monkeypatch):
    saved = _patch(monkeypatch)
    rc = m.main(["--allow-network", "--save", "-p", "hi"])
    assert rc == 0
    assert len(saved) == 1
    assert saved[0].private is False


def test_private_lockdown_persisted_with_save(monkeypatch):
    """--private --save persists the opt-in lockdown (private=True)."""
    saved = _patch(monkeypatch)
    # Loopback base_url so the private-mode base_url gate passes.
    rc = m.main(["--private", "--base-url", "http://127.0.0.1:1234/v1", "--save", "-p", "hi"])
    assert rc == 0
    assert len(saved) == 1
    assert saved[0].private is True


def test_main_build_provider_error_returns_2(monkeypatch, capsys):
    """finding #34: a build_provider ValueError -> rc 2 with 'error:' on stderr."""
    monkeypatch.setattr(m, "load_config", lambda *a, **k: Config(provider="local"))
    monkeypatch.setattr(m, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(m, "run_once", lambda *a, **k: "")

    def _boom(*a, **k):
        raise ValueError("bad provider config")

    monkeypatch.setattr(m, "build_provider", _boom)
    rc = m.main(["-p", "hi"])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_run_once_shuts_down_mcp_on_agent_error(monkeypatch):
    """finding #34: run_once must shut MCP servers down even when agent.run()
    raises (the finally cleanup guarantee)."""
    import llmcli.repl as rmod
    from llmcli.config import Config as Cfg

    calls = {"shutdown": 0}

    class _RecordingMCP:
        def __init__(self, *a, **k):
            pass

        def start_all(self, **k):
            pass

        def registry(self):
            return {}

        def shutdown_all(self):
            calls["shutdown"] += 1

    class _BoomAgent:
        def run(self, prompt):
            raise RuntimeError("agent blew up")

    monkeypatch.setattr(rmod, "MCPManager", _RecordingMCP)
    monkeypatch.setattr(rmod, "load_mcp_config", lambda *a, **k: {})
    monkeypatch.setattr(rmod, "_build_orchestrator", lambda *a, **k: _BoomAgent())
    monkeypatch.setattr(rmod, "_make_console", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        rmod.run_once(object(), Cfg(provider="mock"), "hi", auto_confirm=True)
    assert calls["shutdown"] == 1  # teardown ran despite the exception


def test_repl_incidental_save_does_not_persist_session_privacy(monkeypatch):
    """Finding #2: a session-only --allow-network / --base-url override must NOT
    be persisted by a routine REPL /model|/provider|/effort save.

    Start the REPL with a SESSION config (private=False, off-box base_url) while
    the ON-DISK default is private=True with a loopback base_url. A /model-style
    persist must write the model change but keep private/base_url at their
    on-disk values, mirroring the startup --save contract.
    """
    import llmcli.repl as r
    from llmcli.config import Config

    # On-disk default: private mode ON, loopback base_url.
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="old")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    saved: list[Config] = []
    monkeypatch.setattr(r, "save_config", lambda cfg, *a, **k: saved.append(cfg))
    # Avoid building a real orchestrator/agent or MCP subprocesses.
    monkeypatch.setattr(r, "_build_orchestrator", lambda *a, **k: object())

    class _FakeMCP:
        def __init__(self, *a, **k):
            pass

        def registry(self):
            return {}

        def start_all(self, **k):
            pass

        def shutdown_all(self):
            pass

        def status(self):
            return []

    monkeypatch.setattr(r, "MCPManager", _FakeMCP)

    # SESSION config: --allow-network --base-url http://example.com/v1 (NOT saved).
    session = Config(
        provider="mock", private=False, base_url="http://example.com/v1", model="old"
    )
    repl = r.Repl(config=session, provider=object(), auto_confirm=True)
    assert repl._disk_private is True
    assert repl._disk_base_url == "http://127.0.0.1:1234/v1"

    # A routine /model change persists.
    repl.config.model = "qwen3"
    repl._persist_config()

    assert len(saved) == 1
    written = saved[0]
    assert written.model == "qwen3"            # the deliberate change persists
    assert written.private is True             # session opt-out NOT persisted
    assert written.base_url == "http://127.0.0.1:1234/v1"  # session URL NOT persisted
