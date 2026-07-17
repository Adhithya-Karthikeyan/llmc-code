"""Enforceable PRIVATE mode: loopback validation, tool-set exclusion, base_url
refusal, run_bash OS-level network sandbox, and MCP egress gating.

All offline/deterministic. The sandbox INTEGRATION test (a real network connect
must fail while a local op succeeds) is GUARDED: it is skipped when sandbox-exec
is absent, so the suite stays green on non-macOS / stripped environments.
"""

from __future__ import annotations

import os
import shutil

import pytest

import llmcli.tools as tools_mod
from llmcli.config import Config, is_loopback_url, load_config
from llmcli.orchestration import (
    make_spawn_agent_tool,
    orchestrator_tool_names,
)
from llmcli.repl import build_provider


# ----- is_loopback_url accept/reject -------------------------------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:1234/v1",
    "http://localhost:1234/v1",
    "http://localhost./v1",            # trailing-dot FQDN root normalized away
    "http://127.5.6.7:8080",           # all of 127.0.0.0/8
    "http://[::1]:1234/v1",            # IPv6 loopback
    "https://LOCALHOST/v1",            # case-insensitive
])
def test_is_loopback_url_accepts_loopback(url):
    assert is_loopback_url(url) is True


@pytest.mark.parametrize("url", [
    "http://example.com/v1",
    "https://api.openai.com/v1",
    "http://0.0.0.0:1234/v1",          # unspecified is NOT loopback
    "http://[::]:1234/v1",             # IPv6 unspecified
    "http://8.8.8.8:1234",
    "http://169.254.169.254/latest",   # link-local metadata endpoint
    "http://10.0.0.5:1234/v1",         # private LAN, still off THIS box
    "",
    "not a url",
])
def test_is_loopback_url_rejects_non_loopback(url):
    assert is_loopback_url(url) is False


def test_is_loopback_url_rejects_localhost_prefixed_hostname():
    # 'localhost.evil.com' must NOT be treated as localhost.
    assert is_loopback_url("http://localhost.evil.com/v1") is False


@pytest.mark.parametrize("url", [
    "http://2130706433/v1",       # bare-integer encoding of 127.0.0.1
    "http://0x7f000001/v1",       # 0x-hex encoding of 127.0.0.1
    "http://017700000001/v1",     # all-digit (octal-looking) encoding
])
def test_is_loopback_url_rejects_noncanonical_numeric_host(url):
    # Finding #4: non-canonical numeric host forms are refused so the validator
    # and the HTTP client cannot diverge on what the host means.
    assert is_loopback_url(url) is False


# ----- tool-set exclusion: web_fetch out under private, in under allow ----

def test_private_excludes_web_fetch_from_orchestrator_tools():
    spawn = make_spawn_agent_tool(provider=None, private=True)
    names = orchestrator_tool_names(spawn, mcp_tools=None, private=True)
    assert "web_fetch" not in names
    # The other built-ins are still present.
    assert "read_file" in names and "run_bash" in names and "spawn_agent" in names


def test_allow_network_includes_web_fetch():
    spawn = make_spawn_agent_tool(provider=None, private=False)
    names = orchestrator_tool_names(spawn, mcp_tools=None, private=False)
    assert "web_fetch" in names


def test_web_fetch_fn_refused_in_private(monkeypatch):
    monkeypatch.setattr(tools_mod, "_PRIVATE", True)
    r = tools_mod._web_fetch({"url": "http://example.com"})
    assert r["ok"] is False
    assert "private mode" in r["error"]


# ----- non-loopback base_url refused in private mode ---------------------

def test_build_provider_refuses_non_loopback_in_private():
    with pytest.raises(ValueError) as exc:
        build_provider("local", "m", "http://example.com/v1", private=True)
    assert "non-loopback" in str(exc.value)


def test_build_provider_allows_loopback_in_private():
    # Must not raise (LocalProvider builds the client lazily, so no network).
    p = build_provider("local", "m", "http://127.0.0.1:1234/v1", private=True)
    assert p.name == "local"


def test_build_provider_allows_non_loopback_when_network_allowed():
    p = build_provider("local", "m", "http://example.com/v1", private=False)
    assert p.name == "local"


# ----- finding #1: provider connection pinned to the validated loopback IP --

def test_resolve_loopback_ip_pins_literal_and_hostname():
    from llmcli.config import resolve_loopback_ip

    # A literal loopback IP is returned verbatim (no resolution can change it).
    assert resolve_loopback_ip("http://127.0.0.1:1234/v1") == "127.0.0.1"
    # 'localhost' resolves to a loopback literal we then pin to.
    assert resolve_loopback_ip("http://localhost:1234/v1") in ("127.0.0.1", "::1")
    # Non-loopback / unpinnable hosts fail closed.
    assert resolve_loopback_ip("http://example.com/v1") is None
    assert resolve_loopback_ip("http://2130706433/v1") is None


def test_pin_loopback_base_url_rewrites_hostname_to_literal_ip():
    from llmcli.providers import _pin_loopback_base_url

    # A literal-IP base_url needs no rewrite (no re-resolution is possible).
    url, host = _pin_loopback_base_url("http://127.0.0.1:1234/v1")
    assert url == "http://127.0.0.1:1234/v1"
    assert host is None
    # A hostname base_url is rewritten to the pinned literal IP, and the original
    # host is preserved as the Host header so the local vhost still matches.
    url, host = _pin_loopback_base_url("http://localhost:1234/v1")
    assert host == "localhost"
    assert ("127.0.0.1:1234" in url) or ("[::1]:1234" in url)
    assert "localhost" not in url  # the hostname is gone -> no re-resolution


def test_pin_loopback_base_url_fails_closed_on_non_loopback():
    from llmcli.providers import _pin_loopback_base_url

    with pytest.raises(ValueError) as exc:
        _pin_loopback_base_url("http://example.com/v1")
    assert "non-loopback" in str(exc.value)


def test_local_provider_client_is_pinned_to_literal_ip():
    """The lazily-built private client connects to the literal IP, not a name."""
    p = build_provider("local", "m", "http://localhost:1234/v1", private=True)
    client = p._get_client()  # builds httpx client + pins; no network call.
    # base_url host is the literal loopback IP, so httpx cannot RE-RESOLVE a
    # hostname per request (DNS-rebinding TOCTOU closed).
    assert ("127.0.0.1" in str(client.base_url)) or ("::1" in str(client.base_url))
    assert "localhost" not in str(client.base_url)


def test_local_provider_default_mode_pins_loopback_hostname(monkeypatch):
    """DEFAULT (network-on) mode still IP-pins a LOOPBACK hostname base_url.

    Optional hardening (finding #1): even without --private, a loopback
    base_url like 'localhost' is pinned to its literal IP so httpx cannot
    re-resolve the hostname per request (DNS-rebinding TOCTOU).
    """
    p = build_provider("local", "m", "http://localhost:1234/v1", private=False)
    client = p._get_client()  # builds httpx client + pins; no network call.
    assert ("127.0.0.1" in str(client.base_url)) or ("::1" in str(client.base_url))
    assert "localhost" not in str(client.base_url)


def test_local_provider_default_mode_leaves_external_base_url_untouched():
    """DEFAULT mode does NOT pin a non-loopback base_url (external is opt-in)."""
    p = build_provider("local", "m", "http://example.com/v1", private=False)
    client = p._get_client()
    # The external hostname is preserved (resolved by httpx at request time).
    assert "example.com" in str(client.base_url)


def test_load_config_refuses_persisted_non_loopback_base_url(tmp_path, capsys):
    p = tmp_path / "config.json"
    p.write_text(
        '{"base_url": "http://evil.example.com/v1", "private": true}',
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    # The poisoned URL is NOT honored; it falls back to the safe default.
    assert cfg.base_url != "http://evil.example.com/v1"
    assert is_loopback_url(cfg.base_url)
    assert "non-loopback" in capsys.readouterr().err


def test_load_config_honors_loopback_base_url(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"base_url": "http://127.0.0.1:9999/v1", "private": true}',
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.base_url == "http://127.0.0.1:9999/v1"


def test_load_config_allows_non_loopback_when_private_off(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"base_url": "http://example.com/v1", "private": false}',
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.base_url == "http://example.com/v1"
    assert cfg.private is False


def test_private_defaults_off():
    # The DEFAULT is now network-ENABLED (private=False). --private is opt-in.
    assert Config().private is False


# ----- run_bash wrapped with sandbox-exec when private -------------------

def test_run_bash_wrapped_with_sandbox_exec_when_private(monkeypatch):
    monkeypatch.setattr(tools_mod, "_PRIVATE", True)
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    captured = {}

    class _FakePopen:
        def __init__(self, target, **kw):
            captured["target"] = target
            captured["shell"] = kw.get("shell")
            raise OSError("stop before real exec")

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    tools_mod._run_bash({"command": "echo hi"})
    target = captured["target"]
    assert isinstance(target, list)
    assert target[0] == "sandbox-exec"
    assert target[1] == "-p"
    assert "deny network-outbound" in target[2]
    assert 'remote ip "localhost:*"' in target[2]
    assert target[3:5] == ["/bin/sh", "-c"]
    # pipefail is prepended (finding #21) so a pipeline failure propagates; the
    # original command is preserved verbatim after it.
    assert target[5] == "set -o pipefail; echo hi"
    assert captured["shell"] is False


def test_run_bash_not_wrapped_when_network_allowed(monkeypatch):
    monkeypatch.setattr(tools_mod, "_PRIVATE", False)
    captured = {}

    class _FakePopen:
        def __init__(self, target, **kw):
            captured["target"] = target
            captured["shell"] = kw.get("shell")
            raise OSError("stop before real exec")

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    tools_mod._run_bash({"command": "echo hi"})
    # finding #4: unsandboxed (no sandbox-exec wrapper) but pipefail is now ALSO
    # applied here with shell=False, so exit-code semantics match the private
    # path. No sandbox-exec in the argv. Prefers /bin/bash (honors `echo -n`),
    # falling back to /bin/sh on minimal images.
    _sh = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    assert captured["target"] == [_sh, "-c", "set -o pipefail; echo hi"]
    assert captured["shell"] is False


def test_run_bash_fails_closed_when_sandbox_exec_missing(monkeypatch):
    monkeypatch.setattr(tools_mod, "_PRIVATE", True)
    monkeypatch.setattr(tools_mod.shutil, "which", lambda _: None)
    r = tools_mod._run_bash({"command": "echo hi"})
    assert r["ok"] is False
    assert "sandbox-exec is unavailable" in r["error"]


# ----- GUARDED integration: real sandbox blocks egress, allows local -----

_HAS_SANDBOX = shutil.which("sandbox-exec") is not None


@pytest.mark.skipif(not _HAS_SANDBOX, reason="sandbox-exec not available")
def test_sandbox_blocks_external_connect(tmp_workspace, monkeypatch):
    """A network connect under the sandbox MUST fail (kernel-blocked)."""
    monkeypatch.setattr(tools_mod, "_PRIVATE", True)
    py = (
        "import socket,sys\n"
        "s=socket.socket(); s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('8.8.8.8',53)); print('REACHED')\n"
        "except Exception as e:\n"
        "    print('BLOCKED', type(e).__name__); sys.exit(7)\n"
    )
    # Write the script into the workspace and run it sandboxed.
    (tmp_workspace / "probe.py").write_text(py, encoding="utf-8")
    import sys as _sys
    r = tools_mod._run_bash({"command": f"{_sys.executable} probe.py", "timeout": 15})
    out = (r["result"]["stdout"] + r["result"]["stderr"])
    assert "REACHED" not in out
    assert r["ok"] is False  # the probe exits non-zero when blocked


@pytest.mark.skipif(not _HAS_SANDBOX, reason="sandbox-exec not available")
def test_sandbox_allows_local_echo_and_file_op(tmp_workspace, monkeypatch):
    """A local echo + file write under the sandbox MUST succeed."""
    monkeypatch.setattr(tools_mod, "_PRIVATE", True)
    r = tools_mod._run_bash({"command": "echo hi > out.txt && cat out.txt"})
    assert r["ok"] is True
    assert "hi" in r["result"]["stdout"]
    assert (tmp_workspace / "out.txt").exists()


# =========================================================================
# NEW DEFAULT: network ENABLED out of the box, with ALWAYS-ON SSRF safety.
# These prove the flip (private=False default) AND that the safety guards
# hold even when network is on.
# =========================================================================

# ----- _PRIVATE default + Config default are network-on -------------------

def test_module_private_flag_defaults_off():
    """The process-wide tools flag is network-on by default (set_private flips it)."""
    import importlib
    import llmcli.tools as fresh
    importlib.reload(fresh)
    assert fresh._PRIVATE is False


# ----- web_fetch is in the DEFAULT tool set (no private arg) ---------------

def test_default_orchestrator_tools_include_web_fetch():
    """With NO private arg (the new default), web_fetch is present + so are built-ins."""
    spawn = make_spawn_agent_tool(provider=None)
    names = orchestrator_tool_names(spawn, mcp_tools=None)
    assert "web_fetch" in names
    assert "read_file" in names and "run_bash" in names and "spawn_agent" in names


def test_default_role_tools_include_web_fetch():
    """Sub-agent roles (explorer/coder/reviewer) keep web_fetch by default."""
    from llmcli.orchestration import _role_tools
    assert "web_fetch" in _role_tools("explorer", private=False)
    assert "web_fetch" in _role_tools("coder", private=False)
    assert "web_fetch" in _role_tools("reviewer", private=False)


# ----- run_bash is NOT sandboxed by default --------------------------------

def test_run_bash_not_sandboxed_by_default(monkeypatch):
    """With the default network-on flag, run_bash runs unsandboxed (no sandbox-exec)."""
    monkeypatch.setattr(tools_mod, "_PRIVATE", False)
    captured = {}

    class _FakePopen:
        def __init__(self, target, **kw):
            captured["target"] = target
            raise OSError("stop before real exec")

    monkeypatch.setattr(tools_mod.subprocess, "Popen", _FakePopen)
    tools_mod._run_bash({"command": "curl https://example.com"})
    # No sandbox-exec wrapper: plain shell -c with pipefail (prefers /bin/bash,
    # falls back to /bin/sh on minimal images).
    _sh = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
    assert captured["target"][0] != "sandbox-exec"
    assert captured["target"][:2] == [_sh, "-c"]


# ----- MCP: ALL configured servers start by default ------------------------

def test_mcp_manager_default_is_network_on():
    """A bare MCPManager (no private arg) is network-on, so an unmarked server starts."""
    from llmcli.mcp import MCPManager
    mgr = MCPManager({"x": {"command": "y", "args": [], "env": {}}})
    assert mgr.private is False


# ----- CRITICAL: SSRF guard STILL blocks internal/metadata with network ON --

def test_web_fetch_ssrf_blocks_loopback_in_default_mode(monkeypatch):
    """SAFETY HOLDS WITH NETWORK ON: web_fetch refuses loopback even when
    _PRIVATE is False (the new default)."""
    monkeypatch.setattr(tools_mod, "_PRIVATE", False)
    r = tools_mod._web_fetch({"url": "http://127.0.0.1:1234/"})
    assert r["ok"] is False
    assert "blocked" in r["error"]


def test_web_fetch_ssrf_blocks_metadata_in_default_mode(monkeypatch):
    """SAFETY HOLDS WITH NETWORK ON: the cloud metadata endpoint is refused."""
    monkeypatch.setattr(tools_mod, "_PRIVATE", False)
    r = tools_mod._web_fetch({"url": "http://169.254.169.254/latest/meta-data/"})
    assert r["ok"] is False
    assert "blocked" in r["error"]


def test_web_fetch_ssrf_blocks_private_lan_in_default_mode(monkeypatch):
    """SAFETY HOLDS WITH NETWORK ON: a private-LAN address is refused."""
    monkeypatch.setattr(tools_mod, "_PRIVATE", False)
    r = tools_mod._web_fetch({"url": "http://10.0.0.5/"})
    assert r["ok"] is False
    assert "blocked" in r["error"]


def test_web_fetch_ssrf_rejects_non_http_scheme_in_default_mode(monkeypatch):
    """SAFETY HOLDS WITH NETWORK ON: non-http(s) schemes are refused (no file://)."""
    monkeypatch.setattr(tools_mod, "_PRIVATE", False)
    assert tools_mod._web_fetch({"url": "file:///etc/passwd"})["ok"] is False


# ----- --private STILL fully locks down ------------------------------------

def test_private_still_locks_down_web_fetch_and_base_url(monkeypatch):
    """--private (opt-in) remains a full lockdown: web_fetch absent from the tool
    set, web_fetch fn refused, and a non-loopback base_url refused."""
    # web_fetch removed from the orchestrator tool set.
    spawn = make_spawn_agent_tool(provider=None, private=True)
    names = orchestrator_tool_names(spawn, mcp_tools=None, private=True)
    assert "web_fetch" not in names
    # web_fetch fn refuses outright.
    monkeypatch.setattr(tools_mod, "_PRIVATE", True)
    assert tools_mod._web_fetch({"url": "http://example.com"})["ok"] is False
    # non-loopback base_url refused.
    with pytest.raises(ValueError):
        build_provider("local", "m", "http://example.com/v1", private=True)
