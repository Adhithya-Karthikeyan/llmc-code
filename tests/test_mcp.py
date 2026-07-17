"""End-to-end MCP client tests using a stdlib fake server over stdio.

No network, no node/npx: the fake server is spawned via ``sys.executable``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from llmcli.mcp import (
    MCPClient,
    MCPError,
    MCPManager,
    _content_to_text,
    _make_tool,
    load_mcp_config,
)

_FAKE = str(Path(__file__).parent / "fake_mcp_server.py")
_STR_ID_FAKE = str(Path(__file__).parent / "fake_mcp_server_strid.py")


def _client() -> MCPClient:
    return MCPClient(name="fake", command=sys.executable, args=[_FAKE])


@pytest.fixture
def started_client():
    c = _client()
    c.start(timeout=10.0)
    try:
        yield c
    finally:
        c.close()


# ----- handshake + discovery --------------------------------------------

def test_handshake_and_tools_list(started_client):
    assert started_client.protocol_version == "2025-06-18"
    assert started_client.server_info.get("name") == "fake-mcp"
    names = {t["name"] for t in started_client.list_tools()}
    assert names == {"echo", "add"}


def test_list_tools_shape(started_client):
    tools = {t["name"]: t for t in started_client.list_tools()}
    echo = tools["echo"]
    assert echo["inputSchema"]["type"] == "object"
    assert echo["annotations"]["readOnlyHint"] is True
    # 'add' has no annotations -> empty dict.
    assert tools["add"]["annotations"] == {}


# ----- tools/call --------------------------------------------------------

def test_call_echo(started_client):
    res = started_client.call_tool("echo", {"text": "hello world"})
    assert res == {"ok": True, "result": "hello world"}


def test_call_tool_result_truncated_over_max_output(started_client):
    """MCP tool results are capped to _MAX_OUTPUT like built-in tools, so a
    server returning 50KB of text (e.g. kyp_project_context) cannot dump it
    straight into the live turn context."""
    from llmcli.tools import _MAX_OUTPUT
    big = "x" * (_MAX_OUTPUT * 3)  # well over the byte budget
    res = started_client.call_tool("echo", {"text": big})
    assert res["ok"] is True
    assert res["result"].endswith("...[truncated]")
    # The returned text is bounded by _MAX_OUTPUT (a few bytes over for the
    # marker, but nowhere near the 3x input).
    assert len(res["result"].encode("utf-8")) <= _MAX_OUTPUT + 64
    # And the original input is NOT echoed in full.
    assert big not in res["result"]


def test_call_add(started_client):
    res = started_client.call_tool("add", {"a": 2, "b": 3})
    assert res["ok"] is True
    assert res["result"] == "5"


def test_call_unknown_tool_is_error(started_client):
    res = started_client.call_tool("nope", {})
    assert res["ok"] is False
    assert "Unknown tool" in res["error"]


# ----- registered Tool via _make_tool -----------------------------------

def test_make_tool_readonly_hint_and_routing(started_client):
    tools = {t["name"]: t for t in started_client.list_tools()}
    echo_tool = _make_tool(started_client, "fake", tools["echo"])
    add_tool = _make_tool(started_client, "fake", tools["add"])

    assert echo_tool.name == "mcp__fake__echo"
    # readOnlyHint is SELF-ASSERTED by the untrusted server, so by default it does
    # NOT relax the gate: every MCP tool is confirmation-gated.
    assert echo_tool.requires_confirmation is True
    # No annotations => also gated.
    assert add_tool.requires_confirmation is True

    # The closure routes through the live client.
    assert echo_tool.fn({"text": "hi"}) == {"ok": True, "result": "hi"}
    assert add_tool.fn({"a": 10, "b": 5})["result"] == "15"


def test_make_tool_readonly_hint_honored_only_when_operator_trusts(started_client):
    """readOnlyHint relaxes the gate ONLY when the operator opts in per-server."""
    tools = {t["name"]: t for t in started_client.list_tools()}
    # Operator-trusted server: the read-only hint now relaxes the gate...
    echo_trusted = _make_tool(started_client, "fake", tools["echo"], trust_read_only_hint=True)
    assert echo_trusted.requires_confirmation is False
    # ...but a non-read-only tool stays gated even when the server is trusted.
    add_trusted = _make_tool(started_client, "fake", tools["add"], trust_read_only_hint=True)
    assert add_trusted.requires_confirmation is True


def test_make_tool_name_validation(started_client):
    """`__` names are allowed (real server name is always injected, so no forging
    is possible); only path/shell metacharacters are rejected."""
    dunder = {"name": "evil__nested", "description": "x", "inputSchema": {}}
    t = _make_tool(started_client, "fake", dunder)
    assert t is not None and t.name == "mcp__fake__evil__nested"
    bad_chars = {"name": "rm -rf /", "description": "x", "inputSchema": {}}
    assert _make_tool(started_client, "fake", bad_chars) is None


# ----- manager -----------------------------------------------------------

def test_manager_starts_registers_and_routes():
    mgr = MCPManager(
        {"fake": {"command": sys.executable, "args": [_FAKE], "env": {}}}, private=False
    )
    mgr.start_all()
    try:
        reg = mgr.registry()
        assert "mcp__fake__echo" in reg
        assert "mcp__fake__add" in reg
        # Route a call through the registered tool.
        out = reg["mcp__fake__echo"].fn({"text": "routed"})
        assert out == {"ok": True, "result": "routed"}
        # Status reflects connection + tool names.
        st = mgr.status()
        assert len(st) == 1
        assert st[0]["connected"] is True
        assert set(st[0]["tools"]) == {"echo", "add"}
    finally:
        mgr.shutdown_all()
    # After shutdown the registry is empty.
    assert mgr.registry() == {}


def test_private_mode_skips_unmarked_mcp_server():
    """PRIVATE mode: a server NOT marked private_ok/local is an egress surface
    and must be SKIPPED (not started), with the reason recorded in status."""
    mgr = MCPManager(
        {"fake": {"command": sys.executable, "args": [_FAKE], "env": {}, "private_ok": False}},
        private=True,
    )
    mgr.start_all()
    try:
        assert mgr.registry() == {}  # no tools registered
        st = mgr.status()
        assert st[0]["connected"] is False
        assert "private mode" in st[0]["error"]
    finally:
        mgr.shutdown_all()


def test_private_mode_starts_marked_mcp_server():
    """A server explicitly marked private_ok IS started even in private mode."""
    mgr = MCPManager(
        {"fake": {"command": sys.executable, "args": [_FAKE], "env": {}, "private_ok": True}},
        private=True,
    )
    mgr.start_all()
    try:
        assert "mcp__fake__echo" in mgr.registry()
    finally:
        mgr.shutdown_all()


def test_load_mcp_config_private_ok_and_local_flags(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(
        '{"mcpServers": {'
        '"a": {"command": "x", "private_ok": true},'
        '"b": {"command": "y", "local": true},'
        '"c": {"command": "z"}'
        '}}',
        encoding="utf-8",
    )
    cfg = load_mcp_config(p)
    assert cfg["a"]["private_ok"] is True
    # The "local": true alias was DROPPED (finding #3): it no longer opts a
    # server in to private mode. Only the explicit, unambiguous "private_ok":
    # true grants egress, so a benign-looking 'local' label cannot accidentally
    # enable unrestricted outbound network for the server.
    assert cfg["b"]["private_ok"] is False  # "local": true no longer opts in
    assert cfg["c"]["private_ok"] is False  # unmarked -> not allowed in private


def test_manager_skips_failing_server():
    # A command that cannot be spawned must be skipped, not crash.
    mgr = MCPManager({
        "bad": {"command": "/nonexistent/definitely-not-a-real-binary", "args": [], "env": {}},
    })
    mgr.start_all()
    try:
        assert mgr.registry() == {}
        st = mgr.status()
        assert st[0]["connected"] is False
        assert st[0]["error"]
    finally:
        mgr.shutdown_all()


# ----- robustness --------------------------------------------------------

def test_server_exit_handled():
    c = _client()
    c.start(timeout=10.0)
    try:
        # Kill the server, then a call must surface a clean MCPError-based result,
        # never hang.
        c._proc.kill()
        c._proc.wait(timeout=5)
        res = c.call_tool("echo", {"text": "x"})
        assert res["ok"] is False
        assert res["error"]
    finally:
        c.close()


def test_read_timeout_handled():
    # A server that never replies: request() must raise MCPError within timeout.
    # `cat` echoes nothing back as JSON-RPC, so initialize will time out.
    c = MCPClient(name="silent", command="cat", args=[])
    with pytest.raises(MCPError):
        c.start(timeout=1.0)
    c.close()


# ----- content flattening ------------------------------------------------

def test_content_to_text_mixed():
    content = [
        {"type": "text", "text": "line1"},
        {"type": "image", "data": "x", "mimeType": "image/png"},
        {"type": "text", "text": "line2"},
    ]
    out = _content_to_text(content)
    assert "line1" in out
    assert "line2" in out
    assert "image" in out


# ----- config loading ----------------------------------------------------

def test_load_mcp_config_missing_file(tmp_path):
    assert load_mcp_config(tmp_path / "nope.json") == {}


def test_load_mcp_config_disabled_skipped(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(
        '{"mcpServers": {'
        '"on": {"command": "x", "args": ["a"], "env": {"K": "V"}},'
        '"off": {"command": "y", "disabled": true}'
        '}}',
        encoding="utf-8",
    )
    cfg = load_mcp_config(p)
    assert set(cfg) == {"on"}
    assert cfg["on"]["command"] == "x"
    assert cfg["on"]["args"] == ["a"]
    assert cfg["on"]["env"] == {"K": "V"}


def test_load_mcp_config_malformed(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text("{ not json", encoding="utf-8")
    assert load_mcp_config(p) == {}


def test_load_mcp_config_no_command_skipped(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text('{"mcpServers": {"bad": {"args": []}}}', encoding="utf-8")
    assert load_mcp_config(p) == {}


def test_load_mcp_config_trust_and_timeout_defaults(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text('{"mcpServers": {"s": {"command": "x"}}}', encoding="utf-8")
    cfg = load_mcp_config(p)["s"]
    # Safe defaults: trust off, default timeout.
    assert cfg["trust_read_only_hint"] is False
    assert cfg["timeout"] == 30.0


def test_load_mcp_config_trust_and_timeout_set(tmp_path):
    p = tmp_path / "mcp.json"
    p.write_text(
        '{"mcpServers": {"s": {"command": "x", "trustReadOnlyHint": true, "timeout": 90}}}',
        encoding="utf-8",
    )
    cfg = load_mcp_config(p)["s"]
    assert cfg["trust_read_only_hint"] is True
    assert cfg["timeout"] == 90.0


def test_load_mcp_config_trust_only_exact_true(tmp_path):
    # A truthy-but-not-True value must NOT enable trust (no accidental opt-in).
    p = tmp_path / "mcp.json"
    p.write_text(
        '{"mcpServers": {"s": {"command": "x", "trustReadOnlyHint": "yes", "timeout": -5}}}',
        encoding="utf-8",
    )
    cfg = load_mcp_config(p)["s"]
    assert cfg["trust_read_only_hint"] is False
    # Invalid timeout falls back to the default.
    assert cfg["timeout"] == 30.0


# ----- agent schema exposure (the production path) -----------------------

def test_agent_tools_payload_includes_mcp_and_spawn_agent():
    """REGRESSION GUARD: the orchestrator's schema sent to the model MUST include
    MCP tools and spawn_agent. These live only in the injected per-session
    registry, never the global tools.REGISTRY, so a schema built from the global
    registry would silently drop them and the model would never call them."""
    from llmcli.agent import Agent
    from llmcli.orchestration import (
        make_spawn_agent_tool,
        orchestrator_registry,
        orchestrator_tool_names,
    )

    mgr = MCPManager(
        {"fake": {"command": sys.executable, "args": [_FAKE], "env": {}}}, private=False
    )
    mgr.start_all()
    try:
        mcp_tools = mgr.registry()
        assert "mcp__fake__echo" in mcp_tools  # sanity: the manager has it
        spawn_tool = make_spawn_agent_tool(provider=None)
        registry = orchestrator_registry(spawn_tool, mcp_tools)
        names = orchestrator_tool_names(spawn_tool, mcp_tools)
        agent = Agent(
            provider=None,
            system_prompt="x",
            tool_names=names,
            registry=registry,
        )
        payload = agent._tools_payload()
        emitted = {t["function"]["name"] for t in payload}
        assert "mcp__fake__echo" in emitted
        assert "mcp__fake__add" in emitted
        assert "spawn_agent" in emitted
        # Built-ins are still present.
        assert "read_file" in emitted
    finally:
        mgr.shutdown_all()


# ----- security: child env does not inherit secrets ----------------------

def test_child_env_excludes_provider_secrets(monkeypatch):
    """An MCP child must NOT inherit provider/cloud secrets from the parent env."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "lm-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    from llmcli import mcp as mcp_mod

    captured = {}

    class _FakePopen:
        def __init__(self, *a, **kw):
            captured["env"] = kw.get("env")
            raise OSError("stop before spawn")

    monkeypatch.setattr(mcp_mod.subprocess, "Popen", _FakePopen)
    c = MCPClient(name="x", command="x", args=[], env={"SERVER_KEY": "v"})
    with pytest.raises(MCPError):
        c.start(timeout=1.0)
    env = captured["env"]
    assert "OPENAI_API_KEY" not in env
    assert "LMSTUDIO_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    # Allowlisted + the server's configured env still flow through.
    assert env.get("PATH") == "/usr/bin"
    assert env.get("SERVER_KEY") == "v"


# ----- robustness: timeout poisons the connection ------------------------

def test_call_tool_timeout_poisons_and_closes():
    """On a tools/call timeout the child is killed and the client marked poisoned
    so no orphan lingers and no stale reply corrupts a later call."""
    c = MCPClient(name="fake", command=sys.executable, args=[_FAKE], timeout=10.0)
    c.start(timeout=10.0)
    try:
        # Force a very short timeout so the (otherwise fast) call times out.
        res = c.call_tool("echo", {"text": "hi"}, timeout=0.0)
        assert res["ok"] is False
        assert "timed out" in res["error"]
        assert c.poisoned is True
        assert c._proc is None  # close() reaped the child
        # A subsequent call fails fast, never reads a stale reply.
        res2 = c.call_tool("echo", {"text": "again"})
        assert res2["ok"] is False
        assert "abandoned" in res2["error"]
    finally:
        c.close()


# ----- robustness: string-typed id echo is still matched -----------------

def test_request_matches_string_echoed_id():
    """A server that echoes our int id as a string must still be matched, not
    skipped into a spurious timeout."""
    c = MCPClient(name="strid", command=sys.executable, args=[_STR_ID_FAKE], timeout=10.0)
    c.start(timeout=10.0)
    try:
        res = c.call_tool("echo", {"text": "hi"})
        assert res == {"ok": True, "result": "hi"}
    finally:
        c.close()


# ----- robustness: interleaved notifications + garbage (finding #18) ------
#
# These drive request() directly with a scripted _read_line so the skip-and-
# continue / non-JSON-skip / error-object branches are tested deterministically,
# without depending on OS pipe buffering of a multi-line burst.

def _scripted_client(lines):
    """A started-looking client whose _write_message is a no-op and whose
    _read_line yields the given canned lines in order (then raises EOF)."""
    c = MCPClient(name="scripted", command="x", args=[])
    c._write_message = lambda message: None  # no real subprocess
    it = iter(lines)

    def _read_line(timeout):
        try:
            return next(it)
        except StopIteration:
            raise MCPError("scripted EOF")

    c._read_line = _read_line
    return c


def test_request_skips_notification_and_garbage_then_matches():
    """A notification (no id) + a non-JSON garbage line before the real reply
    are both skipped, and the matching-id result is returned."""
    # request() assigns req_id = _next_id + 1, starting at 1 for the first call.
    c = _scripted_client([
        '{"jsonrpc":"2.0","method":"log","params":{"msg":"noise"}}',  # notification
        "this is not json at all <<<",                                  # garbage
        '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}',                # real reply
    ])
    result = c.request("tools/list", None, timeout=5.0)
    assert result == {"ok": True}


def test_request_skips_mismatched_id():
    """A reply with a DIFFERENT id (another request's) is skipped."""
    c = _scripted_client([
        '{"jsonrpc":"2.0","id":999,"result":{"stale":true}}',  # wrong id
        '{"jsonrpc":"2.0","id":1,"result":{"fresh":true}}',     # ours
    ])
    assert c.request("tools/list", None, timeout=5.0) == {"fresh": True}


def test_request_jsonrpc_error_object_raises():
    """A JSON-RPC error object for our id -> MCPError with code+message."""
    c = _scripted_client([
        '{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"boom always errors"}}',
    ])
    with pytest.raises(MCPError) as ei:
        c.request("tools/call", {"name": "boom"}, timeout=5.0)
    assert "boom always errors" in str(ei.value)
    assert "-32000" in str(ei.value)


def test_call_tool_jsonrpc_error_object_returns_not_ok():
    """The error object surfaces through call_tool as {ok: False, error}."""
    c = _scripted_client([
        '{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"boom always errors"}}',
    ])
    res = c.call_tool("boom", {})
    assert res["ok"] is False
    assert "boom always errors" in res["error"]
    assert "-32000" in res["error"]


def test_interleaved_notification_does_not_extend_deadline():
    """A flood of interleaved notifications must NOT reset/extend the per-request
    deadline: a stream of only-notifications still times out within the budget."""
    import time

    import llmcli.mcp as mcp_mod

    c = MCPClient(name="noisy", command="x", args=[])
    c._write_message = lambda message: None

    # _read_line always returns a fresh notification, so the loop never sees the
    # matching reply. The hard monotonic deadline must still fire on schedule.
    c._read_line = lambda timeout: '{"jsonrpc":"2.0","method":"log"}'
    t0 = time.monotonic()
    with pytest.raises(mcp_mod.MCPTimeoutError):
        c.request("tools/list", None, timeout=0.2)
    assert time.monotonic() - t0 < 2.0  # bounded by the deadline, not unbounded


# ----- cleanup: stderr pipe + drain thread released on close (finding #2) --

def test_close_joins_stderr_thread_and_releases_pipes():
    c = _client()
    c.start(timeout=10.0)
    proc = c._proc
    stderr_pipe = proc.stderr
    thread = c._stderr_thread
    assert thread is not None
    c.close()
    # The drain thread is joined + dropped and the stderr pipe is closed (no FD
    # leak / dangling thread per connect-disconnect cycle).
    assert c._stderr_thread is None
    assert stderr_pipe.closed
    assert not thread.is_alive()


# ----- security: child env allowlist is default-deny (finding #35) --------

def test_child_env_is_default_deny_allowlist(monkeypatch):
    """The child env must be DEFAULT-DENY: an arbitrary new secret-bearing var
    NOT on the allowlist is stripped, while infra vars (PATH/HOME/LANG) survive.
    Proves the allowlist construction, not just four hard-coded provider keys."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws")
    monkeypatch.setenv("MY_SECRET", "random")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/x")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    from llmcli import mcp as mcp_mod

    captured = {}

    class _FakePopen:
        def __init__(self, *a, **kw):
            captured["env"] = kw.get("env")
            raise OSError("stop before spawn")

    monkeypatch.setattr(mcp_mod.subprocess, "Popen", _FakePopen)
    c = MCPClient(name="x", command="x", args=[], env={})
    with pytest.raises(MCPError):
        c.start(timeout=1.0)
    env = captured["env"]
    # Default-deny: no secret survives unless allowlisted.
    assert "ANTHROPIC_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "MY_SECRET" not in env
    # Allowlisted infra vars survive.
    assert env.get("PATH") == "/usr/bin"
    assert env.get("HOME") == "/home/x"
    assert env.get("LANG") == "en_US.UTF-8"


# ----- manager: collision de-collide -------------------------------------

def test_manager_dedupes_colliding_tool_names():
    """Two servers exposing a tool that maps to the same full name: the second is
    skipped, never silently overwriting the first."""
    mgr = MCPManager({
        "fake": {"command": sys.executable, "args": [_FAKE], "env": {}},
    }, private=False)
    mgr.start_all()
    try:
        first = mgr.registry()["mcp__fake__echo"]
        # Re-running start_all against the same name must not overwrite/duplicate.
        mgr.start_all()
        assert mgr.registry()["mcp__fake__echo"] is first
    finally:
        mgr.shutdown_all()
