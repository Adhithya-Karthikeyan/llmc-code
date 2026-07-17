"""Sync stdio Model Context Protocol (MCP) client + manager.

LIGHT, stdlib-only MCP support. Speaks the MCP **stdio transport**: a hand-rolled
synchronous JSON-RPC 2.0 client over a subprocess's stdin/stdout. There is NO
dependency on the official ``mcp`` SDK (or httpx / pydantic / anyio); the whole
thing is the standard library. Only the **stdio** transport is implemented;
HTTP/SSE servers are unsupported.

Wire format (CRITICAL): MCP stdio framing is **newline-delimited JSON**, one
compact JSON object per line terminated by ``\\n``, UTF-8. This is NOT the
Content-Length header framing used by LSP. Messages must not contain embedded
newlines.

Lifecycle per server:
  1. spawn the subprocess (stdin/stdout piped; stderr drained on a thread so a
     chatty server can never fill its stderr pipe and deadlock);
  2. ``initialize`` handshake (send protocolVersion + clientInfo, read the
     result, verify the negotiated version is one we support);
  3. ``notifications/initialized`` (a notification: NO id);
  4. ``tools/list`` to discover the tools the server exposes.

Each request reads stdout via ``select()`` with a timeout so a wedged server can
never hang the CLI: every read is bounded, notifications and id-mismatched
replies are skipped, and a timeout raises ``MCPError`` rather than blocking.

SECURITY: MCP tool results are UNTRUSTED model-facing data, exactly like
``web_fetch`` output. The server's ``readOnlyHint`` annotation is SELF-ASSERTED
by the untrusted server, so it is NOT a trust boundary on its own: by default
ALL MCP tools are confirmation-gated regardless of the hint. The hint only
relaxes the gate when the OPERATOR opts in per-server with
``trustReadOnlyHint: true`` in mcp.json — the trust decision is the user's, not
the server's. Server-controlled tool names are charset-validated (path/shell
metacharacters rejected) and de-collided (a duplicate full name is skipped, never
silently overwritten). The child process env is a minimal allowlist (no provider keys).
"""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import threading
from pathlib import Path

from .tools import Tool, _MAX_OUTPUT, _truncate

# Protocol versions this client understands, newest first. We advertise the
# newest in ``initialize``; the server may negotiate DOWN to any of these.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

CLIENT_NAME = "llmcli"
CLIENT_VERSION = "0.1.0"

# Default per-request read timeout (seconds). A wedged server can never hang the
# CLI: every read is bounded by this.
DEFAULT_TIMEOUT = 30.0
# Grace period (seconds) given to a child to exit after terminate() before kill.
_CLOSE_GRACE = 3.0
# Bounds on what a server's tools/list may contribute (finding #25): a hostile or
# buggy server must not be able to spike memory / bloat the model schema with an
# enormous tool array or a giant inputSchema.
_MAX_TOOLS_PER_SERVER = 256
_MAX_TOOL_SCHEMA_BYTES = 100_000
# Minimal parent-env vars forwarded to MCP children. Deliberately EXCLUDES
# provider/cloud secrets (OPENAI_API_KEY, LMSTUDIO_API_KEY, AWS_*, GITHUB_TOKEN,
# SSH_*, ...): an MCP server is third-party code and must not silently inherit
# them. PATH/HOME let the server start; locale/temp keep it well-behaved.
_CHILD_ENV_ALLOWLIST = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TERM", "TZ",
)
# Config file (Claude Desktop format).
MCP_CONFIG_PATH = Path.home() / ".llm-cli" / "mcp.json"


class MCPError(Exception):
    """Raised on any MCP protocol / transport failure (incl. read timeout)."""


class MCPTimeoutError(MCPError):
    """Raised specifically when a read exceeds the timeout budget.

    A subclass of :class:`MCPError` so existing ``except MCPError`` handlers keep
    working, while callers that must treat a timeout as connection-poisoning (the
    in-flight reply is abandoned in the pipe) can catch it precisely.
    """


def load_mcp_config(path: Path = MCP_CONFIG_PATH) -> dict[str, dict]:
    """Load ``~/.llm-cli/mcp.json`` (Claude Desktop format) -> {name: spec}.

    Format::

        {"mcpServers": {"<name>": {"command","args":[],"env":{},"disabled":false,
                                    "trustReadOnlyHint":false,"timeout":30}}}

    MCP is OPT-IN: a missing file (or a malformed one) yields ``{}`` so the CLI
    runs exactly as before. Servers flagged ``"disabled": true`` are skipped.
    Only entries with a string ``command`` survive.

    ``trustReadOnlyHint`` (default False) is a USER-set, per-server opt-in: when
    True, this operator trusts the server's ``readOnlyHint`` annotation enough to
    auto-run tools it marks read-only. It is NOT server-controlled. ``timeout``
    (default :data:`DEFAULT_TIMEOUT`) is the per-request read timeout in seconds.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return {}
    out: dict[str, dict] = {}
    for name, spec in servers.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            continue
        if spec.get("disabled") is True:
            continue
        command = spec.get("command")
        if not isinstance(command, str) or not command:
            continue
        args = spec.get("args")
        env = spec.get("env")
        timeout = spec.get("timeout")
        out[name] = {
            "command": command,
            "args": list(args) if isinstance(args, list) else [],
            "env": dict(env) if isinstance(env, dict) else {},
            # USER-set trust flag; only an exact True opts in. Anything else
            # (missing, false, truthy-but-not-True) keeps the safe default.
            "trust_read_only_hint": spec.get("trustReadOnlyHint") is True,
            # PRIVATE-mode allow flag: in private mode an MCP server is an egress
            # surface (it could phone home — MCP children are NOT network-
            # sandboxed like run_bash), so it is SKIPPED unless the operator
            # EXPLICITLY marks it safe with "private_ok": true. Only an exact
            # True opts in. The old "local": true alias was DROPPED (finding #3):
            # 'local' reads as a benign descriptor, not an egress grant, so it
            # made the unrestricted-egress decision too easy to enable by
            # accident. The egress-trust decision must be unambiguous: require
            # the pointed "private_ok". (A local vault writer like kyp-mem should
            # be marked "private_ok": true.)
            "private_ok": spec.get("private_ok") is True,
            # Per-server request timeout (seconds); fall back to the default.
            "timeout": (
                float(timeout)
                if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0
                else DEFAULT_TIMEOUT
            ),
        }
    return out


def _content_to_text(content) -> str:
    """Flatten a tools/call ``content`` array to a single string.

    The array can hold mixed item types (text/image/audio/resource_link/
    resource). Text items contribute their ``text``; non-text items contribute a
    short placeholder so the model still learns one was returned. A bare string
    (non-spec but seen in the wild) passes through.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        itype = item.get("type")
        if itype == "text":
            parts.append(str(item.get("text", "")))
        elif itype in ("image", "audio"):
            parts.append(f"[{itype} {item.get('mimeType', '')}]".strip())
        elif itype == "resource_link":
            parts.append(f"[resource_link {item.get('uri', '')}]")
        elif itype == "resource":
            res = item.get("resource")
            if isinstance(res, dict) and isinstance(res.get("text"), str):
                parts.append(res["text"])
            else:
                uri = res.get("uri", "") if isinstance(res, dict) else ""
                parts.append(f"[resource {uri}]")
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


class MCPClient:
    """A synchronous stdio MCP client bound to one server subprocess."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        # Per-request read timeout for this server's calls (operator-configurable
        # so a legitimately long-running tool is not always cut at the default).
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._next_id = 0
        self._stderr_thread: threading.Thread | None = None
        self.protocol_version = ""
        self.server_info: dict = {}
        self._tools: list[dict] = []
        # Set when a tools/call times out: the in-flight reply is left unread in
        # the pipe, so the connection is desynced. We close() and mark it poisoned
        # so no stale reply ever corrupts a later call and no orphan child lingers.
        self.poisoned = False

    # ----- lifecycle -----------------------------------------------------
    def start(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Spawn the server, run the handshake, and load its tools.

        Raises :class:`MCPError` on any failure so the manager can SKIP this one
        server without crashing the CLI.
        """
        # Child env = a MINIMAL allowlisted base (so the server still finds its
        # interpreter/locale/temp) overlaid with the server's configured env. We
        # do NOT copy the full parent environment: that would hand provider API
        # keys (OPENAI_API_KEY/LMSTUDIO_API_KEY) and any other shell secrets to an
        # arbitrary third-party child, which could exfiltrate them. A server that
        # legitimately needs a secret must be given it explicitly via its mcp.json
        # ``env`` block.
        base = {k: os.environ[k] for k in _CHILD_ENV_ALLOWLIST if k in os.environ}
        child_env = {**base, **{str(k): str(v) for k, v in self.env.items()}}
        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=child_env,
                # Run the server in the user's project dir so any server that
                # resolves "the current project" from its cwd (e.g. kyp-mem's
                # cwd-basename fallback) sees the RIGHT project, not wherever the
                # Python process happened to start.
                cwd=os.getcwd(),
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered: flush on every newline we write
            )
        except (OSError, ValueError) as exc:
            raise MCPError(f"failed to spawn '{self.command}': {exc}") from exc

        # Drain stderr on a daemon thread so a server that logs heavily to stderr
        # can never fill the pipe buffer and deadlock the stdout read.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        try:
            self._handshake(timeout)
            self._tools = self._fetch_tools(timeout)
        except Exception:
            # A partial start must not leave a zombie behind.
            self.close()
            raise

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        # Read in fixed-size CHUNKS, not line-by-line (finding #25): a server that
        # emits a single huge line with no newline would force readline to buffer
        # the whole line in memory before discarding it. Chunked reads bound the
        # transient memory regardless of line length. "" => EOF (server closed).
        try:
            while True:
                chunk = proc.stderr.read(65536)
                if not chunk:
                    break  # discard; keeps the pipe from filling
        except (OSError, ValueError):
            pass

    def _handshake(self, timeout: float) -> None:
        result = self.request(
            "initialize",
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
            timeout=timeout,
        )
        if not isinstance(result, dict):
            raise MCPError("initialize returned a non-object result")
        # Version negotiation is NOT symmetric: the server may return a DIFFERENT
        # version than we asked for. If we cannot support what it returned, bail.
        negotiated = result.get("protocolVersion")
        if isinstance(negotiated, str) and negotiated:
            if negotiated not in SUPPORTED_PROTOCOL_VERSIONS:
                raise MCPError(
                    f"server requires unsupported protocol version "
                    f"'{negotiated}' (we support {', '.join(SUPPORTED_PROTOCOL_VERSIONS)})"
                )
            self.protocol_version = negotiated
        else:
            self.protocol_version = LATEST_PROTOCOL_VERSION
        info = result.get("serverInfo")
        self.server_info = info if isinstance(info, dict) else {}
        # Tell the server we are ready. This is a NOTIFICATION (no id, no reply).
        self._notify("notifications/initialized")

    def _fetch_tools(self, timeout: float) -> list[dict]:
        # tools/list params are optional (cursor only); omit them entirely.
        result = self.request("tools/list", None, timeout=timeout)
        if not isinstance(result, dict):
            return []
        tools = result.get("tools")
        if not isinstance(tools, list):
            return []
        # Cap the accepted tools (finding #25): a malicious/buggy server could
        # return an arbitrarily large array that we'd hold in memory and
        # serialize into the model schema. Keep the first N well-formed tools and
        # skip any with an oversized inputSchema.
        out: list[dict] = []
        for t in tools:
            if not (isinstance(t, dict) and isinstance(t.get("name"), str)):
                continue
            schema = t.get("inputSchema")
            if schema is not None and len(json.dumps(schema, ensure_ascii=False)) > _MAX_TOOL_SCHEMA_BYTES:
                continue  # skip a tool whose schema alone would bloat the prompt
            out.append(t)
            if len(out) >= _MAX_TOOLS_PER_SERVER:
                break
        return out

    # ----- JSON-RPC primitives ------------------------------------------
    def _write_message(self, message: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise MCPError(f"server '{self.name}' is not running")
        # Compact, single line, no embedded newlines, newline-terminated.
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise MCPError(f"failed to write to server '{self.name}': {exc}") from exc

    def _notify(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._write_message(message)

    def _read_line(self, timeout: float) -> str:
        """Read one stdout line, bounded by ``timeout``. Raises on timeout/EOF."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise MCPError(f"server '{self.name}' has no stdout")
        # select() on the underlying fd gives a hard wall-clock bound so a wedged
        # server can never hang us. POSIX-only, which is fine (macOS/Linux).
        ready, _, _ = select.select([proc.stdout], [], [], timeout)
        if not ready:
            raise MCPTimeoutError(
                f"timed out after {timeout}s waiting for server '{self.name}'"
            )
        line = proc.stdout.readline()
        if line == "":
            # EOF: the server exited / closed stdout.
            raise MCPError(f"server '{self.name}' closed the connection (exited?)")
        return line

    def request(self, method: str, params: dict | None = None, timeout: float = DEFAULT_TIMEOUT):
        """Send a request and return its ``result`` (or raise :class:`MCPError`).

        Skips notifications and replies whose id doesn't match (a server may
        interleave its own notifications). Honors a hard ``timeout`` budget
        across all the lines read, so it never hangs.
        """
        self._next_id += 1
        req_id = self._next_id
        message = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            message["params"] = params
        self._write_message(message)

        import time as _time

        deadline = _time.monotonic() + timeout
        while True:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                raise MCPTimeoutError(
                    f"timed out after {timeout}s waiting for '{method}' "
                    f"from server '{self.name}'"
                )
            line = self._read_line(remaining).strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON noise on stdout (a misbehaving server). Skip it rather
                # than crash; the timeout still bounds the loop.
                continue
            if not isinstance(msg, dict):
                continue
            msg_id = msg.get("id")
            # Notifications have no id; other-request replies have a different id.
            # JSON-RPC requires the response id to equal the request id, but a
            # server's encoder may not preserve the numeric type (e.g. echo "1"
            # as a string). Match on value OR string form so a string/float echo
            # of our int id is still recognized, not skipped into a spurious
            # timeout. (Python already treats 1 == 1.0, so float echoes match.)
            if msg_id != req_id and str(msg_id) != str(req_id):
                continue
            if "error" in msg and msg["error"] is not None:
                err = msg["error"]
                code = err.get("code") if isinstance(err, dict) else "?"
                emsg = err.get("message") if isinstance(err, dict) else str(err)
                raise MCPError(f"{method} failed (JSON-RPC {code}): {emsg}")
            return msg.get("result")

    # ----- tools ---------------------------------------------------------
    def list_tools(self) -> list[dict]:
        """Return the discovered tools as {name, description, inputSchema, annotations}."""
        out: list[dict] = []
        for t in self._tools:
            schema = t.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            ann = t.get("annotations")
            out.append({
                "name": t["name"],
                "description": t.get("description") or "",
                "inputSchema": schema,
                "annotations": ann if isinstance(ann, dict) else {},
            })
        return out

    def call_tool(self, name: str, arguments: dict, timeout: float | None = None) -> dict:
        """Invoke ``tools/call`` and return the app's standard {ok, result|error}.

        - JSON-RPC errors (unknown tool, bad params) -> {ok: False, error}.
        - A result with ``isError: true`` -> {ok: False, error} (tool-level
          failure, NOT a protocol error).
        - Success -> {ok: True, result: <flattened text>}.

        On a TIMEOUT the connection is poisoned: the server's in-flight reply is
        abandoned in the pipe, so reusing the connection would let that stale
        reply corrupt a later call. We :meth:`close` the client (terminate+kill,
        no orphan) and mark it ``poisoned`` so the manager drops it. Subsequent
        calls fail fast instead of reading a stale reply.
        """
        if self.poisoned:
            return {"ok": False, "error": f"server '{self.name}' connection was abandoned after a timeout"}
        try:
            result = self.request(
                "tools/call",
                {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}},
                timeout=self.timeout if timeout is None else timeout,
            )
        except MCPTimeoutError as exc:
            # Desynced: kill the child and mark unusable so no orphan survives and
            # no stale reply ever poisons a later call on this client.
            self.poisoned = True
            self.close()
            return {"ok": False, "error": str(exc)}
        except MCPError as exc:
            return {"ok": False, "error": str(exc)}
        if not isinstance(result, dict):
            return {"ok": False, "error": "tools/call returned a non-object result"}
        text = _content_to_text(result.get("content"))
        if result.get("isError") is True:
            return {"ok": False, "error": _truncate(text) or "tool reported an error"}
        # Truncate the flattened result to the same byte budget built-in tools use
        # (_MAX_OUTPUT). An MCP server can return arbitrarily large text (e.g.
        # kyp_project_context dumps 20-50KB) straight into the LIVE turn context,
        # which the stale-trim pass does NOT touch. Capping here keeps the per-turn
        # context sane and mirrors what _truncate does for read_file/web_fetch.
        return {"ok": True, "result": _truncate(text)}

    def close(self) -> None:
        """Close stdin, terminate the child, kill after a grace period. No zombies."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except (OSError, ValueError):
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=_CLOSE_GRACE)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=_CLOSE_GRACE)
                    except subprocess.TimeoutExpired:
                        pass
            # Reap and release BOTH read pipes (finding #2): closing only stdout
            # leaked the stderr FD on every connect/disconnect cycle. The drain
            # thread sees EOF once stderr is closed / the child is dead.
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
            # Join the stderr drain thread so its handle isn't dropped while
            # still running; bounded so a wedged grandchild can't hang teardown.
            thread = self._stderr_thread
            if thread is not None:
                thread.join(timeout=1.0)
                self._stderr_thread = None
        finally:
            self._proc = None


# Conservative charset for a server-asserted tool name. Allows the normal
# identifier characters MCP tools use (letters, digits, underscore, dot, hyphen)
# — MUST include ``_`` since almost every real tool name is snake_case
# (e.g. ``kyp_search``, ``read_file``). Path/shell-metacharacter rejection is
# handled in :func:`_make_tool`; ``__`` is intentionally ALLOWED (the real server
# name is always injected into the full name, so a tool name cannot forge another
# server's namespace).
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _make_tool(
    client: MCPClient, server: str, tool_def: dict, trust_read_only_hint: bool = False
) -> Tool | None:
    """Wrap one discovered MCP tool as a local :class:`Tool` (or ``None``).

    Name = ``mcp__<server>__<tool>``. Parameters = the tool's inputSchema (with a
    permissive fallback). The fn closes over ``client`` and routes the call.

    SECURITY: confirmation is ALWAYS required by default. ``readOnlyHint`` is
    asserted by the (untrusted) server itself, so it is NOT a trust boundary: a
    malicious server could mark a destructive tool read-only to auto-run it.
    The hint only relaxes the gate when the OPERATOR has set
    ``trustReadOnlyHint: true`` for this server in mcp.json — i.e. the trust
    decision is the user's, never the server's.

    The server-controlled ``tool_name`` is validated against a conservative
    charset; a name with path/shell metacharacters is REJECTED (returns ``None``).
    ``__`` is allowed: the real server name is always injected into the full
    ``mcp__<server>__<tool>`` name, so a tool name cannot forge another namespace.
    """
    tool_name = tool_def["name"]
    # Reject only path/shell metacharacters (the charset). We do NOT reject ``__``:
    # the full name is always built as mcp__<server>__<tool> with the REAL,
    # operator-configured server name injected here, so a server's tool_name can
    # never forge ANOTHER server's namespace. Collisions are handled by the
    # dedupe in start_all. (Rejecting ``__`` needlessly dropped legitimate tools
    # like kyp-mem's '____kyp_instructions'.)
    if not isinstance(tool_name, str) or not _TOOL_NAME_RE.match(tool_name):
        return None
    full_name = f"mcp__{server}__{tool_name}"
    schema = tool_def.get("inputSchema")
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}}
    annotations = tool_def.get("annotations") or {}
    read_only = bool(annotations.get("readOnlyHint")) if isinstance(annotations, dict) else False
    # Only an operator-trusted server's read-only hint may relax the gate.
    requires_confirmation = not (read_only and trust_read_only_hint)

    def _fn(args: dict) -> dict:
        return client.call_tool(tool_name, args if isinstance(args, dict) else {})

    desc = tool_def.get("description") or f"MCP tool '{tool_name}' from server '{server}'."
    return Tool(
        name=full_name,
        description=f"[mcp:{server}] {desc}",
        parameters=schema,
        fn=_fn,
        requires_confirmation=requires_confirmation,
    )


class MCPManager:
    """Starts configured MCP servers, registers their tools, routes calls."""

    def __init__(self, configs: dict[str, dict] | None = None, private: bool = False):
        # {name: spec} as returned by load_mcp_config.
        self.configs = configs if configs is not None else {}
        # --private LOCKDOWN: in private mode only servers the operator explicitly
        # marked safe (private_ok) are started; others are an egress surface and
        # are SKIPPED. In the DEFAULT (network-on) mode ALL configured servers
        # start.
        self.private = private
        self.clients: dict[str, MCPClient] = {}
        # {full_tool_name: Tool} — the SESSION registry of MCP tools. Never the
        # global import-time tools.REGISTRY.
        self.tools: dict[str, Tool] = {}
        # {server: error-string} for servers that failed to start.
        self.failures: dict[str, str] = {}

    def start_all(self, console=None) -> None:
        """Start every configured server. A failing server is logged + SKIPPED.

        In PRIVATE mode, a server NOT marked ``private_ok``/``local`` in mcp.json
        is skipped with a dim warning (it is an egress surface; do not auto-trust
        arbitrary servers).

        Servers are spawned + handshaked CONCURRENTLY (finding #27): each
        client.start() runs on its own thread, so total startup latency is
        max(server times) instead of sum — a single slow/wedged server no longer
        adds its full timeout to the time before the first prompt. Tool
        registration afterward stays SEQUENTIAL in config order so the
        de-collide (first-wins) behavior is deterministic."""
        # Decide which servers to actually start (private-mode gate first).
        to_start: list[tuple[str, dict, MCPClient]] = []
        for name, spec in self.configs.items():
            if self.private and not spec.get("private_ok", False):
                self.failures[name] = "skipped in private mode (not marked private_ok)"
                if console is not None:
                    console.print(
                        f"mcp: {name} SKIPPED in private mode — an MCP server is an "
                        "egress surface and is NOT network-sandboxed. To grant it "
                        "egress, add \"private_ok\": true to its mcp.json entry "
                        "(this allows UNRESTRICTED outbound network for that "
                        "server), or run --allow-network.",
                        style="dim",
                    )
                continue
            client = MCPClient(
                name=name,
                command=spec.get("command", ""),
                args=spec.get("args", []),
                env=spec.get("env", {}),
                timeout=spec.get("timeout", DEFAULT_TIMEOUT),
            )
            to_start.append((name, spec, client))

        # Spawn + handshake every client in parallel; collect failures per server.
        # The clients are independent stdio subprocesses, so parallel start is
        # safe. Threads join with no extra wall-clock bound here — each start()
        # already self-bounds via its per-request select() timeout.
        def _start_one(spec: dict, client: MCPClient) -> Exception | None:
            try:
                client.start(timeout=spec.get("timeout", DEFAULT_TIMEOUT))
            except Exception as exc:  # noqa: BLE001 - never crash on one server
                return exc
            return None

        threads: list[tuple[str, dict, MCPClient, list]] = []
        for name, spec, client in to_start:
            box: list = [None]
            th = threading.Thread(
                target=lambda s=spec, c=client, b=box: b.__setitem__(0, _start_one(s, c)),
                daemon=True,
            )
            th.start()
            threads.append((name, spec, client, [th, box]))

        # Register tools SEQUENTIALLY in config order (deterministic de-collide).
        for name, spec, client, (th, box) in threads:
            th.join()
            exc = box[0]
            if exc is not None:
                self.failures[name] = str(exc)
                client.close()
                if console is not None:
                    console.print(f"mcp: {name} failed: {exc}", style="dim")
                continue
            self.clients[name] = client
            trust = spec.get("trust_read_only_hint", False) is True
            tools = client.list_tools()
            registered = 0
            for tdef in tools:
                tool = _make_tool(client, name, tdef, trust_read_only_hint=trust)
                # Skip names rejected (forgeable/invalid) or that would overwrite
                # an already-registered tool: never silently shadow an existing
                # one (a later collision could mask a benign tool with a malicious
                # same-named one).
                if tool is None or tool.name in self.tools:
                    if console is not None:
                        bad = tdef.get("name")
                        console.print(
                            f"mcp: {name} skipped tool {bad!r} (invalid or duplicate name)",
                            style="dim",
                        )
                    continue
                self.tools[tool.name] = tool
                registered += 1
            if console is not None:
                console.print(
                    f"mcp: {name} connected ({registered} tools)", style="dim"
                )

    def is_running(self) -> bool:
        """True if any server is currently started (used by the /mcp on toggle to
        avoid re-spawning subprocesses when MCP is already up)."""
        return bool(self.clients)

    def registry(self) -> dict[str, Tool]:
        """The session MCP tool registry {full_name: Tool}."""
        return dict(self.tools)

    def tool_names(self) -> list[str]:
        return list(self.tools)

    def status(self) -> list[dict]:
        """Per-server status: {name, connected, tools:[names], error}."""
        out: list[dict] = []
        for name in self.configs:
            if name in self.clients:
                prefix = f"mcp__{name}__"
                names = [n[len(prefix):] for n in self.tools if n.startswith(prefix)]
                out.append({"name": name, "connected": True, "tools": names, "error": ""})
            else:
                out.append({
                    "name": name,
                    "connected": False,
                    "tools": [],
                    "error": self.failures.get(name, "not started"),
                })
        return out

    def shutdown_all(self) -> None:
        """Cleanly close every started client. Safe to call multiple times."""
        for client in self.clients.values():
            try:
                client.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
        self.clients.clear()
        self.tools.clear()
