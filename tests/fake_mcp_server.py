"""A minimal, stdlib-only MCP server over stdio, for tests only.

Speaks the MCP stdio transport: newline-delimited JSON-RPC 2.0 on stdin/stdout.
Implements just enough to exercise the client end-to-end:
  - initialize / notifications/initialized
  - tools/list (an 'echo' tool with readOnlyHint:true and an 'add' tool)
  - tools/call (echo, add, and an unknown-tool -> JSON-RPC error path)

Logs nothing to stdout (that would corrupt the message stream); any diagnostics
go to stderr. Run via ``sys.executable tests/fake_mcp_server.py``.
"""

from __future__ import annotations

import json
import sys

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the provided text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        # No annotations -> readOnlyHint defaults to false (confirmation gated).
    },
]


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _result(req_id, result: dict) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "result": result})


def _error(req_id, code: int, message: str) -> None:
    _write({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _handle(msg: dict) -> None:
    method = msg.get("method")
    req_id = msg.get("id")

    # Notifications have no id and expect no reply.
    if method == "notifications/initialized":
        return

    if method == "initialize":
        _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "0.0.1"},
        })
        return

    if method == "tools/list":
        _result(req_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            text = str(args.get("text", ""))
            _result(req_id, {"content": [{"type": "text", "text": text}], "isError": False})
        elif name == "add":
            try:
                total = float(args.get("a", 0)) + float(args.get("b", 0))
            except (TypeError, ValueError):
                _result(req_id, {
                    "content": [{"type": "text", "text": "add: non-numeric args"}],
                    "isError": True,
                })
                return
            # Render an integer cleanly (3 not 3.0) for a tidy assertion.
            num = int(total) if total == int(total) else total
            _result(req_id, {"content": [{"type": "text", "text": str(num)}]})
        else:
            # Unknown tool -> JSON-RPC protocol error (not a tool error).
            _error(req_id, -32602, f"Unknown tool: {name}")
        return

    if req_id is not None:
        _error(req_id, -32601, f"Method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict):
            _handle(msg)


if __name__ == "__main__":
    main()
