"""A fake MCP server (tests only) that echoes the request id as a STRING.

Identical to ``fake_mcp_server`` except it replies with ``id`` coerced to a
string (e.g. 1 -> "1"). JSON-RPC requires the response id to equal the request
id, but a server's encoder need not preserve the numeric type; this locks in
that the client matches a string-echoed id rather than skipping it into a
spurious timeout. stdout carries ONLY JSON-RPC frames; diagnostics go to stderr.
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
]


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _result(req_id, result: dict) -> None:
    # Coerce the id to a STRING to exercise the client's id-type tolerance.
    _write({"jsonrpc": "2.0", "id": str(req_id), "result": result})


def _handle(msg: dict) -> None:
    method = msg.get("method")
    req_id = msg.get("id")

    if method == "notifications/initialized":
        return
    if method == "initialize":
        _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-strid", "version": "0.0.1"},
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
        else:
            _write({"jsonrpc": "2.0", "id": str(req_id),
                    "error": {"code": -32602, "message": f"Unknown tool: {name}"}})
        return
    if req_id is not None:
        _write({"jsonrpc": "2.0", "id": str(req_id),
                "error": {"code": -32601, "message": f"Method not found: {method}"}})


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
