# llm-cli — Architecture

tags: architecture, llm-cli, design, modules
created: 2026-06-22
updated: 2026-06-22

## llm-cli — Architecture

Lightweight LOCAL-ONLY agentic CLI (Claude Code-style) in Python. Source at `/Users/adhithya/Projects/apps/llm-cli`. Entry point: `llmc`. 7 source modules, 95 tests, 95 pytest pass.

## Project Layout

```
llm-cli/
├── pyproject.toml          # setuptools; deps: openai>=2,<3, rich>=13.7.0,<16, prompt_toolkit>=3.0.43,<4; py>3.14; entry: llmc
├── requirements.txt        # stale; openai>=1.40.0 no upper bound
├── README.md
├── llmcli/                 # source package (7 modules)
│   ├── __init__.py         # __version__ = "0.1.0"
│   ├── __main__.py         # CLI entry: arg parsing, --save, -p one-shot, REPL
│   ├── config.py           # Config dataclass, load/save ~/.llm-cli/config.json
│   ├── providers.py        # Provider ABC, LocalProvider, MockProvider, fenced-JSON fallback
│   ├── prompts.py          # ORCHESTRATOR/EXPLORER/CODER/REVIEWER role prompts, OUTPUT_CONTRACT, WRITING_DISCIPLINE
│   ├── tools.py            # Tool dataclass, REGISTRY (7 built-in), openai_schema(), security guards
│   ├── agent.py            # Agent class: run(), compact(), render_details()
│   ├── repl.py             # prompt_toolkit REPL, slash commands, confirm wiring, run_once()
│   ├── orchestration.py    # spawn_agent tool, ROLE_TOOLS, orchestrator_registry/tool_names
│   └── mcp.py              # stdio MCP client (JSON-RPC 2.0, stdlib only), MCPManager
├── llmcli.egg-info/        # auto-generated build metadata
├── tests/                  # 13 test files, all offline, MockProvider-only
│   ├── conftest.py         # tmp_workspace + mock_provider fixtures
│   ├── fake_mcp_server.py  # stdlib MCP server fake
│   ├── fake_mcp_server_strid.py  # MCP server echoing request id as string
│   └── test_*.py           # 11 test modules
└── .omc/                   # orchestration memory state (JSON session state)
```

## Module Details

### `__main__.py` — CLI Entry Point
- `_parse_args()`: argparse; flags: `--provider`, `--model`, `--base-url`, `-p/--prompt`, `-y/--yes`, `--max-iterations`, `--effort`, `--save`.
- `main(argv)`: Loads config → applies CLI overrides → calls `save_config` if `--save` → builds provider → either `run_once()` (one-shot `-p`) or `Repl().run()`.
- **Design rule**: CLI flags are session-only overrides unless `--save` explicitly opted in. Never clobbers defaults on throwaway runs.

### `config.py` — Configuration
- `Config` dataclass: `provider`, `model`, `base_url`, `max_iterations`, `effort`. Serializable via `asdict()`.
- `load_config()`: Loads `~/.llm-cli/config.json`; warns on corrupt JSON, falls back; safe type-checking on every field.
- `save_config()`: Writes JSON; best-effort (swallows OSError).
- `get_api_key()`: Env-var resolution chain: `OPENAI_API_KEY` → `LMSTUDIO_API_KEY` → `"lm-studio"` (hardcoded default). Never written to disk.
- Constants: `PROVIDERS = ("local", "mock")`, `EFFORT_LEVELS = ("off", "low", "medium", "high")`, `DEFAULT_BASE_URL = "http://localhost:1234/v1"`.

### `providers.py` — Provider System
- **Provider ABC**: Abstract `stream_chat(messages, tools) → Iterator[dict]`.
- **Normalized events**: `{type: "text", "text": str}`, `{type: "tool_call", "id", "name", "arguments": dict}`, `{type: "done", "finish_reason", "output_tokens", "gen_elapsed"}`. `done` is always exactly one, always last.
- **LocalProvider**: Wraps OpenAI SDK. Lazy `openai` import (only on `_get_client()` call). `stream_chat` handles:
  - Native tool calls via `tool_calls` deltas
  - Fenced-JSON text fallback for models without native tool support (`parse_tool_block()`, `count_tool_blocks()`, `_fence_is_sole_content()`)
  - `stream_options.include_usage=True` for token counting
  - `_GEN_TIMEOUT = 600s` for generation, `_MODELS_TIMEOUT = 5s` for model listing
  - `effort_extra_body()`: maps effort level to `reasoning_effort` + `chat_template_kwargs` via `extra_body`
- **MockProvider**: Deterministic scripted provider. `_step_from_history()` derives step from message count (stateless, correct across REPL turns). Scripts: `hello` (write + run), `plain` (text), `read` (read_file + text).
- **Fenced-JSON safety guards**: Only executes a fence when it's the SOLE content AND exactly one tool-shaped fence exists. Prevents executing fences that are examples inside prose.
- **Key non-obvious**: qwen3.6-35b-a3b and deepseek-coder do NOT emit native OpenAI tool_calls via LM Studio; works because system prompt enforces a text tool-call protocol + fallback parser.

### `prompts.py` — Prompt Engineering
- `OUTPUT_CONTRACT`: Hard rule for terse, skimmable output (up to 5 headers, bullets, no preamble). Appended to every role prompt.
- `WRITING_DISCIPLINE`: Separate discipline for durable content written to files (complete, specific, structured — opposite of terse chat reply). Appended to ORCHESTRATOR and CODER only.
- Role prompts:
  - `ORCHESTRATOR`: core loop EXPLORE→PLAN→ACT→VALIDATE, spawn_agent delegation
  - `EXPLORER`: read-only investigation
  - `CODER`: full tools, implements + validates
  - `REVIEWER`: read-only critique
- `SUMMARIZER_PROMPT`: for `Agent.compact()`.
- `ROLE_PROMPTS` dict: maps role name → assembled system prompt.

### `tools.py` — Tool System
- `Tool` dataclass: `name`, `description`, `parameters` (JSON Schema), `fn(args)→dict`, `requires_confirmation`.
- `REGISTRY`: module-level dict, populated at import via `register()`.
- `openai_schema()`: builds OpenAI `tools` array from registry. Takes optional `names` subset and `registry` override (critical for orchestrator with injected MCP tools).
- **7 built-in tools**: `read_file`, `write_file`, `edit_file`, `run_bash`, `glob`, `grep`, `web_fetch`.
- `READ_ONLY = ["read_file", "glob", "grep", "web_fetch"]`, `FULL = all 7`.
- **Security**: `_workspace_root()` confines file ops to `cwd`. `_within_workspace()` resolves and checks parentage. `run_bash` strips `OPENAI_API_KEY`/`LMSTUDIO_API_KEY` from child env, uses `start_new_session=True` + `SIGKILL` on timeout.
- `web_fetch`: SSRF guard via `_resolve_safe_ip()` (validates every resolved IP, pins connection). Follows redirects with per-hop re-validation. `_TextExtractor` HTML→text parser. Max 5 redirects, 2MB download, 15s timeout.
- Tool results: always `{"ok": bool, "result"|"error": ...}`. Never raise.

### `agent.py` — Agent Loop
- `Agent` class: core agentic loop. Owns `messages` (conversation history), `provider`, `tool_names`, `registry`.
- `run(user_text) → final_text`: Main loop (up to `max_iterations`):
  1. User input → `{role: "user", content: text}`
  2. Build tools payload from `self.tool_names` + `self.registry` (not global REGISTRY — critical)
  3. Stream loop: text accumulated in `text_acc` (buffered, rendered once); tool_calls accumulated; on `done`: captures `output_tokens`, `finish_reason`, `gen_elapsed`
  4. No tool_calls → final answer: normalize preamble, render Markdown, tok/s footer, append to messages, return
  5. Has tool_calls → append assistant tool_calls message → execute each tool (confirm gate) → append tool-result messages → loop
- Consecutive parse-error circuit-breaker: abort after 3 turns.
- `_from_text_fence` flag: prevents storing consumed narration as assistant content (would pollute history since model sees it as both spoken and executed).
- `normalize_final_answer()`: Strips common preamble openers ("Sure,", "Let me", etc.) except code fences or headers.
- `_print_markdown()`: Buffers turn text, renders once as Rich Markdown.
- Collapsed tool lines: `⊕ tool_name summary` — one line per turn.
- `compact()`: Summarizes history via provider, replaces early messages with summary. Returns `(before_tokens, after_tokens)`. Raises `RuntimeError` on failure (preserves history). Edge case: 2 user turns = no-op.
- `render_details(console)`: Ctrl+O reveal of `last_turn_details` buffer.

### `repl.py` — REPL System
- `Repl` class: `prompt_toolkit` session, Rich console, slash commands, MCP lifecycle.
- **Slash commands**: `/help`, `/provider`, `/models`, `/model`, `/effort`, `/compact`, `/mcp`, `/clear`, `/exit`, `/quit`.
- `/model`: Verifies against server model list; allows when unreachable (warns); mock skips check entirely.
- `make_ptk_confirm(session)`: prompt_toolkit-safe y/N confirmation (avoids builtin `input()` deadlock).
- `run_once(provider, config, prompt, auto_confirm)`: One-shot mode. Creates MCPManager, starts servers, builds orchestrator with MCP tools merged in, runs agent, shuts down MCP.
- `_new_agent()`: Builds orchestrator with MCP tools + spawn_agent.
- `_status()`: Shows `provider=... model=... effort=... [base_url=...]`.
- Ctrl+O keybinding: Calls `agent.render_details(console)`, invalidates prompt. Only works at prompt (line-based), not mid-stream.

### `orchestration.py` — Multi-Agent System
- `make_spawn_agent_tool()`: Creates `spawn_agent` Tool with `_spawn()` closure. Creates fresh `Agent` with role-scoped system prompt + tools, runs it, returns summary. `requires_confirmation=True` (delegation itself is gated).
- `ROLE_TOOLS`: `explorer→READ_ONLY`, `coder→FULL`, `reviewer→READ_ONLY`.
- `orchestrator_registry(spawn_tool, mcp_tools)`: Builds `{full_tools + spawn_agent + mcp_tools}`. Never mutates global `REGISTRY`.
- `orchestrator_tool_names(...)`: Returns `FULL + [spawn_agent] + mcp_names`.

### `mcp.py` — MCP System
- `MCPClient`: stdio JSON-RPC 2.0 client. Handshake (`initialize`→`notifications/initialized`→`tools/list`). `select()`-bounded reads prevent hangs. `close()` terminates/kills child with grace period.
- `MCPManager`: Starts configured servers from `~/.llm-cli/mcp.json`, registers tools as `mcp__<server>__<tool>` names. Dedupes collisions. `load_mcp_config()` reads Claude Desktop format.
- **Security**: Child env = allowlisted base + server config. NO provider/cloud secrets forwarded. Tool names validated against `[A-Za-z0-9._-]+`. `readOnlyHint` only relaxes confirmation when operator sets `trustReadOnlyHint: true`.
- **Timeout poisoning**: On `tools/call` timeout, child is killed, client marked `poisoned`, subsequent calls fail fast.
- **ID type tolerance**: Matches `str(msg_id) == str(req_id)` to handle servers that echo int ids as strings.

## Agent Loop — Detailed Flow

1. User input → `{role: "user", content: text}` appended to `self.messages`
2. Tools payload built from `self.tool_names` + `self.registry` (not global REGISTRY)
3. Stream loop: text→`text_acc` (buffered, rendered once), tool_calls→accumulated list, done→captures `output_tokens`, `finish_reason`, `gen_elapsed`
4. No tool_calls → final answer: normalize preamble, render Markdown, tok/s footer, append to messages, return
5. Has tool_calls → append assistant tool_calls message → for each tool: parse error check → execute tool → record detail → append tool-result message → one collapsed dim line for batch → circuit-breaker: 3 consecutive parse-error turns → abort → loop back
6. Max iterations → dim warning, return partial text

## Key Edge Cases & Hacks

- **Fenced-JSON fallback**: Models without native tool-call support use ` ```json {"tool":"name","input":{...}} ``` `. Only executed when SOLE content AND exactly one tool-shaped fence.
- **`_from_text_fence` flag**: Prevents storing consumed narration as assistant content (pollutes history since model sees it as both spoken and executed).
- **Reasoning token timing**: `gen_elapsed` from provider starts from first token of ANY kind (including `reasoning_content`), matching `completion_tokens`. Agent's local `t_first` fallback only uses visible tokens.
- **`compute_tok_stats`**: `bool` is subclass of `int` in Python; explicitly guards against `isinstance(output_tokens, bool)` to prevent `True` counting as 1 token.
- **Normalize final answer**: Only strips if answer doesn't start with code fence or markdown header. Only strips when sentence break exists after opener.
- **MCP `readOnlyHint`**: Self-asserted by untrusted server. NEVER trusted by default. Only relaxes gate when operator sets `trustReadOnlyHint: true` per-server.
- **MCP timeout poisoning**: On timeout, child killed + client marked poisoned, so stale replies from wedged server can never corrupt subsequent calls.
- **Rich `markup=False`**: All console text treated as plain — raw model output often contains brackets that would break Rich markup.
- **MCP JSON-RPC id type tolerance**: `str(msg_id) == str(req_id)` handles servers echoing int ids as strings.
- **read_file extensionless fallback**: Tries `.md`, `.txt`, `.rst` suffixes for files without extensions.
- **run_bash process groups**: `start_new_session=True` + `os.killpg()` SIGKILL on timeout kills entire process group, preventing orphaned grandchild processes.

## Test Coverage (95 tests, all offline)

| File | Tests | What's Tested |
|------|-------|---------------|
| `test_agent_loop.py` | 15 | Tool execution, confirmation, auto-confirm, max_iterations, detail buffer, parse-error abort, sub-agent line_prefix, empty reasoning turn, length truncation, REPL confirm_fn wiring |
| `test_cli_persist.py` | 3 | Session-only flag override, --save persists, no flags = no persist |
| `test_compact.py` | 6 | History reduction, provider raise→RuntimeError, 2-turn no-op, small-history no-op, empty summary→RuntimeError, error event→RuntimeError |
| `test_config_load.py` | 3 | Corrupt JSON warning, missing file defaults, valid config loading |
| `test_mcp.py` | 20+ | Handshake, tools/list, call/echo/add/unknown, _make_tool routing, readOnlyHint trust, config loading, agent schema exposure, child env security, timeout poisoning, string-id echo, dedup |
| `test_mcp_toolname.py` | 5 | Snake_case, hyphen server+underscore tools, double-underscore, metachar rejection, readOnlyHint gate |
| `test_model_validate.py` | 6 | /model rejects unknown, accepts known, allows when unreachable, mock skips, /effort set/rebuild |
| `test_normalize.py` | 4 | Preamble stripping, clean answers unchanged, code fences/headers untouched, no-breakpoint single line, bool output_tokens guard |
| `test_providers.py` | 17 | Effort extra_body, mock event shapes, parse_tool_block, fence isolation, local prose-with-fence, two-tool-fence, sole-fence, output_tokens, compute_tok_stats, format_footer, import-without-openai |
| `test_tools.py` | 20 | Registry contents, confirmation flags, write+read, edit unique/non-unique, glob, grep, bash success/timeout, workspace confinement, vendor pruning, extensionless fallback, secret stripping |
| `test_web_fetch.py` | 15 | SSRF blocking, scheme validation, redirect-to-private blocking, HTML stripping, download truncation |

**Gaps**: No tests for `config.save_config()` disk writes, `Repl.run()` end-to-end, `__main__.main()` with real provider.

## Security Posture

- write/edit/read/glob/grep confined to workspace root (= cwd). 
- web_fetch: http/https only, SSRF guard resolves-once + IP-pins connection + re-checks redirects, blocks private/loopback/link-local/reserved/CGNAT/metadata.
- run_bash strips provider API keys from child env, process group isolation, timeout SIGKILL.
- MCP child env = allowlisted base + server config. NO provider/cloud secrets forwarded.
- `readOnlyHint` never trusted by default.

## KNOWN LIMITATIONS

- Workspace root is cwd-derived (launching from $HOME exposes home tree); no `--workspace` flag yet.
- run_bash detached/backgrounded processes survive timeout SIGKILL.
- Confirm gating is the main guard for run_bash/write/edit (auto-confirmed under `--yes`).
- qwen3.6 IGNORES `reasoning_effort` / `enable_thinking` (always reasons); `/effort` is best-effort only.

## Run

- `cd <dir> && source .venv/bin/activate && llmc` (qwen default).
- One-shot: `llmc -p "task" --yes`.

## Related
  1.00 > Objective (llm-cli/Objective.md)
  0.95 > llm-cli — Build Status (llm-cli/Status.md)
