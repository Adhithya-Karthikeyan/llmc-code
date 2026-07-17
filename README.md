# llm-cli

A lightweight, personal **Claude Code-style agentic CLI** in pure Python — for
**LOCAL LLMs only**.

It gives you a streaming terminal chat with an agentic tool-use loop, multi-agent
orchestration (role-scoped sub-agents), codebase intelligence (explore-before-act),
and a build → validate → fix workflow. There is **no hosted/cloud provider** — it
drives a local model served by [LM Studio](https://lmstudio.ai)'s OpenAI-compatible
server. A deterministic **mock** provider exists purely for offline testing.

Providers:

- **local** — your LM Studio model (native OpenAI-style tool calls, with a
  fenced-JSON text fallback for models that lack native tool support).
- **mock** — deterministic and offline, for testing with no keys/network.

## Install

Install it as a **global command** with [pipx](https://pipx.pypa.io) (or `uv tool`).
This puts the `llmc` command on your PATH inside an isolated environment, so it
runs from any directory with no venv to activate:

```bash
pipx install git+https://github.com/Adhithya-Karthikeyan/llm-cli.git
# or:  uv tool install git+https://github.com/Adhithya-Karthikeyan/llm-cli.git
```

Then run `llmc` anywhere. Upgrade later with `pipx upgrade llmcli`.

**Requires Python ≥ 3.10.**

### Development install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `llmc` console script. You can also run it as a module with
`python -m llmcli`.

## Usage

Interactive REPL (uses your saved provider/model, defaults to `local`):

```bash
llmc                       # or: python -m llmcli
llmc --provider mock       # offline, scripted demo
llmc --provider local --model <your-lm-studio-model-id>
```

One-shot, non-interactive (`--yes` auto-confirms dangerous tools so it runs
unattended):

```bash
llmc --provider mock --yes -p "create hello.py that prints hi and run it"
llmc -p "explain what this repo does"
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--provider {local,mock}` | Pick the backend (default `local`) |
| `--model <name>` | Model id for the active provider |
| `--base-url <url>` | LM Studio base URL (default `http://localhost:1234/v1`) |
| `-p, --prompt "<text>"` | Run one prompt and exit |
| `-c, --continue` | Resume this project's saved session at startup (see [Session memory](#session-memory)) |
| `--yes, -y` | Auto-confirm write/edit/bash tools (untrusts model + fetched content) |
| `--max-iterations <n>` | Cap provider turns per message (use `>=2` for tool tasks) |
| `--effort <level>` | Reasoning effort: off\|low\|medium\|high (best-effort) |
| `--theme <name>` | Color theme: `amber` (**default**, warm polished look) \| `auto` (truecolor) \| `ansi` (Dark mode, ANSI colors only) \| `orange`. Session-only unless `--save` |
| `--mcp {on,off}` | Enable/disable MCP servers (`~/.llm-cli/mcp.json`). `off` starts no servers and sends no MCP tools — smaller prompt, faster tok/s. Session-only unless `--save` |
| `--context N\|auto\|fixed\|off` | Working-context budget (~tokens). llmcli auto-trims history to it after each turn so decode stays fast; adaptive (flexes per request) by default. Session-only unless `--save` |
| `--private` | **Opt into the offline lockdown** for this session: no external egress (see below). Add `--save` to persist |
| `--allow-network` | Accepted **no-op alias** for back-compat (network is already the default). Explicitly keeps network on. Add `--save` to persist |
| `--save` | Persist the given flags as the new default in `~/.llm-cli/config.json` |

### Slash commands (in the REPL)

```
/help                 Show help
/provider <name>      Switch provider: local | mock
/models               List models available on the server
/model <name>         Set the model (verified against the server's list)
/effort <level>       Reasoning effort: off | low | medium | high (best-effort)
/theme <name>         Color theme: amber (default) | auto | ansi | orange
/compact              Aggressively summarize ALL history into one tight note,
                      keeping only the last exchange (frees the most context)
/context [N|auto|fixed|off]
                      Working-context budget — llmcli auto-trims history to this
                      after each turn so decode stays fast. auto = flex per
                      request (default); fixed = flat N; off = trim only near
                      the model's window. Persisted.
/audit [path]         Map-reduce audit: review the repo in small isolated chunks
                      (fast; small context) → merged report. Default path '.'
/speed                Tips to raise tok/s (LM Studio settings + context size)
/mcp [on|off]         No arg: list MCP servers + status. on/off: enable/disable
                      MCP (off = fewer tools per prompt → faster). Persisted.
/clear                Clear the conversation history
/resume               Reload this project's saved session (local-only memory)
/forget               Delete this project's saved session
/exit, /quit          Leave
Ctrl+O                Reveal full detail (args+results) of the last turn
```

## Session memory

llm-cli remembers the conversation **per project directory** (Claude-Code-style:
auto-save, opt-in resume), so you can pick up where you left off.

- **Auto-save.** After every completed turn (and on a clean exit) the running
  conversation is saved for the current working directory. A crash never loses
  it. A history of just the system prompt is skipped — nothing to remember.
- **Opt-in resume.** A fresh launch stays **light** and does *not* reload the
  history automatically. When a saved session exists, the REPL prints a dim hint:

  ```
  ↩ last session (5m ago, 12 msgs): fix the config parser — /resume to continue (or relaunch with -c)
  ```

  Reload it with **`/resume`** inside the REPL, or launch with **`-c` /
  `--continue`** to load it before the first turn. **`/forget`** deletes this
  project's saved session.
- **Where + privacy.** Sessions are stored **locally only**, as JSON under
  `~/.llm-cli/sessions/<dir>-<hash>.json`, keyed by the absolute path of the
  directory. They are **never sent anywhere**. The saved conversation may include
  file contents the model read while working — fine for a local tool, but the
  file stays on your machine. When you resume, the model-aware auto-compaction
  already bounds how much of that history re-enters the context window.

One-shot (`-p`) runs also save their conversation, and `-p -c` prepends the saved
history first — so an unattended one-shot can build on itself across runs.

## Networking & safety (default: network enabled)

llm-cli runs **network-enabled by default**: `web_fetch` is available, `run_bash`
can reach the network, a non-loopback `base_url` is allowed, all configured MCP
servers start, and proxy env vars are honored. The startup banner and the REPL
status line show the active state: `private mode: OFF — network enabled` (or
`private mode: ON — offline lockdown` under `--private`).

The on-device local model on loopback (LM Studio on `127.0.0.1`) is still the
default backend; enabling network does not change where your model runs.

### Always-on safety (active in BOTH default and `--private` modes)

These guards are **never disabled** — they hold even with network enabled, so the
tool only reaches **safe** URLs/endpoints:

- **`web_fetch` SSRF guard.** Only `http`/`https` URLs are allowed. The resolved
  IP is validated and the tool **refuses** loopback, private (RFC 1918),
  link-local, reserved, CGNAT (RFC 6598), multicast, and unspecified ranges, plus
  the cloud-metadata endpoint `169.254.169.254`. The host is resolved and the
  connection is **pinned to the validated IP** (DNS-rebinding defense), and the
  host is **re-validated on every HTTP redirect hop**. Download size and timeout
  are capped. This is fully active whenever `web_fetch` runs (i.e. by default).
- **`run_bash` is confirmation-gated** (unless `--yes`), its writes are
  **confined to the workspace root**, and its child env is built from a
  **secret-scrubbing allowlist** (provider keys / `AWS_*` / `GITHUB_TOKEN` etc.
  are not inherited).
- **`write_file` / `edit_file` are workspace-confined** — they refuse to touch
  paths resolving outside the directory the CLI was launched from.
- **No telemetry.** There is zero analytics / usage / crash reporting.

### `--private` (opt-in offline lockdown)

Pass `--private` to opt into a strict, no-egress mode (add `--save` to persist).
This is **enforced**, not advisory, and is **on top of** the always-on guards
above:

- **Provider base_url must be loopback.** A non-loopback `base_url` (anything
  other than `127.0.0.0/8`, `::1`, or `localhost`) is **REFUSED with a clear
  error — never silently used**, and the loopback host is **IP-pinned**. Validated
  at all three entry points: the `--base-url` flag, the persisted
  `~/.llm-cli/config.json`, and every in-session rebuild (`/provider`, `/model`,
  `/effort`).
- **`web_fetch` is removed from the tool set.** The model can't even call it (and
  a direct call is refused).
- **`run_bash` network egress is blocked at the OS level.** Commands run under a
  macOS `sandbox-exec` no-network profile (`deny network-outbound`, re-allowing
  loopback only); the child **and all its descendants** are kernel-blocked from
  opening external sockets. If `sandbox-exec` is unavailable, `run_bash`
  **fails closed** (refuses) rather than running unsandboxed.
- **No proxy tunneling.** The OpenAI client is built with
  `httpx.Client(trust_env=False)` so `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY`
  cannot tunnel even loopback traffic to an external proxy.
- **MCP servers are gated.** Only servers explicitly marked **`"private_ok": true`**
  in `mcp.json` are started; others are skipped with a warning. (In the default
  network-on mode, all configured servers start.)

**Honest residual notes:**

- In the default (network-on) mode `run_bash` is **not** network-sandboxed, so a
  command you approve can reach the network — the confirmation gate is the
  boundary there.
- Under `--private`, `sandbox-exec` is **deprecated** (its man page is dated 2017)
  but remains functional on current macOS; Apple could remove it in a future
  release.
- An MCP server you allow (`private_ok`, or any server in the default mode) can
  egress — the allow decision is yours.

## MCP (Model Context Protocol)

llm-cli can connect to local **MCP servers** over the **stdio transport** and
expose their tools to the agent — a stdlib-only, synchronous JSON-RPC 2.0 client
(no `mcp` SDK, no new dependencies).

It is **opt-in**: with no config file, MCP is simply off and nothing changes.
To enable it, create `~/.llm-cli/mcp.json` in the Claude Desktop format:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/some/dir"],
      "env": {},
      "disabled": false,
      "trustReadOnlyHint": false,
      "private_ok": false,
      "timeout": 30
    }
  }
}
```

- On REPL start (and on every one-shot `-p` run) each enabled server is spawned,
  handshaken (`initialize` -> `notifications/initialized` -> `tools/list`), and
  its tools are merged into the orchestrator's tool set. A connect line prints,
  e.g. `mcp: filesystem connected (11 tools)`. A server that fails to start logs
  a dim `mcp: <name> failed: ...` and is **skipped** — it never crashes the CLI.
- MCP tools are named `mcp__<server>__<tool>` and render in the collapsed line as
  `server:tool <arg>`.
- `/mcp` lists each configured server, whether it connected, and its tool names.
- Servers are cleanly shut down on exit (stdin closed, terminate, then kill).

### Safety notes / limitations

- **`--private` gating.** By default (network on) **all** configured servers
  start. Under `--private`, an MCP server is an egress surface (it is **not**
  network-sandboxed), so it is **skipped unless** you explicitly mark it
  `"private_ok": true`. A local vault writer like `kyp-mem` should be marked
  `"private_ok": true` so it still works under `--private`.
- **stdio transport only.** HTTP / SSE / Streamable-HTTP MCP servers are not
  supported.
- **Tool results are untrusted.** MCP output flows into the model like any other
  tool result (same as `web_fetch`).
- **Confirmation is on by default.** Every MCP tool is confirmation-gated. A
  tool's `readOnlyHint` is **self-asserted by the (untrusted) server**, so it is
  NOT trusted on its own — a malicious server could mark a destructive tool
  read-only to auto-run it. The hint only relaxes the gate when **you** opt in
  per-server with `"trustReadOnlyHint": true`. The trust decision is the
  operator's, never the server's.
- **No secret inheritance.** Child processes get a **minimal allowlisted env**
  (`PATH`, `HOME`, locale, temp) plus the server's configured `env`. Provider
  and cloud secrets (`OPENAI_API_KEY`, `LMSTUDIO_API_KEY`, `AWS_*`,
  `GITHUB_TOKEN`, ...) are **not** forwarded — give a server a secret only
  explicitly via its `env` block.
- **Per-call timeout.** Each request is bounded by `"timeout"` seconds (default
  30). On a `tools/call` timeout the connection is treated as poisoned: the
  child is killed (no orphan) and its tools are dropped, so a stale reply can
  never corrupt a later call. Raise `timeout` for legitimately long operations.
- **Name hardening.** Server-controlled tool names are validated (no `__`
  namespace forging) and de-collided — a duplicate full name is skipped, never
  silently overwriting another tool.
- **Tool flooding.** A server advertising many tools enlarges the model's tool
  list; keep the configured set focused.

### Concise output, collapsed tool lines, tok/s

The CLI presents like a senior engineer: terse and skimmable.

- **Concise answers** — the orchestrator + sub-agents follow an output contract:
  up to 5 markdown headers (and none for short answers), tight one-idea-per-line
  bullets, the answer first, no preamble or process narration. Competence is
  unchanged; only presentation is constrained.
- **Collapsed tool output** — by default each tool call is ONE line, e.g.
  `⏺ read_file README.md` for reads or `⏺ run_bash pytest -q (exit 0)` for bash
  (only `run_bash` shows an exit-code hint). A failure shows a dim `✗ <error>`.
  Full args and results are not dumped; they are stored for the current turn.
- **`Ctrl+O` to expand** — press it **at the `> ` prompt** to immediately print
  the full args + results of the turn that just finished. Limitation: the REPL is
  line-based, so `Ctrl+O` is handled only while the prompt is active — it cannot
  interrupt a running turn; press it at the next prompt. Only the orchestrator's
  detail is revealed; sub-agent tool calls still stream as nested collapsed lines.
- **tok/s footer** — after each visible assistant message, a dim footer reports
  the generation rate, e.g. `38.4 tok/s`. The local provider reads the server's
  `completion_tokens` (via `stream_options.include_usage`); when absent it
  approximates from the text.
- **Duck working-indicator** — while the CLI is busy (waiting on the model, or
  running tools) a small ASCII duck waddles to-and-fro in place. It is purely
  decorative (no words) and **terminal-only**: it is fully disabled when output
  is piped or not a TTY (one-shot in a pipe, captured tests), so it never leaks
  stray characters into redirected output. It always erases its own line before
  any real output (the answer, the collapsed tool line, the footer, or a `y/N`
  confirm prompt). Set `LLMCLI_NO_SPINNER=1` to turn the duck off on a TTY too.
- **`/compact`** — summarizes the conversation so far via the active provider into
  a compact context note (preserving decisions, files touched, open tasks, key
  facts), replaces the long history with `[system summary + the last 1–2 turns]`,
  and prints a rough before→after token estimate. If the provider call fails it
  no-ops and leaves history untouched.

## Pointing at LM Studio

1. Open **LM Studio** → load a model (one that supports tool/function calling
   gives the best results).
2. Start its **local server** (Developer tab → Start Server). It exposes an
   OpenAI-compatible API at `http://localhost:1234/v1` by default.
3. Point llm-cli at it:

   ```bash
   llmc --provider local --model <the-model-id> --base-url http://localhost:1234/v1
   ```

The API key is read from `OPENAI_API_KEY` or `LMSTUDIO_API_KEY`, defaulting to the
harmless string `lm-studio` (LM Studio ignores it, but the OpenAI SDK requires a
non-empty value).

### Fenced-JSON tool-call fallback

If the loaded model can't do native tool calls, the local provider falls back to
parsing a fenced ` ```json ` tool-call block from the model's text output:

````
```json
{"tool": "read_file", "input": {"path": "main.py"}}
```
````

`tool` maps to the tool name and `input` to its arguments.

### Mock (offline)

```bash
llmc --provider mock --yes -p "create hello.py that prints hi and run it"
```

The mock provider is deterministic and needs no keys or network. Its `hello`
scenario scripts a `write_file` then a `run_bash` tool call so you can validate
the full agent loop offline. It's also what the test suite uses.

## How it works

The "intelligence" is emergent from three things working together:

1. **Good exploration tools** — `repo_map` (compact structural index), `grep`,
   and `read_file` (with `offset`/`limit` for slices) so the model learns the
   code without dumping whole files into context.
2. **A strong orchestrator prompt** that enforces **explore → plan → act → validate**
   and **delegates** multi-file reading to sub-agents to keep its context small.
3. **Delegation** — `spawn_agent` spins up focused, role-scoped sub-agents; the
   `/audit` command map-reduces a whole-repo review across small isolated chunks.

### Architecture

```
llmcli/
  config.py         Settings, LM Studio defaults, ~/.llm-cli/config.json persistence
  prompts.py        System prompts: ORCHESTRATOR, EXPLORER, CODER, REVIEWER
  tools.py          Tool registry: read_file (offset/limit), write_file, edit_file, run_bash, glob, grep, repo_map
  audit.py          Map-reduce /audit: chunk the repo, review each chunk in an isolated context, merge
  providers.py      Provider ABC + LocalProvider (LM Studio) + MockProvider
                    (normalized to {type: text|tool_call|done} events)
  agent.py          The agentic loop: stream → run tools → feed results → repeat
  orchestration.py  spawn_agent delegation tool + role toolsets
  mcp.py            stdio MCP client + manager (sync JSON-RPC 2.0, stdlib only)
  repl.py           prompt_toolkit input + rich streamed output; provider factory
  __main__.py       CLI: flags, one-shot -p mode, `llmc` entry point
tests/              pytest suite (MockProvider only; no network)
```

### Normalized provider events

Every provider's `stream_chat(messages, tools)` yields plain dicts with a `type`:

- `{"type": "text", "text": <str>}` — incremental assistant text (deltas).
- `{"type": "tool_call", "id", "name", "arguments": <dict>}` — args already
  `json.loads`'d into a dict.
- `{"type": "done", "finish_reason": <str>, "output_tokens": <int|None>}` —
  exactly one, always last. `output_tokens` is the completion token count for the
  message (the local provider's `completion_tokens`, or `None` when the server
  did not report it so the agent approximates for the tok/s footer).

All tool calls are emitted before the single terminal `done`.

### Agent loop

1. Send the conversation + tool schemas to the provider; stream the reply.
2. If the model returns tool calls, execute each (with confirmation for
   `write_file` / `edit_file` / `run_bash` unless `--yes`), append the structured
   results as `role:"tool"` messages.
3. Loop until the model returns a plain-text final answer (or the iteration guard
   trips).

### Roles

| Role | Tools | Purpose |
| --- | --- | --- |
| orchestrator | all + `spawn_agent` | Plans, delegates, validates |
| explorer | read-only (`repo_map`, `read_file`, `glob`, `grep`) | Investigates the codebase |
| coder | full | Implements a scoped change, then validates |
| reviewer | read-only | Critiques a change before finalizing |

## Development / tests

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

All tests run offline against the mock provider — no API keys, no network. The
`openai` import is lazy (only inside `LocalProvider`), so the agent/tool/mock
stack imports and runs without `openai` installed.

## Notes / status

- The local (LM Studio) provider is fully implemented but needs a running LM
  Studio server with a model loaded to exercise live; it is not network-tested
  here. The mock provider and the full agent/tool/orchestration stack are covered
  by the test suite.
- Config (provider, model, base URL) persists to `~/.llm-cli/config.json`.
  API keys are read from the environment only and never written to disk.
