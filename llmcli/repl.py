"""Interactive REPL + one-shot runner.

- prompt_toolkit for line input (history, editing).
- rich Console for output; streamed assistant text renders live.
- Slash commands: /model /models /provider /effort /maxout /compact /clear
  /resume /forget /help /exit (see HELP_TEXT — the single canonical command list).
- Ctrl+O reveals the orchestrator's last-turn detail.
- Shows the active provider/model.
- Per-project session memory: the conversation auto-saves after every turn and
  on exit; /resume (or launching with -c/--continue) reloads it (see session.py).

Providers are ONLY 'local' (LM Studio) and 'mock' (offline).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from .agent import _TURN_READ_NUDGE_BYTES, Agent, confirm_label
from dataclasses import replace

from . import checkpoint
from . import cooldown
from . import gitint
from . import hooks
from . import images
from . import memory
from . import mentions
from . import rules
from . import session
from . import tools
from .code_index import make_code_search_tool

from .config import (
    DEFAULT_BASE_URL,
    EFFORT_LEVELS,
    PERMISSION_MODES,
    PROVIDER_LOCAL,
    PROVIDER_MOCK,
    PROVIDERS,
    THEME_AMBER,
    THEME_ANSI,
    THEME_AUTO,
    THEME_CLEAN,
    THEME_ORANGE,
    THEMES,
    Config,
    get_api_key,
    is_loopback_url,
    load_config,
    save_config,
)
from .mcp import MCPManager, load_mcp_config
from .orchestration import (
    _has_memory_tool,
    make_spawn_agent_tool,
    orchestrator_registry,
    orchestrator_tool_names,
)
from .prompts import orchestrator_prompt
from .providers import (
    LocalProvider,
    MockProvider,
    Provider,
    detect_context_length,
    list_local_models,
)
from .tools import set_private

# Authoritative set of llmc's OWN slash commands. The main input loop only
# intercepts a leading-slash line when its first token is in here; anything else
# (e.g. "/build the app" referencing ANOTHER project's CLI) is sent to the model.
# _dispatch_slash handles exactly these — keep the two in sync.
_KNOWN_COMMANDS = frozenset({
    "/exit", "/quit", "/help", "/clear", "/resume", "/forget", "/memory",
    "/theme",
    "/compact", "/mcp", "/audit", "/speed", "/context", "/provider",
    "/models", "/model", "/effort", "/maxout", "/temp", "/verify", "/image",
    "/gentle", "/cooldown",
    "/rerank",
    # Foundation-wave commands (git/rules/checkpoint/session/clipboard/diagnostics).
    "/undo", "/diff", "/commit", "/init", "/copy", "/mode",
    "/branch", "/fork", "/doctor", "/commands",
})

HELP_TEXT = """\
Commands:
  /help                 Show this help
  /provider <name>      Switch provider: local | mock
  /models               List models available on the server
  /model <name>         Set the model (verified against the server's list)
  /effort <level>       Reasoning effort: off | low | medium | high (best-effort;
                        ignored by models that don't support reasoning_effort)
  /maxout <N> | off     Per-request generation cap (max_tokens); off | 0 | -1 =
                        unbounded (default). A low cap counts reasoning tokens.
  /temp [value]         Sampling temperature 0.0-2.0 (default 0.2; lower = more
                        deterministic, fewer malformed tool calls). No arg: show.
  /verify [cmd|off]     Auto-run <cmd> after a turn edits files but runs nothing
                        (output fed back to fix failures). No arg: show; off: disable.
  /gentle [on|off|tokens <n>|gap <s>|sgap <s>]
                        Gentle mode (default ON): lower average GPU load/heat by
                        capping output tokens (shorter bursts) + a cool-down
                        between turns. sgap spaces out sequential sub-agent
                        spawns. Does NOT cap GPU %. No arg: show status.
  /cooldown [on|off|interval <s>|duration <s>]
                        Thermal cooldown (default ON): pause generation for a
                        short break every N seconds of continuous work — even
                        mid-turn — to let the GPU cool. No arg: show status.
  /image [path [msg]]   Attach an image to your NEXT message (vision models only).
                        Stage multiple by repeating; add a msg to send at once.
                        No arg: list staged; /image clear: drop all staged.
  /theme <name>         Color theme: clean (default, minimal dark, low-key grey)
                        | amber (warm polished look) | auto (truecolor) | ansi
                        (Dark mode, ANSI colors only — uses your terminal's own
                        16-color palette) | orange (orange-on-black inline code)
  /compact              Aggressively summarize ALL history into one tight note,
                        keeping only the last exchange (frees the most context)
  /context [N|auto|fixed|off]
                        Working-context budget (auto-trims after each turn so
                        decode stays fast). auto = flex per request (default);
                        fixed = flat N; off = only trim near the model's window
  /audit [path]         Map-reduce audit: review the repo in small isolated
                        chunks (fast; keeps context small) → merged report.
                        Does not touch your conversation. Default path: '.'
  /speed                Tips to raise tok/s (LM Studio settings + how context
                        size affects speed)
  /mcp [on|off]         No arg: list MCP servers + status. on/off: enable or
                        disable MCP (off stops the servers + drops their tools,
                        shrinking each prompt → faster). Persisted.
  /undo                 Revert the last file write/edit this session made
                        (restores the file-snapshot checkpoint; session-scoped)
  /diff [path]          Show the working-tree git diff (optionally for one PATH)
  /commit [msg]         Commit all changes; auto-writes a message from the diff
                        when none is given
  /init                 Write a starter AGENTS.md project-rules file
  /copy [N]             Copy the last (or Nth-from-last) assistant answer to the
                        clipboard
  /mode [name]          Show or set the permission mode (default | acceptEdits |
                        plan | …); no arg: show current
  /branch [tag]         Save (no arg: list) a named snapshot of this conversation
  /fork <tag>           Load a saved branch snapshot and continue from it
  /doctor               Run environment/health checks (provider, git, sandbox, …)
  /commands             List project macros in .llmcli/commands/*.md
  /clear                Clear the conversation history
  /resume               Reload this project's saved session (local-only memory)
  /forget               Delete this project's saved session
  /exit, /quit          Leave
  Ctrl+O                Reveal the orchestrator's last-turn detail (args+results)
Anything else is sent to the agent.
!<cmd> runs <cmd> in YOUR shell directly — it is NOT sandboxed, even in
--private mode (it's your own shell, not a gated agent tool). Review before use.
Lines starting with an unknown /command are sent to the model (so you can work on
projects that use their own / commands, e.g. /build or /deploy).
Start a line with // to send a literal leading slash to the model (e.g.
"//model …" chats ABOUT the command instead of running llmc's /model).

NETWORK (default): network enabled; web_fetch is SSRF-safe (blocks
internal/metadata, validates redirects, http/https only) and confirmation-gated;
run_bash is confirmation-gated and has FULL filesystem + network access — the
y/N prompt is the boundary, so review each command (and don't use --yes with
untrusted tasks). --private adds an offline no-network sandbox for run_bash
(file access stays full) + drops web_fetch.

The SSRF/confinement guards above are ALWAYS ON — they hold even with network
enabled. --private (opt-in lockdown) additionally: refuses a non-loopback
base_url (loopback-pinned); removes web_fetch from the tool set; sandboxes
run_bash with a macOS no-network profile (sandbox-exec, FAIL CLOSED if missing);
starts ONLY MCP servers marked "private_ok": true in mcp.json; and ignores proxy
env (trust_env=False).

Tool activity collapses to one dim line after each turn:
  ⏺ N tools · Ctrl+O to expand
Ctrl+O reveals the full ⏺/⎿ tree — one two-line entry per call:
  ⏺ Read(README.md)
    ⎿  Read 120 lines
The "⏺" glyph is green, the result summary line is dim. A failed call shows a
dim-red "✗ <short reason>" (declined -> "✗ declined"; unknown -> "✗ unknown tool").
Ctrl+O is line-based: it cannot interrupt a running turn — press it at the next
prompt to reveal the turn that just finished. NOTE: it reveals only the
ORCHESTRATOR's own tool calls; a delegated sub-agent's activity shows as a
prefixed "↳ ⏺ N tools" line during the run and is not in this buffer.
"""


SPEED_TIPS = """\
Speed (tok/s) — what controls it and how to raise it

1) BIGGEST factor: context size. The model re-reads the whole conversation for
   every token it writes, so tok/s drops as the chat (and tool output) grows.
   - llmcli AUTO-trims the live context to an adaptive budget after each turn
     (tight for simple asks, larger for big ones). Tune with /context <N|auto|
     fixed|off>; smaller = faster.
   - /compact (aggressive — summarize all but the last exchange) or /clear to
     reset history when it gets long.
   - /audit instead of asking the model to read the whole repo in one go: it
     reviews the project in small isolated chunks, keeping context small.
   - Let the model use repo_map + grep + read_file(offset,limit) instead of
     reading whole files; turn /mcp off if you don't need its tools.

2) LM Studio model-load settings (set these where you load the model):
   - Context Length: set it to what you actually use (e.g. 16k–32k), not the
     max. A smaller KV cache = faster per-token AND less GPU memory pressure.
   - Flash Attention: ON — faster attention, especially at longer context.
   - Speculative Decoding / draft model: enable a small draft model for the same
     family — often a large decode speedup at no quality cost.
   - KV Cache Quantization (Q8): fits more context / frees memory, but is a bit
     SLOWER per token on Apple Metal — use it to FIT, not for speed.

3) Hardware reality: a local Mac is memory-bandwidth-limited, so big contexts
   slow it far more than a datacenter GPU. Keeping context small (point 1) is
   the lever you control; the rest is model/engine tuning (point 2).
"""


# Diff-preview caps: skip a file that is too big to safely read for a preview,
# and cap the rendered unified diff so a huge change can't flood the prompt.
_DIFF_PREVIEW_MAX_FILE_BYTES = 1_000_000
_DIFF_PREVIEW_MAX_LINES = 60


def _read_text_for_preview(path: str) -> str | None:
    """Return a file's text for a diff preview, or None when it can't be shown.

    None on: missing file, a directory, an oversized file, an unreadable file,
    or content that looks binary (contains a NUL byte). Never raises — a preview
    is best-effort and must never block the y/N prompt.
    """
    try:
        p = os.path.expanduser(str(path))
        if not os.path.isfile(p):
            return None
        if os.path.getsize(p) > _DIFF_PREVIEW_MAX_FILE_BYTES:
            return None
        with open(p, "rb") as fh:
            raw = fh.read()
        if b"\x00" in raw:
            return None  # binary — a textual diff is meaningless
        # Normalize to LF (universal-newline semantics) so the computed proposed
        # content matches what tools._edit_file writes — it reads with universal
        # newlines (CRLF/CR -> LF). Without this, a CRLF file's preview would
        # diverge from the actual write.
        text = raw.decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, ValueError):
        return None


def _apply_edit_for_preview(text: str, old: str, new: str) -> str | None:
    """Compute edit_file's would-be result, mirroring tools._edit_file semantics.

    Exact-unique replace first; on no exact match, fall back to the SAME
    normalized (line-ending/trailing-whitespace) unique match tools._edit_file
    uses. Returns None when the target isn't uniquely locatable (nothing to
    preview). Never raises.
    """
    try:
        count = text.count(old)
        if count == 1:
            return text.replace(old, new, 1)
        if count > 1:
            return None  # ambiguous — tools._edit_file would reject it
        # No exact match: reuse the tolerant normalized matcher for parity.
        from .tools import _normalize_lines_with_spans

        norm_old, _ = _normalize_lines_with_spans(old)
        if not norm_old:
            return None
        norm_text, spans = _normalize_lines_with_spans(text)
        matches: list[int] = []
        search_from = 0
        while True:
            idx = norm_text.find(norm_old, search_from)
            if idx < 0:
                break
            matches.append(idx)
            search_from = idx + len(norm_old)
        if len(matches) != 1:
            return None
        a = matches[0]
        b = a + len(norm_old)
        return text[: spans[a][0]] + new + text[spans[b - 1][1]:]
    except Exception:  # noqa: BLE001 - preview is best-effort; never raise
        return None


def _diff_preview_lines(tool_name: str, args: dict) -> list[tuple[str, str]] | None:
    """Build a concise (line, rich-style) preview for a write_file/edit_file call.

    Returns a list of ``(text, style)`` rows (style is "green"/"red"/"dim") ready
    to print before the y/N prompt, or None when there is nothing safe to show
    (unknown tool, bad args, binary/missing/oversized file, or an edit whose
    target isn't uniquely locatable). Capped to ~60 lines. Never raises.
    """
    try:
        if not isinstance(args, dict):
            return None
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return None
        if tool_name == "write_file":
            content = args.get("content", "")
            if not isinstance(content, str):
                return None
            current = _read_text_for_preview(path)
            if current is None:
                # New file (or unreadable/binary/oversized): show a compact note
                # instead of a full diff so the prompt still previews the action.
                if not os.path.exists(os.path.expanduser(path)):
                    n = len(content.splitlines())
                    return [(f"＋ new file: {path} ({n} lines)", "green")]
                return None
            proposed = content
        elif tool_name == "edit_file":
            old = args.get("old")
            new = args.get("new")
            if not isinstance(old, str) or not isinstance(new, str):
                return None
            current = _read_text_for_preview(path)
            if current is None:
                return None
            proposed = _apply_edit_for_preview(current, old, new)
            if proposed is None or proposed == current:
                return None
        else:
            return None

        diff = difflib.unified_diff(
            current.splitlines(),
            proposed.splitlines(),
            fromfile=path,
            tofile=path,
            lineterm="",
        )
        rows: list[tuple[str, str]] = []
        for line in diff:
            if line.startswith("+"):
                style = "green"
            elif line.startswith("-"):
                style = "red"
            else:
                style = "dim"
            rows.append((line, style))
        if not rows:
            return None
        if len(rows) > _DIFF_PREVIEW_MAX_LINES:
            rows = rows[:_DIFF_PREVIEW_MAX_LINES]
            rows.append(("…(truncated)", "dim"))
        return rows
    except Exception:  # noqa: BLE001 - preview is best-effort; never raise
        return None


def make_ptk_confirm(session, config=None):
    """A prompt_toolkit-compatible y/N confirm_fn (no builtin input()).

    The builtin ``input()`` used by agent._default_confirm conflicts with
    prompt_toolkit's event loop and breaks confirmation inside the REPL. This
    asks via the SAME PromptSession. It runs during agent.run (between provider
    turns), NOT inside a key-binding handler, so a plain session.prompt() is safe
    (no nested event loop).

    When ``config`` is given and ``config.diff_preview`` is on, a concise unified
    diff of a pending write_file/edit_file is rendered BEFORE the y/N prompt.
    ``config`` defaults to None (back-compat: no preview, unchanged behavior).
    """

    def _confirm(tool, args) -> bool:
        # Optional diff preview for write/edit (finding: diff_preview). Rendered to
        # stdout via a rich Console so it is captured in tests and themed
        # dim/green/red. Best-effort: any failure just skips the diff and prompts.
        if (
            config is not None
            and getattr(config, "diff_preview", False)
            and getattr(tool, "name", "") in ("write_file", "edit_file")
        ):
            rows = _diff_preview_lines(getattr(tool, "name", ""), args)
            if rows:
                try:
                    from rich.console import Console

                    con = Console(markup=False, highlight=False)
                    for text, style in rows:
                        con.print(text, style=style)
                except Exception:  # noqa: BLE001 - never block the prompt on render
                    pass
        # Use the SAME collapsed summary + byte-hint logic as the loop and the
        # input()-based fallback, centralized in agent.confirm_label (finding #29)
        # — no duplicated label-building here. Full args stay for Ctrl+O reveal.
        label, hint = confirm_label(tool, args)
        # This confirm reuses the interactive PromptSession, which carries a ghost
        # `placeholder` ("Ask anything · …") for the MAIN input. Left as-is it
        # renders glued onto the y/N line. A per-call `placeholder=None` is a NO-OP
        # in prompt_toolkit (it only overrides when the value is not None), so pass
        # an EMPTY placeholder ("") to actually suppress the ghost. prompt() mutates
        # `session.placeholder` in place, so SAVE + RESTORE it — otherwise the main
        # input would silently lose its ghost after the first confirm. patch_stdout
        # (raw=True) coordinates any background stdout write landing during the y/N
        # so it can't leave residue on that row (same fix as the main-input read).
        from prompt_toolkit.patch_stdout import patch_stdout

        saved_placeholder = getattr(session, "placeholder", None)
        try:
            # Leading newline so the y/N prompt isn't glued to the preceding dim
            # tok/s footer when the model narrates AND calls a gated tool.
            with patch_stdout(raw=True):
                answer = session.prompt(
                    f"\nRun {label}?{hint} [y/N] ", placeholder=""
                )
        except (EOFError, KeyboardInterrupt):
            return False
        finally:
            # Restore the session's ghost placeholder for the next main input
            # (best-effort: never let a restore failure break the confirm result).
            try:
                session.placeholder = saved_placeholder
            except Exception:  # noqa: BLE001
                pass
        return answer.strip().lower() in ("y", "yes")

    return _confirm


def gentle_wait(gap: float, now: float, last_end: float) -> float:
    """Seconds to sleep before the next generation for gentle-mode pacing.

    PURE (no sleeping, no clock reads) so the math is unit-testable. ``gap`` is
    the minimum cool-down between the END of one generation (``last_end``) and
    the START of the next (``now``), both monotonic seconds. Returns the positive
    remaining wait only when the user came back WITHIN the gap; returns 0.0 when
    enough time already passed (the typing time covered the gap) or when ``gap``
    is <= 0 (gentle effectively off). The caller decides whether to apply it and
    must TTY-gate the actual sleep so tests/piped runs never block.
    """
    try:
        gap = float(gap)
    except (TypeError, ValueError):
        return 0.0
    if gap <= 0.0:
        return 0.0
    wait = gap - (float(now) - float(last_end))
    return wait if wait > 0.0 else 0.0


def build_provider(
    name: str, model: str, base_url: str, effort: str = "", private: bool = False,
    cache_prompt: bool = False, max_output_tokens: int | None = None,
    embed_model: str | None = None, temperature: float = 0.2,
    gentle_mode: bool = False, gentle_max_tokens: int = 1024,
    seed: int | None = None,
    id_slot: int | None = None,
) -> Provider:
    """Construct a provider by name. Only 'local' and 'mock' are supported.

    PRIVATE-mode enforcement (entry point #3, the in-session rebuild): when
    ``private`` is True the local provider's ``base_url`` MUST be loopback, or we
    REFUSE with a clear ValueError — never silently send project data off-box.
    This is the single chokepoint every in-session rebuild (/provider, /model,
    /effort) and the CLI startup both route through, so a non-loopback URL
    cannot sneak in from any path. The mock provider has no base_url and is
    always safe.
    """
    if name == PROVIDER_MOCK:
        return MockProvider(temperature=temperature, seed=seed)
    if name == PROVIDER_LOCAL:
        if private and not is_loopback_url(base_url):
            raise ValueError(
                f"private mode: refusing a non-loopback base_url ({base_url!r}). "
                "The provider must be on the local machine (127.0.0.0/8, ::1, or "
                "localhost) so project data never leaves the box. "
                "Re-run with --allow-network to use an external server."
            )
        return LocalProvider(
            model=model, base_url=base_url, api_key=get_api_key(),
            effort=effort, private=private, cache_prompt=cache_prompt,
            max_output_tokens=max_output_tokens, embed_model=embed_model,
            temperature=temperature,
            gentle_mode=gentle_mode, gentle_max_tokens=gentle_max_tokens,
            seed=seed, id_slot=id_slot,
        )
    raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}.")


# Feature 2: conservative phrases that mark a FREE-TEXT prompt as a whole-project
# / whole-codebase request, for which /audit (a fast map-reduce review that keeps
# context small) is a better fit than one big agent turn on a local model. Kept as
# a module-level frozenset so the list is easy to see and tune. Matched as
# lowercased substrings — deliberately specific so normal single-file questions
# (e.g. "what does foo() in bar.py do?") never trigger it.
_WHOLE_REPO_PHRASES = frozenset({
    "whole codebase", "entire codebase", "whole code base", "entire code base",
    "whole project", "entire project",
    "whole repo", "entire repo", "the whole repository", "entire repository",
    "review everything", "audit everything", "go through everything",
    "explain this codebase", "explain the codebase", "explain this project",
    "explain the whole", "explain the entire",
    "go through all the files", "go through all files", "read all the files",
    "read the whole", "read the entire",
    "across the whole", "across the entire", "across the codebase",
})
# Phrasings a fixed substring can't capture cleanly (variable middle words).
_WHOLE_REPO_PATTERNS = (
    re.compile(r"how does (the|this) (whole|entire) .+ work", re.IGNORECASE),
)


def looks_like_whole_repo_request(text: str) -> bool:
    """True when free-text looks like a whole-project/whole-codebase request.

    Conservative: matches the explicit phrases in ``_WHOLE_REPO_PHRASES`` plus a
    couple of regexes. Designed NOT to fire on normal single-file questions. Pure
    string signals — no model call. Used only to print a one-line /audit hint; it
    never blocks or changes the request.
    """
    if not text:
        return False
    low = text.lower()
    if any(p in low for p in _WHOLE_REPO_PHRASES):
        return True
    return any(p.search(text) for p in _WHOLE_REPO_PATTERNS)


def _effective_soft_limit(provider: Provider, config: Config) -> int:
    """Auto-compaction budget, sized to the loaded model's REAL context window.

    ``config.context_soft_limit`` is a conservative FLOOR. For a model with a
    large window (qwen is loaded at 256k) we raise the budget to ~80% of its
    context so compaction never fires ~10x too early; for a small window
    (gemma 32k) it stays near that. Best-effort: a mock/no-endpoint provider or a
    failed lookup falls back to the floor. The detected length is cached on the
    provider so we don't re-query on every agent rebuild (/model, /clear, ...).
    """
    floor = config.context_soft_limit
    base_url = getattr(provider, "base_url", None)
    model = getattr(provider, "model", None)
    if not base_url or not model:
        return floor  # mock provider / no endpoint
    ctx = getattr(provider, "_ctx_len", "unset")
    if ctx == "unset":
        ctx = detect_context_length(base_url, model)
        try:
            provider._ctx_len = ctx  # cache (incl. None) to avoid re-querying
        except Exception:  # noqa: BLE001
            pass
    if isinstance(ctx, int) and ctx > 0:
        return max(floor, int(ctx * 0.8))
    return floor


def _fmt_tok(n: int) -> str:
    """Compact token count for the context gauge: 4400 -> '4.4k', 12000 -> '12k'."""
    if n < 1000:
        return str(int(n))
    k = n / 1000
    if k >= 10 or k == int(k):
        return f"{k:.0f}k"
    return f"{k:.1f}k"


def _stdout_is_tty() -> bool:
    """True iff stdout is a real terminal (best-effort, never raises).

    Used by the ANSI theme to decide whether to pin color_system="standard":
    we pin it only for an interactive terminal so a piped/redirected run stays
    ANSI-free (rich auto-disables color for a non-tty ONLY when color_system is
    left to auto-detect)."""
    try:
        return bool(sys.stdout.isatty())
    except Exception:  # noqa: BLE001 - a missing/odd stdout must not crash startup
        return False


# Grab the "@word" ending at the cursor for @-mention file completion: an ``@``
# followed by any run of non-whitespace (path chars). Word-boundary matching is
# left to prompt_toolkit's get_word_before_cursor(pattern=...).
_AT_WORD_RE = re.compile(r"@\S*")

# Short one-line meta shown beside each slash-command completion. Missing entries
# fall back to a generic label; kept compact so it never bloats the completer.
_COMMAND_META = {
    "/help": "show help",
    "/exit": "quit",
    "/quit": "quit",
    "/clear": "reset conversation",
    "/resume": "reload saved session",
    "/forget": "drop saved session",
    "/memory": "conversation memory",
    "/theme": "switch theme",
    "/compact": "shrink context now",
    "/mcp": "MCP servers on/off",
    "/audit": "map-reduce project review",
    "/speed": "speed guide",
    "/context": "context budget",
    "/provider": "switch provider",
    "/models": "list server models",
    "/model": "set model",
    "/effort": "reasoning effort",
    "/maxout": "generation cap",
    "/temp": "sampling temperature",
    "/verify": "auto-verify command",
    "/image": "attach an image",
    "/gentle": "GPU-heat pacing",
    "/rerank": "retrieval reranker",
    "/undo": "revert last file changes",
    "/diff": "show working-tree diff",
    "/commit": "git commit changes",
    "/init": "scaffold rules file",
    "/copy": "copy last answer",
    "/mode": "permission mode",
    "/branch": "git branch",
    "/fork": "fork the session",
    "/doctor": "environment diagnostics",
    "/commands": "list project macros",
}


def _build_input_completer(repl):
    """Build the REPL input-line completer (lazily imports prompt_toolkit).

    Returns a prompt_toolkit ``Completer`` that (a) completes llmc slash commands
    and loaded macros while the line's first token starts with ``/``, and (b)
    fuzzy-completes workspace-relative files for the ``@word`` under the cursor.
    The completer NEVER raises (an exception would break line input) and yields
    nothing when it has no suggestions, so it is a safe no-op on a dumb terminal.
    prompt_toolkit is imported here (not at module top) to keep ``import repl``
    cheap and consistent with the rest of this module's lazy prompt_toolkit use.
    """
    from prompt_toolkit.completion import Completer, Completion, FuzzyCompleter
    from prompt_toolkit.document import Document

    class _ProjectFileCompleter(Completer):
        """Yield every cached workspace file as a raw candidate (Fuzzy filters)."""

        def get_completions(self, document, complete_event):
            word = document.text_before_cursor
            for rel in repl._completion_files():
                yield Completion(rel, start_position=-len(word), display=rel)

    # FuzzyCompleter does the ranking; the path-aware pattern lets a query span
    # "/" and "." (e.g. "src/ap" -> src/app.py). It requires a STRING pattern
    # anchored with "^" (not a compiled regex).
    file_fuzzy = FuzzyCompleter(
        _ProjectFileCompleter(), pattern=r"^[A-Za-z0-9_./\-]+"
    )

    class _InputCompleter(Completer):
        def get_completions(self, document, complete_event):
            try:
                lstripped = document.text_before_cursor.lstrip()
                # Slash completion only while typing the FIRST token (no space yet)
                # so "/model gpt" does not try to complete "gpt" as a command.
                if (
                    lstripped.startswith("/")
                    and " " not in lstripped
                    and "\t" not in lstripped
                ):
                    token = lstripped.lower()
                    names = set(_KNOWN_COMMANDS) | set(repl._macros().keys())
                    for name in sorted(names):
                        if name.startswith(token):
                            meta = _COMMAND_META.get(
                                name,
                                "command" if name in _KNOWN_COMMANDS else "macro",
                            )
                            yield Completion(
                                name,
                                start_position=-len(lstripped),
                                display=name,
                                display_meta=meta,
                            )
                    return
                # @-mention: fuzzy-complete files for the "@word" under the cursor.
                word = document.get_word_before_cursor(pattern=_AT_WORD_RE)
                if word.startswith("@"):
                    after = word[1:]
                    sub = Document(after, len(after))
                    yield from file_fuzzy.get_completions(sub, complete_event)
            except Exception:  # noqa: BLE001 - a completer must NEVER raise into ptk
                return

    return _InputCompleter()


# Orange theme palette (truecolor hex; downsamples to 256 on lesser terminals).
_ORANGE = "#ff9e3d"
_ORANGE_BRIGHT = "#ffb454"
_ORANGE_DIM = "#c9763d"


def _orange_theme():
    """rich Theme overriding the markdown.* styles for the orange theme.

    Inline code (markdown.code) becomes orange TEXT with NO bgcolor, removing
    rich's default grey/black background box. Only accents + inline code are
    recolored; paragraph/body text is left to the terminal default so prose
    stays readable light-on-black. NO style here sets a bgcolor (no boxes).
    """
    from rich.style import Style
    from rich.theme import Theme

    return Theme({
        "markdown.code": Style(color=_ORANGE, bold=True),
        "markdown.code_block": Style(color=_ORANGE),
        "markdown.h1": Style(color=_ORANGE_BRIGHT, bold=True),
        # h2+ step down to plain orange so the heading hierarchy is distinct.
        "markdown.h2": Style(color=_ORANGE, bold=True),
        "markdown.h3": Style(color=_ORANGE, bold=True),
        "markdown.h4": Style(color=_ORANGE, bold=True),
        "markdown.h5": Style(color=_ORANGE),
        "markdown.h6": Style(color=_ORANGE),
        "markdown.strong": Style(color=_ORANGE, bold=True),
        "markdown.item.bullet": Style(color=_ORANGE, bold=True),
        "markdown.item.number": Style(color=_ORANGE, bold=True),
        "markdown.link": Style(color=_ORANGE_BRIGHT, underline=True),
        "markdown.link_url": Style(color=_ORANGE_DIM),
        "markdown.block_quote": Style(color=_ORANGE_DIM),
        "markdown.hr": Style(color=_ORANGE_DIM),
    })


# Amber theme palette (the default). Warm orange accents + a GOLD bold so
# **important words** pop, headers in bright amber, inline code in orange with
# NO background box. Truecolor hex downsamples to 256/16 on lesser terminals.
_AMBER = "#ff9e3d"          # primary accent (gutter, prompt, bullets, code)
_AMBER_GOLD = "#ffcf6b"     # bright gold — bold words + h1/h2 headers
_AMBER_DIM = "#c9763d"      # muted amber — links/urls/quotes/rules


def _amber_theme():
    """rich Theme for the polished default "amber" look.

    Builds on the orange theme: inline code is orange TEXT with NO bgcolor box,
    headers are bright amber, **bold** text reads in GOLD so important words pop,
    bullets/numbers are amber. Body prose stays the terminal default (readable
    light-on-black). No style sets a bgcolor (no boxes anywhere).
    """
    from rich.style import Style
    from rich.theme import Theme

    return Theme({
        "markdown.code": Style(color=_AMBER, bold=True),
        "markdown.code_block": Style(color=_AMBER),
        # h1 is the single brightest GOLD heading; h2+ step down to amber so the
        # heading hierarchy is visually distinct (not a flat wall of gold).
        "markdown.h1": Style(color=_AMBER_GOLD, bold=True),
        "markdown.h2": Style(color=_AMBER, bold=True),
        "markdown.h3": Style(color=_AMBER, bold=True),
        "markdown.h4": Style(color=_AMBER, bold=True),
        "markdown.h5": Style(color=_AMBER),
        "markdown.h6": Style(color=_AMBER),
        # GOLD bold so **important words** are the brightest thing in the answer.
        "markdown.strong": Style(color=_AMBER_GOLD, bold=True),
        "markdown.em": Style(color=_AMBER, italic=True),
        "markdown.item.bullet": Style(color=_AMBER, bold=True),
        "markdown.item.number": Style(color=_AMBER, bold=True),
        "markdown.link": Style(color=_AMBER_GOLD, underline=True),
        "markdown.link_url": Style(color=_AMBER_DIM),
        "markdown.block_quote": Style(color=_AMBER_DIM),
        "markdown.hr": Style(color=_AMBER_DIM),
    })


# Clean theme palette (the DEFAULT). A minimal DARK look: near-monochrome grey
# scale with ONE soft accent. Borders/prompt are a LOW-KEY dim grey (never loud),
# emphasis (**bold**) reads in near-white so important words pop, links keep a
# single restrained blue. Nothing sets a background box. Truecolor hex
# downsamples to 256/16 on lesser terminals.
_CLEAN = "#8b949e"          # soft grey accent — answer box border, prompt, rate
_CLEAN_BRIGHT = "#e6edf3"   # near-white — h1 + emphasis so key words pop
_CLEAN_DIM = "#484f58"      # very dim grey — links/urls/quotes/rules
_CLEAN_TEXT = "#c9d1d9"     # light grey — headers/inline code (calm, readable)
_CLEAN_LINK = "#79c0ff"     # one restrained blue — links only


def _clean_theme():
    """rich Theme for the minimal dark "clean" look.

    Near-monochrome: headers + inline code in a calm light grey, **bold** in
    near-white so important words are the brightest thing, bullets/numbers in the
    soft grey accent, links a single restrained blue. NO style sets a bgcolor
    (no boxes anywhere) — the only frame is the thin answer box from the shared
    layout, drawn in the dim grey accent.
    """
    from rich.style import Style
    from rich.theme import Theme

    return Theme({
        "markdown.code": Style(color=_CLEAN_TEXT),
        "markdown.code_block": Style(color=_CLEAN_TEXT),
        # h1 is the single brightest near-white heading; h2+ step down to the
        # calm light grey so the hierarchy reads without shouting.
        "markdown.h1": Style(color=_CLEAN_BRIGHT, bold=True),
        "markdown.h2": Style(color=_CLEAN_TEXT, bold=True),
        "markdown.h3": Style(color=_CLEAN_TEXT, bold=True),
        "markdown.h4": Style(color=_CLEAN_TEXT, bold=True),
        "markdown.h5": Style(color=_CLEAN_TEXT),
        "markdown.h6": Style(color=_CLEAN_TEXT),
        # Near-white bold so **important words** are the brightest thing.
        "markdown.strong": Style(color=_CLEAN_BRIGHT, bold=True),
        "markdown.em": Style(color=_CLEAN_TEXT, italic=True),
        "markdown.item.bullet": Style(color=_CLEAN, bold=True),
        "markdown.item.number": Style(color=_CLEAN, bold=True),
        "markdown.link": Style(color=_CLEAN_LINK, underline=True),
        "markdown.link_url": Style(color=_CLEAN_DIM),
        "markdown.block_quote": Style(color=_CLEAN_DIM),
        "markdown.hr": Style(color=_CLEAN_DIM),
    })


class Palette:
    """Per-theme accent colours for the banner, answer gutter, prompt, footer,
    and colour-coded status lines. The LAYOUT is identical across themes — only
    the accent differs — so switching themes restyles, never relayouts.

    Fields are rich style strings. ``ptk`` is the prompt_toolkit spelling of the
    accent for the input glyph (prompt_toolkit and rich take different colour
    vocabularies for the basic-ANSI theme).
    """

    def __init__(self, accent, bright, dim, success, ptk, gutter="▌", prompt="❯"):
        self.accent = accent
        self.bright = bright
        self.dim = dim
        self.success = success
        self.ptk = ptk
        self.gutter = gutter
        self.prompt = prompt


# clean is the default minimal-dark grey; amber/orange share the warm palette;
# auto is cool cyan; ansi stays inside the 16 basic ANSI colours (rich names +
# prompt_toolkit "ansi*" spellings) and uses a thin "│" gutter so it needs no
# truecolor. The "gutter" field is the box-border + status accent (the old left
# bar is gone; the shared layout now draws a thin box around the answer).
_PALETTES = {
    THEME_CLEAN: Palette(_CLEAN, _CLEAN_BRIGHT, _CLEAN_DIM, "#3fb950", "#8b949e"),
    THEME_AMBER: Palette(_AMBER, _AMBER_GOLD, _AMBER_DIM, "#7fd17f", "#ff9e3d bold"),
    THEME_ORANGE: Palette(_ORANGE, _ORANGE_BRIGHT, _ORANGE_DIM, "#7fd17f", "#ff9e3d bold"),
    THEME_AUTO: Palette("#5fd7ff", "#aef0ff", "#5f8aa0", "#7fd17f", "#5fd7ff bold"),
    THEME_ANSI: Palette(
        "yellow", "bright_yellow", "bright_black", "green",
        "ansiyellow bold", gutter="│",
    ),
}


def palette_for(theme: str) -> Palette:
    """Accent palette for ``theme`` (defaults to the clean theme for an unknown
    name — matching config.DEFAULT_THEME)."""
    return _PALETTES.get(theme, _PALETTES[THEME_CLEAN])


def _code_theme_for(theme: str) -> str:
    """Pygments code-block theme for the active app theme.

    "ansi" (Dark mode) uses the ANSI-only "ansi_dark" highlighter so fenced code
    stays within the 16 basic ANSI colors; "clean" uses the cool, low-contrast
    "github-dark" style to match its grey palette; "orange"/"amber" use the warm
    dark "native" style; every other theme keeps "monokai".
    """
    if theme == THEME_ANSI:
        return "ansi_dark"
    if theme == THEME_CLEAN:
        # "github-dark" is a calm, low-contrast dark pygments style that suits the
        # minimal grey look; fall back to "native"/"monokai" if a future pygments
        # drops it (both are also dark).
        from pygments.styles import get_style_by_name

        for name in ("github-dark", "native", "monokai"):
            try:
                get_style_by_name(name)
                return name
            except Exception:  # noqa: BLE001 - any lookup failure tries the next
                continue
        return "monokai"
    if theme in (THEME_ORANGE, THEME_AMBER):
        # "native" is a warm dark pygments style; fall back to monokai (also
        # amber-toned) if a future pygments drops it.
        from pygments.styles import get_style_by_name

        try:
            get_style_by_name("native")
            return "native"
        except Exception:  # noqa: BLE001 - any lookup failure keeps the safe default
            return "monokai"
    return "monokai"


def _make_console(theme: str = THEME_AUTO):
    # NOTE: this param default is only a RENDERING fallback for a no-arg call; it
    # is NOT the app-level default theme (that is config.DEFAULT_THEME = "amber").
    # Every real caller passes config.theme explicitly, so the two never conflict.
    from rich.console import Console

    # markup=False: the app uses literal square-bracket labels everywhere
    # ([error], [tool], [provider -> x]) AND streams raw model output that
    # routinely contains brackets (list[int], arr[i], [INFO], regex [a-z]).
    # With markup on, Rich would silently drop those tags or raise MarkupError
    # mid-stream. Treat all console text as plain.
    #
    # highlight=False: Rich's ReprHighlighter would otherwise re-color numbers,
    # paths, and booleans inside EVERY plain console.print — including dim text —
    # rendering the tok/s footer rate as BOLD CYAN and recoloring numeric tokens
    # in status lines (MCP "(16 tools)", auto-compact, max-iterations). The app
    # builds ALL color via explicit Text spans / style= args and never relies on
    # auto-highlight, so disabling it makes dim text render as uniform dim grey,
    # matching the Claude Code footer aesthetic.
    #
    # theme == "ansi" (Dark mode, ANSI colors only): pin color_system="standard"
    # so rich downsamples EVERY style to the 16 basic ANSI colors. The terminal
    # then renders those per the user's own (dark) palette, and NO truecolor
    # escape (\x1b[38;2;R;G;Bm) is emitted — only basic SGR (\x1b[32m, \x1b[2m).
    #
    # CAVEAT we guard against: passing color_system="standard" UNCONDITIONALLY
    # tells rich the stream is colored even when it is NOT a tty, so a piped /
    # redirected run would leak ANSI into the file (rich only auto-disables color
    # for a non-tty when color_system is left to auto-detect). So we pin
    # "standard" only when stdout is an actual terminal; piped runs fall through
    # to the auto-detect console, which correctly emits clean, ANSI-free text —
    # matching the auto theme's piped behavior.
    #
    # The default ("auto") leaves color_system unset so rich auto-detects
    # truecolor/256 as the terminal allows (unchanged behavior).
    #
    # theme == "orange": pass a rich Theme overriding markdown.* (inline code ->
    # orange text, NO bgcolor). color_system is left to auto-detect (orange needs
    # >16 colors; truecolor on a TTY, 256-color fallback otherwise).
    if theme == THEME_ANSI and _stdout_is_tty():
        return Console(color_system="standard", markup=False, highlight=False)
    if theme == THEME_CLEAN:
        return Console(theme=_clean_theme(), markup=False, highlight=False)
    if theme == THEME_AMBER:
        return Console(theme=_amber_theme(), markup=False, highlight=False)
    if theme == THEME_ORANGE:
        return Console(theme=_orange_theme(), markup=False, highlight=False)
    return Console(markup=False, highlight=False)


def _build_orchestrator(
    provider: Provider,
    config: Config,
    console,
    auto_confirm: bool,
    confirm_fn=None,
    mcp_tools: dict | None = None,
    cancel_event=None,
    checkpoint_session: str | None = None,
    suppress_footer: bool = False,
) -> Agent:
    # Code-block highlighting theme derived from the active app theme, threaded
    # into BOTH the orchestrator Agent and every spawned sub-agent so a delegated
    # run's Markdown matches the orchestrator (ansi_dark under Dark mode).
    code_theme = _code_theme_for(config.theme)
    palette = palette_for(config.theme)
    # The current working directory == this session's project. Injected into the
    # system prompt so the model KNOWS its project — critical for memory tools
    # (kyp-mem) which do NOT auto-detect the project and silently save to the
    # wrong/recent one if the model guesses the name.
    workspace = os.getcwd()
    # Code-RAG: a provider-bound, workspace-scoped code_search tool (hybrid BM25 ∪
    # embeddings over THIS project's source). Injected (not a global REGISTRY tool)
    # like spawn_agent, and given to BOTH the orchestrator AND every sub-agent. It
    # is local-only (reads workspace files, no egress) so it is SAFE in --private.
    code_search_tool = make_code_search_tool(
        provider=provider, workspace=workspace, private=config.private,
        config=config,
    )
    spawn_tool = make_spawn_agent_tool(
        provider=provider,
        console=console,
        auto_confirm=auto_confirm,
        max_iterations=config.max_iterations,
        private=config.private,
        # Forward the prompt_toolkit-safe confirm_fn so a spawned coder's gated
        # tool calls use the SAME confirm as the orchestrator, never builtin
        # input() (finding #1).
        confirm_fn=confirm_fn,
        code_theme=code_theme,
        # Accent threads to sub-agents so their footer/glyph match the theme.
        accent=palette.accent,
        # Sub-agents get the SAME project/cwd context as the orchestrator.
        workspace=workspace,
        # Context guard so a delegated coder/explorer compacts like the orchestrator
        # instead of growing unbounded to max_iterations and overflowing the local
        # window (AGENT-2/ORCH-3). Ceiling = absolute near-window safety value.
        context_budget=config.context_budget,
        context_ceiling=_effective_soft_limit(provider, config),
        # Forward the adaptive-budget setting so a sub-agent honours /context off
        # too (otherwise it always used the Agent default True).
        context_adaptive=config.context_adaptive,
        # Give spawned explorer/coder/reviewer sub-agents code_search too.
        code_search_tool=code_search_tool,
        # Gentle sub-agent cool-down: spaces out multi-spawn orchestrator bursts
        # to lower average GPU heat (does NOT cap GPU %). Only fires in a real
        # terminal before the 2nd+ spawn; the first spawn never waits.
        gentle_mode=config.gentle_mode,
        gentle_spawn_gap_seconds=config.gentle_spawn_gap_seconds,
        is_terminal=getattr(console, "is_terminal", False),
        # Propagate the orchestrator's permission mode + cooperative cancel to
        # spawned sub-agents so a delegated coder honours /mode (e.g. plan) and a
        # single Ctrl-C stops the sub-agent's turn too (not just the orchestrator).
        permission_mode=config.permission_mode,
        cancel_event=cancel_event,
        # A long delegated coder run also self-heals failed tool calls and paces
        # the GPU (mirrors the orchestrator's config).
        auto_fix_tools=config.auto_fix_tools,
        auto_fix_max_attempts=config.auto_fix_max_attempts,
        cooldown_enabled=config.cooldown_enabled,
    )
    extra_tools = [code_search_tool]
    registry = orchestrator_registry(
        spawn_tool, mcp_tools, private=config.private, extra_tools=extra_tools
    )
    tool_names = orchestrator_tool_names(
        spawn_tool, mcp_tools, private=config.private, extra_tools=extra_tools
    )
    # The kyp-mem writing guidance is only appended when a memory MCP tool is
    # actually loaded (finding #32), trimming the per-turn system prefix otherwise.
    has_memory_tool = _has_memory_tool(tool_names)
    # Conversation-memory store for PASSIVE hybrid retrieval. Best-effort load
    # (an empty store on a missing/corrupt file — never raises), keyed by the
    # same workspace id as the session so it sits next to the session file. Only
    # the ORCHESTRATOR gets it; sub-agents stay on the no-memory defaults.
    mem = memory.MemoryStore.load(memory.store_path(workspace))
    # System prompt = orchestrator body + (optionally) the project rules block so
    # user-authored conventions (AGENTS.md/LLMCLI.md/…) become binding context.
    # Best-effort: rules_prompt_block returns "" when no rules file exists.
    sp = orchestrator_prompt(has_memory_tool=has_memory_tool, workspace=workspace)
    if config.rules_file_enabled:
        rb = rules.rules_prompt_block(workspace)
        if rb:
            sp += "\n\n" + rb
    return Agent(
        provider=provider,
        system_prompt=sp,
        tool_names=tool_names,
        console=console,
        auto_confirm=auto_confirm,
        max_iterations=config.max_iterations,
        confirm_fn=confirm_fn,
        registry=registry,
        # Foundation-wave wiring (orchestrator only; sub-agents stay on defaults):
        # permission mode, file-snapshot checkpoints for /undo, lifecycle hooks
        # (when enabled), the workspace confinement root, cooperative cancellation,
        # and the model's todo_write checklist tool.
        permission_mode=config.permission_mode,
        checkpoints_enabled=config.checkpoints_enabled,
        hooks=(hooks.load_hooks() if config.hooks_enabled else None),
        workspace_root=workspace,
        cancel_event=cancel_event,
        # Per-launch checkpoint session so this session's /undo only sees its own
        # snapshots (None for one-shot run_once → legacy shared layout).
        checkpoint_session=checkpoint_session,
        todos_enabled=True,
        # Adaptive working budget: keep the live context TIGHT (config.context_budget,
        # flexed per request) so decode stays fast, capped by the near-window
        # safety value (_effective_soft_limit) which is the absolute ceiling.
        context_budget=config.context_budget,
        context_ceiling=_effective_soft_limit(provider, config),
        context_adaptive=config.context_adaptive,
        # PERF read-budget guard threshold (configurable; see config.read_nudge_bytes).
        read_nudge_bytes=config.read_nudge_bytes,
        code_theme=code_theme,
        # Accent + gutter glyph drive the "▌ Answer" block, the footer rate, and
        # the activity glyph for the top-level orchestrator.
        accent=palette.accent,
        gutter_char=palette.gutter,
        # Passive conversation-memory: inject relevant past records each turn and
        # record new Q/A turns. Active only when enabled AND mode != "off".
        memory=mem,
        recall_mode=config.recall_mode,
        memory_top_k=config.memory_top_k,
        memory_enabled=config.memory_enabled,
        # Feature 2/3/4 — top-level orchestrator only. constrained_retry forces a
        # clean native tool call on a malformed one; verify_cmd auto-runs after an
        # edit; review_writes spawns the reviewer sub-agent on a code change. Sub-
        # agents keep the Agent defaults (off / empty) so they never recurse.
        constrained_retry=config.constrained_retry,
        verify_cmd=config.verify_cmd,
        review_writes=config.review_writes,
        # Gated LLM-judge reranker for per-turn memory retrieval. /rerank mirrors
        # the live flag onto the agent so a toggle applies this turn.
        rerank=config.rerank,
        rerank_candidates=config.rerank_candidates,
        # MANDATORY self-healing + thermal cooldown (orchestrator + sub-agents):
        # a failed tool call is auto-corrected + retried before the model sees it,
        # and a long turn breaks mid-way to let the GPU cool.
        auto_fix_tools=config.auto_fix_tools,
        auto_fix_max_attempts=config.auto_fix_max_attempts,
        cooldown_enabled=config.cooldown_enabled,
        # Interactive REPL only: the pinned bottom status bar owns the per-turn
        # Model|Time|Speed stats, so the orchestrator skips its own footer LINE
        # (last_turn_stats is still populated for the bar to read). One-shot `-p`
        # keeps the default False so piped/script output still shows the footer.
        suppress_footer=suppress_footer,
    )


def _load_into_agent(agent: Agent, messages: list) -> None:
    """Replace ``agent.messages`` with a saved history, keeping the system prompt.

    The agent is otherwise untouched (same provider/tools/console). The saved
    messages already include their own system prompt as message[0], so we adopt
    the list wholesale; if a saved file somehow lacks one, we keep the agent's
    current system prompt as the head so the next turn still has a system role.
    """
    if not isinstance(messages, list) or not messages:
        return
    # Defensive: only treat the head as the saved system prompt when it is a
    # well-formed message dict. A non-dict head (corrupt/hand-edited file) falls
    # through to the else branch, which keeps the agent's real system prompt.
    head = messages[0]
    if isinstance(head, dict) and head.get("role") == "system":
        agent.messages = list(messages)
    else:
        agent.messages = [agent.messages[0]] + list(messages)


def run_once(
    provider: Provider, config: Config, prompt: str, auto_confirm: bool,
    resume: bool = False,
) -> str:
    """Run a single prompt through a fresh orchestrator and return final text.

    Connects MCP servers (if ~/.llm-cli/mcp.json exists), runs the prompt with
    their tools merged in, then cleanly shuts them down. With no config file MCP
    is simply off and behavior is unchanged.

    With ``resume`` (the -c/--continue one-shot path) the saved session for the
    cwd is PREPENDED into the agent's history before the prompt, so a one-shot
    keeps the previous conversation's context. Missing/corrupt session -> no-op,
    so a one-shot without a saved session is unchanged. The completed turn is
    saved back so a one-shot can build on itself across runs.
    """
    # JSON output mode: suppress ALL decorative UI (banner, streamed markdown,
    # tool activity, MCP connect lines) by routing the orchestrator's console to
    # os.devnull, then emit ONE machine-readable object to real stdout at the end.
    # Text mode is unchanged.
    json_mode = config.output_format == "json"
    devnull_fh = None
    if json_mode:
        from rich.console import Console

        # Keep a handle so it can be CLOSED in finally (no leaked fd), rather than
        # relying on GC of an anonymous open().
        devnull_fh = open(os.devnull, "w", encoding="utf-8")
        console = Console(file=devnull_fh, markup=False, highlight=False)
    else:
        console = _make_console(config.theme)
    manager = MCPManager(load_mcp_config(), private=config.private)
    # Honor the MCP toggle: when disabled, don't start servers and offer no MCP
    # tools (smaller per-turn prompt, faster decode).
    if config.mcp_enabled:
        manager.start_all(console=console)
    try:
        # MANDATORY thermal cooldown: configure the process-global pacer from this
        # run's config so a long one-shot turn breaks mid-way to cool the GPU.
        cooldown.configure(
            enabled=config.cooldown_enabled,
            interval_seconds=config.cooldown_interval_seconds,
            duration_seconds=config.cooldown_duration_seconds,
        )
        mcp_tools = manager.registry() if config.mcp_enabled else None
        agent = _build_orchestrator(
            provider, config, console, auto_confirm, mcp_tools=mcp_tools
        )
        cwd = os.getcwd()
        if resume:
            data = session.load_session(cwd)
            if data is not None:
                _load_into_agent(agent, data["messages"])
        # JSON mode contract: emit EXACTLY ONE object on stdout whether the run
        # succeeds or a provider/agent error is raised. Text mode is unchanged
        # (the exception propagates as before).
        try:
            result = agent.run(prompt)
        except Exception as exc:  # noqa: BLE001 - one JSON object even on error
            if json_mode:
                print(json.dumps(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                ))
                return ""
            raise
        session.save_session(
            cwd, agent.messages, config.model, session.derive_title(agent.messages)
        )
        # Persist the conversation-memory store so a one-shot builds memory across
        # runs (best-effort; only the orchestrator carries a store).
        if getattr(agent, "memory", None) is not None:
            agent.memory.save(memory.store_path(cwd))
        if json_mode:
            # One parseable object to REAL stdout (the quiet console discarded the
            # decorative UI). json.dumps default-escapes so the answer stays valid.
            print(json.dumps(
                {"ok": True, "answer": result, "model": config.model}
            ))
        return result
    finally:
        manager.shutdown_all()
        if devnull_fh is not None:
            try:
                devnull_fh.close()
            except OSError:
                pass


class Repl:
    def __init__(
        self, config: Config, provider: Provider, auto_confirm: bool = False,
        resume: bool = False,
    ):
        self.config = config
        self.provider = provider
        self.auto_confirm = auto_confirm
        # When True (from -c/--continue), run() loads this project's saved session
        # into the agent at startup before the first turn (Claude-Code-style
        # resume). The default keeps a fresh launch light (no auto-load).
        self.resume = resume
        self.console = _make_console(config.theme)
        # SESSION-ONLY PRIVACY CONTRACT (finding #2): the privacy/egress fields
        # (`private`, `base_url`) may be SESSION-ONLY overrides set by
        # --allow-network / --base-url WITHOUT --save. Routine REPL saves
        # (/provider, /model, /effort) must NOT persist those session overrides
        # to disk — otherwise an unrelated `/model qwen3` would permanently turn
        # private mode OFF. Snapshot the on-disk values now and write THOSE back
        # on every incidental save, so only the field the user actually changed
        # is persisted. (The startup --save path already handles deliberate
        # persistence of the privacy flag in __main__.)
        disk = load_config()
        self._disk_private = disk.private
        self._disk_base_url = disk.base_url
        # The PromptSession is built in run(); the confirm_fn closes over it.
        # Until then it is None and _new_agent uses the Agent default — but the
        # REPL always rebuilds agents after the session exists (see run()).
        self.session = None
        # MCP servers are connected in run() (so the dim connect lines print
        # after the console exists and in the right order). Until then the
        # manager is empty, so the first agent simply has no MCP tools.
        self.mcp = MCPManager(load_mcp_config(), private=config.private)
        # LAZY MCP START: servers are started in a BACKGROUND daemon thread so
        # the first prompt is not blocked ~5s on kyp-mem's startup. The first
        # agent is built WITHOUT MCP tools; once the background start finishes
        # (mid-session) the next turn rebuilds the agent so its tools are offered.
        # _mcp_ready = the manager's tools/clients are populated and safe to
        # read (set AFTER start_all returns, so _new_agent never sees a
        # half-mutated registry). _mcp_integrated = the LIVE self.agent already
        # has MCP tools baked in (so we rebuild only once on the flip).
        self._mcp_thread: threading.Thread | None = None
        self._mcp_ready = False
        self._mcp_integrated = False
        # Images staged via /image, attached to the NEXT user message and then
        # cleared. Each entry is (label, encoded_part) — label for /image listing,
        # part is the OpenAI vision content dict passed to agent.run(images=...).
        self._staged_images: list[tuple[str, dict]] = []
        # GENTLE-mode pacing: monotonic time the last generation ENDED. Used to
        # space rapid back-to-back turns with a cool-down (gentle_gap_seconds).
        # 0.0 means "no prior generation" so the first turn never waits.
        self._last_gen_end: float = 0.0
        # Cooperative interrupt: SET by a SIGINT handler installed only DURING a
        # turn (see _submit) so Ctrl-C makes the agent stop mid-stream and finalize
        # cleanly instead of raising. Cleared before each turn. Shared with the
        # agent via _build_orchestrator(cancel_event=...) in _new_agent.
        self._cancel_event = threading.Event()
        # Lazily-computed cached list of workspace-relative files for the @-mention
        # completer, so a big repo isn't re-walked on every keystroke. None = not
        # yet computed; _refresh_completion_files() invalidates it.
        self._completion_file_cache: list[str] | None = None
        # Per-LAUNCH checkpoint session token so /undo only ever affects the
        # file-snapshot checkpoints THIS REPL session created (QA: a stale /undo
        # from a different session must not resurrect an ancient checkpoint). A
        # fresh launch → a new token → "nothing to undo" until this session writes.
        self._ckpt_session = uuid.uuid4().hex[:12]
        # Pinned bottom status-bar cache (model · ctx% · git · tok/s · time).
        # _status_bar (the prompt_toolkit bottom_toolbar callable) reads THIS on
        # every redraw/keystroke, so it is only ever RECOMPUTED by
        # _refresh_status_bar (once at startup + once per completed turn) — never
        # per keystroke (which would run git + re-estimate tokens on each key).
        self._status_cache = ""
        self.agent = self._new_agent()

    def _persist_config(self) -> None:
        """Persist model/provider/effort WITHOUT leaking session-only privacy.

        Writes the live config but with the privacy/egress fields forced back to
        their on-disk snapshot, so a session-only --allow-network / --base-url
        override is never silently persisted by an incidental REPL save
        (finding #2). The startup --save path remains the only way to persist a
        privacy change deliberately.
        """
        save_config(
            replace(
                self.config,
                private=self._disk_private,
                base_url=self._disk_base_url,
            )
        )

    def _save_session(self) -> None:
        """Auto-save the running conversation for the cwd (best-effort, no raise).

        Called after every completed user turn and on clean exit so a crash never
        loses the conversation. save_session itself swallows OSError and skips a
        history of <=1 message (just the system prompt), so this is always safe.
        """
        session.save_session(
            os.getcwd(),
            self.agent.messages,
            self.config.model,
            session.derive_title(self.agent.messages),
        )
        # Persist the conversation-memory store alongside the session (best-effort;
        # save() itself never raises). Only the orchestrator carries a store.
        if getattr(self.agent, "memory", None) is not None:
            self.agent.memory.save(memory.store_path(os.getcwd()))

    def _confirm_fn(self):
        """Return a prompt_toolkit-compatible confirm_fn once a session exists.

        Returns None before the session is built so Agent falls back to its
        default; run() rebuilds the agent with the real confirm_fn afterwards.
        """
        if self.session is None:
            return None
        return make_ptk_confirm(self.session, self.config)

    def _new_agent(self) -> Agent:
        # MCP tools are offered ONLY when enabled AND the background start has
        # completed (_mcp_ready). When off or still loading, pass None so neither
        # the tool schemas nor the kyp-mem writing-guidance bloat the prompt,
        # and so _new_agent never reads a half-mutated registry during start.
        mcp_tools = (
            self.mcp.registry()
            if (self.config.mcp_enabled and self._mcp_ready)
            else None
        )
        return _build_orchestrator(
            self.provider,
            self.config,
            self.console,
            self.auto_confirm,
            confirm_fn=self._confirm_fn(),
            mcp_tools=mcp_tools,
            # Cooperative Ctrl-C: the agent checks this event mid-stream + between
            # rounds; _submit installs a SIGINT handler that SETs it during a turn.
            cancel_event=self._cancel_event,
            # Session-scoped checkpoints: /undo only affects THIS launch's writes.
            checkpoint_session=self._ckpt_session,
            # The interactive REPL shows Model|Time|Speed in the pinned bottom
            # status bar, so the orchestrator suppresses its own footer line here
            # (run_once/`-p` keeps the default False so piped output still shows it).
            suppress_footer=True,
        )

    def _start_mcp_background(self, console) -> None:
        """Start MCP servers on a BACKGROUND daemon thread.

        The initial start is non-blocking so the first prompt is not delayed ~5s
        by kyp-mem's startup. The first agent is built WITHOUT MCP tools; once
        this thread finishes (mid-session) ``_mcp_ready`` flips True and the next
        turn rebuilds the agent with MCP tools offered (see ``_submit``).
        Mirrors the lazy-start path ``/mcp on`` already uses. If MCP is disabled
        or no servers are configured, this is a no-op.
        """
        if not self.config.mcp_enabled:
            return
        if not getattr(self.mcp, "configs", None):
            return
        if self._mcp_thread is not None and self._mcp_thread.is_alive():
            return  # already starting

        def _bg() -> None:
            started = False
            try:
                self.mcp.start_all(console=console)
                started = self.mcp.is_running()
            except Exception as exc:  # noqa: BLE001 - never crash the REPL on MCP
                if console is not None:
                    try:
                        console.print(
                            f"[dim]MCP startup failed: {exc}. "
                            f"Use /mcp on to retry.[/dim]"
                        )
                    except Exception:  # noqa: BLE001 - console printing must not crash
                        pass
            # Mark safe-to-read ONLY when start_all actually brought servers up,
            # so a silent failure never leaves the user with no MCP tools AND a
            # misleadingly-true ready flag. /mcp on can still retry synchronously.
            self._mcp_ready = started

        t = threading.Thread(target=_bg, name="mcp-start", daemon=True)
        self._mcp_thread = t
        t.start()

    def _ensure_mcp_started_sync(self, console) -> None:
        """Synchronous MCP start used by ``/mcp on``.

        If a background start is in flight, wait for it (so we don't double-spawn
        subprocesses). If MCP still isn't running after that, start synchronously.
        Then mark ready + integrated so the rebuilt agent picks up the tools.
        """
        t = self._mcp_thread
        if t is not None and t.is_alive():
            t.join()
        if not self.mcp.is_running():
            self.mcp.start_all(console=console)
        self._mcp_ready = True
        self._mcp_integrated = True

    def _status(self) -> str:
        # Show effort always (unset shows "unset") for a stable status line, and
        # surface base_url only when it differs from the default so a user on a
        # custom LM Studio port sees where requests go.
        s = f"provider={self.config.provider} model={self.config.model}"
        s += f" effort={self.config.effort or 'unset'}"
        s += f" theme={self.config.theme}"
        s += f" mcp={'on' if self.config.mcp_enabled else 'off'}"
        if self.config.context_budget > 0:
            s += f" ctx={self.config.context_budget}{'~' if self.config.context_adaptive else ''}"
        else:
            s += " ctx=off"
        if self.config.base_url != DEFAULT_BASE_URL:
            s += f" base_url={self.config.base_url}"
        # Private-mode indicator. DEFAULT: network enabled with always-on safety
        # (SSRF-safe + confirmation-gated web_fetch; confirmation-gated run_bash,
        # which has FULL filesystem access — the prompt is the boundary).
        # --private => offline no-network sandbox for run_bash + no web_fetch.
        if self.config.private:
            s += " | private mode: ON — offline lockdown, no external egress"
        else:
            s += (
                " | private mode: OFF — network enabled (web_fetch SSRF-safe; "
                "run_bash confirmation-gated, full FS access)"
            )
        return s

    # ----- colour-coded status lines + banner ----------------------------
    def _ok(self, msg: str) -> None:
        """Accent-coloured confirmation (theme/model/provider switches, etc.)."""
        self.console.print(msg, style=palette_for(self.config.theme).accent)

    def _err(self, msg: str) -> None:
        """Red error line (failed switch, bad input that aborts an action)."""
        self.console.print(msg, style="red")

    def _print_banner(self) -> None:
        """Framed startup banner: provider/model, a green 'ready' dot, and a
        compact theme/privacy line — replacing the old flat one-liner."""
        from rich import box
        from rich.align import Align
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        pal = palette_for(self.config.theme)
        head = Text()
        head.append("◆ ", style=pal.accent)
        # Short model only (drop the "<provider> ·" prefix AND any "org/" prefix,
        # e.g. qwen/qwen3.6-35b-a3b -> qwen3.6-35b-a3b); the pinned status bar
        # carries the fuller live context.
        short_model = (self.config.model or "").rsplit("/", 1)[-1]
        head.append(short_model, style=pal.bright)  # make the model pop
        head.append("   ")
        head.append("● ", style=pal.success)
        head.append("ready", style="dim")
        sub = Text(style=pal.dim)  # muted accent sub-line (on-theme, not grey)
        sub.append(f"theme {self.config.theme}")
        sub.append("  ·  ")
        sub.append("private on" if self.config.private else "private off")
        sub.append("  ·  ")
        sub.append("mcp on" if self.config.mcp_enabled else "mcp off")
        sub.append("  ·  ")
        sub.append("gentle on" if self.config.gentle_mode else "gentle off")
        if self.config.base_url != DEFAULT_BASE_URL:
            sub.append("  ·  ")
            sub.append(self.config.base_url)
        self.console.print()
        # Center the header block horizontally (reference design): an expand=False
        # panel hugs its content, and Align.center then centers that hugged box in
        # the terminal width. (justify="center" alone does NOT center an
        # expand=False panel.)
        self.console.print(Align.center(Panel(
            Group(head, sub), box=box.ROUNDED, border_style=pal.accent,
            title="llm-cli", title_align="left", padding=(0, 1),
            expand=False,  # hug the content instead of spanning the full width
        )))
        # Keep the privacy posture visible on startup (security transparency),
        # compact and dim under the frame.
        # Kept short enough to fit ~80 cols without wrapping under the frame.
        if self.config.private:
            privacy = "offline lockdown · no external egress"
        else:
            privacy = ("network on · web_fetch SSRF-safe+gated · "
                       "run_bash gated (full FS access)")
        # Centered to match the centered header block above (justify="center"
        # centers plain text within the terminal width).
        self.console.print(privacy, style="dim", justify="center")
        self.console.print(
            "Type /help for commands · Ctrl+O reveals tool detail",
            style="dim", justify="center",
        )
        # Breathing room: a blank line below the banner so the first prompt and
        # answer box are not glued to the header (clean, uncongested startup).
        self.console.print()

    def _print_mcp_status(self) -> None:
        """List configured MCP servers, their connection state + tools."""
        statuses = self.mcp.status()
        if not statuses:
            self.console.print(
                "No MCP servers configured. Add them to ~/.llm-cli/mcp.json "
                "(Claude Desktop format).",
                style="dim",
            )
            return
        for st in statuses:
            if st["connected"]:
                tools = st["tools"]
                self.console.print(f"  {st['name']}: connected ({len(tools)} tools)")
                for tn in tools:
                    self.console.print(f"      - {tn}", style="dim")
            else:
                self.console.print(
                    f"  {st['name']}: not connected ({st['error']})", style="dim",
                )

    def _model_ok(self, name: str) -> bool:
        """Verify a model exists on the local server before switching to it.

        Returns True to allow the switch: the model is in the server's list, OR
        the server can't be reached to verify (we warn and allow, since the user
        may be configuring before starting it). Returns False (and prints the
        available models) only when the server IS reachable and the name is not
        in its list — so a typo like '/model fkbf' is rejected.
        """
        self.console.print(
            f"querying server at {self.config.base_url}...", style="dim"
        )
        try:
            models = list_local_models(self.config.base_url, get_api_key(), self.config.private)
        except Exception as exc:  # noqa: BLE001 - unverifiable; allow with a warning
            self.console.print(
                f"[warn] couldn't reach the server to verify '{name}' "
                f"({type(exc).__name__}); switching anyway.",
                style="dim",
            )
            return True
        if name in models:
            return True
        self.console.print(f"Unknown model '{name}'. Available on the server:")
        for m in models:
            self.console.print(f"  - {m}")
        return False

    def _dispatch_slash(self, line: str) -> bool:
        """Handle a /slash command. Return False to signal exit, True to continue."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            self.console.print(HELP_TEXT)
        elif cmd == "/clear":
            self.agent = self._new_agent()
            # Wipe the visible scrollback too (not just the history), so /clear is
            # a true fresh start. TTY-only: console.clear() emits the screen-clear
            # control sequence, which is meaningless when piped/redirected, so we
            # gate it on a real terminal (a no-op otherwise — keeps piped output
            # and the slash-command tests escape-free).
            if getattr(self.console, "is_terminal", False):
                self.console.clear()
            self._ok("[cleared conversation]")
        elif cmd == "/resume":
            data = session.load_session(os.getcwd())
            # A loadable file with no real history (empty/only-system after
            # validation) is treated like "none" so we never report a phantom
            # resume of just the system prompt.
            if data is None or len(data["messages"]) <= 1:
                self.console.print("no saved session for this directory.")
            else:
                # Rebuild the agent so the next turn uses a clean orchestrator,
                # then replace its history with the saved messages (keeping the
                # provider/tools/console). The agent is otherwise untouched.
                self.agent = self._new_agent()
                _load_into_agent(self.agent, data["messages"])
                rel = session.relative_time(data.get("updated_at", ""))
                self.console.print(
                    f"resumed {len(self.agent.messages)} messages from {rel}."
                )
        elif cmd == "/forget":
            cwd = os.getcwd()
            session.clear_session(cwd)
            # Clear conversation MEMORY too: delete the persisted store and reset
            # the live one so a forgotten project starts truly fresh (no recalled
            # records). Best-effort/no-raise, mirroring clear_session.
            try:
                memory.store_path(cwd).unlink(missing_ok=True)
            except OSError:
                pass
            if getattr(self.agent, "memory", None) is not None:
                self.agent.memory = memory.MemoryStore()
            self.console.print("forgot this project's saved session and memory.")
        elif cmd == "/memory":
            # Memory management: purge (remove records) and compact (reduce store size).
            # Usage: /memory purge <all|r0 r2 ...>  — remove specific records or all
            #        /memory compact [N]            — shrink to N records (default MAX)
            if not arg:
                self.console.print(
                    "Usage: /memory purge <all|r0 r2 ...>  — remove records\n"
                    "       /memory compact [N]            — shrink to N records"
                )
                return True
            parts = arg.split(maxsplit=1)
            action = parts[0].lower()
            if action == "purge":
                # Clear the on-disk memory file.
                cwd = os.getcwd()
                try:
                    memory.store_path(cwd).unlink(missing_ok=True)
                except OSError:
                    self._err("failed to delete memory file")
                    return True
                # Purge live memory if present.
                if getattr(self.agent, "memory", None) is not None:
                    remaining = self.agent.memory.purge()
                    self.console.print(
                        f"purged {remaining} record(s) from memory."
                    )
                else:
                    self.console.print("purged memory file from disk.")
            elif action == "compact":
                if getattr(self.agent, "memory", None) is None:
                    self.console.print("no memory store loaded.")
                    return True
                try:
                    n = int(parts[1]) if len(parts) > 1 else None
                except (ValueError, IndexError):
                    n = None  # use default MAX_RECORDS
                removed = self.agent.memory.compact(max_records=n)
                if removed:
                    self.console.print(
                        f"compacted: removed {removed} oldest record(s). "
                        f"{len(self.agent.memory.records)} remaining."
                    )
                else:
                    self.console.print(
                        "[nothing to compact] store already within limit."
                    )
            else:
                self.console.print(f"Unknown memory action: {action!r}. Use purge or compact.")
        elif cmd == "/theme":
            if arg not in THEMES:
                self.console.print(f"Usage: /theme {{{'|'.join(THEMES)}}}")
            else:
                # Rebuild the console under the new color_system (a Console's
                # color system is fixed at construction), then rebuild the agent
                # so it both points at the new console AND uses the matching
                # code_theme (ansi_dark vs monokai) for Markdown code blocks.
                self.config.theme = arg
                self.console = _make_console(arg)
                self.agent = self._new_agent()
                # Restyle the live input "❯" glyph to the new theme's accent.
                # PromptSession reads .style per prompt() call, so the next prompt
                # picks it up with no relayout. (self.session is None until run()
                # builds it — a no-op for the slash-command unit tests.)
                if self.session is not None:
                    from prompt_toolkit.styles import Style as PTKStyle

                    self.session.style = PTKStyle.from_dict(
                        {"prompt": palette_for(arg).ptk}
                    )
                self._persist_config()
                self._ok(f"[theme -> {arg}]  {self._status()}")
        elif cmd == "/compact":
            old_n = len(self.agent.messages)
            try:
                # Aggressive: summarize EVERYTHING, keep only the last exchange.
                before, after = self.agent.compact(keep_turns=1)
            except Exception as exc:  # noqa: BLE001 - never crash / never lose history
                self._err(f"[compact failed: {type(exc).__name__}] history unchanged")
            else:
                new_n = len(self.agent.messages)
                if before == after and new_n == old_n:
                    # Nothing to summarize (e.g. a single short turn) — don't
                    # claim a compaction that didn't happen.
                    self._ok("[nothing to compact] history already minimal")
                else:
                    # NOTE: ~N is a rough chars/4 estimate over history (a different
                    # measure than the per-message tok/s footers, which use the
                    # provider's real token count). Labeled to avoid implying they
                    # are comparable.
                    self._ok(
                        f"[compacted] ~{before} -> ~{after} est. tok "
                        f"(chars/4; history {old_n} -> {new_n} msgs)"
                    )
        elif cmd == "/mcp":
            a = arg.lower()
            if a in ("on", "off"):
                if a == "on":
                    # Start servers if not already up (avoid double-spawning), then
                    # rebuild the agent so their tools are offered again. Use the
                    # sync helper so a background start in flight is JOINED first
                    # (no double-spawn) and the ready/integrated flags are set.
                    self.config.mcp_enabled = True
                    self._ensure_mcp_started_sync(self.console)
                else:
                    # Stop the server subprocesses AND drop their tools; the rebuilt
                    # agent then sends a smaller prompt → faster tok/s.
                    self.mcp.shutdown_all()
                    self.config.mcp_enabled = False
                    self._mcp_ready = False
                    self._mcp_integrated = False
                self.agent = self._new_agent()
                self._persist_config()
                self._ok(f"[mcp -> {a}]  {self._status()}")
                if self.config.mcp_enabled:
                    self._print_mcp_status()
            elif self.config.mcp_enabled:
                # No arg: report state + the live server/tool listing.
                self._ok("MCP is on.  (disable with /mcp off)")
                self._print_mcp_status()
            else:
                self._ok("MCP is off.  (enable with /mcp on)")
                configured = list(getattr(self.mcp, "configs", {}) or {})
                if configured:
                    self.console.print(
                        f"  configured but not loaded: {', '.join(configured)}",
                        style="dim",
                    )
        elif cmd == "/audit":
            # Map-reduce audit: reviews the repo in small isolated chunks instead
            # of one giant context. Does NOT touch the main conversation history.
            from .audit import run_audit

            pal = palette_for(self.config.theme)
            try:
                run_audit(
                    self.provider, self.config, self.console,
                    accent=pal.accent, code_theme=_code_theme_for(self.config.theme),
                    path=arg or ".",
                )
            except KeyboardInterrupt:
                self.console.print("\n[audit interrupted]", style="red")
            except Exception as exc:  # noqa: BLE001 - never kill the REPL
                self._err(f"[audit error] {type(exc).__name__}: {exc}")
        elif cmd == "/speed":
            self.console.print(SPEED_TIPS)
        elif cmd == "/context":
            a = arg.lower()
            changed = True
            if not a or a == "status":
                changed = False
                b = self.config.context_budget
                mode = "adaptive" if self.config.context_adaptive else "fixed"
                state = f"{b} tok ({mode})" if b > 0 else "off (near-window only)"
                self._ok(f"context budget: {state}")
                # Live per-turn snapshot of the CURRENT history (Feature 1). Some
                # attributes only exist after a turn has run, so guard everything
                # with getattr/defaults so /context works before the first turn.
                agent = self.agent
                est = agent._estimate_tokens(getattr(agent, "messages", []) or [])
                # Active turn budget target: the live soft_limit set by the last
                # turn's adaptive computation; before any turn it's 0, so fall back
                # to the configured base budget.
                turn_budget = getattr(agent, "context_soft_limit", 0) or b
                ceiling = getattr(agent, "context_ceiling", 0)
                nudge = getattr(agent, "read_nudge_bytes", _TURN_READ_NUDGE_BYTES)
                read_kb = getattr(agent, "_read_bytes", 0) / 1000
                # Health hint: compare est against the active budget target (or the
                # ceiling when no budget). OK when comfortably under; warn near it.
                limit = turn_budget or ceiling
                if limit > 0 and est >= 0.85 * limit:
                    health = "near limit — /compact or /clear to speed up"
                else:
                    health = "OK"
                self.console.print(
                    f"  live: ~{est} est tok · budget ~{turn_budget} · "
                    f"ceiling {ceiling} · read {read_kb:.1f}/{nudge / 1000:.0f}KB this turn",
                    style="dim",
                )
                self.console.print(f"  health: {health}", style="dim")
                self.console.print(
                    "  /context <N> set budget · auto adaptive · fixed flat · off disable",
                    style="dim",
                )
            elif a == "off":
                self.config.context_budget = 0
            elif a == "auto":
                self.config.context_adaptive = True
                if self.config.context_budget <= 0:
                    self.config.context_budget = Config().context_budget  # restore default
            elif a == "fixed":
                self.config.context_adaptive = False
            else:
                try:
                    n = int(arg)
                except ValueError:
                    self.console.print("Usage: /context <N> | auto | fixed | off")
                    return True
                self.config.context_budget = max(0, n)
            # Any actual change rebuilds the agent (so it applies next turn) and
            # persists; status-only is a no-op.
            if changed:
                self.agent = self._new_agent()
                self._persist_config()
                b = self.config.context_budget
                mode = "adaptive" if self.config.context_adaptive else "fixed"
                state = f"{b} tok ({mode})" if b > 0 else "off"
                self._ok(f"[context -> {state}]")
        elif cmd == "/provider":
            if arg not in PROVIDERS:
                self.console.print(f"Usage: /provider {{{ '|'.join(PROVIDERS) }}}")
            else:
                try:
                    self.provider = build_provider(
                        arg, self.config.model, self.config.base_url,
                        self.config.effort, self.config.private,
                        self.config.cache_prompt, self.config.max_output_tokens,
                        embed_model=self.config.embed_model,
                        temperature=self.config.temperature,
                        gentle_mode=self.config.gentle_mode,
                        gentle_max_tokens=self.config.gentle_max_tokens,
                        seed=self.config.seed, id_slot=self.config.id_slot,
                    )
                except ValueError as exc:
                    self._err(f"[error] {exc}")
                else:
                    self.config.provider = arg
                    self.agent = self._new_agent()
                    self._persist_config()
                    self._ok(f"[provider -> {arg}]  {self._status()}")
        elif cmd == "/models":
            self.console.print(
                f"querying server at {self.config.base_url}...", style="dim"
            )
            try:
                models = list_local_models(self.config.base_url, get_api_key(), self.config.private)
            except Exception as exc:  # noqa: BLE001
                self._err(f"[error] couldn't list models: {type(exc).__name__}: {exc}")
            else:
                self.console.print("Available models:")
                for m in models:
                    self.console.print(f"  - {m}" + ("  (active)" if m == self.config.model else ""))
        elif cmd == "/model":
            if not arg:
                self.console.print("Usage: /model <name>  (see /models)")
            elif self.config.provider == PROVIDER_LOCAL and not self._model_ok(arg):
                pass  # _model_ok already printed why; keep the current model
            else:
                try:
                    self.provider = build_provider(
                        self.config.provider, arg, self.config.base_url,
                        self.config.effort, self.config.private,
                        self.config.cache_prompt, self.config.max_output_tokens,
                        embed_model=self.config.embed_model,
                        temperature=self.config.temperature,
                        gentle_mode=self.config.gentle_mode,
                        gentle_max_tokens=self.config.gentle_max_tokens,
                        seed=self.config.seed, id_slot=self.config.id_slot,
                    )
                except ValueError as exc:
                    self._err(f"[error] {exc}")
                else:
                    self.config.model = arg
                    self.agent = self._new_agent()
                    self._persist_config()
                    self._ok(f"[model -> {arg}]  {self._status()}")
        elif cmd == "/effort":
            if arg not in EFFORT_LEVELS:
                self.console.print(f"Usage: /effort {{{'|'.join(EFFORT_LEVELS)}}}")
            else:
                # "unset" is the sentinel for the server-default (send nothing)
                # state; map it to "" so /effort can return to that state after
                # any level has been set (finding #22). "off" is NOT the same —
                # it sends reasoning_effort=minimal + enable_thinking=False.
                effort = "" if arg == "unset" else arg
                try:
                    self.provider = build_provider(
                        self.config.provider, self.config.model,
                        self.config.base_url, effort, self.config.private,
                        self.config.cache_prompt, self.config.max_output_tokens,
                        embed_model=self.config.embed_model,
                        temperature=self.config.temperature,
                        gentle_mode=self.config.gentle_mode,
                        gentle_max_tokens=self.config.gentle_max_tokens,
                        seed=self.config.seed, id_slot=self.config.id_slot,
                    )
                except ValueError as exc:
                    self._err(f"[error] {exc}")
                    return True
                self.config.effort = effort
                self.agent = self._new_agent()
                self._persist_config()
                self._ok(f"[effort -> {arg}]  {self._status()}")
                self.console.print(
                    "note: best-effort — only models/servers that support "
                    "reasoning_effort honor this (qwen3.6 on LM Studio ignores it).",
                    style="dim",
                )
        elif cmd == "/maxout":
            a = arg.lower()
            if not arg or a == "status":
                cur = self.config.max_output_tokens
                self.console.print(f"max output tokens: {cur if cur else 'unbounded'}")
                return True
            # "off"/"none"/"0"/"-1" all clear the cap (unbounded). A positive int
            # sets it; anything else is rejected.
            if a in ("off", "none", "0", "-1"):
                new_cap: int | None = None
            else:
                try:
                    n = int(arg)
                except ValueError:
                    self.console.print("Usage: /maxout <N> | off | status")
                    return True
                if n <= 0:
                    new_cap = None
                else:
                    new_cap = n
            try:
                self.provider = build_provider(
                    self.config.provider, self.config.model,
                    self.config.base_url, self.config.effort, self.config.private,
                    self.config.cache_prompt, new_cap,
                    embed_model=self.config.embed_model,
                    temperature=self.config.temperature,
                    gentle_mode=self.config.gentle_mode,
                    gentle_max_tokens=self.config.gentle_max_tokens,
                    seed=self.config.seed, id_slot=self.config.id_slot,
                )
            except ValueError as exc:
                self.console.print(f"[error] {exc}")
                return True
            self.config.max_output_tokens = new_cap
            self.agent = self._new_agent()
            self._persist_config()
            cur = self.config.max_output_tokens
            self._ok(
                f"[maxout -> {cur if cur else 'unbounded'}]  {self._status()}"
            )
            # A low cap counts reasoning tokens on thinking models, so it can cut
            # off the visible answer before any text is produced.
            if new_cap is not None and new_cap < 2048:
                self.console.print(
                    "note: on reasoning/thinking models the cap counts reasoning "
                    "tokens, so a low cap may cut off the visible answer.",
                    style="dim",
                )
        elif cmd == "/temp":
            # View/set the sampling temperature (Feature 1). No arg -> show current.
            if not arg or arg.lower() == "status":
                self.console.print(f"temperature: {self.config.temperature}")
                return True
            try:
                t = float(arg)
            except ValueError:
                self.console.print("Usage: /temp <0.0-2.0>  (lower = more deterministic)")
                return True
            if not 0.0 <= t <= 2.0:
                self._err("[error] temperature must be between 0.0 and 2.0")
                return True
            try:
                self.provider = build_provider(
                    self.config.provider, self.config.model,
                    self.config.base_url, self.config.effort, self.config.private,
                    self.config.cache_prompt, self.config.max_output_tokens,
                    embed_model=self.config.embed_model, temperature=t,
                    gentle_mode=self.config.gentle_mode,
                    gentle_max_tokens=self.config.gentle_max_tokens,
                    seed=self.config.seed, id_slot=self.config.id_slot,
                )
            except ValueError as exc:
                self._err(f"[error] {exc}")
                return True
            self.config.temperature = t
            self.agent = self._new_agent()
            self._persist_config()
            self._ok(f"[temp -> {t}]  {self._status()}")
        elif cmd == "/seed":
            # View/set the deterministic seed (Feature: reproducibility).
            # No arg -> show current.
            if not arg or arg.lower() == "status":
                self.console.print(f"seed: {self.config.seed}")
                return True
            try:
                s = int(arg)
            except ValueError:
                self.console.print("Usage: /seed <non-negative int>  (None = no seed)")
                return True
            if s < 0:
                self.console.print("seed must be >= 0")
                return True
            self.config.seed = s
            self._persist_config()
            self._ok(f"[seed -> {s}]  {self._status()}")
        elif cmd == "/rerank":
            # View/set the gated LLM-judge reranker for weak-signal retrieval.
            # No arg -> show current. on/off toggle the flag; it is mirrored onto
            # the LIVE agent (self.agent.rerank) so the change applies this turn
            # without rebuilding the agent (and losing history). code_search reads
            # config.rerank live too, so both retrieval paths pick it up at once.
            if not arg or arg.lower() == "status":
                self.console.print(f"rerank: {'on' if self.config.rerank else 'off'}")
                return True
            val = arg.lower()
            if val in ("on", "true", "yes", "1"):
                self.config.rerank = True
            elif val in ("off", "false", "no", "0"):
                self.config.rerank = False
            else:
                self.console.print("Usage: /rerank [on|off]")
                return True
            # Reflect on the live agent so memory.retrieve honours it this turn.
            self.agent.rerank = self.config.rerank
            self.agent.rerank_candidates = self.config.rerank_candidates
            self._persist_config()
            self._ok(
                f"[rerank -> {'on' if self.config.rerank else 'off'}]  {self._status()}"
            )
        elif cmd == "/verify":
            # View/set the auto-verify command (Feature 3). No arg -> show current;
            # "off" clears it (back to the prose build-nudge).
            if not arg or arg.lower() == "status":
                cur = self.config.verify_cmd
                self.console.print(f"verify command: {cur if cur else 'off (disabled)'}")
                return True
            self.config.verify_cmd = "" if arg.lower() == "off" else arg
            self.agent = self._new_agent()
            self._persist_config()
            cur = self.config.verify_cmd
            self._ok(f"[verify -> {cur if cur else 'off'}]")
            if cur:
                self.console.print(
                    "note: this runs automatically after a turn that edits files "
                    "but never runs a command; its output is fed back to fix failures.",
                    style="dim",
                )
        elif cmd == "/image":
            self._cmd_image(arg)
        elif cmd == "/gentle":
            self._cmd_gentle(arg)
        elif cmd == "/cooldown":
            self._cmd_cooldown(arg)
        elif cmd == "/undo":
            self._cmd_undo(arg)
        elif cmd == "/diff":
            self._cmd_diff(arg)
        elif cmd == "/commit":
            self._cmd_commit(arg)
        elif cmd == "/init":
            self._cmd_init(arg)
        elif cmd == "/copy":
            self._cmd_copy(arg)
        elif cmd == "/mode":
            self._cmd_mode(arg)
        elif cmd == "/branch":
            self._cmd_branch(arg)
        elif cmd == "/fork":
            self._cmd_fork(arg)
        elif cmd == "/doctor":
            self._cmd_doctor(arg)
        elif cmd == "/commands":
            self._cmd_commands(arg)
        else:
            # CUSTOM MACRO fallback: a project may define <cwd>/.llmcli/commands/
            # <name>.md files; a matching /<name> expands the file (with $ARGUMENTS
            # substituted) and is submitted as if the user typed that text.
            macro = self._macros().get(cmd)
            if macro is not None:
                expanded = self._expand_macro(macro, arg)
                if expanded is not None:
                    self._submit(expanded)
                    return True
            self.console.print(f"Unknown command: {cmd}. Try /help.")
        return True

    # Honest one-line description of what gentle mode does (and does NOT do).
    _GENTLE_NOTE = (
        "lowers average GPU load/heat by shortening bursts + spacing turns; "
        "it does NOT cap GPU %."
    )

    def _cmd_gentle(self, arg: str) -> None:
        """Handle /gentle: status | on | off | tokens <n> | gap <seconds>.

        Mirrors /temp and /verify: changes persist via _persist_config. The
        on/off and tokens settings affect the per-request output-token cap, so
        they rebuild the provider (which holds the gentle cap); gap is pacing-only
        and needs no rebuild. All user-facing text is HONEST: gentle does not cap
        GPU utilization %, it only lowers average load/heat.
        """
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        val = parts[1].strip() if len(parts) > 1 else ""
        if not sub or sub == "status":
            state = "on" if self.config.gentle_mode else "off"
            self.console.print(
                f"gentle: {state}  ·  token cap {self.config.gentle_max_tokens}  ·  "
                f"cool-down {self.config.gentle_gap_seconds}s"
            )
            self.console.print(self._GENTLE_NOTE, style="dim")
            return
        if sub in ("on", "off"):
            self.config.gentle_mode = sub == "on"
            # Rebuild so the provider's effective cap reflects the new gentle state.
            if not self._rebuild_provider():
                return
            self.agent = self._new_agent()
            self._persist_config()
            self._ok(f"[gentle -> {sub}]")
            self.console.print(self._GENTLE_NOTE, style="dim")
            return
        if sub == "tokens":
            try:
                n = int(val)
            except ValueError:
                self.console.print("Usage: /gentle tokens <positive int>")
                return
            if n <= 0:
                self._err("[error] gentle tokens must be a positive integer")
                return
            self.config.gentle_max_tokens = n
            if not self._rebuild_provider():
                return
            self.agent = self._new_agent()
            self._persist_config()
            self._ok(f"[gentle tokens -> {n}]")
            return
        if sub == "gap":
            try:
                g = float(val)
            except ValueError:
                self.console.print("Usage: /gentle gap <seconds >= 0>")
                return
            if g < 0:
                self._err("[error] gentle gap must be >= 0")
                return
            self.config.gentle_gap_seconds = g
            self._persist_config()
            self._ok(f"[gentle gap -> {g}s]")
            return
        if sub == "sgap":
            try:
                g = float(val)
            except ValueError:
                self.console.print("Usage: /gentle sgap <seconds >= 0>")
                return
            if g < 0:
                self._err("[error] gentle spawn-gap must be >= 0")
                return
            self.config.gentle_spawn_gap_seconds = g
            self._persist_config()
            self._ok(f"[gentle spawn-gap -> {g}s]")
            return
        self.console.print("Usage: /gentle [on | off | tokens <n> | gap <seconds> | sgap <seconds>]")

    # Honest one-line description of what thermal cooldown does.
    _COOLDOWN_NOTE = "pauses generation to let the GPU cool; it does NOT cap GPU %."

    def _reconfigure_cooldown(self) -> None:
        """Push the current cooldown config into the process-global pacer."""
        cooldown.configure(
            enabled=self.config.cooldown_enabled,
            interval_seconds=self.config.cooldown_interval_seconds,
            duration_seconds=self.config.cooldown_duration_seconds,
        )

    def _cmd_cooldown(self, arg: str) -> None:
        """Handle /cooldown: status | on | off | interval <s> | duration <s>.

        Mirrors /gentle: on/off/interval/duration persist via _persist_config and
        reconfigure the process-global thermal pacer so the change takes effect
        this session. All text is HONEST: cooldown pauses generation to lower GPU
        heat; it does not cap GPU utilization %.
        """
        parts = arg.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        val = parts[1].strip() if len(parts) > 1 else ""
        if not sub or sub == "status":
            st = cooldown.status()
            state = "on" if self.config.cooldown_enabled else "off"
            self.console.print(
                f"cooldown: {state}  ·  every {st['interval_seconds']:g}s "
                f"break {st['duration_seconds']:g}s  ·  next break in "
                f"{st['seconds_until_next_break']:g}s"
            )
            self.console.print(self._COOLDOWN_NOTE, style="dim")
            return
        if sub in ("on", "off"):
            self.config.cooldown_enabled = sub == "on"
            self._reconfigure_cooldown()
            self._persist_config()
            # Rebuild so the LIVE agent's loop-top pause gate reflects the new
            # state: the Agent captures cooldown_enabled at build time (interval/
            # duration flow through module globals read live by maybe_pause, but
            # on/off does not), so a session started with --no-cooldown could
            # otherwise never be turned on mid-session. Mirrors /gentle.
            self.agent = self._new_agent()
            self._ok(f"[cooldown -> {sub}]")
            self.console.print(self._COOLDOWN_NOTE, style="dim")
            return
        if sub == "interval":
            try:
                n = float(val)
            except ValueError:
                self.console.print("Usage: /cooldown interval <seconds > 0>")
                return
            if n <= 0:
                self._err("[error] cooldown interval must be > 0")
                return
            self.config.cooldown_interval_seconds = n
            self._reconfigure_cooldown()
            self._persist_config()
            self._ok(f"[cooldown interval -> {n:g}s]")
            return
        if sub == "duration":
            try:
                n = float(val)
            except ValueError:
                self.console.print("Usage: /cooldown duration <seconds >= 0>")
                return
            if n < 0:
                self._err("[error] cooldown duration must be >= 0")
                return
            self.config.cooldown_duration_seconds = n
            self._reconfigure_cooldown()
            self._persist_config()
            self._ok(f"[cooldown duration -> {n:g}s]")
            return
        self.console.print(
            "Usage: /cooldown [on | off | interval <seconds> | duration <seconds>]"
        )

    def _rebuild_provider(self) -> bool:
        """Rebuild self.provider from the current config; False on ValueError.

        Centralizes the build_provider call used by /gentle (private mode can
        refuse a non-loopback base_url with a ValueError, which we surface).
        """
        try:
            self.provider = build_provider(
                self.config.provider, self.config.model, self.config.base_url,
                self.config.effort, self.config.private, self.config.cache_prompt,
                self.config.max_output_tokens, embed_model=self.config.embed_model,
                temperature=self.config.temperature,
                gentle_mode=self.config.gentle_mode,
                gentle_max_tokens=self.config.gentle_max_tokens,
                seed=self.config.seed, id_slot=self.config.id_slot,
            )
        except ValueError as exc:
            self._err(f"[error] {exc}")
            return False
        return True

    def _cmd_image(self, arg: str) -> None:
        """Handle /image: stage attachments for the next turn (or send at once).

        - no arg              -> list currently-staged attachments
        - "clear"             -> drop all staged attachments
        - "<path>"            -> validate + stage one image for the next message
        - "<path> <message>"  -> stage AND send <message> with it immediately
        Validation goes through images.encode_image; on error nothing is staged.
        """
        arg = arg.strip()
        if not arg:
            if not self._staged_images:
                self.console.print(
                    "no images staged. Use /image <path> [message].", style="dim"
                )
            else:
                self.console.print(f"staged images ({len(self._staged_images)}):")
                for label, _ in self._staged_images:
                    self.console.print(f"  • {label}", style="dim")
            return
        if arg.lower() == "clear":
            n = len(self._staged_images)
            self._staged_images.clear()
            self._ok(f"[cleared {n} staged image{'s' if n != 1 else ''}]")
            return
        # First token is the path; the rest (if any) is a message to send now.
        parts = arg.split(maxsplit=1)
        path = parts[0]
        message = parts[1].strip() if len(parts) > 1 else ""
        try:
            part = images.encode_image(path, max_bytes=self.config.max_image_bytes)
            size_kb = os.path.getsize(path) / 1024
        except (ValueError, OSError) as exc:
            # Validation failed -> stage NOTHING and report the actionable error.
            self._err(f"[error] {exc}")
            return
        name = os.path.basename(path)
        self._staged_images.append((f"{name} ({size_kb:.0f} KB)", part))
        if message:
            # Stage-and-send: run the turn now with all staged images, then clear.
            self._submit(message)
        else:
            self._ok(
                f"attached {name} ({size_kb:.0f} KB); "
                "it will be sent with your next message"
            )

    # ----- Foundation-wave command helpers -------------------------------
    def _cmd_undo(self, arg: str) -> None:
        """Restore the most recent file-snapshot checkpoint (git-free /undo)."""
        try:
            res = checkpoint.undo(os.getcwd(), session=self._ckpt_session)
        except Exception as exc:  # noqa: BLE001 - never crash the REPL
            self._err(f"[undo error] {type(exc).__name__}: {exc}")
            return
        msg = res.get("message", "")
        if res.get("undone"):
            self._ok(msg)
        else:
            self.console.print(msg, style="dim")
        for rel in res.get("restored", []):
            self.console.print(f"  restored {rel}", style="dim")
        for rel in res.get("deleted", []):
            self.console.print(f"  deleted {rel}", style="dim")
        for err in res.get("errors", []):
            self._err(f"  {err}")

    def _cmd_diff(self, arg: str) -> None:
        """Show the working-tree diff (optionally for one PATH) via git."""
        cwd = os.getcwd()
        if not gitint.is_repo(cwd):
            self.console.print("not a git repo", style="dim")
            return
        text = gitint.diff(cwd, arg or None)
        if text.strip():
            self.console.print(text)
        else:
            self.console.print("no changes", style="dim")

    def _cmd_commit(self, arg: str) -> None:
        """Commit all changes; generate a message from the diff when none given."""
        cwd = os.getcwd()
        if not gitint.is_repo(cwd):
            self._err("[error] not a git repository")
            return
        msg = arg.strip()
        if not msg:
            diff_text = gitint.diff(cwd)
            msg = self._generate_commit_message(diff_text) or "update"
        res = gitint.commit_all(cwd, msg)
        if res.get("ok"):
            h = res.get("commit_hash", "")
            short = h[:9] if h else "(unknown)"
            self._ok(f"[committed {short}] {msg}")
        else:
            self._err(f"[commit failed] {res.get('error', 'unknown error')}")

    def _generate_commit_message(self, diff_text: str) -> str:
        """Ask the provider for a one-line conventional-commit summary of a diff.

        Reuses the summarizer pattern (stream_chat + collect text events). The
        provider was built with the low config.temperature, so no extra tuning is
        needed here. Returns "" on any failure so the caller falls back to a
        default. Never raises.
        """
        if not diff_text.strip():
            return ""
        req = [
            {"role": "system", "content": (
                "You write a single-line Conventional Commit message (e.g. "
                "'fix: ...', 'feat: ...') summarizing a git diff. Reply with ONLY "
                "that one line — no body, no quotes, no code fences."
            )},
            {"role": "user", "content": diff_text[:8000]},
        ]
        acc: list[str] = []
        try:
            for event in self.provider.stream_chat(req, None):
                if event.get("type") == "text":
                    acc.append(event.get("text", ""))
                elif event.get("type") == "done":
                    break
        except Exception:  # noqa: BLE001 - fall back to a default message
            return ""
        line = " ".join("".join(acc).split()).strip()
        if line.startswith("[provider error") or line.startswith("[stream error"):
            return ""
        return line[:72]

    def _cmd_init(self, arg: str) -> None:
        """Create a starter AGENTS.md rules file, or report an existing one."""
        cwd = os.getcwd()
        existing = rules.find_rules_file(cwd)
        if existing is not None:
            self.console.print(
                f"a rules file already exists: {existing.name} — edit it directly.",
                style="dim",
            )
            return
        target = os.path.join(cwd, "AGENTS.md")
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(rules.default_template())
        except OSError as exc:
            self._err(f"[error] could not write AGENTS.md: {exc}")
            return
        self._ok(f"[created AGENTS.md] edit {target} to describe this project's rules.")

    def _cmd_copy(self, arg: str) -> None:
        """Copy the Nth-last assistant answer (default last) to the clipboard."""
        try:
            n = int(arg) if arg.strip() else 1
        except ValueError:
            self._err("Usage: /copy [N]  (N = how many answers back; default 1)")
            return
        if n < 1:
            n = 1
        answers = [
            m.get("content") for m in getattr(self.agent, "messages", [])
            if isinstance(m, dict) and m.get("role") == "assistant"
            and isinstance(m.get("content"), str) and m.get("content").strip()
        ]
        if not answers or n > len(answers):
            self.console.print("no assistant answer to copy.", style="dim")
            return
        text = answers[-n]
        if self._copy_to_clipboard(text):
            self._ok(f"[copied {len(text)} chars to clipboard]")
        else:
            self.console.print(
                "no clipboard tool found (pbcopy/xclip/wl-copy/clip.exe); "
                "here is the answer:", style="dim",
            )
            self.console.print(text)

    @staticmethod
    def _copy_to_clipboard(text: str) -> bool:
        """Copy ``text`` via an OS clipboard tool. True on success. Never raises."""
        candidates = (
            ["pbcopy"],                              # macOS
            ["wl-copy"],                             # Linux/Wayland
            ["xclip", "-selection", "clipboard"],    # Linux/X11
            ["xsel", "--clipboard", "--input"],      # Linux/X11 (alt)
            ["clip.exe"],                            # WSL / Windows
        )
        for argv in candidates:
            if shutil.which(argv[0]) is None:
                continue
            try:
                proc = subprocess.run(
                    argv, input=text, text=True, timeout=10,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if proc.returncode == 0:
                    return True
            except (OSError, subprocess.SubprocessError):
                continue
        return False

    def _cmd_mode(self, arg: str) -> None:
        """View or set the permission mode (config.permission_mode)."""
        name = arg.strip()
        if not name:
            self._ok(f"permission mode: {self.config.permission_mode}")
            self.console.print(
                f"  available: {', '.join(PERMISSION_MODES)}", style="dim"
            )
            return
        if name not in PERMISSION_MODES:
            self._err(
                f"[error] unknown mode '{name}'. Valid: {', '.join(PERMISSION_MODES)}"
            )
            return
        self.config.permission_mode = name
        self.agent = self._new_agent()
        self._persist_config()
        self._ok(f"[mode -> {name}]  {self._status()}")

    def _branch_path(self, tag: str):
        """Path to a tagged conversation snapshot for the cwd (~/.llm-cli/sessions).

        Keyed by ``<project-id>--branch-<tag>`` so tagged snapshots sit next to the
        regular session file and never collide across projects.
        """
        safe = session._sanitize(tag)
        sid = session.session_id(os.getcwd())
        return session.sessions_dir() / f"{sid}--branch-{safe}.json"

    def _cmd_branch(self, arg: str) -> None:
        """Save (no arg: list) named conversation snapshots for this project."""
        tag = arg.strip()
        if not tag:
            sid = session.session_id(os.getcwd())
            prefix = f"{sid}--branch-"
            tags: list[str] = []
            try:
                for p in session.sessions_dir().glob(f"{prefix}*.json"):
                    tags.append(p.name[len(prefix):-len(".json")])
            except OSError:
                pass
            if tags:
                self.console.print("saved branches:")
                for t in sorted(tags):
                    self.console.print(f"  • {t}", style="dim")
            else:
                self.console.print(
                    "no saved branches. Use /branch <tag> to snapshot this conversation.",
                    style="dim",
                )
            return
        payload = {
            "cwd": os.path.abspath(os.getcwd()),
            "tag": tag,
            "model": self.config.model,
            "messages": self.agent.messages,
        }
        try:
            checkpoint._atomic_write_bytes(
                self._branch_path(tag),
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )
        except OSError as exc:
            self._err(f"[error] could not save branch: {exc}")
            return
        self._ok(f"[branched '{tag}'] {len(self.agent.messages)} messages saved.")

    def _cmd_fork(self, arg: str) -> None:
        """Load a tagged snapshot into the current agent to continue from it."""
        tag = arg.strip()
        if not tag:
            self._err("Usage: /fork <tag>  (see /branch to list saved tags)")
            return
        try:
            data = json.loads(self._branch_path(tag).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._err(f"[error] no saved branch '{tag}'.")
            return
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            self._err(f"[error] branch '{tag}' has no usable history.")
            return
        messages = [m for m in messages if isinstance(m, dict)]
        _load_into_agent(self.agent, messages)
        self._ok(
            f"[forked '{tag}'] continuing from {len(self.agent.messages)} messages."
        )

    def _cmd_doctor(self, arg: str) -> None:
        """Run environment/health checks and print pass/✗ with fix hints."""
        cwd = os.getcwd()
        self._ok("llm-cli doctor:")
        success_style = palette_for(self.config.theme).success

        def _line(ok: bool, label: str, detail: str = "") -> None:
            mark = "✓" if ok else "✗"
            self.console.print(f"  {mark} {label}", style=success_style if ok else "red")
            if detail:
                self.console.print(f"      {detail}", style="dim")

        # Provider reachability + model presence (local server only).
        if self.config.provider == PROVIDER_LOCAL:
            try:
                models = list_local_models(
                    self.config.base_url, get_api_key(), self.config.private
                )
                if self.config.model in models:
                    _line(True, f"provider reachable ({self.config.base_url})",
                          f"model '{self.config.model}' available")
                else:
                    _line(False, f"provider reachable ({self.config.base_url})",
                          f"model '{self.config.model}' NOT in server list "
                          f"({len(models)} models) — /models to list, /model to switch")
            except Exception as exc:  # noqa: BLE001 - each check is best-effort
                _line(False, "provider reachable",
                      f"could not reach {self.config.base_url}: {type(exc).__name__} "
                      "— is LM Studio running?")
        else:
            _line(True, f"provider = {self.config.provider}", "no server check needed")

        # Config loads.
        try:
            load_config()
            _line(True, "config loads OK")
        except Exception as exc:  # noqa: BLE001
            _line(False, "config load", f"{type(exc).__name__}: {exc}")

        # Git availability.
        git_ok = gitint.git_available()
        _line(git_ok, "git available",
              "" if git_ok else "install git for /diff and /commit")

        # sandbox-exec (macOS private-mode network sandbox).
        has_sbx = shutil.which("sandbox-exec") is not None
        _line(has_sbx, "sandbox-exec available",
              "" if has_sbx else "private-mode run_bash sandbox needs sandbox-exec (macOS)")

        # MCP status.
        if self.config.mcp_enabled:
            try:
                running = bool(self.mcp.is_running())
            except Exception:  # noqa: BLE001
                running = False
            _line(True, f"mcp enabled ({'running' if running else 'not started'})")
        else:
            _line(True, "mcp disabled", "enable with /mcp on")

        # Rules file present.
        try:
            rf = rules.find_rules_file(cwd)
        except Exception:  # noqa: BLE001
            rf = None
        _line(rf is not None, "project rules file",
              f"found {rf.name}" if rf else "none — /init writes a starter AGENTS.md")

        # Checkpoints dir writable.
        try:
            d = checkpoint.checkpoints_dir()
            d.mkdir(parents=True, exist_ok=True)
            writable = os.access(str(d), os.W_OK)
        except Exception:  # noqa: BLE001
            writable = False
        _line(writable, "checkpoints dir writable",
              "" if writable else "cannot write ~/.llm-cli/checkpoints — /undo disabled")

    def _macros(self) -> dict:
        """Discover project macros in ``<cwd>/.llmcli/commands/*.md`` (cached).

        Returns ``{"/name": Path}``. A missing directory yields ``{}``. Cached per
        session on first use. Never raises.
        """
        cache = getattr(self, "_macro_cache", None)
        if cache is not None:
            return cache
        from pathlib import Path

        macros: dict = {}
        try:
            base = Path(os.getcwd()) / ".llmcli" / "commands"
            if base.is_dir():
                for p in base.glob("*.md"):
                    if p.is_file():
                        macros["/" + p.stem.lower()] = p
        except OSError:
            macros = {}
        self._macro_cache = macros
        return macros

    def _expand_macro(self, path, arg: str) -> str | None:
        """Read a macro file and substitute ``$ARGUMENTS`` with the command args.

        Returns the expanded prompt text, or None on a read error. Never raises.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._err(f"[error] could not read macro: {exc}")
            return None
        return text.replace("$ARGUMENTS", arg)

    def _cmd_commands(self, arg: str) -> None:
        """List available project macros (from ``<cwd>/.llmcli/commands/*.md``)."""
        macros = self._macros()
        if not macros:
            self.console.print(
                "no project macros. Add <name>.md files under .llmcli/commands/ "
                "to define /<name> shortcuts ($ARGUMENTS is substituted).",
                style="dim",
            )
            return
        self.console.print("available macros:")
        for name in sorted(macros):
            self.console.print(f"  {name}", style="dim")

    def _completion_files(self) -> list[str]:
        """Cached workspace-relative file list for the @-mention completer.

        Computed once (lazily) per session so a big repo is not re-walked on every
        keystroke. ``mentions.project_files`` never raises, but the whole thing is
        guarded anyway so a completion attempt can never break input.
        """
        cache = self._completion_file_cache
        if cache is not None:
            return cache
        try:
            files = mentions.project_files(os.getcwd())
        except Exception:  # noqa: BLE001 - completion helper must never raise
            files = []
        self._completion_file_cache = files
        return files

    def _refresh_completion_files(self) -> None:
        """Invalidate the cached @-mention file list (recomputed on next use)."""
        self._completion_file_cache = None

    def _make_input_completer(self):
        """Build the prompt_toolkit completer for the input line (see module fn)."""
        return _build_input_completer(self)

    def _mention_web_fetch(self, url: str) -> str:
        """Text-returning web_fetch for @url mentions (never raises).

        Wraps the REGISTRY ``web_fetch`` tool (which returns a dict). web_fetch may
        be gated (private mode) or absent; any failure yields a short notice string
        rather than raising, so mention expansion always degrades gracefully.
        """
        tool = tools.REGISTRY.get("web_fetch")
        if tool is None:
            return "[web_fetch unavailable]"
        try:
            res = tool.fn({"url": url})
        except Exception as exc:  # noqa: BLE001
            return f"[web_fetch failed: {exc}]"
        if isinstance(res, dict):
            if res.get("ok"):
                return str(res.get("result", ""))
            return f"[web_fetch: {res.get('error', 'unavailable')}]"
        return str(res)

    def _mention_read_file(self, relpath: str) -> str:
        """Text-returning read_file for @file mentions (never raises)."""
        tool = tools.REGISTRY.get("read_file")
        if tool is None:
            raise FileNotFoundError("read_file unavailable")
        res = tool.fn({"path": relpath})
        if isinstance(res, dict):
            if res.get("ok"):
                return str(res.get("result", ""))
            # Surface the tool's error to expand_mentions, which turns a raised
            # exception into a "could not read" notice block.
            raise OSError(str(res.get("error", "read failed")))
        return str(res)

    def _context_gauge_line(self) -> str:
        """One dim line summarizing live context usage, e.g.
        ``context: 37% (~4.4k / 12k tok)``. Best-effort; returns "" on any error.

        Uses the SAME estimator the agent uses for auto-compaction
        (``Agent._estimate_tokens`` over ``self.agent.messages``) and the SAME
        ceiling (``_effective_soft_limit``) so the gauge matches when compaction
        actually fires.
        """
        try:
            used = Agent._estimate_tokens(self.agent.messages)
            ceiling = _effective_soft_limit(self.provider, self.config)
            if ceiling <= 0:
                return ""
            pct = int(round(100 * used / ceiling))
            return (
                f"context: {pct}% (~{_fmt_tok(used)} / {_fmt_tok(ceiling)} tok)"
            )
        except Exception:  # noqa: BLE001 - a status gauge must never raise
            return ""

    def _status_bar(self):
        """prompt_toolkit ``bottom_toolbar`` callable: return the CACHED bar text.

        Invoked on EVERY redraw/keystroke, so it must be cheap — it only reads the
        pre-rendered cache (a FormattedText built by ``_refresh_status_bar``, which
        runs once at startup and once per completed turn). It never touches git or
        re-estimates tokens here (a git subprocess per keystroke would be awful).
        """
        return self._status_cache

    def _refresh_status_bar(self) -> None:
        """Recompute the pinned bottom status bar and store it in ``_status_cache``.

        The bar reads ``model · ctx N% · <branch>[*] · <rate> tok/s · <time>``,
        e.g. `` qwen3.6-35b-a3b · ctx 37% · main* · 223 tok/s · 0.42s``. Called
        once at startup and at the end of each turn (NOT per keystroke) so git +
        the token estimate are hit at most once per turn. Every segment is guarded
        so the bar can never raise; the rate/time segments are omitted until a
        turn has populated ``self.agent.last_turn_stats``.
        """
        from prompt_toolkit.formatted_text import FormattedText

        segments: list[str] = []
        # model — short form (strip any "org/" prefix, e.g. qwen/qwen3.6 -> qwen3.6)
        try:
            model = getattr(self.provider, "model", "") or self.config.model or ""
            short = model.rsplit("/", 1)[-1] if model else ""
            if short:
                segments.append(short)
        except Exception:  # noqa: BLE001 - never let the bar break input
            pass
        # ctx N% — reuse the SAME estimator + ceiling as the (removed) gauge line.
        try:
            used = Agent._estimate_tokens(self.agent.messages)
            ceiling = _effective_soft_limit(self.provider, self.config)
            if ceiling > 0:
                segments.append(f"ctx {int(round(100 * used / ceiling))}%")
        except Exception:  # noqa: BLE001
            pass
        # git — branch (+ "*" when dirty); omit the whole segment when not a repo.
        try:
            cwd = os.getcwd()
            if gitint.is_repo(cwd):
                branch = gitint.current_branch(cwd) or ""
                if branch:
                    try:
                        if gitint.is_dirty(cwd):
                            branch += "*"
                    except Exception:  # noqa: BLE001
                        pass
                    segments.append(branch)
        except Exception:  # noqa: BLE001
            pass
        # tok/s + time — only once a turn has populated last_turn_stats.
        try:
            stats = getattr(self.agent, "last_turn_stats", None)
            if stats:
                rate = stats.get("toks_per_sec")
                if rate is not None:
                    segments.append(f"{rate:.0f} tok/s")
                elapsed = stats.get("elapsed")
                if elapsed is not None:
                    segments.append(f"{elapsed:.2f}s")
        except Exception:  # noqa: BLE001
            pass

        text = " · ".join(segments)
        if not text:
            self._status_cache = ""
            return
        # Style: dim base with a subtle accent on the NUMBERS. re.split's capture
        # group puts digit-runs at odd indices; they read in the theme accent
        # (pal.ptk — the ptk-safe accent spelling), the rest stays dim grey.
        pal = palette_for(self.config.theme)
        frags = []
        for i, seg in enumerate(re.split(r"(\d[\d.]*)", " " + text)):
            if not seg:
                continue
            frags.append(((pal.ptk if i % 2 == 1 else "fg:ansibrightblack"), seg))
        self._status_cache = FormattedText(frags)

    def _run_shell_passthrough(self, cmd: str) -> None:
        """Run a user-typed ``!<cmd>`` shell command and print its output.

        User-initiated (the user explicitly typed a shell command) so no sandbox
        is applied — this is like running the command in their own shell. Output
        (stdout+stderr merged) is byte-capped like ``tools._run_bash`` so a huge
        result cannot flood the UI, and a short timeout prevents a wedged command
        from hanging the REPL. Never raises.
        """
        cmd = cmd.strip()
        if not cmd:
            return
        shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        try:
            proc = subprocess.run(
                [shell, "-c", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._err("[shell] command timed out")
            return
        except OSError as exc:
            self._err(f"[shell] could not run: {exc}")
            return
        capped = tools._truncate_tail(proc.stdout or "", tools._MAX_OUTPUT)
        if capped.strip():
            self.console.print(capped.rstrip("\n"))
        if proc.returncode != 0:
            self.console.print(f"[exit {proc.returncode}]", style="dim")

    def _submit_or_stage(self, line: str) -> None:
        """Drag-and-drop aware entry point for a non-slash input line.

        Detects dropped image PATHS in ``line`` (bare/quoted/escaped/``file://``,
        single or multiple, with or without typed text). Behaviour:
        - no image paths     -> fast path: submit the ORIGINAL line unchanged so a
                                normal text turn is byte-for-byte identical to before;
        - paths + more text  -> stage each image, then submit the remaining text now;
        - paths + no text    -> stage the image(s) and print a hint, DO NOT run an
                                empty turn (the staged images carry to the next msg).
        Per-file encode errors are printed and skipped; other files still attach.
        """
        # `!<cmd>` SHELL PASSTHROUGH: the user explicitly typed a shell command, so
        # run it locally and print the output — never send it to the model. A bare
        # "!" (no command) falls through to the normal path.
        stripped = line.strip()
        if stripped.startswith("!") and len(stripped) > 1:
            self._run_shell_passthrough(stripped[1:])
            return
        paths, remaining = images.extract_image_paths(line)
        if not paths:
            # Fast path: no dropped images -> original behaviour, original text.
            self._submit(line)
            return
        attached: list[str] = []
        for path in paths:
            try:
                part = images.encode_image(path, max_bytes=self.config.max_image_bytes)
                size_kb = os.path.getsize(path) / 1024
            except (ValueError, OSError) as exc:
                # One bad file does not abort the rest; report and continue.
                self._err(f"[error] {exc}")
                continue
            name = os.path.basename(path)
            self._staged_images.append((f"{name} ({size_kb:.0f} KB)", part))
            attached.append(name)
        remaining = remaining.strip()
        if remaining:
            # Image(s) + typed text -> send now with everything staged.
            self._submit(remaining)
        elif attached:
            # Pure drop, no text -> stage and wait for the user's question rather
            # than firing an empty turn. Staged images persist to the next message.
            self._ok(
                f"📎 attached {', '.join(attached)} — now type your question "
                "about the image and press enter."
            )

    def _submit(self, line: str) -> None:
        """Run one agent turn, attaching any staged images, then clear staging.

        Shared by the normal input loop and /image's stage-and-send path so both
        consume the staged attachments and auto-save identically.
        """
        # LAZY MCP integration: if the background start finished since the live
        # agent was built, rebuild the agent so MCP tools are now offered. The
        # first turn proceeds WITHOUT MCP tools if servers are still coming up;
        # once they're ready, this rebuild flips them in for the next turn.
        if (
            self.config.mcp_enabled
            and self._mcp_ready
            and not self._mcp_integrated
        ):
            self._mcp_integrated = True
            self.agent = self._new_agent()
        imgs = [part for _, part in self._staged_images]
        # Read the monotonic clock ONCE for BOTH the thermal-cooldown idle-reset
        # and the gentle inter-turn cool-down, so this turn makes a single pre-run
        # clock read (the post-run _last_gen_end update below is the only other).
        now = time.monotonic()
        # THERMAL COOLDOWN idle-reset: the pacer targets CONTINUOUS work, so if the
        # user sat idle at the prompt longer than a break's duration, reset its
        # baseline — idle time must not accumulate toward a break. Guard the first
        # turn (no prior generation yet -> _last_gen_end is 0.0).
        if (
            self.config.cooldown_enabled
            and self._last_gen_end > 0.0
            and now - self._last_gen_end > self.config.cooldown_duration_seconds
        ):
            cooldown.reset()
        # GENTLE-mode inter-turn cool-down: pace ONLY real model turns (slash
        # commands route through _dispatch_slash and never reach here). TTY-GATED
        # so the test suite / piped runs NEVER actually sleep (mirrors the spinner
        # and screen-clear gating on console.is_terminal). If the user spent
        # longer than the gap typing, gentle_wait returns 0 and nothing sleeps.
        if self.config.gentle_mode and getattr(self.console, "is_terminal", False):
            wait = gentle_wait(
                self.config.gentle_gap_seconds, now, self._last_gen_end
            )
            if wait > 0:
                self.console.print("(gentle: brief cool-down…)", style="dim")
                time.sleep(wait)
        # @-MENTION EXPANSION: if the line references files/dirs/urls/@diff, resolve
        # them into a context preamble the model sees FIRST. The user-visible line
        # (already echoed by prompt_toolkit as they typed) is left unchanged; only
        # the text PASSED to the model is augmented. No "@" -> byte-for-byte
        # identical to before (expand_mentions early-returns on no "@").
        send_text = line
        if "@" in line:
            try:
                _text, blocks = mentions.expand_mentions(
                    line,
                    os.getcwd(),
                    web_fetch=self._mention_web_fetch,
                    git_diff=lambda p=None: gitint.diff(os.getcwd(), p),
                    read_file=self._mention_read_file,
                )
                if blocks:
                    send_text = mentions.render_blocks(blocks) + "\n\n" + line
            except Exception:  # noqa: BLE001 - mention expansion must never abort a turn
                send_text = line
        # COOPERATIVE INTERRUPT (TWO-STAGE): clear the cancel flag, then (only in
        # the main thread on a real TTY) install a SIGINT handler.
        #   1st Ctrl-C in a turn → just SET the event so the agent stops
        #      cooperatively and finalizes cleanly (it checks cancel_event
        #      mid-stream + between rounds).
        #   2nd Ctrl-C in the SAME turn → restore the previous SIGINT handler and
        #      re-raise KeyboardInterrupt so a wedged read (e.g. the pre-first-token
        #      wait, where PEP 475 otherwise retries the socket read and swallows
        #      the first signal) force-breaks; the `except KeyboardInterrupt`
        #      below catches it → "[interrupted]".
        # The counter resets each turn; the previous handler is restored in the
        # finally. The `except KeyboardInterrupt` also stays as a fallback (e.g.
        # handler not installed).
        self._cancel_event.clear()
        prev_sigint = None
        sigint_installed = False
        sigint_count = [0]
        if (
            threading.current_thread() is threading.main_thread()
            and _stdout_is_tty()
        ):
            def _on_sigint(signum, frame):  # noqa: ARG001
                sigint_count[0] += 1
                if sigint_count[0] >= 2:
                    # Force-break a wedged/blocking read: restore the original
                    # handler and re-raise so the read actually unwinds.
                    try:
                        signal.signal(signal.SIGINT, prev_sigint)
                    except (ValueError, OSError):
                        pass
                    raise KeyboardInterrupt
                self._cancel_event.set()

            try:
                prev_sigint = signal.signal(signal.SIGINT, _on_sigint)
                sigint_installed = True
            except (ValueError, OSError):  # not main thread / unsupported
                sigint_installed = False
        try:
            self.agent.run(send_text, images=imgs or None)
            # A completed turn may have created/deleted files; invalidate the
            # cached @-mention file list so new files autocomplete without a
            # restart. Cheap/best-effort (recomputed lazily on next use).
            self._refresh_completion_files()
        except KeyboardInterrupt:
            self.console.print("\n[interrupted]")
        except Exception as exc:  # noqa: BLE001
            self.console.print(f"[error] {type(exc).__name__}: {exc}")
        finally:
            # Restore the previous SIGINT handler so Ctrl-C at the input PROMPT
            # keeps its normal behaviour (raise KeyboardInterrupt -> the run loop's
            # "Ctrl-D or /exit to quit" hint).
            if sigint_installed:
                try:
                    signal.signal(signal.SIGINT, prev_sigint)
                except (ValueError, OSError):
                    pass
            # Staged images are consumed by this turn whether it succeeded or not
            # (they were "attached to the next message").
            self._staged_images.clear()
            # Record when this generation ENDED so the next turn's cool-down is
            # measured from here (even an errored turn consumed GPU time).
            self._last_gen_end = time.monotonic()
        # Refresh the pinned bottom status bar now that this turn populated
        # last_turn_stats + changed context usage (ctx% now lives in the bar, not
        # a standalone printed gauge line). Recomputed here — once per turn — so
        # the bottom_toolbar callable only ever reads the cache, never runs git or
        # re-estimates tokens per keystroke. Fully guarded; never raises.
        self._refresh_status_bar()
        # Auto-save after EVERY completed turn (even an errored one) so a later
        # crash never loses the conversation. Best-effort/no-raise.
        self._save_session()

    def run(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.application import run_in_terminal
        from prompt_toolkit.key_binding import KeyBindings

        # Ctrl+O: reveal the most-recent turn's full detail. LIMITATION: this is
        # handled only while the input prompt is active (line-based REPL). It
        # cannot interrupt a mid-stream model turn; press it at the next "> "
        # prompt to reveal the just-finished turn.
        kb = KeyBindings()

        @kb.add("c-o")
        def _(event):
            # render_details prints to the rich console while prompt_toolkit owns
            # the screen; route it through run_in_terminal so the app suspends its
            # render, lets the detail print cleanly, then restores the prompt (no
            # bleed onto the pinned bottom_toolbar / placeholder row).
            run_in_terminal(lambda: self.agent.render_details(self.console))
            event.app.invalidate()  # redraw the prompt; user is at the line input

        # NOTE: the prompt_toolkit session is named ``ptk_session`` (not
        # ``session``) so it does NOT shadow the module-level ``session`` import
        # used for per-project persistence below.
        from prompt_toolkit.styles import Style as PTKStyle

        pal = palette_for(self.config.theme)
        # "bottom-toolbar": override prompt_toolkit's default reverse-video bar to
        # a flat, non-inverted line so the pinned status bar reads as calm dim text
        # (its fragments carry their own fg; see _refresh_status_bar).
        ptk_style = PTKStyle.from_dict(
            {"prompt": pal.ptk, "bottom-toolbar": "noreverse"}
        )
        # Persistent line history (also powers prompt_toolkit's built-in Ctrl-R
        # reverse search). Best-effort: a failure to create the file/dir must not
        # break input, so we fall back to an in-memory (None) history.
        from prompt_toolkit.history import FileHistory

        history = None
        try:
            hist_path = os.path.expanduser("~/.llm-cli/history")
            os.makedirs(os.path.dirname(hist_path), exist_ok=True)
            history = FileHistory(hist_path)
        except OSError:
            history = None
        # completer: fuzzy / and @ completion (see _build_input_completer);
        # complete_while_typing shows it live; enable_open_in_editor wires the
        # built-in Ctrl-X Ctrl-E to compose a long prompt in $EDITOR.
        # Ghost placeholder (dim) surfacing the input affordances, and the pinned
        # bottom status bar (bottom_toolbar) — both render ONLY in a real
        # interactive prompt, so piped/one-shot runs are untouched. The bar reads a
        # cached string via _status_bar (recomputed per turn, never per keystroke).
        from prompt_toolkit.formatted_text import FormattedText

        placeholder = FormattedText(
            [("fg:ansibrightblack", "Ask anything · / commands · @ files")]
        )
        ptk_session = PromptSession(
            key_bindings=kb,
            style=ptk_style,
            history=history,
            completer=self._make_input_completer(),
            complete_while_typing=True,
            enable_open_in_editor=True,
            bottom_toolbar=self._status_bar,
            placeholder=placeholder,
        )
        # Connect MCP servers (if any are configured AND mcp is enabled) on a
        # BACKGROUND daemon thread so the first prompt is not blocked ~5s by
        # kyp-mem's startup. The first agent is built WITHOUT MCP tools; once the
        # background start finishes, the next turn rebuilds the agent with its
        # tools offered (see _submit). When mcp is disabled, nothing starts.
        # A server that fails to start is logged + skipped (in the thread), never
        # crashing the REPL.
        self._start_mcp_background(self.console)
        # Wire the session into agents now that it exists: rebuild the agent so
        # its confirm_fn is the prompt_toolkit-safe one (no builtin input()) and
        # it carries any MCP tools that just connected.
        self.session = ptk_session
        self.agent = self._new_agent()
        # Seed the pinned status bar once now that the agent exists (resting form:
        # model · ctx 0% · branch — no tok/s until the first turn completes). It is
        # then refreshed at the end of every turn in _submit, so git/ctx are never
        # recomputed per keystroke.
        self._refresh_status_bar()
        # Ensure the tools layer agrees with this session's mode (run_bash sandbox
        # + web_fetch guard read this process-wide flag).
        set_private(self.config.private)

        # MANDATORY thermal cooldown: configure the process-global pacer from this
        # session's config so a long turn breaks mid-way to let the GPU cool.
        cooldown.configure(
            enabled=self.config.cooldown_enabled,
            interval_seconds=self.config.cooldown_interval_seconds,
            duration_seconds=self.config.cooldown_duration_seconds,
        )

        cwd = os.getcwd()
        # --continue/-c: auto-load this project's saved session BEFORE the first
        # turn so the conversation resumes where it left off. Best-effort: a
        # missing/corrupt session just leaves a fresh agent.
        if self.resume:
            data = session.load_session(cwd)
            if data is not None:
                _load_into_agent(self.agent, data["messages"])
                rel = session.relative_time(data.get("updated_at", ""))
                self.console.print(
                    f"continuing previous session "
                    f"({len(self.agent.messages)} msgs, {rel}).",
                    style="dim",
                )

        # Startup hint: when a saved session exists (and we did NOT just resume
        # it), mention it on a dim line so a fresh launch stays light but the user
        # knows /resume (or -c) can pick it back up.
        if not self.resume:
            meta = session.session_meta(cwd)
            if meta is not None:
                rel = session.relative_time(meta["updated_at"])
                self.console.print(
                    f"↩ last session ({rel}, {meta['message_count']} msgs): "
                    f"{meta['title']} — /resume to continue (or relaunch with -c)",
                    style="dim",
                )

        self._print_banner()
        # GIT DIRTY WARNING: if the workspace has uncommitted changes, warn once so
        # the user knows agent edits will mix with their own (mention /diff /commit).
        # Best-effort, never raises.
        try:
            if (
                gitint.git_available()
                and gitint.is_repo(cwd)
                and gitint.is_dirty(cwd)
            ):
                self.console.print(
                    "note: this repo has uncommitted changes — agent edits will mix "
                    "with yours (use /diff to review, /commit to snapshot).",
                    style="dim",
                )
        except Exception:  # noqa: BLE001
            pass
        # Styled input glyph (accent "❯"). HTML drives the prompt_toolkit style
        # class defined above; falls back cleanly to plain text on a dumb term.
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.patch_stdout import patch_stdout

        prompt_glyph = HTML(f"<prompt>{pal.prompt}</prompt> ")
        try:
            while True:
                try:
                    # patch_stdout coordinates ANY stdout write that lands while
                    # the prompt is displayed (the background MCP-connect daemon
                    # thread's "mcp: … connected" line, and the first rich tool-tree
                    # line after a turn) with prompt_toolkit's erase/redraw — WITHOUT
                    # this, those writes glue onto the rendered prompt+ghost
                    # placeholder row and leak placeholder fragments (e.g. "…es",
                    # "…@ files"). raw=True is REQUIRED here: those lines are rich
                    # output carrying ANSI colour escapes, and the default
                    # (raw=False) proxy routes them through Vt100_Output.write(),
                    # which replaces ESC (\x1b) with "?" — stripping colour and
                    # printing "?[2m…" garbage. write_raw() (raw=True) passes the
                    # escapes through verbatim so colour renders correctly above the
                    # prompt. Scoped to the read only, so a turn's rich output during
                    # generation (outside this block) is untouched.
                    with patch_stdout(raw=True):
                        line = ptk_session.prompt(prompt_glyph)
                except KeyboardInterrupt:
                    self.console.print("\n(Ctrl-D or /exit to quit)", style="dim")
                    continue
                except EOFError:
                    self._save_session()  # persist the conversation on clean exit
                    self.console.print("\nBye.")
                    return
                if not line.strip():
                    continue
                stripped = line.lstrip()
                if stripped.startswith("//"):
                    # Escape hatch for collisions: a line starting with // is CHAT
                    # about a command (possibly one of ours), not a command to run.
                    # Drop exactly ONE leading slash and fall through to the model,
                    # e.g. "//model is broken in project A" sends the model
                    # "/model is broken in project A" instead of running /model.
                    line = stripped[1:]
                elif stripped.startswith("/") and (
                    stripped.split(maxsplit=1)[0].lower() in _KNOWN_COMMANDS
                    or stripped.split(maxsplit=1)[0].lower() in self._macros()
                ):
                    # One of llmc's OWN commands OR a loaded project macro
                    # (/<name> from .llmcli/commands/) -> handle it locally.
                    if not self._dispatch_slash(line):
                        self._save_session()  # /exit | /quit: persist before leaving
                        self.console.print("Bye.")
                        return
                    continue
                # else: either a normal (non-slash) message, OR a leading-slash line
                # whose first token is NOT an llmc command (e.g. "/build the app"
                # for another project's CLI). Either way the WHOLE line goes to the
                # model below — llmc only intercepts its own commands, never prints
                # "Unknown command" here.
                # Feature 2: a whole-project free-text question gets a one-line
                # hint pointing at /audit (map-reduce, keeps context small). We
                # only HINT — never auto-run audit (it spawns many sub-agents) and
                # never change the request; the agent runs normally below.
                if looks_like_whole_repo_request(line):
                    self.console.print(
                        "tip: this looks like a whole-project question — /audit runs a "
                        "fast map-reduce review that keeps context small (faster on "
                        "local models).",
                        style="dim",
                    )
                # _submit_or_stage first detects any dragged-in image paths
                # (staging them, possibly without sending); otherwise it falls
                # through to _submit, which attaches images staged via /image,
                # runs the turn, clears staging, and auto-saves.
                self._submit_or_stage(line)
        finally:
            # Always tear down server subprocesses cleanly on exit. Join the
            # background start thread first so it isn't mid-mutation during
            # shutdown_all().
            t = self._mcp_thread
            if t is not None and t.is_alive():
                t.join(timeout=10)
            self.mcp.shutdown_all()
