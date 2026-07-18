"""Interactive REPL + one-shot runner.

- prompt_toolkit for line input (history, editing).
- rich Console for output; streamed assistant text renders live.
- Slash commands: /model /models /provider /effort /maxout /compact /clear
  /resume /forget /help /exit (see _HELP_SECTIONS — the canonical command list).
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
from dataclasses import dataclass, replace

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
    THEME_BLOSSOM,
    THEME_CLEAN,
    THEME_EMBER,
    THEME_FROST,
    THEME_MIDNIGHT,
    THEME_NEON,
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

# Authoritative set of llmcode's OWN slash commands. The main input loop only
# intercepts a leading-slash line when its first token is in here; anything else
# (e.g. "/build the app" referencing ANOTHER project's CLI) is sent to the model.
# _dispatch_slash handles exactly these — keep the two in sync.
_KNOWN_COMMANDS = frozenset({
    "/exit", "/quit", "/help", "/clear", "/resume", "/forget", "/memory",
    "/theme",
    "/compact", "/mcp", "/audit", "/speed", "/context", "/provider",
    "/models", "/model", "/effort", "/maxout", "/temp", "/verify", "/image",
    "/gentle", "/cooldown",
    "/rerank", "/codeembed",
    # Foundation-wave commands (git/rules/checkpoint/session/clipboard/diagnostics).
    "/undo", "/diff", "/commit", "/init", "/copy", "/mode",
    "/branch", "/fork", "/doctor", "/commands",
})

# Grouped command reference for /help — the single canonical command list. Each
# section is (title, [(command, one-line description), …]). ``Repl._print_help``
# renders it with an accent /command column + muted descriptions on the themed
# console (byte-clean when piped). The longer NETWORK/SSRF prose lives in
# _HELP_NETWORK (shown by `/help network`). Keep in sync with _KNOWN_COMMANDS /
# _dispatch_slash.
_HELP_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Core", [
        ("/help [network]", "Show this help (network = the network & sandbox notes)"),
        ("/clear", "Clear the conversation history"),
        ("/resume", "Reload this project's saved session (local-only memory)"),
        ("/forget", "Delete this project's saved session"),
        ("/exit, /quit", "Leave"),
        ("Ctrl+O", "Reveal the orchestrator's last-turn detail (args + results)"),
    ]),
    ("Model", [
        ("/provider <name>", "Switch provider: local | mock"),
        ("/models", "List models available on the server"),
        ("/model <name>", "Set the model (verified against the server's list)"),
        ("/effort <level>", "Reasoning effort: off | low | medium | high (best-effort)"),
        ("/maxout <N|off>", "Per-request generation cap; off | 0 | -1 = unbounded"),
        ("/temp [value]", "Sampling temperature 0.0-2.0 (default 0.2; lower = steadier)"),
    ]),
    ("Context", [
        ("/compact", "Summarize ALL history into one note, keep the last exchange"),
        ("/context [N|auto|fixed|off]", "Working-context budget (auto-trims each turn)"),
        ("/audit [path]", "Map-reduce repo review in small isolated chunks"),
        ("/speed", "Tips to raise tok/s (LM Studio settings + context size)"),
        ("/codeembed [on|off]", "Toggle semantic code_search (off = BM25-only, fast)"),
        ("/mcp [on|off]", "List, enable, or disable MCP servers + their tools"),
    ]),
    ("Git & files", [
        ("/undo", "Revert the last file write/edit this session made"),
        ("/diff [path]", "Show the working-tree git diff (optionally one PATH)"),
        ("/commit [msg]", "Commit all changes (auto-writes a message if none given)"),
        ("/init", "Write a starter AGENTS.md project-rules file"),
        ("/copy [N]", "Copy the last (or Nth-from-last) answer to the clipboard"),
        ("/mode [name]", "Show or set the permission mode (default|acceptEdits|plan|…)"),
        ("/branch [tag]", "Save (no arg: list) a named conversation snapshot"),
        ("/fork <tag>", "Load a saved branch snapshot and continue from it"),
        ("/doctor", "Run environment/health checks (provider, git, sandbox, …)"),
        ("/commands", "List project macros in .llmcode/commands/*.md"),
    ]),
    ("Tuning", [
        ("/verify [cmd|off]", "Auto-run <cmd> after a turn edits files (fed back to fix)"),
        ("/gentle [on|off|…]", "Gentle mode (default ON): lower average GPU load/heat"),
        ("/cooldown [on|off|…]", "Thermal cooldown (default ON): periodic cooling breaks"),
        ("/image [path [msg]]", "Attach an image to your NEXT message (vision models)"),
    ]),
    ("Themes", [
        ("/theme <name>",
         "clean/midnight · amber/ember · orange · auto/frost · neon · blossom · ansi"),
    ]),
]

# Footer under the sections in the default /help. Keeps the shell-safety note
# (!<cmd> is NOT sandboxed) + the unknown-slash routing rule visible by default;
# the longer NETWORK/SSRF prose moves behind `/help network`.
_HELP_FOOTER = """\
!<cmd> runs <cmd> in YOUR shell directly — it is NOT sandboxed, even in
--private mode (it's your own shell, not a gated agent tool). Review before use.
Lines starting with an unknown /command are sent to the model; start a line with
// to send a literal leading slash (e.g. "//model …"). Anything else is sent to
the agent.
Run /help network for the network & sandbox security details."""

# Detailed network + sandbox security notes, surfaced on demand via `/help
# network` so the default command list stays scannable.
_HELP_NETWORK = """\
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

Tool activity collapses to one line after each turn:
  ⏺ N tools · Ctrl+O to expand
Ctrl+O reveals the full ⏺/⎿ tree — one two-line entry per call:
  ⏺ Read(README.md)
    ⎿  Read 120 lines
The "⏺" head is accent-coloured, the result summary line is muted. A failed call
shows a "✗ <short reason>" (declined -> "✗ declined"; unknown -> "✗ unknown tool").
Ctrl+O is line-based: it cannot interrupt a running turn — press it at the next
prompt to reveal the turn that just finished. NOTE: it reveals only the
ORCHESTRATOR's own tool calls; a delegated sub-agent's activity shows as a
prefixed "↳ ⏺ N tools" line during the run and is not in this buffer."""


SPEED_TIPS = """\
Speed (tok/s) — what controls it and how to raise it

1) BIGGEST factor: context size. The model re-reads the whole conversation for
   every token it writes, so tok/s drops as the chat (and tool output) grows.
   - llmcode AUTO-trims the live context to an adaptive budget after each turn
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

_MAX_AUTO_COMPACT_CEILING = 48_000  # cap: past ~48k the per-token KV bandwidth cost + eventual re-prefill outweigh keeping more context (decode is bandwidth-bound). config.context_soft_limit (floor) still wins if a user sets it higher.


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


def _diff_preview_lines(
    tool_name: str, args: dict, palette=None
) -> list[tuple[str, str]] | None:
    """Build a concise (line, rich-style) preview for a write_file/edit_file call.

    Returns a list of ``(text, style)`` rows ready to print before the y/N prompt,
    or None when there is nothing safe to show (unknown tool, bad args, binary/
    missing/oversized file, or an edit whose target isn't uniquely locatable).
    Capped to ~60 lines. Never raises.

    Styles are the theme's SEMANTIC tokens when a ``palette`` is passed — ``+``
    lines in ``success``, ``-`` lines in ``error``, context/``@@`` in ``muted`` —
    else the historic "green"/"red"/"dim" literals so the ANSI theme (and any
    palette-less caller) is byte-for-byte unchanged.
    """
    add_style = palette.success if palette else "green"
    del_style = palette.error if palette else "red"
    ctx_style = palette.dim if palette else "dim"
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
                    return [(f"＋ new file: {path} ({n} lines)", add_style)]
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
                style = add_style
            elif line.startswith("-"):
                style = del_style
            else:
                style = ctx_style
            rows.append((line, style))
        if not rows:
            return None
        if len(rows) > _DIFF_PREVIEW_MAX_LINES:
            rows = rows[:_DIFF_PREVIEW_MAX_LINES]
            rows.append(("…(truncated)", ctx_style))
        return rows
    except Exception:  # noqa: BLE001 - preview is best-effort; never raise
        return None


def _enc_can(obj, s: str) -> bool:
    """True when ``obj``'s output encoding can represent ``s`` (glyph guard).

    Mirrors ``Agent._console_can_encode``: guards the ⚠/⏺ glyphs against mojibake
    on a legacy-encoded terminal or ASCII pipe (LANG=C). ``obj`` is anything with
    an ``encoding`` attribute (a rich Console or ``sys.stdout``); a UTF-8 stream —
    which every test's capsys/StringIO is — always returns True, leaving the
    piped-test output byte-for-byte unchanged while still ASCII-degrading where a
    glyph truly can't render."""
    enc = getattr(obj, "encoding", None) or "utf-8"
    try:
        s.encode(enc)
        return True
    except (LookupError, UnicodeEncodeError, AttributeError):
        return False


def _box_for(name: str):
    """Map a ThemeSpec ``box_style`` string to the matching ``rich.box.*`` const.

    Accepts "ROUNDED"/"HEAVY"/"DOUBLE"/"MINIMAL"/"SQUARE"/"ASCII" (case-insensitive);
    an unknown/empty value falls back to ROUNDED (the historic banner frame)."""
    from rich import box

    return getattr(box, (name or "ROUNDED").upper(), box.ROUNDED)


# --------------------------------------------------------------------------- #
# Startup wordmark
# --------------------------------------------------------------------------- #
# STATIC, pre-rendered "llmc-code" block wordmark (figlet 'ANSI Shadow' style),
# embedded verbatim so we ship ZERO figlet dependency (deps stay tiny). Six rows;
# widest line is 74 columns (measured). It is shown ONLY on a wide, UTF-8-capable
# real terminal — every other context (narrow tty, piped/non-tty, LANG=C console)
# falls back to the compact framed banner, so narrow/scripted/legacy runs never
# see broken art or stray ANSI. Regenerate via scratchpad/gen_wordmark.py if the
# product name ever changes; keep _WORDMARK_WIDTH in sync with the widest line.
_WORDMARK = (
    "██╗     ██╗     ███╗   ███╗ ██████╗       ██████╗ ██████╗ ██████╗ ███████╗\n"
    "██║     ██║     ████╗ ████║██╔════╝      ██╔════╝██╔═══██╗██╔══██╗██╔════╝\n"
    "██║     ██║     ██╔████╔██║██║     █████╗██║     ██║   ██║██║  ██║█████╗  \n"
    "██║     ██║     ██║╚██╔╝██║██║     ╚════╝██║     ██║   ██║██║  ██║██╔══╝  \n"
    "███████╗███████╗██║ ╚═╝ ██║╚██████╗      ╚██████╗╚██████╔╝██████╔╝███████╗\n"
    "╚══════╝╚══════╝╚═╝     ╚═╝ ╚═════╝       ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝"
)
_WORDMARK_WIDTH = 74            # widest wordmark line (measured) — gate tty width on this
_WORDMARK_GLYPHS = "█╗║═╝╚╔"    # encoding-guard probe: every block/box char the art uses


def _hex_to_rgb(token) -> tuple[int, int, int] | None:
    """Parse a ``#rrggbb`` palette token to an ``(r, g, b)`` tuple.

    Returns None for anything that is NOT a plain 6-digit hex (an ANSI colour
    NAME like ``"yellow"``, an empty string, or ``None``) so callers can fall
    back to a flat style instead of crashing on the ``ansi`` theme. A trailing
    style attr (e.g. ``"#7aa2f7 bold"``) is tolerated by taking the first word.
    """
    if not isinstance(token, str):
        return None
    t = token.strip().split(" ", 1)[0] if token.strip() else ""
    if not t.startswith("#") or len(t) != 7:
        return None
    try:
        return (int(t[1:3], 16), int(t[3:5], 16), int(t[5:7], 16))
    except ValueError:
        return None


# Rotating, action-oriented input ghost placeholders keyed to turn count (fixes
# the generic "Ask anything · …" ghost). Cycled by ``_placeholder_for_turn`` so
# each fresh prompt nudges a different concrete first move.
_PLACEHOLDER_ROTATION = (
    "what should we build next?",
    "ask, or / for commands",
    "@file to attach context",
    "describe a bug to hunt",
)


def _placeholder_for_turn(n: int) -> str:
    """Ghost placeholder text for turn ``n`` (cycles ``_PLACEHOLDER_ROTATION``)."""
    return _PLACEHOLDER_ROTATION[n % len(_PLACEHOLDER_ROTATION)]


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
        # Resolve the active theme once: it colours BOTH the diff preview and the
        # y/N prompt line (warning ⚠ glyph + accent action label).
        theme = getattr(config, "theme", THEME_CLEAN) if config is not None else THEME_CLEAN
        pal = palette_for(theme)
        # Use the SAME collapsed summary + byte-hint logic as the loop and the
        # input()-based fallback, centralized in agent.confirm_label (finding #29)
        # — no duplicated label-building here. Full args stay for Ctrl+O reveal.
        label, hint = confirm_label(tool, args)
        # Optional diff preview for write/edit (finding: diff_preview). Rendered on
        # the THEMED console so + / - / context lines carry the theme's semantic
        # success/error/muted tokens (byte-clean when piped — a non-tty console
        # emits no ANSI). Best-effort: any failure just skips the diff and prompts.
        if (
            config is not None
            and getattr(config, "diff_preview", False)
            and getattr(tool, "name", "") in ("write_file", "edit_file")
        ):
            rows = _diff_preview_lines(getattr(tool, "name", ""), args, pal)
            if rows:
                try:
                    con = _make_console(theme)
                    # Themed header ABOVE the diff — "⏺ edit_file <path>" in the
                    # accent colour — so the diff visually belongs to the pending
                    # action. The glyph ASCII-degrades on a non-UTF-8 console and
                    # the line stays byte-clean when piped (non-tty console → no
                    # ANSI). ``label`` is already "<tool> <path>".
                    glyph = "⏺ " if _enc_can(con, "⏺") else "* "
                    con.print(glyph + label, style=pal.accent)
                    for text, style in rows:
                        con.print(text, style=style)
                except Exception:  # noqa: BLE001 - never block the prompt on render
                    pass
        # This confirm reuses the interactive PromptSession, which carries a ghost
        # `placeholder` ("Ask anything · …") for the MAIN input. Left as-is it
        # renders glued onto the y/N line. A per-call `placeholder=None` is a NO-OP
        # in prompt_toolkit (it only overrides when the value is not None), so pass
        # an EMPTY placeholder ("") to actually suppress the ghost. prompt() mutates
        # `session.placeholder` in place, so SAVE + RESTORE it — otherwise the main
        # input would silently lose its ghost after the first confirm. patch_stdout
        # (raw=True) coordinates any background stdout write landing during the y/N
        # so it can't leave residue on that row (same fix as the main-input read).
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.patch_stdout import patch_stdout

        # Highest-stakes moment: lead with a warning-coloured ⚠ glyph, colour the
        # action label (the "<tool> <path>" the y/N applies to) in the theme accent,
        # and keep the "[y/N]" in the default fg. Inline prompt_toolkit style
        # fragments (bare hex/colour = fg) so this renders through ptk on a real
        # terminal; the ⚠ glyph ASCII-degrades on a non-UTF-8 console. The prompt is
        # interactive-only (ptk owns the terminal) so it never reaches piped output.
        warn_glyph = "⚠ " if _enc_can(sys.stdout, "⚠") else "! "
        message = FormattedText([
            ("", "\n"),
            (pal.warning, warn_glyph),
            ("", "Run "),
            (pal.ptk, label),
            ("", f"?{hint} [y/N] "),
        ])
        saved_placeholder = getattr(session, "placeholder", None)
        try:
            # Leading newline so the y/N prompt isn't glued to the preceding dim
            # tok/s footer when the model narrates AND calls a gated tool.
            with patch_stdout(raw=True):
                answer = session.prompt(message, placeholder="")
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
    The ~80% ceiling is itself capped at ``_MAX_AUTO_COMPACT_CEILING`` (48k):
    past that, per-token KV bandwidth + eventual re-prefill cost outweigh keeping
    more context, though a larger ``context_soft_limit`` floor still wins.
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
        return max(floor, min(int(ctx * 0.8), _MAX_AUTO_COMPACT_CEILING))
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
    "/codeembed": "toggle semantic code search",
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

    Returns a prompt_toolkit ``Completer`` that (a) completes llmcode slash commands
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


# Per-theme hex is inlined directly into each ThemeSpec in _SPECS below (one
# palette per curated theme, matching the design-spec tables byte-for-byte), so
# there are no shared per-theme colour constants to drift out of sync.


@dataclass(frozen=True)
class ThemeSpec:
    """Single source of truth for ONE theme's look. Both rendering engines —
    the markdown rich ``Theme`` (``to_rich_theme``) and the chrome ``Palette``
    (``to_palette``) — derive from this, so a theme is described in one place.

    Every field is a rich style string unless suffixed ``_ptk`` (prompt_toolkit
    vocabulary). Truecolor hex is primary; rich/ptk auto-downsample except the
    ``ansi`` theme, which uses only the 16 ANSI names. Fields not yet consumed
    by a rendering path carry safe defaults so partial specs never crash.
    """

    # ---- identity -----------------------------------------------------
    name: str

    # ---- accents / neutrals consumed by the chrome Palette ------------
    accent: str                  # answer-box border, prompt, footer values
    accent_bright: str           # brightest accent tier (h1, strong, Palette.bright)
    muted: str                   # secondary/metadata grey (Palette.dim)
    success: str                 # ✓ marks, ready dot, +diff, doctor pass
    accent_ptk: str              # prompt glyph + status-bar numbers (Palette.ptk)

    # ---- prose (markdown.*) ------------------------------------------
    faint: str                   # markdown hr / block_quote / link_url
    inline_code: str             # markdown.code / .code_block (TEXT, no bg box)
    bold_word: str               # markdown.strong
    bullet: str                  # markdown.item.bullet
    heading: str                 # markdown.h2..h6 base
    link: str                    # markdown.link
    accent_secondary: str        # markdown.item.number

    # ---- pygments code fence -----------------------------------------
    code_theme: str

    # ---- optional / not-yet-wired (safe defaults; inert until later) --
    fg: str = ""                 # primary body foreground (Console default today)
    error: str = "red"           # ✗ marks, -diff, _err, doctor fail
    warning: str = "yellow"      # confirm ⚠, caution status
    em: str | None = None        # markdown.em override (None = rich default italic)
    inline_code_bg: str | None = None   # optional bg box for inline code
    inline_code_bold: bool = True       # whether markdown.code is bold
    box_style: str = "ROUNDED"   # answer box + banner frame shape
    border: str = ""             # answer box + banner border color
    banner_glyph: str = "◆"
    ready_glyph: str = "●"
    prompt_glyph: str = "❯"
    spinner_glyph_set: str = "braille"
    spinner: str = ""            # braille spinner glyph color (hex)
    spinner_timer: str = ""      # "· 3s ·" timer segment color (hex)
    gutter: str = "▌"            # box-border / status accent glyph
    muted_ptk: str = ""          # status-bar base text + placeholder
    status_num_ptk: str = ""     # status-bar digit runs
    completion_ptk: str = ""     # completion-menu selected item


class Palette:
    """Per-theme accent colours for the banner, answer gutter, prompt, footer,
    and colour-coded status lines. The LAYOUT is identical across themes — only
    the accent differs — so switching themes restyles, never relayouts.

    Fields are rich style strings. ``ptk`` is the prompt_toolkit spelling of the
    accent for the input glyph (prompt_toolkit and rich take different colour
    vocabularies for the basic-ANSI theme).
    """

    def __init__(self, accent, bright, dim, success, ptk, gutter="▌", prompt="❯",
                 *, error="red", warning="yellow", muted_ptk="", status_num_ptk="",
                 border="", box_style="ROUNDED", spinner="", spinner_timer="",
                 banner_glyph="◆", ready_glyph="●", completion_ptk=""):
        self.accent = accent
        self.bright = bright
        self.dim = dim
        self.success = success
        self.ptk = ptk
        self.gutter = gutter
        self.prompt = prompt
        # Extended tokens (defaulted, keyword-only): carried for later steps so
        # every existing positional Palette(...) construction still works as-is.
        self.error = error
        self.warning = warning
        self.muted_ptk = muted_ptk
        self.status_num_ptk = status_num_ptk
        self.border = border
        self.box_style = box_style
        self.spinner = spinner
        self.spinner_timer = spinner_timer
        self.banner_glyph = banner_glyph
        self.ready_glyph = ready_glyph
        self.completion_ptk = completion_ptk


def to_rich_theme(spec: ThemeSpec) -> "Theme":
    """Build the markdown.* rich ``Theme`` from a ``ThemeSpec``.

    Only markdown.* keys are overridden (as today); rich merges the rest with its
    defaults. Inline code is TEXT with no bg box unless ``inline_code_bg`` is set,
    and ``markdown.em`` is only overridden when ``spec.em`` is set (matching the
    orange theme, which historically kept rich's default italic em).
    """
    from rich.style import Style
    from rich.theme import Theme

    if spec.inline_code_bold:
        code_style = Style(color=spec.inline_code, bgcolor=spec.inline_code_bg,
                           bold=True)
    else:
        code_style = Style(color=spec.inline_code, bgcolor=spec.inline_code_bg)
    d = {
        "markdown.code": code_style,
        "markdown.code_block": Style(color=spec.inline_code),
        "markdown.h1": Style(color=spec.accent_bright, bold=True),
        # h2+ step down to the heading tier so the hierarchy stays distinct.
        "markdown.h2": Style(color=spec.heading, bold=True),
        "markdown.h3": Style(color=spec.heading, bold=True),
        "markdown.h4": Style(color=spec.heading, bold=True),
        "markdown.h5": Style(color=spec.heading),
        "markdown.h6": Style(color=spec.heading),
        "markdown.strong": Style(color=spec.bold_word, bold=True),
        "markdown.item.bullet": Style(color=spec.bullet, bold=True),
        "markdown.item.number": Style(color=spec.accent_secondary, bold=True),
        "markdown.link": Style(color=spec.link, underline=True),
        # rich applies markdown.link_url to the VISIBLE anchor text when hyperlinks
        # are on (the default), so keep it in the themed link color + underlined
        # rather than faint — otherwise link text reads as dim body copy.
        "markdown.link_url": Style(color=spec.link, underline=True),
        "markdown.block_quote": Style(color=spec.faint),
        "markdown.hr": Style(color=spec.faint),
    }
    if spec.em is not None:
        d["markdown.em"] = Style(color=spec.em, italic=True)
    return Theme(d)


def to_palette(spec: ThemeSpec) -> Palette:
    """Chrome ``Palette`` (unchanged public surface) derived from a ``ThemeSpec``.

    The seven fields existing callers read (accent/bright/dim/success/ptk/gutter/
    prompt) map to the spec's accent tier; the extended tokens ride along as
    keyword-only attributes for later steps.
    """
    return Palette(
        spec.accent, spec.accent_bright, spec.muted, spec.success, spec.accent_ptk,
        gutter=spec.gutter, prompt=spec.prompt_glyph,
        error=spec.error, warning=spec.warning,
        muted_ptk=spec.muted_ptk, status_num_ptk=spec.status_num_ptk,
        border=spec.border, box_style=spec.box_style,
        spinner=spec.spinner, spinner_timer=spec.spinner_timer,
        banner_glyph=spec.banner_glyph, ready_glyph=spec.ready_glyph,
        completion_ptk=spec.completion_ptk,
    )


# One ThemeSpec per theme key, each a curated, structurally-distinct dark
# palette (different accent hue, box shape, spinner colour, and code fence). Hex
# is inlined verbatim from the design-spec tables. Legacy keys keep resolving but
# now carry a curated look:
#   clean  -> Midnight (Tokyo Night)   amber/orange -> Ember (Gruvbox)
#   auto   -> Frost (Nord)             ansi -> unchanged 16-colour ANSI
# and neon (Dracula) + blossom (Catppuccin) are new. The descriptive aliases
# midnight/frost/ember resolve to clean/auto/amber via _THEME_ALIASES below.
_SPECS: dict[str, ThemeSpec] = {
    # ----- clean == Midnight (Tokyo Night), the DEFAULT: calm cool-blue --------
    THEME_CLEAN: ThemeSpec(
        name=THEME_CLEAN,
        accent="#7aa2f7", accent_bright="#bb9af7", muted="#565f89",
        success="#9ece6a", accent_ptk="#7aa2f7 bold",
        faint="#3d59a1", inline_code="#7dcfff", bold_word="#c0caf5",
        bullet="#7aa2f7", heading="#c0caf5", link="#7dcfff",
        accent_secondary="#bb9af7", code_theme="native",
        fg="#c0caf5", error="#f7768e", warning="#e0af68", em="#a9b1d6",
        box_style="ROUNDED", border="#7aa2f7", banner_glyph="◆",
        spinner="#7aa2f7", spinner_timer="#565f89",
        muted_ptk="#565f89", status_num_ptk="#7aa2f7 bold",
        completion_ptk="#7aa2f7",
    ),
    # ----- amber == Ember (Gruvbox): warm amber/gold ---------------------------
    THEME_AMBER: ThemeSpec(
        name=THEME_AMBER,
        accent="#fe8019", accent_bright="#fabd2f", muted="#928374",
        success="#b8bb26", accent_ptk="#fe8019 bold",
        faint="#665c54", inline_code="#fe8019", bold_word="#fabd2f",
        bullet="#fe8019", heading="#fabd2f", link="#8ec07c",
        accent_secondary="#8ec07c", code_theme="native",
        fg="#ebdbb2", error="#fb4934", warning="#fabd2f", em="#d3869b",
        box_style="HEAVY", border="#fe8019", banner_glyph="◆",
        spinner="#fe8019", spinner_timer="#928374",
        muted_ptk="#928374", status_num_ptk="#fabd2f bold",
        completion_ptk="#fe8019",
    ),
    # ----- orange == Ember (Gruvbox) too (kept for back-compat) ----------------
    THEME_ORANGE: ThemeSpec(
        name=THEME_ORANGE,
        accent="#fe8019", accent_bright="#fabd2f", muted="#928374",
        success="#b8bb26", accent_ptk="#fe8019 bold",
        faint="#665c54", inline_code="#fe8019", bold_word="#fabd2f",
        bullet="#fe8019", heading="#fabd2f", link="#8ec07c",
        accent_secondary="#8ec07c", code_theme="native",
        fg="#ebdbb2", error="#fb4934", warning="#fabd2f", em="#d3869b",
        box_style="HEAVY", border="#fe8019", banner_glyph="◆",
        spinner="#fe8019", spinner_timer="#928374",
        muted_ptk="#928374", status_num_ptk="#fabd2f bold",
        completion_ptk="#fe8019",
    ),
    # ----- auto == Frost (Nord): monochrome-elegant cool cyan ------------------
    # auto still leaves color_system unset in _make_console (truecolor auto-detect).
    THEME_AUTO: ThemeSpec(
        name=THEME_AUTO,
        accent="#88c0d0", accent_bright="#eceff4", muted="#4c566a",
        success="#a3be8c", accent_ptk="#88c0d0 bold",
        faint="#434c5e", inline_code="#8fbcbb", bold_word="#eceff4",
        bullet="#88c0d0", heading="#e5e9f0", link="#81a1c1",
        accent_secondary="#81a1c1", code_theme="nord",
        fg="#d8dee9", error="#bf616a", warning="#ebcb8b", em="#b48ead",
        # HORIZONTALS (not MINIMAL): clean top+bottom horizontal rules with no
        # side borders — a genuinely minimal but VISIBLE frame, distinct from the
        # other themes' ROUNDED/HEAVY/DOUBLE. MINIMAL renders as blank spaces, and
        # on a rich Panel so does SIMPLE (its top/bottom box rows are blank), so
        # both left Frost's answer box borderless; HORIZONTALS is the box that
        # actually draws visible rules on a title-less Panel.
        box_style="HORIZONTALS", border="#88c0d0", banner_glyph="◆",
        spinner="#88c0d0", spinner_timer="#4c566a",
        muted_ptk="#4c566a", status_num_ptk="#88c0d0 bold",
        completion_ptk="#88c0d0",
    ),
    # ----- neon == Dracula: high-contrast punchy -------------------------------
    THEME_NEON: ThemeSpec(
        name=THEME_NEON,
        accent="#bd93f9", accent_bright="#ff79c6", muted="#6272a4",
        success="#50fa7b", accent_ptk="#bd93f9 bold",
        faint="#44475a", inline_code="#8be9fd", bold_word="#ff79c6",
        bullet="#bd93f9", heading="#f8f8f2", link="#8be9fd",
        accent_secondary="#8be9fd", code_theme="dracula",
        fg="#f8f8f2", error="#ff5555", warning="#f1fa8c", em="#ffb86c",
        box_style="DOUBLE", border="#bd93f9", banner_glyph="◆",
        spinner="#bd93f9", spinner_timer="#6272a4",
        muted_ptk="#6272a4", status_num_ptk="#ff79c6 bold",
        completion_ptk="#bd93f9",
    ),
    # ----- blossom == Catppuccin Mocha: soft pastel ----------------------------
    THEME_BLOSSOM: ThemeSpec(
        name=THEME_BLOSSOM,
        accent="#cba6f7", accent_bright="#f5c2e7", muted="#6c7086",
        success="#a6e3a1", accent_ptk="#cba6f7 bold",
        faint="#45475a", inline_code="#94e2d5", bold_word="#f5c2e7",
        bullet="#cba6f7", heading="#cdd6f4", link="#89b4fa",
        accent_secondary="#89b4fa", code_theme="native",
        fg="#cdd6f4", error="#f38ba8", warning="#f9e2af", em="#b4befe",
        box_style="ROUNDED", border="#cba6f7", banner_glyph="❋",
        spinner="#cba6f7", spinner_timer="#6c7086",
        muted_ptk="#6c7086", status_num_ptk="#89b4fa bold",
        completion_ptk="#cba6f7",
    ),
    # ----- ansi: the 16 basic ANSI colours only (rich names + ptk "ansi*") -----
    # UNCHANGED: no truecolor hex; a thin "│" gutter. to_rich_theme still runs so
    # it finally gets a markdown Theme (strips rich's default grey code box), but
    # every colour is an ANSI name that downsamples to the user's own palette.
    THEME_ANSI: ThemeSpec(
        name=THEME_ANSI,
        accent="yellow", accent_bright="bright_yellow", muted="bright_black",
        success="green", accent_ptk="ansiyellow bold",
        faint="bright_black", inline_code="yellow", bold_word="bright_yellow",
        bullet="yellow", heading="yellow", link="bright_yellow",
        accent_secondary="yellow", code_theme="ansi_dark",
        fg="", gutter="│", error="red", warning="ansiyellow",
    ),
}


# Descriptive aliases (zero-break): each points at an existing _SPECS key so old
# saved configs AND the new names all resolve to a spec. midnight≡clean,
# frost≡auto, ember≡amber. Resolved by _resolve_theme wherever a theme key is
# consumed (palette_for / _make_console / _code_theme_for). neon/blossom are real
# _SPECS keys (not aliases).
_THEME_ALIASES: dict[str, str] = {
    THEME_MIDNIGHT: THEME_CLEAN,
    THEME_FROST: THEME_AUTO,
    THEME_EMBER: THEME_AMBER,
}


def _resolve_theme(theme: str) -> str:
    """Canonical _SPECS key for a theme name, following descriptive aliases.

    Every legacy key (clean/amber/orange/auto/ansi) and every new key
    (neon/blossom + the midnight/frost/ember aliases) resolves to a real spec;
    anything unknown falls back to the clean/default spec.
    """
    if theme in _SPECS:
        return theme
    alias = _THEME_ALIASES.get(theme)
    if alias in _SPECS:
        return alias
    return THEME_CLEAN


_PALETTES = {k: to_palette(v) for k, v in _SPECS.items()}


# Backward-compatible thin shims for the prior per-theme rich Theme builders,
# now derived from the shared spec (their look is the curated palette above).
def _clean_theme():
    return to_rich_theme(_SPECS[THEME_CLEAN])


def _amber_theme():
    return to_rich_theme(_SPECS[THEME_AMBER])


def _orange_theme():
    return to_rich_theme(_SPECS[THEME_ORANGE])


def palette_for(theme: str) -> Palette:
    """Accent palette for ``theme`` (descriptive aliases resolved; an unknown
    name falls back to the clean theme — matching config.DEFAULT_THEME)."""
    return _PALETTES[_resolve_theme(theme)]


def _code_theme_for(theme: str) -> str:
    """Pygments code-block theme for the active app theme, from ``_SPECS``.

    "ansi" (Dark mode) uses the ANSI-only "ansi_dark" highlighter so fenced code
    stays within the 16 basic ANSI colors; clean/amber/orange/blossom use
    "native", auto uses "nord", and neon uses "dracula". Descriptive aliases are
    resolved first. Any named style an installed pygments lacks (and an unknown
    theme) falls back through native -> monokai so it never raises.
    """
    spec = _SPECS.get(_resolve_theme(theme))
    if spec is None:
        return "monokai"
    name = spec.code_theme
    # ansi_dark is always available and must be returned verbatim (16-color).
    if name == "ansi_dark":
        return name
    from pygments.styles import get_style_by_name

    for candidate in (name, "native", "monokai"):
        try:
            get_style_by_name(candidate)
            return candidate
        except Exception:  # noqa: BLE001 - any lookup failure tries the next
            continue
    return "monokai"


def _make_console(theme: str = THEME_AUTO):
    # NOTE: this param default is only a RENDERING fallback for a no-arg call; it
    # is NOT the app-level default theme (that is config.DEFAULT_THEME = "clean").
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
    # EVERY theme now gets a markdown rich Theme built from its spec (auto and
    # ansi included), so inline code renders as themed TEXT with no default grey
    # code box. Descriptive aliases (midnight/frost/ember) resolve to their canon-
    # ical spec; an unknown name falls back to the clean/default spec.
    resolved = _resolve_theme(theme)
    rich_theme = to_rich_theme(_SPECS[resolved])
    # ansi (Dark mode) on a real terminal ALSO pins color_system="standard" so
    # rich downsamples every style to the 16 basic ANSI colors. Piped/non-tty runs
    # fall through to auto-detect (color_system unset) and stay ANSI-free — the one
    # special case, unchanged.
    if resolved == THEME_ANSI and _stdout_is_tty():
        return Console(theme=rich_theme, color_system="standard",
                       markup=False, highlight=False)
    return Console(theme=rich_theme, markup=False, highlight=False)


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
        # Accent (+ the full palette) thread to sub-agents so their footer/glyph
        # AND their spinner match the theme, mirroring the orchestrator.
        accent=palette.accent,
        palette=palette,
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
    # user-authored conventions (AGENTS.md/LLMCODE.md/…) become binding context.
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
        # the activity glyph for the top-level orchestrator. The FULL palette is
        # threaded too so the tool tree / footer / activity line reach the semantic
        # tokens (success/error/muted) and the spinner reaches its theme colours.
        accent=palette.accent,
        palette=palette,
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

    Connects MCP servers (if ~/.llmcode/mcp.json exists), runs the prompt with
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
        # Rotating input-ghost placeholder index (see _placeholder_for_turn):
        # advanced once per prompt shown so each turn nudges a different action.
        self._placeholder_i = 0
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
        """Error line (failed switch, bad input that aborts an action), coloured
        with the theme's semantic ``error`` token (ANSI 'red' for the ansi theme,
        a themed hex otherwise)."""
        self.console.print(msg, style=palette_for(self.config.theme).error)

    def _print_help(self, arg: str = "") -> None:
        """Render /help as grouped sections on the themed console.

        Each section prints an accent title, then a row per command with the
        ``/command`` column in the theme accent and its description in the muted
        token. Presentation only — no command BEHAVIOUR changes. ``/help network``
        shows the longer network + sandbox security notes instead. Piped/non-tty
        runs stay plain and ANSI-free (the console emits no colour off a terminal),
        so the command strings are always present for scripts/tests.
        """
        from rich.text import Text

        if arg.strip().lower() == "network":
            self.console.print(_HELP_NETWORK)
            return
        pal = palette_for(self.config.theme)
        for title, rows in _HELP_SECTIONS:
            self.console.print(Text(title, style=f"bold {pal.accent}"))
            # Align the /command column within THIS section (short-command sections
            # aren't padded out to the widest global command).
            width = max((len(cmd) for cmd, _ in rows), default=0)
            for cmd, desc in rows:
                line = Text("  ")
                line.append(cmd.ljust(width), style=pal.accent)
                line.append("  ")
                line.append(desc, style=pal.dim)
                self.console.print(line)
            self.console.print()
        self.console.print(_HELP_FOOTER, style=pal.dim)

    def _print_banner(self) -> None:
        """Startup banner. On a wide, UTF-8-capable terminal it opens with a bold
        block-letter ``llmc-code`` wordmark for a strong first impression; on a
        narrow, piped/non-tty, or LANG=C console it falls back to the compact
        framed banner (provider/model + a green ``ready`` dot + theme/privacy
        line). Both paths then print the shared privacy/help/first-run footer.

        Gating the wordmark on ALL of {real terminal, enough width, encodable
        glyphs} guarantees narrow terminals, piped/non-tty runs (byte-clean,
        ANSI-free), and legacy consoles never see broken art."""
        pal = palette_for(self.config.theme)
        show_wordmark = (
            getattr(self.console, "is_terminal", False)
            and self.console.size.width >= _WORDMARK_WIDTH
            and _enc_can(self.console, _WORDMARK_GLYPHS)
        )
        if show_wordmark:
            # Returning-run compression (fixes audit #2): once the first-run hero
            # has been seen (a ~/.llmcode/seen marker), subsequent launches collapse
            # to a two-line ribbon + prompt — fast entry, no wall of text. First run
            # shows the full hero (wordmark + tagline + ribbon + tail) then writes
            # the marker. Only the wide-tty wordmark path branches on the marker;
            # piped/narrow/LANG=C stays the unchanged compact/clean path below.
            if self._seen_before():
                self._print_returning_ribbon(pal)
            else:
                self._print_wordmark_header(pal)
                self._print_banner_tail(pal)
                self._mark_seen()
        else:
            self._print_compact_header(pal)   # the historic framed banner (fallback)
            self._print_banner_tail(pal)

    def _seen_marker_path(self) -> str:
        """Path to the first-run marker in the app config dir (``~/.llmcode/seen``)."""
        return os.path.expanduser("~/.llmcode/seen")

    def _seen_before(self) -> bool:
        """True when a prior run already showed the first-run hero. Best-effort:
        an unreadable dir reads as first-run (never crashes startup)."""
        try:
            return os.path.exists(self._seen_marker_path())
        except OSError:
            return False

    def _mark_seen(self) -> None:
        """Record that the first-run hero has been shown (best-effort). An
        unwritable config dir is swallowed — the user just keeps the first-run
        look, we never error on startup."""
        try:
            path = self._seen_marker_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("1")
        except OSError:
            pass

    def _print_returning_ribbon(self, pal) -> None:
        """Compressed returning-run startup: a single ``◆ llmc-code  <model>
        <badge>   /help · @ files`` ribbon (the prompt is the second line). Skips
        the big wordmark + verbose teaser/footer. Same tty/encoding gates as the
        hero; ◆/⬡ degrade to */# on a legacy console."""
        from rich.text import Text

        core = pal.banner_glyph if _enc_can(self.console, pal.banner_glyph) else "*"
        lock = "⬡" if _enc_can(self.console, "⬡") else "#"
        short_model = (self.config.model or "").rsplit("/", 1)[-1]
        line = Text()
        line.append(core + " llmc-code", style=pal.accent)
        line.append("   ")
        if short_model:
            line.append(short_model, style=pal.bright)
            line.append("   ")
        # Honest lock badge: only claim offline when private mode is on.
        line.append(lock + (" offline" if self.config.private else " local"),
                    style=pal.success)
        line.append("      ")
        line.append("/help · @ files", style=pal.dim)
        self.console.print()
        self.console.print(line)

    def _core_caret_message(self, pal):
        """FormattedText for the ``◆ ❯`` core-caret input prompt.

        The ◆ core carries the permission indicator (zero extra chrome): plan/
        read-only → ``pal.warning``, full-auto → ``pal.success``, else the accent
        (``pal.ptk``). The ❯ caret uses the completion accent. ◆/❯ degrade to */>
        on a console whose encoding can't represent them."""
        from prompt_toolkit.formatted_text import FormattedText

        core_glyph = "◆ " if _enc_can(self.console, "◆") else "* "
        caret = pal.prompt if _enc_can(self.console, pal.prompt) else ">"
        mode = getattr(self.config, "permission_mode", "default")
        if mode in ("plan", "read-only"):
            core_style = pal.warning
        elif mode == "full-auto":
            core_style = pal.success
        else:
            core_style = pal.ptk
        caret_style = pal.completion_ptk or pal.ptk
        return FormattedText([(core_style, core_glyph), (caret_style, caret + " ")])

    def _print_wordmark_header(self, pal) -> None:
        """The "Local Reactor" startup hero: a diagonal-gradient ``llmc-code``
        wordmark + a value-prop tagline + the reactor ribbon.

        Only reached for a wide UTF-8 terminal (see ``_print_banner`` gating).
        Each wordmark cell is coloured by lerping RGB ``pal.accent`` →
        ``accent_secondary`` across ``(col+row)`` for a static diagonal gradient
        (built ONCE, no per-frame loop). The tagline sells the local pitch and the
        ribbon shows ``<model> · ◆ core ready · <honest lock badge>``. Every new
        glyph degrades to ASCII via ``_enc_can`` and this whole path is tty-gated,
        so piped/narrow/LANG=C runs never reach it."""
        from rich.align import Align
        from rich.style import Style
        from rich.text import Text

        # Diagonal gradient endpoints. Palette may lack ``accent_secondary`` (it
        # rides on the ThemeSpec, not the chrome Palette), so fall back to bright.
        c0 = _hex_to_rgb(pal.accent)
        c1 = _hex_to_rgb(getattr(pal, "accent_secondary", pal.bright))
        lines = _WORDMARK.split("\n")
        maxrow = max(len(lines) - 1, 1)
        maxcol = max((len(ln) for ln in lines), default=1) - 1
        denom = (maxcol + maxrow) or 1
        art = Text(no_wrap=True)
        for row, line in enumerate(lines):
            if c0 is not None and c1 is not None:
                # Per-cell RGB lerp — one Text, computed once at startup.
                for col, ch in enumerate(line):
                    t = (col + row) / denom
                    rr = round(c0[0] + (c1[0] - c0[0]) * t)
                    gg = round(c0[1] + (c1[1] - c0[1]) * t)
                    bb = round(c0[2] + (c1[2] - c0[2]) * t)
                    art.append(ch, style=Style(color=f"#{rr:02x}{gg:02x}{bb:02x}",
                                               bold=True))
            else:
                # Non-hex palette (e.g. the ansi theme): flat accent, never crash.
                art.append(line, style=f"bold {pal.accent}")
            if row < len(lines) - 1:
                art.append("\n")
        self.console.print()          # breathing room above the wordmark
        # Align.center (not Text.justify) is what actually centers a renderable
        # block against the terminal width — the same idiom the compact banner
        # uses for its panel. Every art line is the same width, so they stay aligned.
        self.console.print(Align.center(art))
        self.console.print()          # breathing room below the wordmark
        # Value-prop tagline (the differentiator, not "ready"): the core glyph in
        # the accent, the pitch in bright. ◆ degrades to * on a legacy console.
        core = pal.banner_glyph if _enc_can(self.console, pal.banner_glyph) else "*"
        tagline = Text()
        tagline.append(core + "  ", style=pal.accent)
        tagline.append(
            "a coding agent that runs on your machine — private, local, yours",
            style=pal.bright,
        )
        self.console.print(tagline, justify="center")
        self.console.print()
        # Reactor ribbon: <short-model> · ◆ core ready · honest lock badge.
        self.console.print(self._reactor_ribbon(pal), justify="center")

    def _reactor_ribbon(self, pal):
        """The one-line reactor ribbon shared by the first-run hero.

        ``<short-model>`` (bright) · ``◆ core ready`` (success) · the HONEST lock
        badge. HONESTY: the model is ALWAYS local, but we only claim offline when
        private mode is on — ``⬡ offline · no egress`` (private) vs ``⬡ local``
        (network on). ``◆``/``⬡`` degrade to ``*``/``#`` on a legacy console."""
        from rich.text import Text

        core = pal.banner_glyph if _enc_can(self.console, pal.banner_glyph) else "*"
        lock = "⬡" if _enc_can(self.console, "⬡") else "#"
        short_model = (self.config.model or "").rsplit("/", 1)[-1]
        ribbon = Text()
        if short_model:
            ribbon.append(short_model, style=pal.bright)   # make the model pop
            ribbon.append("   ")
        ribbon.append(core + " core ready", style=pal.success)
        ribbon.append("   ")
        if self.config.private:
            ribbon.append(lock + " offline · no egress", style=pal.success)
        else:
            ribbon.append(lock + " local", style=pal.success)
        return ribbon

    def _print_compact_header(self, pal) -> None:
        """The compact framed banner (fallback path): a boxed provider/model +
        green ``ready`` dot ``head`` over a muted theme/privacy/mcp/gentle
        ``sub`` line, centered. Kept intact as the guaranteed-safe banner for
        narrow/piped/legacy consoles."""
        from rich.align import Align
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        # Per-theme banner glyphs (◆/❋ head, ● ready dot). ASCII-degrade them via
        # the same encoding guard the tool tree uses so a legacy/ASCII console
        # (LANG=C) shows "*"/"o" instead of "?" mojibake — the banner is the FIRST
        # thing users see and previously had no guard.
        diamond = pal.banner_glyph if _enc_can(self.console, pal.banner_glyph) else "*"
        ready_dot = pal.ready_glyph if _enc_can(self.console, pal.ready_glyph) else "o"
        head = Text()
        head.append(diamond + " ", style=pal.accent)
        # Short model only (drop the "<provider> ·" prefix AND any "org/" prefix,
        # e.g. qwen/qwen3.6-35b-a3b -> qwen3.6-35b-a3b); the pinned status bar
        # carries the fuller live context.
        short_model = (self.config.model or "").rsplit("/", 1)[-1]
        head.append(short_model, style=pal.bright)  # make the model pop
        head.append("   ")
        head.append(ready_dot + " ", style=pal.success)
        head.append("ready", style=pal.dim)
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
            Group(head, sub), box=_box_for(pal.box_style), border_style=pal.accent,
            title="llmc-code", title_align="left", padding=(0, 1),
            expand=False,  # hug the content instead of spanning the full width
        )))

    def _print_banner_tail(self, pal) -> None:
        """Shared banner footer for BOTH paths: the privacy posture, the /help
        hint, a first-run example (tty only), and trailing breathing room. Piped/
        non-tty runs stay byte-clean — the console emits no ANSI off a real
        terminal and the example line is tty-gated."""
        # Keep the privacy posture visible on startup (security transparency),
        # compact and dim under the frame.
        # Kept short enough to fit ~80 cols without wrapping under the frame.
        if self.config.private:
            privacy = "offline lockdown · no external egress"
        else:
            privacy = ("network on · web_fetch SSRF-safe+gated · "
                       "run_bash gated (full FS access)")
        # Centered to match the centered header block above (justify="center"
        # centers plain text within the terminal width). Sub-lines use the theme's
        # muted token (still recessive, now theme-tinted rather than flat grey).
        self.console.print(privacy, style=pal.dim, justify="center")
        self.console.print(
            "Type /help for commands · Ctrl+O reveals tool detail",
            style=pal.dim, justify="center",
        )
        # First-run affordance: ONE quiet example line so a new user has an obvious
        # first move. Gated on is_terminal so piped/one-shot/sub-agent runs stay
        # byte-clean (never printed off a real terminal).
        if getattr(self.console, "is_terminal", False):
            self.console.print(
                "try: explain this repo · /help for commands",
                style=pal.dim, justify="center",
            )
        # Breathing room: a blank line below the banner so the first prompt and
        # answer box are not glued to the header (clean, uncongested startup).
        self.console.print()

    def _print_mcp_status(self) -> None:
        """List configured MCP servers, their connection state + tools."""
        statuses = self.mcp.status()
        if not statuses:
            self.console.print(
                "No MCP servers configured. Add them to ~/.llmcode/mcp.json "
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
            self._print_help(arg)
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
                # code_theme (ansi_dark / nord / dracula / native ...) for Markdown
                # code blocks. Everything is re-derived from the resolved spec so
                # there is no parallel theming path.
                self.config.theme = arg
                self.console = _make_console(arg)
                self.agent = self._new_agent()
                # Re-derive the whole live ptk Style from the resolved spec so the
                # input glyph, status bar, placeholder, and completion menu all
                # restyle on switch. PromptSession reads .style per prompt() call,
                # so the next prompt picks it up with no relayout. (self.session is
                # None until run() builds it — a no-op for the slash-command unit
                # tests.)
                if self.session is not None:
                    from prompt_toolkit.styles import Style as PTKStyle

                    spec = _SPECS[_resolve_theme(arg)]
                    self.session.style = PTKStyle.from_dict(
                        {
                            "prompt": spec.accent_ptk,
                            "bottom-toolbar": f"noreverse {spec.muted_ptk}".strip(),
                            "placeholder": spec.muted_ptk,
                            "completion-menu.completion.current": spec.completion_ptk,
                        }
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
        elif cmd == "/codeembed":
            # View/set semantic code_search. Mirrors /rerank: no arg -> show
            # current; on/off set the mode. "on" = semantic ("auto"), "off" =
            # BM25-only ("bm25"). code_search reads config.code_search_recall LIVE
            # per call, so the change applies this turn without rebuilding the agent.
            _semantic = self.config.code_search_recall in ("auto", "embed")
            if not arg or arg.lower() == "status":
                self.console.print(f"code_search embeddings: {'on' if _semantic else 'off'}")
                return True
            val = arg.lower()
            if val in ("on", "true", "yes", "1"):
                self.config.code_search_recall = "auto"
            elif val in ("off", "false", "no", "0"):
                self.config.code_search_recall = "bm25"
            else:
                self.console.print("Usage: /codeembed [on|off]")
                return True
            self._persist_config()
            _semantic = self.config.code_search_recall in ("auto", "embed")
            self._ok(
                f"[codeembed -> {'on' if _semantic else 'off'}]  {self._status()}"
            )
            self.console.print(
                "on = semantic (may swap the embedding model on a single-GPU/local "
                "server); off = BM25-only (fast, no swap).",
                style="dim",
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
            # CUSTOM MACRO fallback: a project may define <cwd>/.llmcode/commands/
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
        """Path to a tagged conversation snapshot for the cwd (~/.llmcode/sessions).

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
        self._ok("llmc-code doctor:")
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
              "" if writable else "cannot write ~/.llmcode/checkpoints — /undo disabled")

        # Server tuning (local LLM): ranked, copy-pasteable load-time advice.
        self.console.print("  Server tuning (local LLM):", style=success_style)

        def _tip(text: str) -> None:
            self.console.print(f"      · {text}", style="dim")

        detected = getattr(self.provider, "_ctx_len", None)
        if not isinstance(detected, int) or detected <= 0:
            # Fall back to a best-effort lookup; NEVER raise (mock / no endpoint).
            base_url = getattr(self.provider, "base_url", None) or self.config.base_url
            model = getattr(self.provider, "model", None) or self.config.model
            try:
                detected = detect_context_length(base_url, model) if (base_url and model) else None
            except Exception:  # noqa: BLE001 - best-effort; any failure -> skip
                detected = None
        if not isinstance(detected, int) or detected <= 0:
            self.console.print(
                "      server tuning advice unavailable (no local endpoint detected)",
                style="dim",
            )
        else:
            budget = self.config.context_budget
            # The EFFECTIVE auto-compaction ceiling (raw floor grown toward ~80% of
            # the server ctx, capped at _MAX_AUTO_COMPACT_CEILING) — NOT the raw
            # context_soft_limit floor, which understates it ~2x on large servers.
            soft = _effective_soft_limit(self.provider, self.config)
            if detected > 4 * budget:
                _tip(f"Server loaded at {detected} ctx but working budget is ~{budget} "
                     f"(auto-compaction ceiling ~{soft}). Reload the model with a smaller "
                     f"context (llama.cpp `-c 32768` / LM Studio 'Context Length' ≈ 32k–48k) "
                     f"— KV memory is allocated at load and scales linearly with context, so "
                     f"this frees VRAM/bandwidth.")
            _tip("Enable flash attention (llama.cpp `-fa on` / LM Studio toggle) — mainly "
                 "needed to unlock KV-cache quantization; small direct tok/s effect.")
            _tip("KV cache quantization: `--cache-type-k q8_0 --cache-type-v q8_0` halves KV "
                 "VRAM so you can FIT more context. HONEST NOTE: on Apple Metal q8 KV can be "
                 "slightly SLOWER per token — use it to fit, not for speed.")
            _tip("`--cache-reuse 256` recovers the post-compaction tail re-prefill that the "
                 "prompt-cache (prefix-only) misses.")
            _tip("Speculative decoding: load a small draft model (llama.cpp `-md <draft>` / "
                 "LM Studio speculative decoding) — ~1.3–1.8x decode at low temperature. "
                 "Guard: draft weights shrink the KV budget, so only with free VRAM.")
            self.console.print(
                "      Do NOT: avoid YaRN/RoPE context stretching (grows KV, hurts quality) "
                "and `--mlock` unless you actually see swapping.",
                style="dim",
            )
            self.console.print(
                "      Note: these are mostly server LOAD-TIME settings the harness can't set "
                "for you (LM Studio uses GUI toggles) — most are memory/VRAM wins that help "
                "tok/s only indirectly by keeping all weights on-GPU.",
                style="dim",
            )

    def _macros(self) -> dict:
        """Discover project macros in ``<cwd>/.llmcode/commands/*.md`` (cached).

        Returns ``{"/name": Path}``. A missing directory yields ``{}``. Cached per
        session on first use. Never raises.
        """
        cache = getattr(self, "_macro_cache", None)
        if cache is not None:
            return cache
        from pathlib import Path

        macros: dict = {}
        try:
            base = Path(os.getcwd()) / ".llmcode" / "commands"
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
        """List available project macros (from ``<cwd>/.llmcode/commands/*.md``)."""
        macros = self._macros()
        if not macros:
            self.console.print(
                "no project macros. Add <name>.md files under .llmcode/commands/ "
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
        """Recompute the pinned "reactor" status HUD and cache it in ``_status_cache``.

        Segmented powerline, core-led + lock-trailing::

            ◆ qwen3.6-35b-a3b · ⬤⬤⬤⬜⬜ 58% · main* · ▸ 226 tok/s · 0.42s   ⬡ offline

        The DATA + per-turn caching are unchanged (this runs once at startup and
        once per completed turn — NOT per keystroke — so git + the token estimate
        are hit at most once per turn; ``_status_bar`` just reads the cache). Only
        the styling is rebuilt into explicit prompt_toolkit fragments:

        - a leading ``◆`` core + the active model in the accent;
        - a colour-shifting context gauge — five ``⬤/⬜`` cells whose FILLED cells
          read success (0-60%) → warning (60-85%) → error (85%+), empty cells dim,
          plus the ``NN%`` (ASCII: ``[###--] NN%``);
        - tok/s as the HERO: a ``▸`` prefix + the number in the brightest accent
          tier, unit dim — the one metric that is *yours*;
        - the git branch + time dim; everything unlabelled reads dim;
        - a right-aligned, HONEST ``⬡`` lock badge — ``offline`` only in private
          mode, else ``local`` (a networked local model is never "offline").

        Every glyph degrades to ASCII via ``_enc_can`` (``◆``→``*``, ``⬤/⬜``→``#/-``,
        ``▸``→``>``, ``⬡``→``#``). Every segment is guarded so the bar can never
        raise; the badge is DROPPED (never wrapped) when the toolbar is too narrow.
        """
        from prompt_toolkit.formatted_text import FormattedText, fragment_list_width

        pal = palette_for(self.config.theme)
        con = self.console
        # prompt_toolkit-vocabulary style tokens ONLY (hex or ptk/ansi names —
        # never a rich-only spelling like "bright_yellow", which ptk's colour
        # parser rejects). accent/muted reuse the existing ptk tokens.
        muted = pal.muted_ptk or "fg:ansibrightblack"
        accent = pal.status_num_ptk or pal.ptk or "fg:ansiyellow bold"
        # tok/s hero = the brightest tier. accent_bright (pal.bright) only when it
        # is a hex value ptk can parse; else a bold accent (the ansi theme, whose
        # accent_bright is the rich-only name "bright_yellow", falls here).
        _bright = pal.bright if (isinstance(pal.bright, str)
                                 and pal.bright.startswith("#")) else pal.accent
        hero = f"{_bright} bold"
        # success/warning/error are hex or W3C/ANSI names in every theme -> ptk-safe.
        ok_c = pal.success or "ansigreen"
        warn_c = pal.warning or "ansiyellow"
        err_c = pal.error or "ansired"

        core = "◆" if _enc_can(con, "◆") else "*"
        cells_ok = _enc_can(con, "⬤⬜")
        arrow = "▸" if _enc_can(con, "▸") else ">"
        lock = "⬡" if _enc_can(con, "⬡") else "#"
        sep = (muted, " · ")

        left: list[tuple[str, str]] = []
        # model — short form (strip any "org/" prefix, qwen/qwen3.6 -> qwen3.6).
        try:
            model = getattr(self.provider, "model", "") or self.config.model or ""
            short = model.rsplit("/", 1)[-1] if model else ""
        except Exception:  # noqa: BLE001 - never let the bar break input
            short = ""
        if short:
            left.append((accent, f"{core} {short}"))
        # context gauge — 5 colour-shifting cells + NN%.
        try:
            used = Agent._estimate_tokens(self.agent.messages)
            ceiling = _effective_soft_limit(self.provider, self.config)
            if ceiling > 0:
                pct = int(round(100 * used / ceiling))
                clamped = max(0, min(100, pct))
                filled = max(0, min(5, int(round(clamped / 100 * 5))))
                level = ok_c if clamped < 60 else (warn_c if clamped < 85 else err_c)
                if left:
                    left.append(sep)
                if cells_ok:
                    if filled:
                        left.append((level, "⬤" * filled))
                    if filled < 5:
                        left.append((muted, "⬜" * (5 - filled)))
                    left.append((level, f" {pct}%"))
                else:
                    left.append((level, "[" + "#" * filled))
                    left.append((muted, "-" * (5 - filled)))
                    left.append((level, f"] {pct}%"))
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
                    if left:
                        left.append(sep)
                    left.append((muted, branch))
        except Exception:  # noqa: BLE001
            pass
        # tok/s HERO + time — only once a turn has populated last_turn_stats.
        try:
            stats = getattr(self.agent, "last_turn_stats", None)
            if stats:
                rate = stats.get("toks_per_sec")
                if rate is not None:
                    if left:
                        left.append(sep)
                    left.append((muted, f"{arrow} "))
                    left.append((hero, f"{rate:.0f}"))
                    left.append((muted, " tok/s"))
                elapsed = stats.get("elapsed")
                if elapsed is not None:
                    if left:
                        left.append(sep)
                    left.append((muted, f"{elapsed:.2f}s"))
        except Exception:  # noqa: BLE001
            pass

        if not left:
            self._status_cache = ""
            return
        # Leading pad space (the bar historically starts with a space).
        frags: list[tuple[str, str]] = [(muted, " ")]
        frags.extend(left)

        # ⬡ lock badge, RIGHT-aligned + HONEST: "offline" only in private mode; a
        # networked (non-private) local model reads "local", never "offline". The
        # badge is DROPPED (not wrapped) when the toolbar cannot fit it.
        try:
            private = bool(getattr(self.config, "private", False))
            badge = (ok_c, f"{lock} " + ("offline" if private else "local"))
            width = shutil.get_terminal_size(fallback=(80, 24)).columns
            left_w = fragment_list_width(frags)
            badge_w = fragment_list_width([badge])
            gap = 3  # minimum spaces between the left group and the badge
            # -1 keeps the last column free so the toolbar never wraps to a 2nd row.
            if width and left_w + gap + badge_w <= width - 1:
                frags.append((muted, " " * (width - 1 - left_w - badge_w)))
                frags.append(badge)
        except Exception:  # noqa: BLE001 - badge is optional; never break the bar
            pass

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
        from prompt_toolkit.output.color_depth import ColorDepth

        # Match prompt_toolkit's color depth to rich's truecolor when the terminal
        # advertises it, so the prompt glyph, bottom status bar and y/N confirm emit
        # exact 38;2;r;g;b instead of a 256-color approximation. None lets ptk
        # auto-detect (unchanged behaviour for non-truecolor terminals).
        _cd = (
            ColorDepth.DEPTH_24_BIT
            if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
            else None
        )

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
        # (its fragments carry their own fg; see _refresh_status_bar). The full dict
        # MIRRORS the /theme live-rebuild (placeholder + completion-menu keys) so
        # the input glyph, placeholder, and completion menu are themed from the very
        # FIRST render — not only after a theme switch.
        ptk_style = PTKStyle.from_dict(
            {
                "prompt": pal.ptk,
                "bottom-toolbar": f"noreverse {pal.muted_ptk}".strip(),
                "placeholder": pal.muted_ptk,
                "completion-menu.completion.current": pal.completion_ptk,
            }
        )
        # Persistent line history (also powers prompt_toolkit's built-in Ctrl-R
        # reverse search). Best-effort: a failure to create the file/dir must not
        # break input, so we fall back to an in-memory (None) history.
        from prompt_toolkit.history import FileHistory

        history = None
        try:
            hist_path = os.path.expanduser("~/.llmcode/history")
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

        # Rotating action-oriented ghost (replaces the generic "Ask anything · …"):
        # the loop advances _placeholder_i and re-sets .placeholder before every
        # prompt, so this initial value is just turn 0's affordance.
        placeholder = FormattedText(
            [(pal.muted_ptk or "fg:ansibrightblack",
              _placeholder_for_turn(self._placeholder_i))]
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
            color_depth=_cd,
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
        # The core caret prompt: ◆ ❯ where ◆ IS the permission indicator (recoloured
        # per mode) and ❯ the caret. Rebuilt per turn (below) so a mid-session /mode
        # or /theme switch retints without touching session.style. FormattedText is
        # already imported above (placeholder build); only patch_stdout is new here.
        from prompt_toolkit.patch_stdout import patch_stdout

        try:
            while True:
                try:
                    # Rotate the action-oriented ghost placeholder by turn count
                    # (keeps the prior gotcha: a FormattedText value, never None —
                    # the confirm still saves/restores + suppresses with "").
                    ptk_session.placeholder = FormattedText(
                        [(pal.muted_ptk or "fg:ansibrightblack",
                          _placeholder_for_turn(self._placeholder_i))]
                    )
                    self._placeholder_i += 1
                    prompt_message = self._core_caret_message(pal)
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
                        line = ptk_session.prompt(prompt_message)
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
                    # One of llmcode's OWN commands OR a loaded project macro
                    # (/<name> from .llmcode/commands/) -> handle it locally.
                    if not self._dispatch_slash(line):
                        self._save_session()  # /exit | /quit: persist before leaving
                        self.console.print("Bye.")
                        return
                    continue
                # else: either a normal (non-slash) message, OR a leading-slash line
                # whose first token is NOT an llmcode command (e.g. "/build the app"
                # for another project's CLI). Either way the WHOLE line goes to the
                # model below — llmcode only intercepts its own commands, never prints
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
