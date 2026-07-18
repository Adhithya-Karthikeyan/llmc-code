"""The agentic tool-use loop.

``Agent`` drives one provider through a stream -> tool-call -> tool-result loop
until the model returns a plain-text final answer (or the iteration guard
trips). It is reusable for sub-agents: each instance owns its own message
history, system prompt, and a restricted tool subset.

The loop relies ONLY on the normalized event stream from the provider:
  - text events accumulate into the running assistant reply (and render live).
  - tool_call events are collected during the iteration; after ``done``, if any
    were seen, the agent appends a properly-shaped assistant tool_calls message,
    executes each tool (with confirmation gating), appends one tool-result
    message per call, and loops. Otherwise the accumulated text is the final
    answer.

Presentation:
  - Each tool call renders as ONE collapsed line ("⏺ name summary") with a dim
    success/fail suffix. Full args + results are stored in ``last_turn_details``
    (reset per user turn) and revealed on demand through a SINGLE channel: the
    Ctrl+O reveal (``render_details``). They are never printed live, so detail
    is shown exactly once and never duplicated.
  - Each visible assistant message gets a dim tok/s footer.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from typing import Callable

from . import checkpoint as checkpoint_mod
from . import cooldown
from . import hooks as hooks_mod
from . import remediation
from . import tools as tools_mod
from .images import text_of
from .prompts import SUMMARIZER_PROMPT
from .providers import Provider
from .spinner import Spinner


# ---------------------------------------------------------------------------
# Answer rendering (Markdown look + the thin accent answer BOX)
# ---------------------------------------------------------------------------
# rich centers ``# h1`` headings, which reads oddly in a terminal answer column.
# These two classes (defined ONCE at import, not per render call) left-align
# every heading level; colour/weight still come from the theme's markdown.h*
# styles. _LeftAlignedMarkdown overrides ONLY the heading element so paragraphs,
# lists, fences and rules keep rich's defaults.
def _make_markdown_classes():
    from rich.markdown import Heading, Markdown

    class _LeftHeading(Heading):
        def __rich_console__(self, console, options):
            text = self.text.copy()
            text.justify = "left"  # never center — even for h1
            yield text

    class _LeftAlignedMarkdown(Markdown):
        elements = {**Markdown.elements, "heading_open": _LeftHeading}

    return _LeftHeading, _LeftAlignedMarkdown


_LeftHeading, _LeftAlignedMarkdown = _make_markdown_classes()


def build_answer_markdown(body: str, code_theme: str):
    """A rich ``Markdown`` whose headings render LEFT-aligned (never centered),
    with fenced code highlighted by ``code_theme``."""
    return _LeftAlignedMarkdown(body, code_theme=code_theme)


def confirm_label(tool: tools_mod.Tool, args: dict) -> tuple[str, str]:
    """Return ``(label, hint)`` for a gated tool's y/N confirmation prompt.

    Shared by both confirm_fns (this module's input()-based fallback and the
    REPL's prompt_toolkit one) so the label/byte-hint logic lives in ONE place
    (finding #29). ``label`` is the same collapsed summary the loop shows
    ("write_file path"); ``hint`` is a write_file byte count, else "".
    """
    safe_args = args if isinstance(args, dict) else {}
    summary = _tool_summary(tool.name, safe_args)
    label = f"{tool.name} {summary}".rstrip()
    hint = ""
    if tool.name == "write_file":
        content = safe_args.get("content")
        if isinstance(content, str):
            hint = f" (content: {len(content)} bytes)"
    return label, hint


def _default_confirm(tool: tools_mod.Tool, args: dict) -> bool:
    """Interactive y/N confirmation for a dangerous tool call.

    NOTE: this uses builtin ``input()`` and conflicts with prompt_toolkit's
    event loop. It is fine for one-shot/non-interactive use; the REPL injects a
    prompt_toolkit-compatible confirm_fn instead (see repl.make_ptk_confirm).
    """
    # Show the same collapsed summary as the loop ("⏺ write_file path") instead
    # of a raw JSON args dump, consistent with the rest of the clean stream.
    label, hint = confirm_label(tool, args)
    try:
        # Leading newline so the y/N prompt isn't glued to the preceding dim
        # tok/s footer when the model narrates AND calls a gated tool.
        answer = input(f"\nRun {label}?{hint} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Pure, unit-testable helpers
# ---------------------------------------------------------------------------

def _oneline(s: str) -> str:
    """Collapse any whitespace run (incl. embedded newlines) to a single space.

    A weak or hostile model can emit a tool arg containing a literal '\\n'
    (e.g. pattern='*.py\\nINJECTED'). Since the collapsed tool line is rendered
    via rich Text with no_wrap=True/overflow=ellipsis — which only disables
    SOFT wrap, NOT literal newlines — an un-sanitized value would split the
    one-line tool tree across multiple physical lines. Collapsing here keeps the
    documented "one collapsed line per call" contract for both the head label
    and the path-bearing result summaries.
    """
    return " ".join(str(s).split())


def _tool_summary(name: str, args: dict) -> str:
    """A compact one-token-ish hint derived from a tool call's args.

    Used for the collapsed "⏺ name summary" line. Returns "" (just the name)
    when there is nothing meaningful to show.
    """
    if not isinstance(args, dict):
        return ""
    if name in ("write_file", "edit_file", "read_file"):
        return str(args.get("path", ""))
    if name == "run_bash":
        cmd = str(args.get("command", "")).replace("\n", " ")
        return cmd[:50] + ("..." if len(cmd) > 50 else "")
    if name == "glob":
        return str(args.get("pattern", ""))
    if name == "grep":
        pat = str(args.get("pattern", ""))
        path = str(args.get("path", ""))
        if path and path != ".":
            return f"{pat} in {path}"
        return pat
    if name == "spawn_agent":
        return str(args.get("role", ""))
    if name == "web_fetch":
        return str(args.get("url", ""))
    if name.startswith("mcp__"):
        # The collapsed line ALREADY prints the full "mcp__server__tool" name, so
        # here we only surface the first short scalar arg (no repeating the name)
        # — e.g. "⏺ mcp__kyp-mem__kyp_search <query>".
        for v in args.values():
            if isinstance(v, (str, int, float, bool)):
                return str(v).replace("\n", " ")[:50].rstrip()
        return ""
    return ""


def _reactor_pulse_to(palette) -> str | None:
    """The brightest accent hex for the reactor-pulse PEAK: ``accent_bright`` if
    the palette exposes it, else ``bright``. ``None`` when there's no palette (so
    the spinner stays on the classic braille frame — no behaviour change)."""
    if palette is None:
        return None
    return getattr(palette, "accent_bright", None) or getattr(palette, "bright", None)


def _spinner_verb(name: str, args: dict) -> str:
    """A short, plain-ASCII activity verb for the reactor spinner while a tool
    runs — ``reading providers.py`` / ``running <cmd>`` / ``searching <pat>`` /
    ``editing <file>`` — so the pulse says what the machine is doing, not just
    "working". Falls back to ``running <tool>`` for anything unmapped."""
    a = args if isinstance(args, dict) else {}

    def _base(p: str) -> str:
        p = str(p)
        return os.path.basename(p.rstrip("/")) or p

    if name == "read_file":
        return f"reading {_base(a.get('path', ''))}".strip()
    if name == "write_file":
        return f"writing {_base(a.get('path', ''))}".strip()
    if name == "edit_file":
        return f"editing {_base(a.get('path', ''))}".strip()
    if name == "run_bash":
        cmd = str(a.get("command", "")).replace("\n", " ").strip()
        return f"running {cmd[:24]}".strip() if cmd else "running"
    if name in ("grep", "code_search"):
        return f"searching {str(a.get('pattern', a.get('query', ''))).strip()}".strip()
    if name == "glob":
        return f"globbing {str(a.get('pattern', '')).strip()}".strip()
    if name == "spawn_agent":
        return f"delegating {str(a.get('role', '')).strip()}".strip()
    friendly = name.split("__")[-1] if name.startswith("mcp__") else name
    return f"running {friendly}".strip()


# Friendly, Claude-Code-like tool display names. Anything not listed keeps its
# raw registry name (so a novel/local tool still renders intelligibly).
_DISPLAY_NAMES = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "run_bash": "Bash",
    "glob": "Glob",
    "grep": "Grep",
    "web_fetch": "Fetch",
    "spawn_agent": "Task",
}


def display_name(name: str) -> str:
    """Map a raw tool name to its Claude-Code-style friendly display name.

    ``mcp__<server>__<tool>`` collapses to ``<server>:<tool>``. Everything else
    uses the static map, falling back to the raw name when unmapped.
    """
    if not isinstance(name, str) or not name:
        return str(name)
    if name.startswith("mcp__"):
        parts = name.split("__")
        if len(parts) >= 3:
            return f"{parts[1]}:{'__'.join(parts[2:])}"
        return name
    return _DISPLAY_NAMES.get(name, name)


def tool_call_label(name: str, args: dict) -> str:
    """The "DisplayName(args)" portion of a tool line (no glyph, no color).

    Empty args render as a bare "DisplayName" (no empty parens), matching
    Claude Code's look for argument-less calls.
    """
    # Collapse any embedded newline/whitespace so the head stays a single line
    # even when the model emits a multi-line arg (no_wrap only disables soft
    # wrap, not literal '\n').
    summary = _oneline(_tool_summary(name, args if isinstance(args, dict) else {}))
    disp = display_name(name)
    return f"{disp}({summary})" if summary else disp


def _first_nonempty_line(text: str) -> str:
    for ln in str(text).splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def result_summary(name: str, result: dict) -> str:
    """A SHORT, Claude-Code-style one-line summary of a successful tool result.

    Never the full dump (that stays in last_turn_details for Ctrl+O). Tolerant
    of malformed/missing payloads from a weak model — always returns a string.
    """
    if not isinstance(result, dict):
        return ""
    payload = result.get("result")
    if name == "read_file":
        if isinstance(payload, str):
            lines = payload.splitlines()
            # A sliced read prepends a "[lines a-b of N]" / "[no lines: ...]"
            # header — don't count it toward the displayed line total.
            if lines and lines[0].startswith(("[lines ", "[no lines")) and lines[0].endswith("]"):
                n = len(lines) - 1
            else:
                n = len(lines)
        else:
            n = 0
        return f"Read {n} lines"
    if name == "write_file":
        if isinstance(payload, dict):
            nbytes = payload.get("bytes_written")
            if isinstance(nbytes, int):
                return f"Wrote {nbytes} bytes"
            path = payload.get("path")
            if path:
                return f"Created {_oneline(path)}"
        return "Wrote file"
    if name == "edit_file":
        path = payload.get("path") if isinstance(payload, dict) else None
        return f"Updated {_oneline(path)}" if path else "Updated file"
    if name == "run_bash":
        if isinstance(payload, dict):
            stdout = payload.get("stdout")
            if isinstance(stdout, str) and stdout.strip():
                lines = [ln for ln in stdout.splitlines() if ln.strip()]
                first = lines[0].strip()[:60]
                extra = len(lines) - 1
                return first + (f" (+{extra} lines)" if extra > 0 else "")
            code = payload.get("exit_code")
            if code is not None:
                return f"exit {code}"
        return "exit 0"
    if name == "glob":
        matches = payload.get("matches") if isinstance(payload, dict) else None
        n = len(matches) if isinstance(matches, list) else 0
        return f"{n} file" if n == 1 else f"{n} files"
    if name == "grep":
        matches = payload.get("matches") if isinstance(payload, dict) else None
        n = len(matches) if isinstance(matches, list) else 0
        return f"{n} match" if n == 1 else f"{n} matches"
    if name == "web_fetch":
        text = payload.get("text") if isinstance(payload, dict) else None
        n = len(text) if isinstance(text, str) else 0
        return f"Fetched {n} chars"
    if name == "spawn_agent":
        role = ""
        if isinstance(payload, str):
            # spawn_agent's result is the sub-agent's summary string; we don't
            # have the role here, so a generic preview is the best short hint.
            return (_first_nonempty_line(payload)[:60]) or "done"
        return f"{role} done".strip()
    if name.startswith("mcp__"):
        # ~60-char text preview of whatever the MCP tool returned.
        if isinstance(payload, str):
            preview = _first_nonempty_line(payload)
        else:
            preview = _first_nonempty_line(json.dumps(payload, ensure_ascii=False)) if payload is not None else ""
        return preview[:60]
    # Unknown tool: a short preview of the payload, never the full dump.
    if isinstance(payload, str):
        return _first_nonempty_line(payload)[:60]
    return ""


# Max length of a one-line failure summary in the tool tree. The tree is
# hard-clipped to terminal width anyway (no_wrap + ellipsis), so this is just a
# sane upper bound shared by both the explicit-error and exit-code branches.
_ERR_SUMMARY_MAX = 70


def error_summary(result: dict) -> str:
    """Short first-line error text for a failed tool call (no glyph/color)."""
    if not isinstance(result, dict):
        return "failed"
    err = result.get("error")
    if err:
        first = _first_nonempty_line(err) or "failed"
        return first[:_ERR_SUMMARY_MAX]
    # run_bash (and any tool that signals failure via a nonzero exit code) marks
    # ok=False with NO "error" field — the diagnostics live in result["result"]
    # {exit_code, stderr}. Surfacing a bare "failed" there hides WHY the command
    # failed (missing file, grep no-match, permission denied, ...). Show the exit
    # code plus the first stderr line so the failure is actually actionable.
    payload = result.get("result")
    if isinstance(payload, dict) and payload.get("exit_code") is not None:
        code = payload.get("exit_code")
        stderr = payload.get("stderr")
        if isinstance(stderr, str) and stderr.strip():
            return f"exit {code}: {_first_nonempty_line(stderr).strip()}"[:_ERR_SUMMARY_MAX]
        return f"exit {code}"
    return "failed"


# Content-free preamble openers. NOTE: "here is " / "here's " / "here are " are
# DELIBERATELY EXCLUDED (finding #9): they almost always precede the answer's
# actual subject ("Here is how recursion works: a function calls itself."), so
# stripping the clause after them drops real content. Only true filler openers
# that carry no information are listed.
_PREAMBLE_OPENERS = (
    "sure, ", "sure! ", "sure ",
    "let me ", "i will ", "i'll ", "okay, ", "ok, ", "certainly, ",
)

# Content-free filler that can follow an opener clause-internally and still
# carry NO answer content (finding #2). Only when the WHOLE clause between the
# opener and the first sentence break is one of these (after stripping trailing
# punctuation) do we strip it. Anything else — e.g. "the file has 3 functions"
# — is real content and is preserved.
_PREAMBLE_FILLER = (
    "do it", "do that", "do this", "do it for you",
    "help you", "help you with that", "help with that",
    "explain", "explain that", "explain it", "answer that",
    "answer your question", "i can help", "i can do that", "of course",
    "no problem", "happy to help", "let's do it", "lets do it",
)


# Markup that means the model wrote a TOOL CALL as plain text instead of issuing
# it — a <function=NAME>/<tool_call> block, a deepseek/qwen control sigil, a
# Mistral [TOOL_CALLS], or a Llama python_tag. If this shows up in what would be
# the FINAL answer, the call never ran ("it said it would do X but didn't"), so
# the agent re-prompts instead of printing the raw markup.
_TOOL_MARKUP_RE = re.compile(
    r"<function\s*=\s*\w|<tool_call\b|</tool_call>|\[TOOL_CALLS\]|<\|python_tag\|>"
    "|｜tool▁call",
)


# Fenced code blocks (terminated OR unterminated-to-end) and inline `code`, plus
# a leading <think> block — the SAME example-guard regions the provider's
# extract_text_tool_calls skips. Stripped before the markup check so an answer
# that legitimately SHOWS <function=…>/<tool_call> syntax inside a code fence
# doesn't trigger a spurious re-prompt.
_GUARD_SPAN_RE = re.compile(r"```.*?```|```.*\Z|`[^`\n]*`", re.DOTALL)
_LEADING_THINK_RE = re.compile(r"\A\s*<think>.*?</think>", re.DOTALL)


def looks_like_unexecuted_tool_call(text: str) -> bool:
    """True if ``text`` (a would-be final answer) actually contains tool-call
    markup that was never executed — the model narrated/emitted a call as text.

    Markup that appears only inside a code fence / inline code / leading <think>
    is an EXAMPLE, not a real call (mirrors the provider's guard spans), so those
    regions are stripped first to avoid a spurious recovery generation."""
    if not text:
        return False
    stripped = _LEADING_THINK_RE.sub("", text)
    stripped = _GUARD_SPAN_RE.sub("", stripped)
    return _TOOL_MARKUP_RE.search(stripped) is not None


def normalize_final_answer(text: str) -> str:
    """Conservatively strip a single content-free preamble opener.

    The output contract forbids "Sure", "Let me", "I will" openers. A local
    model often emits them anyway. This is a string-level (no model call)
    normalizer that strips ONE leading preamble clause, and ONLY when the answer
    does not start with a code fence or markdown header (so it can never corrupt
    code blocks or structured output).

    We strip ONLY when the clause is genuinely content-free (finding #2):
      1. the opener IS the entire first sentence (opener immediately followed by
         a sentence break, e.g. "Sure. <answer>" / "Okay, <answer>" where the
         text after the opener begins the real answer), OR
      2. the clause between the opener and the first sentence break is a known
         filler phrase carrying no information ("let me help you with that.").

    We NEVER cut across a clause that contains other words — e.g.
    "Sure, the file has 3 functions. They are foo, bar, baz." keeps the count.
    A ':'-introduced clause is never a strip point (a colon introduces content).
    """
    if not text:
        return text
    stripped = text.lstrip()
    # Never touch fenced code or already-structured (header) answers.
    if stripped.startswith("```") or stripped.startswith("#"):
        return text
    low = stripped.lower()
    for opener in _PREAMBLE_OPENERS:
        if low.startswith(opener):
            rest = stripped[len(opener):]
            # Find the first true SENTENCE break ('\n' or '. '); a ':' usually
            # introduces the real content, so it is NOT a strip point.
            cut = -1
            sep_len = 0
            for sep in ("\n", ". "):
                idx = rest.find(sep)
                if idx != -1 and (cut == -1 or idx < cut):
                    cut = idx
                    sep_len = len(sep)
            if cut == -1:
                break
            clause = rest[:cut].strip()
            tail = rest[cut + sep_len:].lstrip()
            if not tail:
                break
            # Only strip when the clause carries no content: empty (opener was
            # the whole first sentence) or a recognised filler phrase.
            clause_key = clause.lower().rstrip(".!?,").strip()
            if clause == "" or clause_key in _PREAMBLE_FILLER:
                return tail
            break
    return text


def compute_tok_stats(text: str, output_tokens, elapsed: float) -> tuple[int, float]:
    """Return ``(tokens, rate)`` for one assistant message.

    - If ``output_tokens`` is a positive int, use it.
    - Else approximate from ``text``: max(chars/4, word_count) to avoid
      undercounting (0 when text is empty).
    - rate = tokens / max(elapsed, 1e-6) (divide-by-zero guard -> finite rate).
    """
    tokens = 0
    if isinstance(output_tokens, int) and not isinstance(output_tokens, bool) and output_tokens > 0:
        tokens = output_tokens
    elif text:
        tokens = max(len(text) // 4, len(text.split()))
    rate = tokens / max(elapsed, 1e-6)
    return tokens, rate


def format_footer(rate: float) -> str:
    """Dim footer showing ONLY the generation speed, e.g. ``47.9 tok/s``."""
    return f"{rate:.1f} tok/s"


# Never trim the working budget below this — even a "simple" turn needs room for
# the system prompt + the answer.
_MIN_TURN_BUDGET = 4_000
# Context hygiene: once a turn is COMPLETE, its bulky tool outputs (file reads,
# test-run stdout, repo maps) and the file content inside its write/edit tool
# calls are rarely needed again — but they sit in history and are re-sent on
# EVERY later turn, so a one-line follow-up drags tens of thousands of stale
# tokens. These caps shrink that stale bulk (keeping a head snippet + a note);
# the live turn's own tool output stays full. Info is recoverable (file on disk
# + memory recall), so trimming is safe.
_STALE_TOOL_RESULT_CAP = 500   # chars of a past tool RESULT kept verbatim
_STALE_ARG_FIELD_CAP = 160     # chars of a past write/edit content/old/new kept
# Collision-proof idempotency sentinel — must NOT be a word that can appear in
# real tool output / file content (a plain "trimmed" would let any read of code
# or test output that merely mentions the word escape trimming). The guillemets
# make an accidental match effectively impossible.
_TRIM_MARKER = "⟪ctx-trimmed⟫"   # ⟪ctx-trimmed⟫
# Per-turn read-budget guard (PERF): how many bytes of CODE/output the context-
# bloating tools may pull into ONE user turn before run() appends a ONE-TIME
# nudge telling the model to stop reading and delegate/answer. ~32KB ≈ ~8K
# tokens, the point where extra context starts visibly slowing local decode.
_TURN_READ_NUDGE_BYTES = 32_000
# Verbs that mark an MCP tool name as a likely state MUTATION (blocked in
# read-only/plan). Matched against the FIRST word of a tool's final ``__``-
# segment (e.g. ``mcp__crm__create_document`` -> "create"). This is a best-effort
# NAME heuristic only: it can't see a server's real side effects, so read-style
# names (get/list/search/read/fetch/report/analyze) are treated as safe.
_MCP_MUTATION_VERBS = frozenset({
    "create", "update", "delete", "write", "insert", "drop", "submit",
    "commit", "run", "execute", "set", "add", "remove", "move", "rename",
    "merge", "push",
})


def _mcp_name_is_mutation(name: str) -> bool:
    """Best-effort: True if an ``mcp__…`` tool name looks like a mutation.

    Splits the tool name on ``__``, takes the final segment, then its leading
    word (split on ``_``), and reports whether that word is a known mutation
    verb. Read-style verbs return False so reads stay allowed in read-only/plan.
    """
    segment = name.split("__")[-1]
    first_word = segment.split("_", 1)[0].lower()
    return first_word in _MCP_MUTATION_VERBS
# Tools that pull CODE/output INTO context (vs. side-effecting/write tools).
# A module-level frozenset so the set is easy to see and extend.
_BLOATING_TOOL_NAMES = frozenset({
    "read_file", "grep", "repo_map", "code_search", "glob",
})
# Substrings that mark an MCP read tool as context-bloating. MCP tool names are
# dynamic (``mcp__<server>__<tool>``) so exact membership can't match them; a tool
# is bloating if it's in ``_BLOATING_TOOL_NAMES`` OR it's an MCP tool whose
# *suffix* (the part after the last ``__``) contains one of these read-oriented
# substrings (search / context / project / session tools pull large payloads
# into context just like the built-in read tools). The suffix is matched — not
# the whole name — so a server name like ``kyp-mem`` does NOT cause that
# server's write/delete tools (``mcp__kyp-mem__kyp_write``) to be flagged as
# bloating. Kept separate from the exact-match set so the existing membership
# invariant (test_only_known_bloating_tools_in_set) is preserved.
_BLOATING_TOOL_PATTERNS = frozenset({
    "search", "context", "project", "session",
})


def _is_bloating_tool(name: str) -> bool:
    """Whether a tool call pulls code/output INTO context (read-style).

    Exact membership in ``_BLOATING_TOOL_NAMES`` (built-ins) OR an MCP read tool
    matched by substring against its suffix only. Mirrors the exact-match
    convention used for the built-in set while covering dynamic MCP names
    without flagging write tools on a read-named server.
    """
    if name in _BLOATING_TOOL_NAMES:
        return True
    low = name.lower()
    if not low.startswith("mcp__"):
        return False
    suffix = low.rsplit("__", 1)[-1]
    return any(p in suffix for p in _BLOATING_TOOL_PATTERNS)
# Substrings that mark a request as "big" (needs more context kept). Lowercased
# match against the user's message.
_BIG_REQUEST_KEYWORDS = (
    "audit", "refactor", "rewrite", "redesign", "migrate", "implement",
    "whole project", "entire", "all files", "every file", "across", "codebase",
    "trace", "debug", "investigate", "end-to-end", "end to end",
)


def request_weight(text: str) -> float:
    """Heuristic 1.0–3.0 multiplier for how much context a request likely needs.

    Bigger for long prompts and ones naming broad/multi-file work; ~1.0 for a
    short simple question. No model call — pure string signals.
    """
    if not text:
        return 1.0
    low = text.lower()
    weight = 1.0
    n = len(text)
    if n > 400:
        weight += 0.5
    if n > 1200:
        weight += 0.5
    if any(k in low for k in _BIG_REQUEST_KEYWORDS):
        weight += 1.5
    return min(weight, 3.0)


# A token that looks like a filename/path (foo.py, src/bar.ts) — the message
# points at concrete code, so it is at least a "task".
_FILE_TOKEN_RE = re.compile(r"\b[\w./-]+\.[A-Za-z]{1,6}\b")
# Verbs that mark a concrete code ACTION (vs a conversational/meta question).
_ACTION_VERB_RE = re.compile(
    r"\b(add|implement|writ|creat|buil|fix|refactor|renam|mov|delet|remov|"
    r"replac|updat|chang|modif|optimi|migrat|debug|test|install|generat|"
    r"convert|wire|integrat|patch|revert)\w*\b",
    re.IGNORECASE,
)
# Openers of a short conversational / meta message that needs almost no context.
_TRIVIAL_OPENERS = (
    "what happened", "what did you", "what was", "what's the status", "why",
    "how come", "explain", "summar", "recap", "status", "thanks", "thank you",
    "ok", "okay", "cool", "nice", "got it", "hi", "hey", "hello", "who",
)
# Words signalling a bare follow-up that leans on the PRIOR turn's context.
_FOLLOWUP_HINTS = (" it", " that", " this", " them", " again", "same", "redo",
                   "undo", "continue", "go on", "retry", "rerun")
# Function words that carry no DOMAIN content — used to tell a pure meta question
# ("what happened?") from an investigation that names real things ("why is the
# LOGIN BROKEN?"). The latter benefits from context, so it must NOT drop to
# trivial just because it opens with "why".
_STOPWORDS = frozenset(
    "a an the is are was were be been being do did does done you i we it its "
    "that this these those what why how when who whom where which to of in on "
    "for and or not no yes can could would should will about with as at by from "
    "your my our me us your please just now then so".split()
)


def classify_request(text: str) -> str:
    """Rough INTENT of a user message, used to size how much context to pre-load.

    Returns one of:
      "trivial"  — a short conversational/meta question ("what happened?") that
                   needs only the running summary + last exchange.
      "followup" — short, leans on the immediately prior action ("run that again").
      "task"     — names a file/symbol or asks for a concrete code action.
      "broad"    — whole-repo / audit / multi-file / long request.

    Pure string signals — NO model call, deterministic, can't fail a turn. The
    bias is DELIBERATELY upward: when ambiguous it returns "task" (more context),
    because under-loading is recoverable (the model pulls more via read_file/
    code_search/grep) while a starved real task is not. Only a message that is
    BOTH short AND clearly conversational is allowed to drop to trivial/followup.
    """
    if not text or not text.strip():
        return "trivial"
    low = text.lower().strip()
    n = len(low)
    # Broad: explicit whole-repo keywords or a genuinely long prompt.
    if any(k in low for k in _BIG_REQUEST_KEYWORDS) or n > 1000:
        return "broad"
    # Task: points at a file, asks for a code action, or is non-trivially long.
    if _FILE_TOKEN_RE.search(text) or _ACTION_VERB_RE.search(low) or n > 200:
        return "task"
    # Short and no concrete code target: conversational. Distinguish a bare
    # follow-up (refers to prior work) from a pure trivial/meta question.
    if n <= 120 and (low.startswith(_TRIVIAL_OPENERS) or low.endswith("?")):
        if any(h in low for h in _FOLLOWUP_HINTS):
            return "followup"
        # Only a question with (almost) no DOMAIN content words is truly trivial.
        # One that names real things ("why is the LOGIN BROKEN?") is an
        # investigation that benefits from context -> task (the safe, larger side).
        content_words = [
            w for w in re.findall(r"[a-z0-9_]+", low) if w not in _STOPWORDS
        ]
        if len(content_words) <= 1:
            return "trivial"
        return "task"
    # Everything else defaults UP to task (safe direction).
    return "task"


def _tool_call_batch_sig(tool_calls: list[dict]) -> tuple:
    """Canonical signature of a tool-call batch for the duplicate-loop guard.

    Returns a sorted tuple of ``(name, args_json)`` per call so two batches that
    issue the SAME work match regardless of call/key order. Each call's args are
    JSON-serialized with ``sort_keys=True`` (so key order in the model's JSON
    doesn't matter) and a ``default=str`` fallback (best-effort for any odd
    non-JSON value). Used ONLY to detect an exact-repeat loop; never for
    execution or rendering.
    """
    parts: list[tuple] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        try:
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            args_str = repr(sorted(str(a) for a in args.items()))
        parts.append((name, args_str))
    return tuple(sorted(parts))


def _msg_chars(m: dict) -> int:
    """Rough CHAR count (NOT divided by 4) of one message for token estimation.

    Same formula as ``Agent._estimate_tokens`` per message: the text-length of the
    message content (``text_of`` drops base64 image parts) plus the serialized
    ``tool_calls`` JSON. Kept at the module level so the incremental running
    estimate (``Agent._estimate_tokens_cached``) can add deltas for newly appended
    messages without re-serializing the whole history each iteration (the O(n^2)
    hot path: ``_maybe_auto_compact`` ran ``_estimate_tokens`` every loop iteration
    and re-``json.dumps``-ed every past tool_call's arguments). Returning chars
    (not tokens) preserves the integer-division semantics of ``_estimate_tokens``
    — the cached total is divided by 4 once at query time, exactly matching the
    full recompute.
    """
    total = len(text_of(m.get("content")))
    tcs = m.get("tool_calls")
    if tcs:
        total += len(json.dumps(tcs, ensure_ascii=False))
    return total


class Agent:
    def __init__(
        self,
        provider: Provider,
        system_prompt: str,
        tool_names: list[str],
        console=None,
        auto_confirm: bool = False,
        max_iterations: int = 12,
        confirm_fn: Callable[[tools_mod.Tool, dict], bool] | None = None,
        registry: dict | None = None,
        line_prefix: str = "",
        context_soft_limit: int = 0,
        code_theme: str = "monokai",
        accent: str | None = None,
        palette=None,
        gutter_char: str = "▌",
        context_budget: int = 0,
        context_ceiling: int = 0,
        context_adaptive: bool = True,
        memory=None,
        recall_mode: str = "off",
        memory_top_k: int = 3,
        memory_enabled: bool = False,
        render_answer: bool = True,
        read_nudge_bytes: int = _TURN_READ_NUDGE_BYTES,
        constrained_retry: bool = False,
        verify_cmd: str = "",
        review_writes: bool = False,
        rerank: bool = False,
        rerank_candidates: int = 20,
        permission_mode: str = "default",
        checkpoints_enabled: bool = False,
        hooks: dict | None = None,
        cancel_event=None,
        workspace_root: str | None = None,
        checkpoint_session: str | None = None,
        todos_enabled: bool = False,
        auto_fix_tools: bool = False,
        auto_fix_max_attempts: int = 2,
        cooldown_enabled: bool = False,
        suppress_footer: bool = False,
    ):
        self.provider = provider
        self.tool_names = list(tool_names)
        # When False, run() prints the collapsed activity line but SKIPS rendering
        # the final answer markdown + tok/s footer. A spawned sub-agent sets this
        # off: its answer is RETURNED to the orchestrator (which renders it once),
        # so without this the same answer printed twice — the sub-agent's live copy
        # AND the orchestrator's re-rendered copy. The return value is unchanged.
        self.render_answer = bool(render_answer)
        # When True, the "Model | Time | Speed" footer LINE is not printed — the
        # combined REPL owns a pinned bottom status bar that shows those stats
        # instead (read from self.last_turn_stats below). last_turn_stats is still
        # computed + stored when suppressed. DEFAULT False so one-shot `-p` and
        # every existing test print the footer byte-for-byte as before.
        self.suppress_footer = bool(suppress_footer)
        # Snapshot of the most recent turn's footer stats (model/elapsed/rate/
        # tokens), populated by _print_footer from the SAME values the footer
        # renders. None until the first rendered turn; the REPL's status bar reads
        # it to mirror the footer without recomputing.
        self.last_turn_stats: dict | None = None
        self.console = console
        # Pygments theme for fenced code blocks in rendered Markdown answers.
        # "monokai" (default) is a truecolor/256 theme; the ANSI "Dark mode" uses
        # "ansi_dark" so highlighting stays within the 16 basic ANSI colors and
        # emits no truecolor escapes. Threaded from config.theme by the REPL.
        self.code_theme = code_theme
        # Accent colour (a rich style string, e.g. "#8b949e") for the answer
        # box border + title, the tok/s footer rate, and the collapsed activity
        # glyph. When None (sub-agents built directly / tests) the answer renders
        # as plain Markdown and status lines stay dim — byte-for-byte the old
        # behaviour. The thin "Answer" box is applied ONLY to a top-level agent
        # (empty line_prefix); a nested sub-agent ("  ↳ ") keeps plain Markdown so
        # the box never collides with its nesting marker.
        self.accent = accent or None
        # Full theme Palette (repl.Palette) when the REPL threads it in — carries
        # the SEMANTIC tokens (success/error), the muted grey, and the raw-stream
        # spinner colours. None for sub-agents built directly / tests: the tool
        # tree, footer, activity line, and spinner then fall back to the historic
        # literal styles ("green"/"dim"/"dim red") and an uncoloured spinner, so
        # byte-for-byte the old behaviour holds. Every use is gated so PIPED /
        # non-terminal output stays ANSI-free regardless of the palette.
        self.palette = palette
        # Retained for back-compat (threaded from the palette); the old left-bar
        # gutter glyph is no longer drawn now that answers use a thin box.
        self.gutter_char = gutter_char
        # Auto-compaction soft budget (rough chars/4 token estimate). When >0 and
        # the live history exceeds it, run() compacts BEFORE the next provider
        # call so a long file-reading session never grows the prompt past the
        # model's context window (findings #4/#26). 0 (the default for sub-agents
        # and tests) disables it.
        self.context_soft_limit = max(0, int(context_soft_limit))
        # Adaptive working budget. When context_budget > 0, run() RECOMPUTES
        # self.context_soft_limit at the start of EVERY turn from the request
        # (tight by default, flexed up for a big request, down for a small one),
        # capped at context_ceiling (the near-window safety value). 0 keeps the
        # static context_soft_limit above — byte-for-byte the old behaviour, which
        # is what sub-agents and the tests use.
        self.context_budget = max(0, int(context_budget))
        self.context_ceiling = max(0, int(context_ceiling))
        self.context_adaptive = bool(context_adaptive)
        # Anti-thrash floor for auto-compaction. When a compaction attempt can't
        # meaningfully shrink history (the recent context the model needs is
        # itself near the budget, or a weak model summarizes poorly), we record
        # the size here and skip re-attempting — re-summarizing every turn for ~no
        # gain just rewrites history (breaking the KV-cache prefix) and spams the
        # user. Reset to 0 once history drops back under budget.
        self._compact_floor = 0
        # PERF incremental token estimate (Fix 2): a running CHAR total over
        # ``self.messages`` so ``_maybe_auto_compact`` (called every loop iteration)
        # doesn't re-``json.dumps`` every past tool_call's arguments each iteration
        # — the O(n^2) hot path. ``None`` means "dirty": the next
        # ``_estimate_tokens_cached`` call recomputes the full sum and re-arms the
        # cache. Invalidated on any in-place mutation (``_trim_stale_tool_outputs``)
        # or history rewrite (compaction); new appends add a delta when the cache is
        # armed. The cached value is CHARS (divided by 4 at query time) to preserve
        # the integer-division semantics of the static ``_estimate_tokens``.
        self._running_token_chars: int | None = None
        # Reliable "did the last stale-trim rewrite anything" flag (see
        # _trim_stale_tool_outputs); the per-turn caller invalidates the token
        # cache off THIS, not the lossy est-token return. Defensive default so the
        # attribute always exists even if the trim is never run.
        self._last_trim_mutated = False
        # Memoized tool-schema payload (PERF): openai_schema() re-serializes the
        # full (potentially large, MCP-inclusive) tool set every turn though it is
        # identical unless the active tool NAMES change. Cache it keyed on the
        # tool-name tuple; within one Agent instance tool_names/registry are fixed
        # (MCP integration + config changes REBUILD the agent), so this rebuilds
        # only if the key ever changes. None key => not yet built.
        self._tools_payload_cache: list[dict] | None = None
        self._tools_payload_key: tuple[str, ...] | None = None
        # ACBUILD-1: build/test verification-nudge state. _unvalidated_writes is
        # ARMED when a write_file/edit_file succeeds and CLEARED when run_bash
        # runs; _build_nudged caps the verify-nudge at ONE per write-batch so the
        # loop can't spin. Both default off (no writes yet -> never misfires on a
        # read-only/question turn or a sub-agent without run_bash).
        self._unvalidated_writes = False
        self._build_nudged = False
        # Feature 2: constrained-decode retry on a malformed tool call. When on,
        # a tool-call parse failure re-issues the SAME request once with
        # tool_choice="required" before falling back to the corrective-text path.
        # DEFAULT off (sub-agents/tests unchanged); the orchestrator enables it.
        self.constrained_retry = bool(constrained_retry)
        # Feature 3: auto-verify command. When non-empty and a turn wrote/edited
        # files but ran no command, run() executes this once and feeds the result
        # back. Empty = keep the prose build-nudge. DEFAULT empty.
        self.verify_cmd = verify_cmd or ""
        # Feature 4: reviewer gate. When on, a turn that changed code spawns the
        # read-only reviewer sub-agent (via spawn_agent in this agent's registry)
        # before finalizing. DEFAULT off; only the top-level orchestrator (which
        # has spawn_agent) enables it, so sub-agents can never recurse.
        self.review_writes = bool(review_writes)
        # Per-turn latches for Features 3 & 4 (reset at the start of every run()):
        # _auto_verified caps the auto-verify at ONCE per turn; _reviewed_this_turn
        # caps the reviewer gate at ONCE per turn; _wrote_code_this_turn /
        # _changed_files track whether (and which) files a turn changed.
        self._auto_verified = False
        self._reviewed_this_turn = False
        self._wrote_code_this_turn = False
        self._changed_files: list[str] = []
        # PERF read-budget guard: cumulative bytes the context-bloating tools have
        # pulled into context THIS turn, and a per-turn latch so the "stop reading"
        # nudge fires AT MOST ONCE. Both reset at the start of every run() (a new
        # user turn); initialized here so a never-run agent is well-formed.
        self._read_bytes = 0
        self._read_nudge_fired = False
        # Per-turn read-budget threshold (bytes). Configurable via
        # config.read_nudge_bytes; falls back to the module default when not
        # supplied (sub-agents/tests). A non-positive value is coerced to the
        # default so the guard can never be disabled by a bad config.
        self.read_nudge_bytes = (
            int(read_nudge_bytes) if int(read_nudge_bytes) > 0 else _TURN_READ_NUDGE_BYTES
        )
        # Visible nesting marker prepended to this agent's OWN line-oriented
        # output (tool lines + footer). Sub-agents are spawned with a non-empty
        # prefix (e.g. "  ↳ ") so a delegated run's lines are distinguishable
        # from the orchestrator's column-0 lines.
        self.line_prefix = line_prefix
        self.auto_confirm = auto_confirm
        self.max_iterations = max_iterations
        self.confirm_fn = confirm_fn or _default_confirm
        # Registry: defaults to the module REGISTRY but can be overridden so an
        # orchestrator can inject an extra tool (e.g. spawn_agent).
        self.registry = registry if registry is not None else tools_mod.REGISTRY
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # Full detail (name/full args/full result) for the CURRENT user turn,
        # reset at the start of each run(). Revealed on demand via Ctrl+O
        # (render_details). NOTE: orchestrator + sub-agents are SEPARATE Agent
        # instances; the REPL only reveals the ORCHESTRATOR's buffer. Sub-agent
        # (spawn_agent) detail still streams live as nested "↳" lines.
        self.last_turn_details: list[dict] = []
        # Conversation-memory (passive hybrid retrieval). DEFAULTS OFF so
        # sub-agents and every existing caller/test are byte-for-byte unchanged:
        # only the orchestrator (built by the REPL) is handed a real MemoryStore.
        # Memory is ACTIVE only when a store exists AND it is enabled AND the mode
        # is not "off". When active, run() (a) injects the most relevant past
        # records as an ephemeral system note right before the user message and
        # (b) records each completed Q/A turn so later turns can recall it.
        self.memory = memory
        self.recall_mode = recall_mode
        self.memory_top_k = memory_top_k
        self.memory_enabled = memory_enabled
        # Gated LLM-judge reranker for the per-turn memory retrieve. Mirrored
        # onto the live agent by /rerank (self.agent.rerank = ...) so the toggle
        # takes effect this turn without rebuilding the agent. Default False ->
        # the retrieve call site passes rerank=False -> byte-for-byte unchanged.
        self.rerank = bool(rerank)
        self.rerank_candidates = max(1, int(rerank_candidates))
        # ----- WAVE features (all behavior-preserving when unset) ---------
        # Permission mode governs the confirm gate (see _permission_decision):
        #   "default"   -> exactly the historic auto_confirm/confirm_fn behavior;
        #   "read-only"/"plan" -> block write_file/edit_file/run_bash, reads OK;
        #   "auto-edit" -> auto-approve edits, still confirm run_bash;
        #   "full-auto" -> auto-approve everything.
        # DEFAULT "default" so every existing caller/test is byte-for-byte unchanged.
        self.permission_mode = permission_mode or "default"
        # When True, snapshot a file's on-disk bytes right before a write/edit so
        # a bad edit is one /undo away. DEFAULT off — no checkpoint machinery runs.
        self.checkpoints_enabled = bool(checkpoints_enabled)
        # Loaded lifecycle-hook config (llmcode.hooks.load_hooks shape) or None.
        # DEFAULT None -> the pre/post-tool hook calls are complete no-ops.
        self.hooks = hooks or None
        # Optional threading.Event-like with .is_set(); when set mid-run the loop
        # stops cleanly and finalizes. DEFAULT None -> never checked.
        self.cancel_event = cancel_event
        # Confinement anchor for checkpoints/hooks; resolved lazily to os.getcwd()
        # when needed so construction stays side-effect-free. DEFAULT None.
        self.workspace_root = workspace_root
        # Per-launch checkpoint session token: when set, snapshot() isolates this
        # session's file-snapshot checkpoints under a per-session subdir so /undo
        # only ever reverts THIS session's writes. DEFAULT None -> legacy shared
        # layout (fine for one-shot run_once).
        self.checkpoint_session = checkpoint_session
        # MANDATORY tool self-healing (diagnose -> fix -> retry). When True, a
        # FAILED tool execution consults remediation.remediate for a SAFE corrected
        # retry (bounded by auto_fix_max_attempts) BEFORE the model sees the error.
        # DEFAULT off so every existing caller/test is byte-for-byte unchanged; the
        # orchestrator passes True. auto_fix_max_attempts is clamped to 1..5.
        self.auto_fix_tools = bool(auto_fix_tools)
        self.auto_fix_max_attempts = max(1, min(5, int(auto_fix_max_attempts)))
        # Count of successful/attempted auto-fix retries this agent has performed
        # (surfaced in the Ctrl+O details view / return of details). Cumulative.
        self._autofixes = 0
        # MANDATORY thermal cooldown. When True, the iteration loop calls
        # cooldown.maybe_pause() at the TOP of each round so a long turn breaks
        # mid-way (and back-to-back turns accumulate toward the interval). DEFAULT
        # off so tests never pause; the orchestrator + sub-agents pass the config.
        self.cooldown_enabled = bool(cooldown_enabled)
        # Model-maintained checklist state (Feature 6). Empty until todo_write runs.
        self._todos: list[dict] = []
        # Register the model-callable todo_write tool ONLY when opted in, so the
        # default schema/registry (and the 1120-test baseline) is untouched. We
        # copy the registry first so the shared module REGISTRY is never mutated,
        # and append the name to tool_names so _tools_payload (built from
        # tool_names + self.registry) exposes it and its memo key reflects it.
        if todos_enabled:
            todo_tool = tools_mod.Tool(
                name="todo_write",
                description=(
                    "Maintain a visible checklist for a multi-step task. Pass "
                    "items as a list of {\"text\": str, \"status\": "
                    "\"pending\"|\"in_progress\"|\"done\"}. Replaces the current "
                    "list each call; render it whenever it changes."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "done"],
                                    },
                                },
                                "required": ["text"],
                            },
                        }
                    },
                    "required": ["items"],
                },
                fn=self._todo_write_fn,
            )
            self.registry = dict(self.registry)
            self.registry["todo_write"] = todo_tool
            if "todo_write" not in self.tool_names:
                self.tool_names = self.tool_names + ["todo_write"]

    # ----- permission gate ------------------------------------------------
    def _permission_decision(self, tool, args) -> tuple[str, str]:
        """Decide how a tool call proceeds under the active permission mode.

        Returns ``(decision, reason)`` where decision is:
          - "allow": run it (a non-gated tool, or a read under any mode);
          - "auto":  run it WITHOUT the interactive confirm (auto-approved);
          - "block": do NOT run it — ``reason`` explains why (declined/mode).

        "default" preserves the exact historic gate: a requires_confirmation tool
        auto-runs iff auto_confirm, else defers to confirm_fn (a False from which
        blocks with the same "User declined this tool call." message as before).
        """
        name = tool.name
        mode = self.permission_mode
        if mode in ("read-only", "plan"):
            if name in ("write_file", "edit_file", "run_bash"):
                return (
                    "block",
                    f"{mode} mode: {name} disabled — change mode to apply edits.",
                )
            # Delegation would otherwise escape the mode: a spawned sub-agent runs
            # in its own (default) mode and could write/edit/run_bash. Block it so
            # read-only/plan can't be escalated via spawn_agent.
            if name == "spawn_agent":
                return (
                    "block",
                    f"{mode} mode: delegation disabled — "
                    "change mode to make changes.",
                )
            # Best-effort MCP mutation guard: MCP tools (named ``mcp__…``) can also
            # mutate state (create/update/delete/run …), which the 3 built-in
            # blocks above don't cover. Heuristic: block an MCP tool whose final
            # ``__``-segment begins with a known mutation verb; read-style tools
            # (get/list/search/read/fetch/report/analyze) stay allowed. This is a
            # name heuristic only — it can't inspect server-side side effects.
            if name.startswith("mcp__") and _mcp_name_is_mutation(name):
                return (
                    "block",
                    f"{mode} mode: MCP tool '{name}' looks like a mutation "
                    "(create/update/delete/run/…) — disabled; "
                    "change mode to make changes.",
                )
            return ("allow", "")
        if mode == "full-auto":
            return ("auto", "")
        if mode == "auto-edit" and name in ("write_file", "edit_file"):
            return ("auto", "")
        # "default" (and auto-edit for non-edit tools, e.g. run_bash): the
        # historic confirm gate, byte-for-byte.
        if tool.requires_confirmation:
            if self.auto_confirm:
                return ("auto", "")
            if self.confirm_fn(tool, args):
                return ("auto", "")
            return ("block", "User declined this tool call.")
        return ("allow", "")

    # ----- cooperative interrupt (Feature 5) -----------------------------
    def _finalize_cancelled(self, text: str) -> str:
        """Gracefully finalize a turn interrupted via ``cancel_event``.

        Appends a clean assistant message (NO tool_calls — so history has no
        dangling tool_calls awaiting results) carrying the partial answer plus an
        "[interrupted]" note, prints the activity summary, and returns the note.
        """
        body = (text or "").strip()
        note = (body + "\n\n[interrupted]").strip() if body else "[interrupted]"
        self._print_activity_summary()
        self.messages.append({"role": "assistant", "content": note})
        self._account_appended_msg_token_est()
        self._dim("[interrupted]")
        return note

    # ----- model-maintained checklist (Feature 6) ------------------------
    def _todo_write_fn(self, args) -> dict:
        """Store/replace the visible checklist; render it when it changes.

        Tolerates ``{"items": [...]}``, a bare list, and plain-string items
        (coerced to ``{"text": s, "status": "pending"}``). Bad statuses fall back
        to "pending"; empty-text entries are dropped. Returns ``{"ok": True,
        "result": {"count": N}}``.
        """
        items = args.get("items", args) if isinstance(args, dict) else args
        normalized: list[dict] = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, str):
                    text = it.strip()
                    if text:
                        normalized.append({"text": text, "status": "pending"})
                elif isinstance(it, dict):
                    text = str(it.get("text", "")).strip()
                    if not text:
                        continue
                    status = it.get("status", "pending")
                    if status not in ("pending", "in_progress", "done"):
                        status = "pending"
                    normalized.append({"text": text, "status": status})
        changed = normalized != self._todos
        self._todos = normalized
        if changed:
            self._render_todos()
        return {"ok": True, "result": {"count": len(normalized)}}

    def _render_todos(self) -> None:
        """Render the checklist concisely, themed like the rest of the stream."""
        if self.console is None or not self._todos:
            return
        marks = {"done": "[x]", "in_progress": "[~]", "pending": "[ ]"}
        lines = [self.line_prefix + "Todos:"]
        for it in self._todos:
            lines.append(
                f"{self.line_prefix}  {marks.get(it['status'], '[ ]')} {it['text']}"
            )
        self.console.print("\n".join(lines), style="dim")

    # ----- memory helpers ------------------------------------------------
    def _memory_active(self) -> bool:
        """True iff passive retrieval/record-creation should run this turn."""
        return (
            self.memory is not None
            and self.memory_enabled
            and self.recall_mode != "off"
        )

    @staticmethod
    def _render_memory(records: list) -> str:
        """Compactly render retrieved records: numbered, summary-or-text[:240],
        whitespace-collapsed so the whole block stays well under ~600 tokens."""
        lines: list[str] = []
        for i, r in enumerate(records, 1):
            snippet = (getattr(r, "summary", "") or "").strip()
            if not snippet:
                snippet = (getattr(r, "text", "") or "")[:240]
            snippet = " ".join(snippet.split())
            if snippet:
                lines.append(f"{i}. {snippet}")
        return "\n".join(lines)

    @staticmethod
    def _cheap_summary(text: str) -> str:
        """Heuristic gist (NO LLM call): the first sentence, else the first ~160
        chars of the whitespace-collapsed text."""
        t = " ".join((text or "").split())
        if not t:
            return ""
        m = re.search(r"^.*?[.!?](?:\s|$)", t)
        first = m.group(0).strip() if m else t
        if len(first) > 160:
            first = first[:160].rstrip()
        return first

    # ----- output helpers ------------------------------------------------
    def _print_markdown(self, text: str) -> None:
        """Render assistant prose/answer as Markdown for clean terminal output.

        We buffer a turn's text and render it once (rather than streaming raw
        chunks live) so: (a) leading/trailing whitespace the model emits never
        becomes stray blank lines, and (b) headers/bullets/code render properly
        instead of as literal '##' / '`' characters.
        """
        if self.console is None:
            return
        body = text.strip()
        if not body:
            return

        # Tidy Markdown with LEFT-aligned headers + fenced-code highlighting.
        # code_theme reads cleanly per theme — "monokai" (auto), "ansi_dark"
        # (Dark mode), "native" (amber/orange).
        md = build_answer_markdown(body, self.code_theme)

        # Top-level agent on a REAL TERMINAL with an accent => wrap the answer in
        # a thin rounded BOX (the headline look) with a blank line above and below
        # for breathing room, so turns never feel congested. The box is title-less
        # (reference design): a plain bordered frame whose border reads in the
        # theme's accent. The is_terminal gate keeps piped/scripted one-shot output
        # clean PLAIN Markdown — the box glyphs are literal text, so without this
        # guard `llmcode "q" > out.txt` would interleave border chars into the answer.
        # Nested sub-agents (non-empty line_prefix) and accent-less agents always
        # render plain Markdown so the box never collides with the "↳" marker and
        # old (byte-clean) behaviour holds.
        is_tty = bool(getattr(self.console, "is_terminal", False))
        if self.accent and not self.line_prefix and is_tty:
            from rich import box as _box
            from rich.panel import Panel

            # Answer box follows the active theme's box_style so it matches the
            # banner (ember→HEAVY, neon→DOUBLE, frost→SIMPLE, …). Inlined getattr
            # (NOT repl._box_for, which would be a circular import); a palette-less
            # agent falls back to ROUNDED — byte-identical to the old behaviour.
            _bx = getattr(
                _box,
                (getattr(self.palette, "box_style", "ROUNDED") or "ROUNDED").upper(),
                _box.ROUNDED,
            ) if self.palette else _box.ROUNDED
            self.console.print()  # breathing room above the answer box
            self.console.print(
                Panel(
                    md,
                    box=_bx,
                    border_style=self.accent,
                    padding=(0, 1),
                )
            )
            self.console.print()  # breathing room below the answer box
        else:
            self.console.print(md)

    def _info(self, msg: str) -> None:
        if self.console is not None:
            self.console.print(msg)

    def _dim(self, msg: str) -> None:
        """Print a dim, line-prefixed status notice consistent with footers."""
        if self.console is not None:
            self.console.print(self.line_prefix + msg, style="dim")

    def _print_footer(self, assistant_text: str, output_tokens, elapsed: float) -> None:
        """Print the dim ``Model | Time | Speed`` footer for a visible answer.

        Matches the reference design: ``Model: <model> | Time: <s>s | Speed:
        <rate> tok/s``. The model name comes from the provider this agent runs on
        (``self.provider.model``); ``format_footer`` still yields the ``X tok/s``
        speed fragment, which is composed into the full line here."""
        # Compute the stats FIRST (before any early return) so the REPL's pinned
        # status bar always mirrors what the footer would show — from the SAME
        # values, never recomputed differently. Stored even when the line itself
        # is suppressed or skipped (empty answer / console-less agent).
        _tokens, rate = compute_tok_stats(assistant_text, output_tokens, elapsed)
        model = getattr(self.provider, "model", "") or ""
        self.last_turn_stats = {
            "model": model,
            "elapsed": float(elapsed),
            "toks_per_sec": float(rate),
            "output_tokens": int(_tokens),
        }
        # suppress_footer: the combined REPL shows these stats in its bottom bar,
        # so skip the printed line (stats are already stored above). The historic
        # guards (no console / empty answer) still suppress the line as before —
        # so DEFAULT (suppress_footer=False) output is byte-for-byte unchanged.
        if self.suppress_footer or self.console is None or not assistant_text.strip():
            return
        speed = format_footer(rate)  # "47.9 tok/s"
        elapsed_s = f"{elapsed:.2f}s"
        # Middle-dot separator matches the pinned status bar so the two read as
        # the same object, but "·" (U+00B7) mojibakes to "?" on a non-UTF-8
        # console (e.g. LANG=C piped output) — fall back to "|" there, same
        # care as the spinner's _SEP/_SEP_ASCII pair.
        sep = " · " if self._console_can_encode("·") else " | "
        if self.accent:
            # Labels + separators stay muted; the values (model, time, rate) read
            # in the accent colour — a lively footer instead of a uniform grey line.
            # NOTE no base style on the Text (a "dim" base would layer onto the
            # accent spans and mute them); each span carries its own style. muted =
            # the theme's grey when a palette is threaded, else "dim" (palette-less
            # sub-agents / tests stay byte-identical).
            from rich.text import Text

            muted = self.palette.dim if self.palette else "dim"
            line = Text(self.line_prefix)
            if model:  # omit the Model segment entirely when unknown (e.g. mock)
                line.append("Model: ", style=muted)
                line.append(model, style=self.accent)
                line.append(sep, style=muted)
            line.append("Time: ", style=muted)
            line.append(elapsed_s, style=self.accent)
            line.append(sep, style=muted)
            line.append("Speed: ", style=muted)
            num, _, unit = speed.partition(" ")
            line.append(num, style=self.accent)
            if unit:
                line.append(" " + unit, style=muted)
            self.console.print(line)
        else:
            # Plain (piped/non-accent) fallback: same text, no styling — byte-clean.
            model_seg = f"Model: {model}{sep}" if model else ""
            plain = f"{model_seg}Time: {elapsed_s}{sep}Speed: {speed}"
            self.console.print(self.line_prefix + plain, style="dim")

    def _console_can_encode(self, s: str) -> bool:
        """True when the console's output encoding can represent ``s``.

        Guards the ⏺/⎿/✓ glyphs against mojibake on a legacy-encoded terminal or
        an ASCII pipe (e.g. LANG=C, where the encoding is ascii). A UTF-8 console
        — which every test's StringIO/capsys is — always returns True, so gating
        on encoding-capability leaves the piped-test output byte-for-byte
        unchanged while still falling back to ASCII where the glyph truly can't
        render."""
        if self.console is None:
            return False
        enc = getattr(self.console, "encoding", None) or "utf-8"
        try:
            s.encode(enc)
            return True
        except (LookupError, UnicodeEncodeError):
            return False

    def _tree_glyphs(self) -> tuple[str, str]:
        """Return the ``(head, connector)`` glyphs for the tool tree.

        The Claude-Code ⏺/⎿ glyphs when the console can encode them, else ASCII
        ("* " for ⏺, "  L  " for ⎿) so dumb terminals / pipes never show mojibake.
        Gated on encoding-capability rather than ``is_terminal`` on purpose:
        force_terminal=False test consoles (still UTF-8) must keep the glyphs, so
        this preserves the byte-identical piped-test output while remaining a
        correct mojibake guard."""
        if self._console_can_encode("⏺⎿"):
            return "⏺ ", "  ⎿  "
        return "* ", "  L  "

    def _render_tool_tree(
        self, name: str, args: dict, connector: str, is_error: bool
    ) -> None:
        """Render ONE Claude-Code-style two-line tool tree.

        Line 1: a GREEN "⏺" glyph then the default-color "DisplayName(args)".
        Line 2: a DIM "  ⎿  " connector + a short result/error summary. ``connector``
        is the pre-built summary text (already chosen by the caller, since it must
        match the executed/declined/failed outcome). ``is_error`` is the REAL
        outcome the caller already knows (parse error / unknown tool / declined /
        result.ok is not True) — it is passed in rather than re-derived from the
        connector text. Re-deriving via ``startswith("Error:")`` produced false
        positives: a SUCCESSFUL run_bash whose stdout first line begins with
        "Error:" (common — many programs print "Error: ..." to stdout on a 0
        exit) would render dim-red as if it had failed. When ``is_error`` is
        True the connector renders DIM RED; otherwise DIM grey.

        Built as rich ``Text`` (spans + styles), never markup tags, so it is safe
        under the console's markup=False and leaves no ANSI residue when piped.
        """
        if self.console is None:
            return
        from rich.text import Text

        # Semantic tokens from the theme Palette when threaded in; else the
        # historic literals so a palette-less agent (sub-agent built directly /
        # tests) stays byte-for-byte identical. The head glyph matches the
        # collapsed activity line: accent on success, error-red when this call
        # failed (fixes the old always-green head vs red collapsed inconsistency).
        pal = self.palette
        head_style = (pal.error if is_error else pal.accent) if pal else "green"
        check_style = pal.success if pal else "dim green"
        conn_style = pal.dim if pal else "dim"
        summary_style = (
            (pal.error if is_error else pal.dim) if pal
            else ("dim red" if is_error else "dim")
        )

        head_glyph, conn_glyph = self._tree_glyphs()
        head = Text(self.line_prefix)
        head.append(head_glyph, style=head_style)
        head.append(tool_call_label(name, args))
        # A SUCCESSFUL call gets a "✓" (success token) on the head line. TTY-gated
        # (a real terminal only) so piped/scripted output — every test drives a
        # non-terminal console — stays byte-for-byte identical. Failures keep the
        # existing "✗ ..." connector treatment below and get NO ✓.
        if (
            not is_error
            and bool(getattr(self.console, "is_terminal", False))
            and self._console_can_encode("✓")
        ):
            head.append(" ✓", style=check_style)
        self.console.print(head, no_wrap=True, overflow="ellipsis")

        body = Text(self.line_prefix)
        body.append(conn_glyph, style=conn_style)
        body.append(connector, style=summary_style)
        self.console.print(body, no_wrap=True, overflow="ellipsis")

    def _record_detail(self, name, args, result, ok, elapsed) -> None:
        """Store full detail for the current turn.

        Detail is printed through a SINGLE channel: the explicit Ctrl+O reveal
        (``render_details``). We deliberately do NOT print live here, so each
        detail is shown exactly once (on reveal) and never duplicated.
        """
        self.last_turn_details.append({
            "name": name,
            "args": args,
            "result": result,
            "ok": bool(ok),
            "elapsed": elapsed,
        })

    def _print_activity_summary(self) -> None:
        """Reactor tool summary: a persistent, outcome-at-a-glance one-liner that
        always conveys pass/fail counts, plus an auto-expanded tree so work is
        never hidden behind Ctrl+O.

        One-liner: ``◆ N tools · ✓x ✗y · <first failure reason>`` — ``◆`` in the
        accent, ``✓x`` in the success token, ``✗y`` + the first failure reason in
        the error token, and a dim ``ctrl-o details`` hint. The full ⏺/⎿ tree is
        then rendered inline (``render_details``) UNLESS a large batch (>5) all
        succeeded — i.e. it auto-expands whenever any call failed OR the batch is
        modest (≤5), and collapses to just the one-liner only for a big all-green
        batch. Auto-expand + the hint are orchestrator-only (empty ``line_prefix``,
        matching the Ctrl+O reveal, which only the orchestrator's buffer honours);
        sub-agents keep the compact counts line so they never flood the parent.

        Built as rich ``Text`` (spans, never markup) and TTY-gated via the console:
        on a non-terminal (piped / sub-agent) rich strips styling, so the output
        stays byte-clean and ANSI-free. ``◆``/``✓``/``✗`` degrade to ``*``/``v``/``x``
        on a legacy-encoded console."""
        if self.console is None or not self.last_turn_details:
            return
        details = self.last_turn_details
        n = len(details)
        failed = sum(1 for r in details if r.get("ok") is not True)
        ok = n - failed

        # First failure's reason for the one-liner tail, e.g. "run_bash exit 1".
        reason = ""
        if failed:
            for r in details:
                if r.get("ok") is not True:
                    reason = (
                        f"{r.get('name', '')} {error_summary(r.get('result'))}".strip()
                    )
                    break

        # Signature glyphs; each degrades to ASCII on a console that can't encode
        # it (mirrors _tree_glyphs / the banner), keeping piped/LANG=C byte-clean.
        core = "◆" if self._console_can_encode("◆") else "*"
        ok_g = "✓" if self._console_can_encode("✓") else "v"
        fail_g = "✗" if self._console_can_encode("✗") else "x"

        label = f"{n} tool{'s' if n != 1 else ''}"
        hint = "" if self.line_prefix else " · ctrl-o details"

        if self.accent:
            # Semantic tokens from the theme Palette when threaded, else the
            # historic literals so a palette-less agent (sub-agents / tests) stays
            # byte-identical. The ◆ core is ALWAYS the accent now — pass/fail reads
            # from the ✓x/✗y counts, not from re-tinting the glyph.
            from rich.text import Text

            pal = self.palette
            muted = pal.dim if pal else "dim"
            succ = pal.success if pal else "green"
            err = pal.error if pal else "red"

            line = Text(self.line_prefix)
            line.append(f"{core} ", style=self.accent)
            line.append(label, style=muted)
            line.append(" · ", style=muted)
            line.append(f"{ok_g}{ok}", style=succ)
            if failed:
                line.append(f" {fail_g}{failed}", style=err)
                if reason:
                    line.append(" · ", style=muted)
                    line.append(reason, style=err)
            if hint:
                line.append(hint, style=muted)
            self.console.print(line)
        else:
            tail = f" {fail_g}{failed}" if failed else ""
            reason_txt = f" · {reason}" if (failed and reason) else ""
            self.console.print(
                self.line_prefix
                + f"{core} {label} · {ok_g}{ok}{tail}{reason_txt}{hint}",
                style="dim",
            )

        # Auto-expand: never hide the work. Render the full ⏺/⎿ tree inline unless
        # a large all-green batch (>5 tools, no failures) — so any failure OR a
        # modest batch (≤5) always shows its detail. Orchestrator-only: sub-agent
        # trees would flood the parent, and only the orchestrator buffer is
        # Ctrl+O-revealable, so their summary stays the compact counts line.
        if not self.line_prefix and not (n > 5 and failed == 0):
            self.render_details(self.console)

    def render_details(self, console) -> None:
        """Expand the collapsed activity line into the full ⏺/⎿ tool tree."""
        if console is None:
            return
        if not self.last_turn_details:
            console.print("(no tool activity in the last turn)", style="dim")
            return
        for r in self.last_turn_details:
            name, args, result = r["name"], r["args"], r["result"]
            if r.get("ok") is True:
                conn = result_summary(name, result) or "done"
                if name == "spawn_agent" and isinstance(args, dict):
                    role = str(args.get("role", "")).strip()
                    if role:
                        conn = f"{role} done"
                self._render_tool_tree(name, args, conn, False)
            else:
                self._render_tool_tree(name, args, "✗ " + error_summary(result), True)

    # ----- tool schema ---------------------------------------------------
    def _tools_payload(self) -> list[dict] | None:
        # Build the schema from the SAME registry dispatch uses (self.registry),
        # not the global import-time REGISTRY. Otherwise MCP tools and spawn_agent
        # (which live only in the injected registry) are silently dropped from the
        # schema and the model never learns they exist.
        if not self.tool_names:
            return None
        # Memoize on the tool-name tuple: the schema depends ONLY on the active
        # tool names + their (per-instance fixed) registry definitions, so rebuild
        # only when that key changes. Preserves order so a reordering also
        # rebuilds. Avoids re-serializing the whole schema every turn.
        key = tuple(self.tool_names)
        if self._tools_payload_key != key:
            schema = tools_mod.openai_schema(self.tool_names, registry=self.registry)
            self._tools_payload_cache = schema or None
            self._tools_payload_key = key
        return self._tools_payload_cache

    def _get_tool(self, name: str) -> tools_mod.Tool | None:
        return self.registry.get(name)

    # ----- per-turn read-budget guard ------------------------------------
    def _account_tool_read(self, tool_name: str, result_text: str) -> str:
        """Track CODE/output bytes pulled into context by the bloating tools and
        append a ONE-TIME nudge once the per-turn budget is crossed (CHANGE 1).

        Only the context-bloating tools (``_BLOATING_TOOL_NAMES`` plus the MCP
        read tools matched by ``_BLOATING_TOOL_PATTERNS``) are counted; every
        other tool result passes through unchanged. The nudge is appended to the
        crossing call's result text (so the model sees it inline) and fires at
        most once per turn (``_read_nudge_fired``). Returns the (possibly
        nudge-appended) result text.
        """
        if not _is_bloating_tool(tool_name):
            return result_text
        self._read_bytes += len(result_text)
        threshold = getattr(self, "read_nudge_bytes", _TURN_READ_NUDGE_BYTES)
        if self._read_bytes >= threshold and not self._read_nudge_fired:
            self._read_nudge_fired = True
            kb = self._read_bytes // 1000
            # Only point at spawn_agent when THIS agent actually has it (the
            # orchestrator). Sub-agents (explorer/coder/reviewer) lack it, so for
            # them the right move is simply to stop reading and answer.
            if self.registry.get("spawn_agent") is not None:
                advice = (
                    "STOP reading more files — instead delegate broad exploration "
                    'to a read-only explorer sub-agent via spawn_agent(role='
                    '"explorer") (it reads in its own context), or answer now with '
                    "what you have."
                )
            else:
                advice = "STOP reading more files — answer now with what you have."
            return result_text + (
                f"\n\n[context-budget] You've pulled ~{kb}KB of code into context "
                "this turn. Every later token is now slower on local models. "
                + advice
            )
        return result_text

    # ----- end-of-turn gates (auto-verify + reviewer) --------------------
    def _auto_verify_and_feed_back(self) -> None:
        """Feature 3: auto-run ``self.verify_cmd`` via run_bash and feed it back.

        Uses the SAME run_bash machinery (so the configured output truncation that
        keeps failure TAILS applies), records the call for Ctrl+O, prints a dim
        notice, and appends the result as an ephemeral (`_nudge`) user message so
        the model can fix any failures before finalizing. Best-effort: a missing
        run_bash tool or a raising fn never breaks the turn.
        """
        tool = self._get_tool("run_bash")
        if tool is None:
            return
        self._dim(f"[auto-verify] running: {self.verify_cmd}")
        t0 = time.perf_counter()
        try:
            result = tool.fn({"command": self.verify_cmd})
        except Exception as exc:  # noqa: BLE001 - verification must not crash the loop
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        dt = time.perf_counter() - t0
        self._record_detail("run_bash", {"command": self.verify_cmd}, result,
                            result.get("ok"), dt)
        # Build a compact transcript from the (already-truncated) payload.
        if result.get("ok") is True:
            payload = result.get("result") or {}
            code = payload.get("exit_code", 0)
            body = (payload.get("stdout") or "") + (payload.get("stderr") or "")
            outcome = f"PASSED (exit {code})"
        else:
            payload = result.get("result")
            if isinstance(payload, dict):
                code = payload.get("exit_code", "?")
                body = (payload.get("stdout") or "") + (payload.get("stderr") or "")
            else:
                code = "?"
                body = str(result.get("error") or "")
            outcome = f"FAILED (exit {code})"
        self.messages.append({
            "role": "user",
            "_nudge": True,
            "content": (
                f"I auto-ran your verification command after your file changes:\n"
                f"  $ {self.verify_cmd}\n"
                f"Result: {outcome}\n"
                f"Output:\n{body}\n\n"
                "If it FAILED, fix the issues (edit files) and continue; if it "
                "PASSED, give your final answer."
            ),
        })

    def _review_changes_and_feed_back(self, user_text: str) -> bool:
        """Feature 4: spawn the read-only reviewer on the turn's changed files and
        feed its findings back. Returns True when findings were injected (caller
        should loop again), False when the reviewer could not run (finalize as-is).
        """
        spawn = self.registry.get("spawn_agent")
        if spawn is None:
            return False
        files = ", ".join(self._changed_files) if self._changed_files else "(unknown)"
        task = (
            "Review the code change just made in this project for correctness, "
            "bugs, edge cases, security issues, and deviations from the codebase's "
            "conventions. Read the changed file(s) and report concrete, prioritized "
            "findings (issue, where, suggested fix); if it looks good, say so "
            f"plainly.\n\nChanged file(s): {files}\n"
            f"The change was made to accomplish: {user_text}"
        )
        try:
            result = spawn.fn({"role": "reviewer", "task": task})
        except Exception as exc:  # noqa: BLE001 - review must not crash the loop
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._record_detail("spawn_agent", {"role": "reviewer", "task": task},
                            result, result.get("ok"), 0.0)
        if result.get("ok") is not True:
            return False
        findings = str(result.get("result") or "").strip()
        if not findings:
            return False
        self.messages.append({
            "role": "user",
            "_nudge": True,
            "content": (
                "A code reviewer examined your changes and reported:\n\n"
                f"{findings}\n\n"
                "Address any REAL issues (edit the files), then give your final "
                "answer. If a finding is not a real problem, briefly say why and "
                "finalize — do not re-spawn a reviewer."
            ),
        })
        return True

    # ----- one provider call ---------------------------------------------
    def _stream_turn(
        self, req_messages: list[dict], tools_payload: list[dict] | None,
        tool_choice: str | None = None, recovery: bool = False,
    ) -> tuple[str, list[dict], object, str, float]:
        """Run ONE provider call, consuming the normalized event stream.

        Returns ``(assistant_text, tool_calls, output_tokens, finish_reason,
        elapsed)``. Owns the duck spinner + deterministic stream close + the
        generation-time measurement, so both the primary call and the Feature-2
        constrained-decode retry share identical behavior. ``tool_choice`` is
        forwarded to the provider ONLY when set, so the default path keeps the
        2-arg ``stream_chat`` call shape every scripted test provider relies on.

        ``recovery`` marks an empty-turn recovery generation: for THIS call only,
        the provider's gentle token cap is bypassed (so a long think can't
        truncate the answer into emptiness) and thinking is disabled
        (effort="off", so the model emits CONTENT not more reasoning). Both
        changes are made on the shared provider instance and RESTORED in the
        finally, so normal turns stay gentle-capped. NOTE: we deliberately do NOT
        globally exempt reasoning models from the gentle cap — only this
        empty-turn recovery path is exempted; flip that here if ever desired.
        """
        text_acc: list[str] = []
        tool_calls: list[dict] = []
        output_tokens = None
        finish_reason = "stop"

        # Timing: GENERATION speed only — measured from the FIRST generated token
        # to the last. The time before the first token (prompt processing /
        # time-to-first-token) is deliberately EXCLUDED, so the footer reflects the
        # model's pure decode throughput, not request latency.
        t0 = time.perf_counter()
        t_first = None
        gen_elapsed = None
        # Ant scurries during the (often long) provider think/generate wait.
        # TTY-only + self-erasing; disabled when the console is None or not a real
        # terminal, so non-TTY runs are byte-for-byte unchanged. STOPPED below
        # before ANY console write.
        spinner = Spinner(
            self.console,
            color=self.palette.spinner if self.palette else None,
            timer_color=self.palette.spinner_timer if self.palette else None,
            # Reactor pulse: ◆ core breathes color between spinner (valley) and the
            # brightest accent (peak); verb starts "thinking" (awaiting the model)
            # and flips to "forging" on the first token; set_rate feeds live tok/s.
            pulse_to=_reactor_pulse_to(self.palette),
            verb="thinking",
        )
        # Running character tally for the live tok/s estimate (chars/4, matching
        # tokens.estimate_text_tokens). Kept as a single int incremented per chunk
        # so the rate snapshot is O(1) per event (no O(n^2) re-sum of text_acc).
        chars_acc = 0
        # Hold the provider iterator so the finally can close() it: on a break
        # (done) or KeyboardInterrupt the suspended generator is finalized here,
        # which runs the provider's own finally and deterministically releases its
        # HTTP stream/socket (finding #3).
        stream_iter = None
        # RECOVERY GENERATION setup: temporarily disable the gentle cap and
        # thinking on the shared provider for this one call. Saved with a sentinel
        # (so a legitimate False/"" original is restored correctly) and restored in
        # the finally. hasattr-guarded so scripted test providers without these
        # attributes are untouched (and the default recovery=False path is a
        # complete no-op — byte-identical to before).
        _missing = object()
        prov = self.provider
        saved_gentle = _missing
        saved_effort = _missing
        # start() is INSIDE the try so a KeyboardInterrupt delivered between start()
        # and the loop can never leave the duck running with the cursor hidden
        # (finding #11): the finally always calls stop(). The recovery state-swap is
        # ALSO inside the try so the finally restore is airtight even if an
        # assignment raises.
        try:
            if recovery:
                if hasattr(prov, "gentle_mode"):
                    saved_gentle = prov.gentle_mode
                    prov.gentle_mode = False
                if hasattr(prov, "effort"):
                    saved_effort = prov.effort
                    prov.effort = "off"
            spinner.start()
            # Forward tool_choice ONLY when set: the default keeps the historic
            # 2-arg stream_chat(messages, tools) call shape that every scripted
            # MockProvider subclass in the tests overrides with a 2-arg signature.
            if tool_choice is not None:
                stream_iter = self.provider.stream_chat(
                    req_messages, tools_payload, tool_choice=tool_choice
                )
            else:
                stream_iter = self.provider.stream_chat(req_messages, tools_payload)
            for event in stream_iter:
                # Cooperative interrupt (Feature 5): if the caller signalled a
                # cancel mid-stream, stop cleanly and return the text accumulated
                # so far with finish_reason="cancelled". No-op when no event is
                # wired (the default) — byte-for-byte the old loop.
                if self.cancel_event is not None and self.cancel_event.is_set():
                    finish_reason = "cancelled"
                    break
                etype = event.get("type")
                if etype == "text":
                    chunk = event.get("text", "")
                    if chunk and t_first is None:
                        t_first = time.perf_counter()
                        # First token in: the model is now generating, not waiting.
                        spinner.set_verb("forging")
                    text_acc.append(chunk)  # buffered; rendered once at turn end
                    # Live tok/s: publish a thread-safe snapshot (est tokens / gen
                    # window) for the reactor frame. Best-effort — a rough chars/4
                    # estimate that ticks up as your GPU decodes; the footer's real
                    # provider token count stays authoritative. Gated so tiny/early
                    # turns never flash a bogus number (set_rate(None) hides it).
                    if chunk and t_first is not None:
                        chars_acc += len(chunk)
                        _gen = time.perf_counter() - t_first
                        _toks = chars_acc // 4
                        if _gen > 0.2 and _toks > 0:
                            spinner.set_rate(_toks / _gen)
                elif etype == "tool_call":
                    if t_first is None:
                        t_first = time.perf_counter()
                        spinner.set_verb("forging")
                    tool_calls.append(event)
                elif etype == "done":
                    output_tokens = event.get("output_tokens")
                    gen_elapsed = event.get("gen_elapsed")
                    finish_reason = event.get("finish_reason", "stop")
                    break
        finally:
            # Stop+erase the duck the instant the stream ends, BEFORE the buffered
            # answer / narration / footer / collapsed tool line is rendered.
            spinner.stop()
            # Close the (possibly suspended) provider generator so its own finally
            # runs and releases the HTTP stream even on break/Ctrl+C.
            if stream_iter is not None:
                close = getattr(stream_iter, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:  # noqa: BLE001 - teardown must not raise
                        pass
            # Restore the provider's gentle/effort state after a recovery call so
            # subsequent normal turns stay gentle-capped (and at the user's effort).
            if saved_gentle is not _missing:
                prov.gentle_mode = saved_gentle
            if saved_effort is not _missing:
                prov.effort = saved_effort
        # Prefer the provider's generation time: it times from the first token of
        # ANY kind (incl. reasoning tokens the agent never sees as text), so its
        # window matches the completion_tokens count and the server's tok/s. Fall
        # back to the agent's first-visible-token -> done, else total time.
        if gen_elapsed is not None:
            elapsed = gen_elapsed
        elif t_first is not None:
            elapsed = time.perf_counter() - t_first
        else:
            elapsed = time.perf_counter() - t0
        return "".join(text_acc), tool_calls, output_tokens, finish_reason, elapsed

    # ----- the loop ------------------------------------------------------
    def run(self, user_text: str, images: list | None = None) -> str:
        # APPEND-ONLY HISTORY: the user message is the only thing added here. The
        # recalled-memory note is NEVER inserted into self.messages (see below), so
        # there is nothing to strip and the history prefix stays byte-stable turn
        # over turn (PERF-1: keeps the server's cache_prompt KV prefix reusable).
        #
        # MULTIMODAL: ``images`` is an optional list of already-encoded vision
        # parts (see images.encode_image). ONLY when images are present do we
        # build ``content`` as the OpenAI list form [text part, ...image parts];
        # the overwhelmingly common text-only turn keeps ``content`` a plain
        # STRING exactly as before, so the cached history prefix stays byte-stable
        # (PERF-1). ``user_text`` (the plain string) is still what feeds budget /
        # retrieval / memory below — base64 blobs never leak into those paths.
        if images:
            content = [{"type": "text", "text": user_text}] + list(images)
            self.messages.append({"role": "user", "content": content})
        else:
            self.messages.append({"role": "user", "content": user_text})
        # Incremental token estimate (Fix 2): add the just-appended user message
        # to the running cache (no-op if it was dirtied by the trim below).
        self._account_appended_msg_token_est()
        # Track multimodal turns so a model/server that rejects image input can be
        # given an actionable "load a vision model" hint in the error branch below.
        self._sent_images_this_turn = bool(images)
        # Reset the per-user-turn detail buffer (reveal shows the latest turn).
        self.last_turn_details = []
        # Reset the per-turn read-budget accounting: a fresh turn starts with an
        # empty code-read tally and an un-fired nudge latch (CHANGE 1).
        self._read_bytes = 0
        self._read_nudge_fired = False
        # Reset the per-turn auto-verify / reviewer-gate latches (Features 3 & 4):
        # each may fire at most ONCE per user turn, and write-tracking starts empty.
        self._auto_verified = False
        self._reviewed_this_turn = False
        self._wrote_code_this_turn = False
        self._changed_files = []
        # Classify the message ONCE (cheap, deterministic) and reuse it to size
        # the budget, gate per-turn memory retrieval, and hint the model — so a
        # trivial "hi" doesn't drag heavy machinery the message never needed.
        turn_intent = classify_request(user_text)
        # Adaptive context: size THIS turn's working budget to the request, so the
        # live history is trimmed to fit (older turns -> a running summary) and
        # decode stays fast.
        if self.context_budget > 0:
            new_budget = self._compute_turn_budget(user_text)
            if new_budget != self.context_soft_limit:
                # A DIFFERENT per-turn budget invalidates the anti-thrash floor,
                # which is an absolute est-token value recorded under the previous
                # budget; without this a small turn after a big one would skip a
                # genuinely-needed compaction (the floor is still set high).
                self._compact_floor = 0
                self.context_soft_limit = new_budget
        elif self.context_ceiling > 0:
            # Budget OFF (/context off) but a ceiling exists: keep the near-window
            # safety valve so a long session can't overflow the model window (the
            # documented behaviour). Sub-agents/tests have ceiling 0 -> unchanged.
            self.context_soft_limit = self.context_ceiling
        # CONTEXT HYGIENE (runs EVERY turn, before retrieval / the loop): shrink
        # stale tool output + written-file content from earlier COMPLETED turns so
        # this turn only carries the genuinely-needed context. This is the primary
        # defense against the "tiny question, 50k-token prefill, minutes of wait"
        # failure — tool results are 50%+ of a long engineering history and were
        # re-sent verbatim every turn. The current turn's own tools stay full.
        self._trim_stale_tool_outputs()
        # The trim rewrites tool-result content + tool-call argument strings
        # IN PLACE; when it does, the incremental token-estimate cache (Fix 2) is
        # stale and must be invalidated. But on the COMMON steady-state turn there
        # is nothing new to trim (everything past the last user turn already
        # carries the _TRIM_MARKER), so the trim mutates nothing — invalidating
        # then would force a needless O(n) full re-sum of the whole history on the
        # next _maybe_auto_compact(). Only invalidate when bytes actually changed.
        if self._last_trim_mutated:
            self._invalidate_token_est()
        # PASSIVE RETRIEVAL: when memory is active and the store has records, find
        # the most relevant past records for THIS query ONCE per turn (retrieval is
        # expensive — never inside the loop) and render them into ONE ephemeral
        # user-role note. CRITICAL (PERF-1 + AGENT-3): this note is kept OUT of
        # self.messages and is appended only to the FIRST request of the turn
        # (iteration 0) as the FINAL element. That way
        #   - self.messages stays append-only, so its prefix is byte-stable and the
        #     server's cache_prompt KV reuse keeps paying off (a mid-history
        #     insert+strip every turn would backfill the slot and break the cache);
        #     the note rides the naturally-uncached request tail at ~zero cost,
        #   - it is NEVER persisted (not in history -> session-save can't leak it),
        #   - _maybe_auto_compact() can never strip/summarize it (it isn't there).
        # The note carries role "user" (not "system") and is sent ONLY on
        # iteration 0 — see the req_messages build in the loop below for WHY
        # (portability across strict chat templates + tokens + KV-cache).
        # ANY retrieval / embedding failure is swallowed (never breaks a turn).
        mem_block: dict | None = None
        # SKIP retrieval entirely for a trivial/meta message ("hi", "thanks"):
        # it needs no past project records, and retrieval can escalate to an
        # embedding call — pure heavy work the message never needed. The model can
        # still recall via tools if a later message actually requires it.
        if self._memory_active() and self.memory.records and turn_intent != "trivial":
            try:
                records = self.memory.retrieve(
                    user_text, provider=self.provider,
                    mode=self.recall_mode, top_k=self.memory_top_k,
                    rerank=self.rerank, rerank_candidates=self.rerank_candidates,
                )
            except Exception:  # noqa: BLE001 - retrieval must never break a turn
                records = []
            rendered = self._render_memory(records) if records else ""
            if rendered:
                mem_block = {
                    "role": "user",
                    "content": (
                        "RELEVANT MEMORY (from earlier in this project; use if "
                        "helpful):\n" + rendered
                    ),
                }
        # For a trivial message, nudge the model to answer directly instead of
        # "grounding" via project-context / session tools (which a system prompt or
        # MCP server may otherwise push it to do on EVERY turn) — that tool-round
        # detour is the heaviest part of a "hi". Rides the same iteration-0 tail as
        # mem_block (out of self.messages, cache-safe, never persisted).
        hint_block: dict | None = None
        if turn_intent == "trivial":
            hint_block = {
                "role": "user",
                "content": (
                    "[Note: this is a brief conversational message. Reply directly "
                    "and concisely. Do NOT call tools or load project context unless "
                    "the message clearly requires it.]"
                ),
            }
        tools_payload = self._tools_payload()
        final_text = ""
        # Abort early if the model gets stuck re-emitting malformed JSON args,
        # instead of silently burning the whole iteration budget doing nothing.
        consecutive_parse_errors = 0
        # Auto-nudge: a heavy reasoning model can spend a whole turn thinking and
        # emit NO visible answer. Rather than give up, re-prompt it to write the
        # answer — that recovers it far more often than the empty sentinel. Up to
        # TWO shots (two recovery generations), each run UNCAPPED + thinking-OFF
        # via _stream_turn(recovery=True) so the model has room to emit CONTENT.
        nudge_count = 0
        max_nudges = 2
        # One-shot: if a would-be final answer is actually unexecuted tool-call
        # markup, re-prompt ONCE to issue it as a real call (bounded so a model
        # that keeps emitting markup can't loop forever).
        tool_markup_nudged = False
        # When the previous iteration fired an empty-turn nudge, the NEXT provider
        # call is a recovery generation: it bypasses the gentle token cap and
        # disables thinking (see _stream_turn). Consumed (reset to False) right
        # after that call so only the immediately-following generation is special.
        recovery_next = False
        # Duplicate-tool-call loop guard: a model stuck re-issuing the SAME tool
        # batch (same names + same args) turn after turn burns the whole iteration
        # budget doing nothing useful. Break early once the same batch repeats
        # N times in a row. A batch that DIFFERS (different args, a progressed
        # read_file offset, a new file) resets the count — only an exact-repeat
        # loop trips this. See _tool_call_batch_sig for the signature rule.
        _dup_batch_threshold = 4
        _last_batch_sig: tuple | None = None
        _consecutive_dup_batches = 0

        for _iter in range(self.max_iterations):
            # Cooperative interrupt (Feature 5): checked at the TOP of every
            # iteration — BEFORE any assistant(tool_calls) message is appended —
            # so a cancel can never leave a dangling tool_calls message without
            # its matching tool results. Finalizes the partial answer gracefully.
            # No-op (never even evaluated past the None guard) by default.
            if self.cancel_event is not None and self.cancel_event.is_set():
                return self._finalize_cancelled(final_text)

            # MANDATORY thermal cooldown: at a safe checkpoint (top of the loop,
            # before any assistant/tool message is appended) take a short break if
            # an interval of continuous work has elapsed. A cheap no-op when not
            # due; only fires when this agent was built with cooldown_enabled. The
            # process-global pacer seeds its baseline on the first call (no pause).
            if self.cooldown_enabled:
                cooldown.maybe_pause(notify=self._dim)

            # Auto-compaction safety valve (findings #4/#26): before each provider
            # call, if the rough history estimate exceeds the soft budget, compact
            # earlier turns so the prompt prefix can't grow past the model's
            # context window mid-session. A compact failure (provider error) is
            # swallowed — better to send a too-long prompt than to abort the turn.
            self._maybe_auto_compact()

            # Bounds the truncation-recovery retry to ONCE per iteration so a
            # generation that keeps hitting the REAL context window can't loop.
            truncation_retried = False

            # Per-request payload: on the FIRST provider call of the turn
            # (iteration 0) append the ephemeral memory note as the FINAL element;
            # on tool-react iterations (1+) send history unchanged. WHY:
            #   - role "user" (set above), not "system": a trailing user-role
            #     context block is universally accepted by OpenAI-compatible /
            #     llama.cpp chat templates in ANY position. A system-role block
            #     landing AFTER assistant(tool_calls)+tool messages on a
            #     multi-iteration turn can be rejected by strict templates.
            #   - iteration-0-only: the model only needs the recall when it first
            #     reads the user's question; on iteration 0 the request is
            #     [system, ...history, user(actual), user(mem)] with no tool
            #     messages yet — no role-ordering problem. Re-sending it every
            #     iteration just wastes tokens (and would break the request tail's
            #     cacheability on the tool-react rounds).
            #   - self.messages is never mutated, so its prefix stays cache-stable
            #     and the note is never persisted.
            # Iteration-0 extras (memory recall + the trivial-message hint) ride
            # the uncached request tail; later tool-react iterations send history
            # unchanged so the cached prefix keeps paying off.
            if _iter == 0:
                extras = [b for b in (mem_block, hint_block) if b]
                req_messages = self.messages + extras if extras else self.messages
            else:
                req_messages = self.messages
            (
                assistant_text, tool_calls, output_tokens, finish_reason, elapsed
            ) = self._stream_turn(
                req_messages, tools_payload, recovery=recovery_next
            )
            # Recovery applies to exactly ONE generation (the one just made).
            recovery_next = False

            # Mid-stream cancel (Feature 5): _stream_turn broke early and flagged
            # the generation cancelled. Nothing has been appended for this partial
            # generation yet (the assistant tool_calls message is written further
            # below), so history stays well-formed — finalize the partial text.
            if finish_reason == "cancelled":
                return self._finalize_cancelled(
                    normalize_final_answer(assistant_text) or final_text
                )

            # Feature 2: constrained-decode retry. When the model emitted a tool
            # call the system could NOT parse (a "_parse_error" on any call), retry
            # the SAME request ONCE with tool_choice="required" to force a clean
            # native tool call — far better than only feeding back a "re-emit valid
            # JSON" string. Bounded to one retry per provider call. If the server
            # rejects tool_choice (it yields a stream-error -> no tool_calls) OR the
            # retry STILL fails to parse, we keep the original result and fall
            # through to the existing corrective-text behavior below.
            if (
                self.constrained_retry
                and tools_payload
                and tool_calls
                and any(tc.get("_parse_error") for tc in tool_calls)
            ):
                # If the parse error came from the gentle cap CUTTING OFF the
                # tool-call JSON (finish_reason=="length") — the common cause for a
                # big write_file/edit_file — lift the cap on this retry too, so one
                # retry both forces a clean call AND has room to emit the full args.
                # Otherwise the capped retry would just re-truncate, wasting a
                # generation before the truncation-recovery block below fixes it.
                retry_recovery = (
                    finish_reason == "length"
                    and getattr(self.provider, "gentle_mode", False)
                )
                if retry_recovery:
                    truncation_retried = True
                (
                    r_text, r_calls, r_tokens, r_finish, r_elapsed
                ) = self._stream_turn(
                    req_messages, tools_payload, tool_choice="required",
                    recovery=retry_recovery,
                )
                if r_calls and not any(tc.get("_parse_error") for tc in r_calls):
                    assistant_text, tool_calls, output_tokens, finish_reason, elapsed = (
                        r_text, r_calls, r_tokens, r_finish, r_elapsed
                    )

            # Truncation recovery. The gentle output cap (default 1024 tokens)
            # can cut a generation off mid-stream (finish_reason=="length"). For a
            # plain answer that just chops the text; FAR worse, a write_file/
            # edit_file carries the WHOLE file content inside its tool-call JSON
            # arguments, so a truncated call means the model "did the change" but
            # the write never lands (incomplete/parse-error args). Retry the SAME
            # request ONCE with the gentle cap LIFTED (recovery=True) so the full
            # answer / tool call is produced. Scoped to gentle_mode being on (the
            # silently-too-low cap we can fix by lifting); a real context-window
            # truncation isn't helped by retrying, so we don't, and the marker
            # below still fires. tool_choice="required" on the retry only when the
            # truncated generation was already emitting tool calls, so a truncated
            # write is re-emitted as a tool call rather than drifting to prose.
            if (
                finish_reason == "length"
                and not truncation_retried
                and getattr(self.provider, "gentle_mode", False)
            ):
                truncation_retried = True
                (
                    u_text, u_calls, u_tokens, u_finish, u_elapsed
                ) = self._stream_turn(
                    req_messages, tools_payload, recovery=True,
                    tool_choice="required" if tool_calls else None,
                )
                # Adopt the uncapped result ONLY when it is a STRICT improvement:
                # never when it errored (a transient retry blip must not clobber a
                # usable partial answer with a "[provider error]" sentinel), and
                # only when it actually carries content — a recovered tool call or
                # non-empty text. An empty/error retry leaves the original partial
                # (and its truncation marker) intact rather than discarding it.
                if u_finish != "error" and (u_calls or u_text.strip()):
                    assistant_text, tool_calls, output_tokens, finish_reason, elapsed = (
                        u_text, u_calls, u_tokens, u_finish, u_elapsed
                    )

            if not tool_calls:
                # Final answer. Normalize (strip a forbidden preamble opener),
                # render as Markdown, then the dim speed footer.
                final_text = normalize_final_answer(assistant_text)
                if not final_text.strip():
                    # Reasoning-only / empty turn: the model thought the whole turn
                    # and emitted no visible answer. NUDGE it once to actually write
                    # the answer (no extra tools/thinking) before giving up — this
                    # recovers the answer far more often than the empty sentinel.
                    if nudge_count < max_nudges:
                        nudge_count += 1
                        # The NEXT generation is a recovery shot: uncapped +
                        # thinking-off so the model emits a real answer instead of
                        # burning the (gentle-capped) budget on more reasoning.
                        recovery_next = True
                        self.messages.append({
                            "role": "user",
                            # Tagged so compact() does NOT count this synthetic
                            # message as a real user turn (which would make
                            # keep_turns under-keep the actual turns).
                            "_nudge": True,
                            "content": (
                                "You did not produce a final answer for the user. "
                                "Give your final answer now in plain text. Do NOT "
                                "call any tools and do NOT think out loud — just the "
                                "answer."
                            ),
                        })
                        continue
                    # Already nudged and STILL empty: surface the sentinel so the
                    # user / one-shot caller knows the turn produced nothing.
                    self._print_activity_summary()
                    notice = (
                        "[no answer produced — the model returned only reasoning "
                        "or an empty turn]"
                    )
                    self._dim(notice)
                    return notice
                # FOLLOW-THROUGH GUARD: the "final answer" is actually unexecuted
                # tool-call markup — the model wrote a <tool_call>/<function=...>
                # block as text instead of issuing it ("said it would do X but
                # didn't"). Re-prompt ONCE to re-emit it as a real tool call,
                # rather than printing raw markup as the answer.
                if (
                    not tool_markup_nudged
                    and looks_like_unexecuted_tool_call(final_text)
                ):
                    tool_markup_nudged = True
                    recovery_next = True
                    self.messages.append({
                        "role": "user",
                        "_nudge": True,
                        "content": (
                            "Your last message wrote a TOOL CALL as plain text "
                            "(e.g. a <tool_call> / <function=...> block) — it was "
                            "NOT executed. Re-issue it now as a real tool call. If "
                            "you actually meant to answer the user, reply in plain "
                            "text with NO tool markup."
                        ),
                    })
                    continue
                # AGENT-1: a provider/stream FAILURE must NOT be persisted as a
                # real answer. providers.py redacts transport/decode errors to the
                # "[provider error: ...]" / "[stream error: ...]" sentinels (and
                # finish_reason=="error" is the explicit signal); compact() already
                # guards the same sentinels. On failure: surface it dimly but do
                # NOT append an assistant turn and do NOT record it to memory (a
                # poisoned "fact"). Returned so one-shot/spawn callers see the error.
                if (
                    finish_reason == "error"
                    or final_text.lstrip().startswith("[provider error")
                    or final_text.lstrip().startswith("[stream error")
                ):
                    self._print_activity_summary()
                    self._dim(final_text)
                    # If this turn attached an image, the most likely cause is a
                    # text-only model rejecting vision input — point the user at it
                    # rather than leaving a bare transport error.
                    if getattr(self, "_sent_images_this_turn", False):
                        self._dim(
                            "[hint: the loaded model may not support vision — "
                            "load a multimodal model in LM Studio]"
                        )
                    return final_text
                # END-OF-TURN GATES (compose in order, each at most ONCE per turn):
                #   1. auto-verify / build-nudge (ACBUILD-1 + Feature 3)
                #   2. reviewer gate (Feature 4)
                # Each appends an ephemeral message and `continue`s WITHOUT printing
                # the answer, so the final answer is rendered exactly once when both
                # gates have passed (no double print, no infinite loop).
                #
                # The model changed files but ran no command this session. When a
                # verify_cmd is configured (Feature 3) AUTO-RUN it once and feed the
                # result back so the model can fix failures; otherwise keep the
                # ACBUILD-1 prose nudge (once per write-batch). Skipped when no
                # write happened or this agent has no run_bash tool.
                if (
                    self._unvalidated_writes
                    and "run_bash" in (self.tool_names or [])
                ):
                    if self.verify_cmd and not self._auto_verified:
                        self._auto_verified = True
                        # We are running the project's command for it, so the writes
                        # are no longer "unvalidated" (don't also fire the nudge).
                        self._unvalidated_writes = False
                        self._auto_verify_and_feed_back()
                        continue
                    if not self.verify_cmd and not self._build_nudged:
                        self._build_nudged = True
                        self.messages.append({
                            "role": "user",
                            # Tagged like the empty-turn nudge so compact()/session-save
                            # treat it as ephemeral, not a real user turn.
                            "_nudge": True,
                            "content": (
                                "You changed files but never ran the project's "
                                "tests/build this session. Run the tests now (use "
                                "run_bash) and report the REAL result — do not claim "
                                "success without verifying."
                            ),
                        })
                        continue
                # Feature 4: reviewer gate. A turn that changed code spawns the
                # read-only reviewer sub-agent ONCE, then feeds its findings back so
                # the model can address them before finalizing. Only an agent with
                # spawn_agent in its registry (the top-level orchestrator) can do
                # this, so sub-agents never trigger it / recurse.
                if (
                    self.review_writes
                    and self._wrote_code_this_turn
                    and not self._reviewed_this_turn
                    and self.registry.get("spawn_agent") is not None
                ):
                    self._reviewed_this_turn = True
                    if self._review_changes_and_feed_back(user_text):
                        continue
                # Real answer: record it, then render.
                self.messages.append({"role": "assistant", "content": final_text})
                # RECORD CREATION: a real, COMPLETE answer was produced, so
                # remember this turn (lossless Q + A) for later recall. SKIPPED on
                # truncation (finish_reason=="length", checked below): a cut-off
                # answer is a partial fact we must not store as if complete. Only
                # fires for a genuine answer — the empty/nudge-only and no-answer
                # branches return above without reaching here. Guarded so a store
                # failure never breaks the turn.
                if self._memory_active() and finish_reason != "length":
                    try:
                        self.memory.add(
                            text=f"Q: {user_text}\nA: {final_text}",
                            summary=self._cheap_summary(final_text),
                        )
                    except Exception:  # noqa: BLE001 - never break a turn
                        pass
                # One collapsed line for the whole turn's tool work (Ctrl+O expands).
                self._print_activity_summary()
                # Render the answer + footer only for a top-level agent. A spawned
                # sub-agent (render_answer=False) returns its answer to the
                # orchestrator, which renders it ONCE — printing here too would
                # double the whole answer on every delegated turn.
                if self.render_answer:
                    self._print_markdown(final_text)
                    self._print_footer(assistant_text, output_tokens, elapsed)
                if finish_reason == "length":
                    # The server truncated at the token limit; the text above is
                    # incomplete. Flag it on the console AND bake the marker into
                    # the RETURNED text (finding #6) so one-shot/spawn_agent
                    # callers — which consume only the return string — can see the
                    # answer was cut off, mirroring the empty-turn sentinel above.
                    if self.render_answer:
                        self._dim("[output truncated at token limit]")
                    return final_text + "\n[output truncated at token limit]"
                return final_text

            # Mid-turn narration ("Let me check the file...") is SUPPRESSED — the
            # user wants the backend hidden. Only the FINAL answer is shown; the
            # whole turn's tool activity collapses to ONE dim "⏺ N tools" line in
            # the final-answer branch, expandable via Ctrl+O. The narration is
            # still kept as assistant message content below (for history).

            # If a tool call was synthesized from a fenced text block, the
            # accumulated narration IS that fence; storing it as assistant content
            # would pollute history (the model would see it as both spoken and
            # called). Drop the content for fence-derived calls.
            stored_content: str | None = assistant_text or None
            if any(tc.get("_from_text_fence") for tc in tool_calls):
                stored_content = None

            # Append the assistant message carrying the tool calls (OpenAI shape).
            self.messages.append({
                "role": "assistant",
                "content": stored_content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    }
                    for tc in tool_calls
                ],
            })
            # Incremental token estimate (Fix 2): add the assistant(tool_calls)
            # message's char cost (incl. the serialized tool-call args) to the
            # running cache so the next _maybe_auto_compact() doesn't reserialize
            # every past tool_call's arguments.
            self._account_appended_msg_token_est()

            # Execute each tool call. Each renders as ONE Claude-Code-style
            # two-line tree (rendered immediately as it resolves):
            #   ⏺ Read(README.md)
            #     ⎿  Read 120 lines
            # Full per-call args/results still go to last_turn_details (Ctrl+O
            # reveals them). Confirmation prompts for gated tools still print
            # individually (before that tool's tree).
            batch_had_parse_error = False
            # Track whether ANY tool call in the batch was NOT a parse error
            # (executed, declined, or unknown-tool). The circuit-breaker must
            # only fire when the WHOLE batch failed to parse — a turn that lands
            # one good call plus one malformed call is making real progress and
            # must reset the counter (finding #23).
            batch_had_non_parse_call = False
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                # Accept the model's alias keys (file_path->path, old_string->old,
                # cmd->command, ...) BEFORE confirm/render/execute so a call that is
                # correct-but-differently-keyed just works instead of failing with
                # "requires a string 'path'". No-op for tools without aliases.
                args = tools_mod.normalize_tool_args(name, args)
                tool = self._get_tool(name)

                dt = 0.0
                # The REAL outcome, known here at the call site. Passed straight
                # into _render_tool_tree so the result-line color is never
                # re-derived from the connector text (a successful run_bash whose
                # stdout starts with "Error:" must NOT render red).
                is_error = True
                parse_error = tc.get("_parse_error")
                if parse_error:
                    # Provider could not parse the model's tool args as JSON. Don't
                    # execute with empty args; feed the failure back so the model
                    # re-emits valid JSON.
                    result = {
                        "ok": False,
                        "error": (
                            f"Could not parse tool arguments as JSON: {parse_error}. "
                            "Re-emit the tool call with valid JSON arguments."
                        ),
                    }
                    connector = "✗ invalid arguments"
                    batch_had_parse_error = True
                elif tool is None:
                    # Suggest the closest real tool names (stdlib difflib) so a
                    # weak local model can self-correct next turn. Cheap: only on
                    # the error path. self.registry.keys() is the exact set of
                    # callable tool names the model is given in the schema.
                    matches = difflib.get_close_matches(
                        name, list(self.registry.keys()), n=3, cutoff=0.4
                    )
                    if matches:
                        err = f"Unknown tool: {name} — did you mean: {', '.join(matches)}?"
                    else:
                        err = f"Unknown tool: {name}"
                    result = {"ok": False, "error": err}
                    connector = "✗ unknown tool"
                    batch_had_non_parse_call = True
                else:
                    # Permission gate (replaces the old inline confirm elif). In
                    # "default" mode this is byte-for-byte the historic behavior:
                    # auto_confirm / confirm_fn decide, and a declined gated tool
                    # yields the SAME {"ok":False,...}/"✗ declined" shape as before.
                    decision, reason = self._permission_decision(tool, args)
                    if decision == "block":
                        result = {"ok": False, "error": reason}
                        connector = (
                            "✗ declined"
                            if reason == "User declined this tool call."
                            else "✗ blocked"
                        )
                        batch_had_non_parse_call = True
                    else:
                        batch_had_non_parse_call = True
                        # Pre-tool hook veto gate (Feature 4): runs AFTER permission
                        # allow and BEFORE checkpoint/execution. A "block" decision
                        # short-circuits without ever calling tool.fn. No-op (and no
                        # subprocess) when self.hooks is falsy — the default.
                        hook_blocked = False
                        if self.hooks:
                            root = self.workspace_root or os.getcwd()
                            try:
                                pre = hooks_mod.run_pre_tool(
                                    self.hooks, name, args, cwd=root
                                )
                            except Exception:  # noqa: BLE001 - a broken hook never wedges the loop
                                pre = {"decision": "allow"}
                            if pre.get("decision") == "block":
                                hook_blocked = True
                                result = {
                                    "ok": False,
                                    "error": (
                                        "Blocked by pre-tool hook: "
                                        f"{pre.get('reason', '')}"
                                    ),
                                }
                                connector = "✗ blocked by hook"
                        if not hook_blocked:
                            # Checkpoint-before-write (Feature 3): snapshot the
                            # file's current bytes right before a write/edit so
                            # /undo can revert. A checkpoint failure must NEVER
                            # break the turn — swallowed. Off by default.
                            if (
                                self.checkpoints_enabled
                                and name in ("write_file", "edit_file")
                            ):
                                _p = args.get("path")
                                if isinstance(_p, str) and _p:
                                    try:
                                        checkpoint_mod.snapshot(
                                            [_p],
                                            root=self.workspace_root or os.getcwd(),
                                            label=name,
                                            session=self.checkpoint_session,
                                        )
                                    except Exception:  # noqa: BLE001 - checkpoint is best-effort
                                        pass
                            # Duck waddles during tool execution. It is started
                            # AFTER any confirm_fn prompt has already resolved (in
                            # _permission_decision), so the y/N prompt is never
                            # glued to a spinner; stopped before printing anything.
                            t_tool = time.perf_counter()
                            tool_spinner = Spinner(
                                self.console,
                                color=self.palette.spinner if self.palette else None,
                                timer_color=(
                                    self.palette.spinner_timer if self.palette else None
                                ),
                                # Reactor pulse while the tool runs, with the verb set
                                # to what it's actually doing (reading providers.py /
                                # running <cmd> / editing <file>).
                                pulse_to=_reactor_pulse_to(self.palette),
                                verb=_spinner_verb(name, args),
                            )
                            tool_spinner.start()
                            try:
                                result = tool.fn(args)
                            except Exception as exc:  # noqa: BLE001 - tools must not crash the loop
                                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                            finally:
                                tool_spinner.stop()
                            dt = time.perf_counter() - t_tool
                            # Post-tool hook (Feature 4): fire-and-forget after a
                            # successful OR failed execution; never blocks/raises.
                            if self.hooks:
                                root = self.workspace_root or os.getcwd()
                                try:
                                    hooks_mod.run_post_tool(
                                        self.hooks, name, args, result, cwd=root
                                    )
                                except Exception:  # noqa: BLE001 - post-hook must not break the loop
                                    pass
                            # MANDATORY self-heal: a REAL execution failure (we are
                            # past the parse-error / unknown-tool / declined /
                            # permission-block / hook-block branches, so this is a
                            # genuine tool failure) gets a SAFE corrected retry from
                            # the deterministic remediator BEFORE the model ever
                            # sees the error. Bounded by auto_fix_max_attempts and
                            # non-destructive (only a path is ever corrected). A
                            # complete no-op when off or when there is no safe fix.
                            if self.auto_fix_tools and result.get("ok") is not True:
                                for _fix_attempt in range(self.auto_fix_max_attempts):
                                    try:
                                        fix = remediation.remediate(
                                            name, args, result,
                                            root=self.workspace_root or os.getcwd(),
                                        )
                                    except Exception:  # noqa: BLE001 - remediation never wedges the loop
                                        fix = None
                                    if not fix:
                                        break
                                    new_args, expl = fix
                                    self._dim(f"↻ auto-fix: {expl}")
                                    # Snapshot the CORRECTED write/edit target so a
                                    # /undo still reverts the file we actually wrote.
                                    if (
                                        self.checkpoints_enabled
                                        and name in ("write_file", "edit_file")
                                    ):
                                        _cp = new_args.get("path")
                                        if isinstance(_cp, str) and _cp:
                                            try:
                                                checkpoint_mod.snapshot(
                                                    [_cp],
                                                    root=self.workspace_root or os.getcwd(),
                                                    label=name,
                                                    session=self.checkpoint_session,
                                                )
                                            except Exception:  # noqa: BLE001 - checkpoint is best-effort
                                                pass
                                    # Re-run the SAME tool with corrected args via
                                    # the same crash guard. Permission was already
                                    # granted for this tool this batch, so the user
                                    # confirm_fn is NOT re-invoked on retry. Time the
                                    # retry and fold it into dt so _record_detail
                                    # reflects TOTAL wall time (first attempt + every
                                    # retry), not just the first attempt.
                                    t_retry = time.perf_counter()
                                    try:
                                        result = tool.fn(new_args)
                                    except Exception as exc:  # noqa: BLE001 - tools must not crash the loop
                                        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                                    dt += time.perf_counter() - t_retry
                                    args = new_args
                                    if result.get("ok") is True:
                                        # _autofixes counts HEALED calls (successes),
                                        # not attempts, so the surfaced number means
                                        # "calls we actually fixed".
                                        self._autofixes += 1
                                        # Re-run the post-tool hook once with the
                                        # FINAL corrected outcome so observers see the
                                        # real result, not the first (failed) one.
                                        # Fire-and-forget; the pre-tool veto is NOT
                                        # re-run (remediation only narrows the path
                                        # deeper into the workspace, a low-risk move).
                                        if self.hooks:
                                            try:
                                                hooks_mod.run_post_tool(
                                                    self.hooks, name, new_args, result,
                                                    cwd=self.workspace_root or os.getcwd(),
                                                )
                                            except Exception:  # noqa: BLE001 - post-hook must not break the loop
                                                pass
                                        break
                            if result.get("ok") is True:
                                is_error = False
                                connector = result_summary(name, result)
                                # spawn_agent: prefer "<role> done" (we have the role here).
                                if name == "spawn_agent":
                                    role = str(args.get("role", "")).strip()
                                    if role:
                                        connector = f"{role} done"
                                if not connector:
                                    connector = "done"
                            else:
                                # One consistent failure vocabulary across every
                                # result line (parse error / unknown tool /
                                # declined / execution failure all use the "✗ "
                                # glyph). Color is driven by the explicit is_error
                                # flag, not this prefix.
                                connector = "✗ " + error_summary(result)
                            # ACBUILD-1: track unvalidated file changes (only here,
                            # in the actually-executed branch — parse errors /
                            # unknown / declined / blocked never touch the flags). A
                            # successful write/edit ARMS the verify-nudge and
                            # re-arms it for a fresh batch (_build_nudged reset); ANY
                            # run_bash execution CLEARS it (the model ran the
                            # project's commands), so a later final answer is accepted.
                            if name in ("write_file", "edit_file") and result.get("ok") is True:
                                self._unvalidated_writes = True
                                self._build_nudged = False
                                # Feature 4 (quality): remember THIS turn changed
                                # code (and which file) so the reviewer gate can run
                                # on the real paths. Not cleared by run_bash — a turn
                                # that wrote then tested still gets reviewed.
                                self._wrote_code_this_turn = True
                                path = args.get("path")
                                if isinstance(path, str) and path and path not in self._changed_files:
                                    self._changed_files.append(path)
                            elif name == "run_bash":
                                self._unvalidated_writes = False

                # Tool detail is NOT rendered live anymore — the whole turn's
                # activity collapses to one "⏺ N tools" line (final-answer branch)
                # and the full ⏺/⎿ tree is shown on demand via Ctrl+O
                # (render_details). connector/is_error are computed above only so
                # render_details can reuse the same vocabulary later.
                self._record_detail(name, args, result, result.get("ok"), dt)
                # CHANGE 1: this is the single seam where every tool's result text
                # is turned into context. Account the code-read budget here and let
                # the helper append the one-time "stop reading" nudge in place.
                content = self._account_tool_read(name, json.dumps(result))
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": content,
                })
                # Incremental token estimate (Fix 2): add the tool-result message
                # to the running cache.
                self._account_appended_msg_token_est()

            # Circuit-breaker: a model stuck emitting malformed JSON args would
            # otherwise consume the entire iteration budget doing nothing useful.
            # Abort after 3 consecutive turns whose ENTIRE batch was parse errors.
            # A turn that lands even one real call (executed/declined/unknown) is
            # making progress and resets the counter (finding #23): only count it
            # when there was a parse error AND no non-parse call ran.
            if batch_had_parse_error and not batch_had_non_parse_call:
                consecutive_parse_errors += 1
                if consecutive_parse_errors >= 3:
                    self._dim(
                        "stopping: the model repeatedly emitted invalid tool-call JSON."
                    )
                    return final_text or "[stopped: repeated invalid tool-call JSON]"
            else:
                consecutive_parse_errors = 0

            # Duplicate-batch loop guard: a model stuck re-issuing the SAME tool
            # calls (same names + same args) burns the whole iteration budget
            # doing nothing useful. Break early once the same batch repeats N
            # times in a row. A batch that differs (different args, a progressed
            # read_file offset, a new file) resets the count — only an
            # exact-repeat loop trips this. SKIPPED when the batch had a parse
            # error: a malformed-call loop is owned by the parse-error breaker
            # above (and the args of a parse-error call aren't trustworthy for a
            # signature), so the two guards must not both fire on the same loop.
            if not batch_had_parse_error:
                batch_sig = _tool_call_batch_sig(tool_calls)
                if batch_sig == _last_batch_sig:
                    _consecutive_dup_batches += 1
                    if _consecutive_dup_batches >= _dup_batch_threshold:
                        self._dim(
                            "stopping: the model repeatedly issued the same "
                            "tool call without making progress. Re-run, "
                            "rephrase your request, or raise --max-iterations."
                        )
                        return (
                            final_text
                            or "[stopped: repeated identical tool calls]"
                        )
                else:
                    _last_batch_sig = batch_sig
                    _consecutive_dup_batches = 1
            # loop again so the model can react to the tool results

        # Max iterations hit.
        self._dim(
            f"reached the tool-use limit ({self.max_iterations}); stopping. "
            "Re-run or raise --max-iterations."
        )
        return final_text or "[stopped: reached max tool-use iterations]"

    # ----- adaptive context budget --------------------------------------
    def _compute_turn_budget(self, user_text: str) -> int:
        """The working-context budget for this turn (rough chars/4 tokens).

        Base = ``context_budget``; when ``context_adaptive`` it is multiplied by a
        request-weight (1.0–3.0: bigger for long/broad/multi-file requests). The
        result is floored at ``_MIN_TURN_BUDGET`` and capped at ``context_ceiling``
        (the near-window safety value) so it never exceeds what the model can hold.
        """
        base = self.context_budget
        if self.context_adaptive:
            base = int(base * request_weight(user_text))
            # Intent-aware sizing (the "load only what THIS message needs" lever):
            # a trivial/meta question ("what happened?") gets a deliberately SMALL
            # working budget so history compacts hard and the prefill stays tiny —
            # the model pulls anything specific on demand via its tools. task /
            # followup / broad keep the length+keyword weight above (request_weight
            # already grants 2.5x for whole-repo keywords), so this only ever
            # SHRINKS the cheap case and never starves a real task.
            if classify_request(user_text) == "trivial":
                base = min(base, max(_MIN_TURN_BUDGET, self.context_budget // 2))
        budget = max(_MIN_TURN_BUDGET, base)
        if self.context_ceiling > 0:
            # Clamp the ceiling to at least the floor so a misconfigured tiny
            # ceiling can't starve the turn below the documented minimum.
            budget = min(budget, max(self.context_ceiling, _MIN_TURN_BUDGET))
        return budget

    # ----- context hygiene ----------------------------------------------
    def _trim_stale_tool_outputs(self) -> int:
        """Shrink bulky tool output + written-file content from COMPLETED turns.

        The dominant context bloat in an engineering session is tool RESULTS
        (file reads, repeated test-run stdout, repo maps) plus the file content
        inside write_file/edit_file tool CALLS — together 90%+ of a long history
        in practice, all re-sent verbatim on every later turn though rarely
        needed once their turn is done. This caps that stale bulk so a small
        follow-up ("what happened?") isn't forced to prefill 50k+ tokens.

        Conversation-aware: the CURRENT (last) user turn is left FULL — a
        follow-up often needs what just happened — and only messages BEFORE it
        are trimmed. Deterministic (no model call) and structure-preserving (only
        a tool message's `content` and a tool_call's argument STRING are shrunk,
        re-serialized to valid JSON, so the assistant(tool_calls)->tool pairing
        and the message count are untouched). Returns the est-token saving.
        """
        user_idxs = [
            i for i, m in enumerate(self.messages)
            if m.get("role") == "user" and not m.get("_nudge")
        ]
        if not user_idxs:
            return 0
        boundary = user_idxs[-1]  # keep the current/last user turn's tools full
        saved_chars = 0
        # Track whether we ACTUALLY rewrote any content/args this pass. The
        # est-token return (``saved_chars // 4``) is a lossy proxy — integer
        # division can round a genuine byte-change down to 0, and the arg-branch
        # ``max(0, …)`` can record 0 saving even though a field was replaced — so
        # the caller must invalidate the token cache off THIS flag, not the
        # numeric return, to stay correct (never leave a stale cache).
        mutated = False
        for m in self.messages[1:boundary]:  # skip system[0]; stop before last turn
            if m.get("_memory"):
                continue
            role = m.get("role")
            if role == "tool":
                c = m.get("content")
                # Idempotent: the trim runs EVERY turn, so skip a result we already
                # shrank (else we'd re-slice it and corrupt the note each turn).
                if (
                    isinstance(c, str)
                    and len(c) > _STALE_TOOL_RESULT_CAP
                    and _TRIM_MARKER not in c
                ):
                    trimmed = (
                        c[:_STALE_TOOL_RESULT_CAP]
                        + f"\n…[+{len(c) - _STALE_TOOL_RESULT_CAP} chars {_TRIM_MARKER} — "
                        "older tool output; re-run or re-read if you need it]"
                    )
                    saved_chars += len(c) - len(trimmed)
                    m["content"] = trimmed
                    mutated = True
            elif role == "assistant":
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if not isinstance(args, str) or len(args) <= _STALE_ARG_FIELD_CAP:
                        continue
                    try:
                        obj = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(obj, dict):
                        continue
                    changed = False
                    for k in ("content", "old", "new", "old_string", "new_string"):
                        v = obj.get(k)
                        if (
                            isinstance(v, str)
                            and len(v) > _STALE_ARG_FIELD_CAP
                            and _TRIM_MARKER not in v
                        ):
                            obj[k] = (
                                v[:_STALE_ARG_FIELD_CAP]
                                + f"…[+{len(v) - _STALE_ARG_FIELD_CAP}ch {_TRIM_MARKER}]"
                            )
                            changed = True
                    if changed:
                        new_args = json.dumps(obj, ensure_ascii=False)
                        saved_chars += max(0, len(args) - len(new_args))
                        fn["arguments"] = new_args
                        mutated = True
        # Record the reliable mutation signal for the caller (see ``mutated``
        # above); leave the numeric est-token return unchanged for callers/tests
        # that assert on it.
        self._last_trim_mutated = mutated
        return saved_chars // 4  # rough est-token saving

    # ----- auto-compaction ----------------------------------------------
    def _maybe_auto_compact(self) -> None:
        """Compact earlier history when it exceeds the soft token budget.

        Safety valve for long sessions (findings #4/#26): keeps the prompt
        prefix bounded so a file-reading session can't grow past the model's
        context window — across multiple user turns AND within a single heavy
        turn whose own tool rounds overflow (findings #1/#3). No-op when
        ``context_soft_limit`` is 0 or the estimate is under budget. A compact
        failure is swallowed — sending a slightly-too-long prompt is better than
        aborting the user's turn.
        """
        if self.context_soft_limit <= 0:
            return
        est = self._estimate_tokens_cached()
        if est <= self.context_soft_limit:
            self._compact_floor = 0  # back under budget; allow future attempts
            return
        # Already tried at ~this size and it didn't help — don't thrash. Re-attempt
        # only after history has grown meaningfully past the recorded floor.
        if est <= self._compact_floor:
            return
        # Snapshot so a non-useful compaction can be UNDONE: applying it would
        # rewrite the history prefix and invalidate the server KV-cache for ~no
        # token saving (e.g. the kept tail is itself near budget, or a weak model
        # summarizes poorly). Keeping the prefix stable is the bigger win.
        snapshot = self.messages
        try:
            # AGGRESSIVE: keep only the LAST user exchange verbatim (was 2). The
            # 2-turn keep was why a heavy session barely shrank — the 2nd-to-last
            # turn's bulk stayed in the kept tail and was never summarized. The
            # meaningful-saving bar below still rolls back a near-useless apply.
            before, after = self.compact(keep_turns=1)
        except Exception:  # noqa: BLE001 - never let auto-compact break a turn
            self.messages = snapshot
            # History was potentially rewritten then rolled back; the incremental
            # token-est cache (Fix 2) is stale.
            self._invalidate_token_est()
            return
        # "Meaningful" = saves at least ~3% (and >=128 est tok). Below that, the
        # churn isn't worth breaking the cache or interrupting the user. Lowered
        # from 5%: a 3% real reduction is still a real reduction worth keeping,
        # and the more aggressive keep_turns=1 makes most compactions clear this
        # bar comfortably. A near-no-op (e.g. a bloated summary) still rolls back.
        if before - after >= max(128, before // 33):
            self._compact_floor = 0
            self._dim(
                f"[auto-compacted history ~{before} -> ~{after} est. tok "
                "to stay within the context window]"
            )
        else:
            # Roll back the trivial compaction and remember not to retry until
            # history grows ~10% past where it stalled. Stay silent (no noise).
            self.messages = snapshot
            self._compact_floor = int(est * 1.1) + 1
            # Rolled back to the snapshot, but the cache was computed before the
            # attempt; invalidate so the next estimate recomputes from the
            # restored history (Fix 2).
            self._invalidate_token_est()

    # ----- /compact ------------------------------------------------------
    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate (chars/4) over content + serialized tool_calls."""
        total = 0
        for m in messages:
            total += _msg_chars(m)
        return total // 4

    def _invalidate_token_est(self) -> None:
        """Mark the incremental running-token cache dirty (Fix 2).

        Call after any in-place mutation of ``self.messages`` (e.g. the
        per-turn ``_trim_stale_tool_outputs`` rewrites tool-result content and
        tool-call argument strings) or any reassignment of ``self.messages``
        (compaction rewrites history). The next ``_estimate_tokens_cached`` call
        recomputes the full sum and re-arms the cache.
        """
        self._running_token_chars = None

    def _estimate_tokens_cached(self) -> int:
        """Incremental token estimate over ``self.messages`` (Fix 2, PERF).

        Maintains ``self._running_token_chars`` — a running CHAR total over
        ``self.messages`` — so the hot path (``_maybe_auto_compact`` runs every
        loop iteration) does NOT re-``json.dumps`` every past tool_call's
        arguments each iteration. Appends add a cheap per-message delta; in-place
        mutations and history rewrites invalidate via ``_invalidate_token_est``.

        Returns the SAME value ``self._estimate_tokens(self.messages)`` would
        (chars summed, then integer-divided by 4 at query time) so callers see no
        behaviour change. On a dirty cache the full recompute re-arms it, so the
        FIRST call after an invalidation is O(n) and subsequent calls are O(1).
        """
        if self._running_token_chars is None:
            self._running_token_chars = sum(
                _msg_chars(m) for m in self.messages
            )
        return self._running_token_chars // 4

    def _account_appended_msg_token_est(self) -> None:
        """Add the just-appended message's char cost to the running cache (Fix 2).

        Called right after a ``self.messages.append(...)``. No-op when the cache
        is dirty (the next ``_estimate_tokens_cached`` recomputes the full sum, so
        a stale delta would double-count). When armed, the per-message cost is a
        cheap O(1) delta — the win over re-summing the whole history each
        iteration.
        """
        if self._running_token_chars is not None and self.messages:
            self._running_token_chars += _msg_chars(self.messages[-1])

    def compact(self, keep_turns: int = 2) -> tuple[int, int]:
        """Summarize earlier history via the ACTIVE provider, replace it.

        ``keep_turns`` = how many trailing USER turns stay verbatim; everything
        before them is summarized into one note. Default 2 (auto-compaction's
        safe behaviour). Manual ``/compact`` passes 1 — keep only the LAST
        exchange and summarize everything else (maximally small).

        Returns ``(before_tokens, after_tokens)`` rough estimates. Builds the
        new history LOCALLY and assigns to ``self.messages`` only on success, so
        a provider failure leaves history untouched. On ANY provider failure
        (a raised exception OR an error/empty summary event) this raises
        RuntimeError for a uniform contract; the caller (REPL) catches it and
        no-ops, leaving history unchanged.
        """
        keep_turns = max(1, int(keep_turns))
        before_tokens = self._estimate_tokens(self.messages)

        # Nothing meaningful to compact: system + <=1 turn.
        if len(self.messages) <= 3:
            return before_tokens, before_tokens

        system_msg = self.messages[0]

        # Find the tail to KEEP: the last ``keep_turns`` user turns + their
        # replies. Fall back to the first user message when there are fewer.
        # Exclude synthetic nudge messages so keep_turns counts only REAL user turns.
        user_idxs = [
            i for i, m in enumerate(self.messages)
            if m.get("role") == "user" and not m.get("_nudge")
        ]
        if user_idxs:
            keep_from = (
                user_idxs[-keep_turns] if len(user_idxs) >= keep_turns else user_idxs[0]
            )
        else:
            keep_from = len(self.messages)
        # DROP ephemeral `_memory` blocks from BOTH the kept tail and the
        # to-summarize range: they are regenerated every turn (mirrors the
        # `_nudge` handling), so they must NEVER be summarized into — nor
        # duplicated by — a compaction. The kept tail is rebuilt without them so
        # an in-tail compaction can't carry one into a summary either.
        kept_tail = [m for m in self.messages[keep_from:] if not m.get("_memory")]

        # Messages to summarize = everything between system and the kept tail
        # (minus any `_memory` block, per above).
        to_summarize = [m for m in self.messages[1:keep_from] if not m.get("_memory")]

        # Nothing to summarize across user-turn boundaries (e.g. a SINGLE user
        # turn whose own tool rounds are huge — the small-context footgun in
        # findings #1/#3). Anchoring strictly to user-message boundaries leaves
        # to_summarize empty, so the prompt is sent oversized. When we are still
        # over the soft budget, fall back to summarizing the OLDER completed tool
        # rounds INSIDE the kept tail, preserving the user message + the last few
        # rounds and the tool_calls/tool pairing invariant.
        if not to_summarize:
            if (
                self.context_soft_limit > 0
                and before_tokens > self.context_soft_limit
                and len(kept_tail) > 1
            ):
                return self._compact_within_tail(
                    system_msg, kept_tail, before_tokens
                )
            # Otherwise no-op: do NOT call the provider and do NOT inject an
            # empty summary note (that would GROW history by one meaningless
            # message). Return the unchanged estimate.
            return before_tokens, before_tokens

        summary = self._summarize_messages(to_summarize)

        # Build the new history locally; assign only as the FINAL step.
        new_messages = [
            system_msg,
            {"role": "system", "content": "Summary of earlier conversation:\n" + summary},
        ] + kept_tail
        after_tokens = self._estimate_tokens(new_messages)
        self.messages = new_messages
        # History was rewritten (compaction); the incremental token-est cache
        # (Fix 2) is stale — invalidate so the next estimate recomputes from the
        # new, smaller history.
        self._invalidate_token_est()
        return before_tokens, after_tokens

    @staticmethod
    def _serialize_for_summary(messages: list[dict]) -> str:
        """Flatten messages into a transcript for the summarizer.

        Tool results are capped at 500 chars and tool_calls are serialized so the
        most information-dense part of an engineering session (which tools were
        called, with what args) survives — assistant tool_calls messages often
        carry content=None and would otherwise summarize as empty.
        """
        lines: list[str] = []
        for m in messages:
            role = m.get("role", "?")
            # text_of() yields the text parts only, so an attached image's base64
            # blob is never fed into the summarizer prompt.
            content = text_of(m.get("content"))
            if role == "tool":
                content = content[:500]
            tcs = m.get("tool_calls")
            if tcs:
                names_args = []
                for tc in tcs:
                    fn = tc.get("function") or {}
                    # The summarizer only needs the GIST of a call. write_file/
                    # edit_file carry the ENTIRE file content inside ``arguments`;
                    # feeding 50KB verbatim into the summarizer prompt wastes
                    # context and tokens. Cap to the same ~500 chars used for tool
                    # results above (the model already saw the full call when it
                    # was made; the summary is a compressed transcript).
                    args_str = fn.get("arguments")
                    if isinstance(args_str, str) and len(args_str) > 500:
                        args_str = args_str[:500]
                    names_args.append({"name": fn.get("name"), "arguments": args_str})
                tc_str = json.dumps(names_args, ensure_ascii=False)
                content = (content + " " if content else "") + "tool_calls: " + tc_str
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _summarize_messages(self, messages: list[dict]) -> str:
        """One-shot the ACTIVE provider over a transcript and return the summary.

        Raises RuntimeError on any provider failure (raised exception OR an
        error/empty summary event) for a uniform contract — callers leave history
        untouched on failure.
        """
        req_messages = [
            {"role": "system", "content": SUMMARIZER_PROMPT},
            {"role": "user", "content": self._serialize_for_summary(messages)},
        ]
        summary_acc: list[str] = []
        try:
            for event in self.provider.stream_chat(req_messages, None):
                if event.get("type") == "text":
                    summary_acc.append(event.get("text", ""))
                elif event.get("type") == "done":
                    break
        except Exception as exc:  # noqa: BLE001 - uniform contract; history unchanged
            raise RuntimeError(f"compact: provider failed: {type(exc).__name__}: {exc}") from exc
        summary = "".join(summary_acc).strip()
        if (
            not summary
            or summary.startswith("[provider error")
            or summary.startswith("[stream error")
        ):
            raise RuntimeError("compact: provider returned no usable summary")
        return summary

    def _compact_within_tail(
        self, system_msg: dict, kept_tail: list[dict], before_tokens: int
    ) -> tuple[int, int]:
        """Summarize OLDER completed tool rounds inside a single active turn.

        Handles the small-context footgun (findings #1/#3): one user turn whose
        own tool rounds overflow the window. ``kept_tail`` begins with the user
        message; the trailing assistant/tool rounds are collapsed into a summary
        note while preserving the user message, the last few rounds, and the
        tool_calls<->tool pairing invariant. No-op (returns unchanged estimate)
        if there is nothing safe to collapse.
        """
        # kept_tail[0] is the user message; the rest are assistant/tool rounds.
        head = kept_tail[:1]
        body = kept_tail[1:]

        # A "round" is an assistant message (optionally carrying tool_calls)
        # followed by its tool result message(s). Walk the body grouping each
        # assistant message with the tool messages that immediately follow it so
        # we never split a tool_calls message from its results.
        rounds: list[list[dict]] = []
        for m in body:
            if m.get("role") == "assistant" or not rounds:
                rounds.append([m])
            else:
                rounds[-1].append(m)

        # Keep the last 1 round verbatim (the in-progress context the model is
        # actively reasoning over); summarize everything earlier. AGGRESSIVE
        # (was 2): the 2-round keep left the prior round's bulky tool result
        # un-summarized, so a giant single turn barely shrank. The user message
        # + the active round + the pairing invariant are still preserved.
        keep_rounds = 1
        if len(rounds) <= keep_rounds:
            return before_tokens, before_tokens
        to_summarize = [m for r in rounds[:-keep_rounds] for m in r]
        kept_rounds = [m for r in rounds[-keep_rounds:] for m in r]
        if not to_summarize:
            return before_tokens, before_tokens

        # Cheap pre-check: if the block we'd summarize is itself too small to
        # clear the ~3% meaningful-saving bar (matches _maybe_auto_compact), don't
        # spend a provider call only to roll it back (the auto-compact caller
        # would discard it anyway).
        if self._estimate_tokens(to_summarize) < max(128, before_tokens // 33):
            return before_tokens, before_tokens

        summary = self._summarize_messages(to_summarize)

        new_messages = (
            [system_msg]
            + head
            + [{"role": "system",
                "content": "Summary of earlier tool activity this turn:\n" + summary}]
            + kept_rounds
        )
        after_tokens = self._estimate_tokens(new_messages)
        self.messages = new_messages
        # History was rewritten (in-tail compaction); invalidate the incremental
        # token-est cache (Fix 2).
        self._invalidate_token_est()
        return before_tokens, after_tokens
