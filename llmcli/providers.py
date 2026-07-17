"""Providers: normalize different backends to a single streamed event model.

LOCAL ONLY. Two providers:
  - LocalProvider: LM Studio's OpenAI-compatible server. Lazily imports
    ``openai`` INSIDE the class so the rest of the stack runs offline.
  - MockProvider: deterministic, scripted, no network, no openai import.

NORMALIZED EVENTS (every stream_chat yields plain dicts with a ``type`` key):

  {"type": "text", "text": <str>}                      # incremental delta
  {"type": "tool_call", "id", "name", "arguments": <dict>}  # args already parsed
  {"type": "done", "finish_reason": <str>, "output_tokens": <int|None>}  # last

The "done" event always carries an ``output_tokens`` key (the completion token
count for the just-streamed message, or None when the server did not report it
so the agent can approximate). It is exactly one, always last.

Invariants the agent relies on:
  - Within a turn: zero+ text, then zero+ tool_call, then exactly one done.
  - tool_call.arguments is ALWAYS a dict (json.loads'd), never a JSON string.
  - On malformed JSON: arguments={}, plus optional "_parse_error".
"""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import re
import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import Iterator

from .config import is_loopback_url, resolve_loopback_ip
from .images import text_of


def _pin_loopback_base_url(base_url: str) -> tuple[str, str | None]:
    """Pin ``base_url`` to its VALIDATED loopback IP (DNS-rebinding defense).

    Resolves + validates the host once (every answer must be loopback) and
    rewrites the URL host to the resulting literal IP so the HTTP client cannot
    RE-RESOLVE the hostname per request and ship project data off-box (finding
    #1). The original host is returned as the ``Host`` header value so the local
    server still sees the intended vhost.

    Returns ``(pinned_url, host_header)``. If the host is already a literal IP,
    no rewrite is needed and ``host_header`` is ``None``. Raises ``ValueError``
    when the URL cannot be pinned to a loopback IP (fail-closed: a non-loopback
    or unresolvable host must never reach the client unpinned).
    """
    pinned_ip = resolve_loopback_ip(base_url)
    if pinned_ip is None:
        raise ValueError(
            f"private mode: refusing a non-loopback/unpinnable base_url "
            f"({base_url!r}). It must resolve EXCLUSIVELY to a loopback address "
            "so project data never leaves the machine."
        )
    parsed = urllib.parse.urlparse(base_url)
    original_host = parsed.hostname or ""
    # Already a literal IP host: no re-resolution is possible, leave it as-is.
    if original_host == pinned_ip:
        return base_url, None
    # Rewrite the host to the pinned literal IP, preserving scheme/port/path.
    port = parsed.port
    # Bracket IPv6 literals in the authority.
    host_part = f"[{pinned_ip}]" if ":" in pinned_ip else pinned_ip
    netloc = f"{host_part}:{port}" if port is not None else host_part
    pinned_url = urllib.parse.urlunparse(parsed._replace(netloc=netloc))
    return pinned_url, original_host


# ---------------------------------------------------------------------------
# Text-fallback tool-call parser (module scope; stdlib only)
# ---------------------------------------------------------------------------

# Matches a fenced ```json { ... } ``` block.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_tool_block(text: str) -> dict | None:
    """Extract a fenced ```json {"tool":.., "input":..} ``` block from text.

    Returns {"name": <tool>, "arguments": <dict>} if a complete, valid block
    is present, else None. Tolerant: only fires on a well-formed block that
    actually looks like a tool call (has a "tool"/"name" key).
    """
    if not text or "```" not in text:
        return None
    for m in _FENCE_RE.finditer(text):
        blob = m.group(1)
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("tool") or data.get("name")
        if not isinstance(name, str) or not name:
            continue
        # PRIMARY: wrapper shape {"tool":..,"input":{..}} / {"name":..,"arguments":{..}}.
        if "input" in data or "arguments" in data:
            raw_args = data.get("input", data.get("arguments", {}))
            args = raw_args if isinstance(raw_args, dict) else {}
        else:
            # FALLBACK (weak local models): FLAT shape {"tool":..,"path":..}.
            # With no input/arguments wrapper, the remaining top-level keys ARE
            # the arguments; dropping them would silently lose call params.
            args = {k: v for k, v in data.items() if k not in ("tool", "name")}
        return {"name": name, "arguments": args}
    return None


def count_tool_blocks(text: str) -> int:
    """Count fenced blocks that look like a tool call (have a tool/name key).

    Used to refuse AMBIGUOUS multi-tool-fence turns: if a model emits two
    tool-shaped fences, executing only the first silently drops the rest, so we
    decline to auto-execute and let the model re-emit one call at a time.
    """
    if not text or "```" not in text:
        return 0
    n = 0
    for m in _FENCE_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("tool") or data.get("name"), str):
            name = data.get("tool") or data.get("name")
            if name:
                n += 1
    return n


def _fence_is_sole_content(text: str) -> bool:
    """True if a fenced ```json block is the only substantive content in text.

    Removing every fenced block must leave only whitespace; otherwise the model
    intended the fence as an EXAMPLE inside a prose answer, not as a tool call,
    and we must NOT execute it.
    """
    if not text:
        return False
    stripped = _FENCE_RE.sub("", text).strip()
    return stripped == ""


# ---------------------------------------------------------------------------
# HIGH-CONFIDENCE chat-template tool-call extraction (module scope; stdlib only)
# ---------------------------------------------------------------------------
#
# Many local models emit tool/function calls in their OWN chat-template markup
# as PLAIN TEXT when LM Studio/llama.cpp does NOT convert them to native OpenAI
# ``tool_calls`` (server-side template mismatch). The agent would then treat the
# call as prose, and the model goes on to HALLUCINATE the tool's OUTPUT plus a
# prose answer (we observed deepseek-coder-v2 "write" a file that never existed).
#
# These are model CONTROL tokens; their presence OUTSIDE a code fence is an
# unambiguous tool-call signal. So for each known format we extract the FIRST
# tool-call group and DISCARD any trailing text — the hallucinated tool OUTPUT +
# prose answer that some servers let the model continue with is by-definition
# fake and must never reach the user as an answer. (Unlike the generic ```json
# fence path below, these formats do NOT require "sole content".)

# DeepSeek's control-token delimiters are NON-ASCII. Build the literals from
# their code points so the source can never be silently corrupted by an editor
# that normalizes look-alike glyphs:
#   U+FF5C  ｜  FULLWIDTH VERTICAL LINE  (token "bar")
#   U+2581  ▁  LOWER ONE EIGHTH BLOCK   (token "underscore")
_DS_BAR = "｜"
_DS_USC = "▁"
_DS_CALL_BEGIN = f"<{_DS_BAR}tool{_DS_USC}call{_DS_USC}begin{_DS_BAR}>"
_DS_CALL_END = f"<{_DS_BAR}tool{_DS_USC}call{_DS_USC}end{_DS_BAR}>"
_DS_SEP = f"<{_DS_BAR}tool{_DS_USC}sep{_DS_BAR}>"

# A LEADING <think>...</think> reasoning block (Qwen3): markup inside the model's
# OWN reasoning is documentation, never a real call, so the whole block is treated
# as an example-guard span (its char range is added to ``guard_spans``).
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
# Single-backtick inline-code span ``...`` (FALSE-POSITIVE GUARD, HIGH-3): inline
# code is documentation too, so a sigil whose start falls inside one is an example,
# not a real call. Bounded to a SINGLE line (never crosses a newline) so a stray
# unbalanced backtick cannot turn a whole multi-line region into one bogus span.
# (The CLOSED/UNTERMINATED triple-``` fence guard is handled positionally by
# ``_outer_fence_spans``; we no longer DELETE any span from the text — deleting
# corrupted backticks/fences inside a real call's argument values, HIGH-A, and
# re-formed a sigil split across a stripped span into a live token, MED-B.)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# A synthesized tool name must look like a sane identifier; anything else is
# garbage we refuse to emit (LOW-6). Downstream maps unknown names to an error
# anyway, but this stops obviously-bogus names from ever becoming a call.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
# DeepSeek args fence: an inner ```json { ... } ``` local to one call block.
_DS_ARGS_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# DeepSeek name: text after the sep, up to a newline / code fence / first brace.
_DS_NAME_RE = re.compile(r"\s*([^\n{`]+)")
# Qwen2.5/Qwen3 + Nous Hermes-2-Pro/Hermes-3 emit identical <tool_call> blocks.
_QWEN_OPEN = "<tool_call>"
_QWEN_CLOSE = "</tool_call>"
# Mistral/Mixtral/Nemo prefix a JSON array of calls with this literal token.
_MISTRAL_TAG = "[TOOL_CALLS]"
# Llama 3.x: <function=NAME>{...}</function> (name in the tag attribute).
_LLAMA_FN_RE = re.compile(r"<function=([^>]+)>")
_LLAMA_FN_CLOSE = "</function>"
# Llama 3.x: <|python_tag|>{...} (ASCII pipes, distinct from DeepSeek's ｜).
_LLAMA_PY_TAG = "<|python_tag|>"


def _valid_tool_name(name: object) -> bool:
    """True iff ``name`` is a sane tool identifier (LOW-6 defensive check)."""
    return isinstance(name, str) and _TOOL_NAME_RE.match(name) is not None


def _sorted_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort ``spans`` by start and MERGE overlaps so they are non-overlapping.

    ``_index_in_spans`` does an O(log n) ``bisect`` membership test, which is only
    correct on a sorted, NON-overlapping range list: with nested/overlapping spans
    (e.g. a ```...``` fence inside a ``<think>`` block, or an inline run whose
    closing backtick is also a fence's opener) the bisect candidate could be a
    LATER span that misses ``idx`` while an earlier span actually contains it.
    Merging once (O(n log n)) collapses any such overlaps into their union, which
    is EXACTLY what the membership test depends on (True iff idx is in ANY span),
    so guard semantics are preserved. Touching spans (s == prev_end) are merged
    too — equivalent for a half-open ``[start, end)`` membership test.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlapping or touching -> extend the current span
            if e > le:
                merged[-1] = (ls, e)
        else:
            merged.append((s, e))
    return merged


def _index_in_spans(
    idx: int, spans: list[tuple[int, int]], starts: list[int]
) -> bool:
    """True iff char position ``idx`` falls inside any ``[start, end)`` span.

    ``spans`` MUST be sorted by start and non-overlapping (see ``_sorted_spans``)
    and ``starts`` its parallel list of start offsets, PRECOMPUTED ONCE by the
    caller so each membership test is O(log n) instead of an O(n) linear scan
    (the old ``any(...)`` was O(n) per call -> O(n^2) on degenerate backtick runs).
    """
    pos = bisect.bisect_right(starts, idx)
    return pos > 0 and idx < spans[pos - 1][1]


def _outer_fence_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of ```...``` GUARD fences (computed once for both scan paths).

    These are fenced code blocks that hold documentation/examples, so any
    tool-call markup inside them must NOT execute. The span set is:
      * each CLOSED pair of triple-backtick markers, in order; PLUS
      * an ODD trailing ``` with no closer -> a span running to end-of-string
        (HIGH-1: a forgotten / token-capped closing fence still hides its body).
    A surviving DeepSeek call's OWN inner ```json args fence is EXCLUDED: that
    fence is a REAL call's argument body, not an example, so counting its two
    markers could mis-pair the remaining fences and hide the real call (MED-4).
    """
    # 1) Protect each DeepSeek call's inner args fence: a ```...``` that appears
    #    right after <｜tool▁sep｜>NAME, bounded to that one call's segment.
    protected: list[tuple[int, int]] = []
    pos = 0
    while True:
        s = text.find(_DS_SEP, pos)
        if s == -1:
            break
        after_start = s + len(_DS_SEP)
        seg_end = len(text)
        for tok in (_DS_CALL_BEGIN, _DS_CALL_END):
            j = text.find(tok, after_start)
            if j != -1:
                seg_end = min(seg_end, j)
        m = _DS_ARGS_FENCE_RE.search(text[after_start:seg_end])
        if m:
            protected.append((after_start + m.start(), after_start + m.end()))
        pos = after_start
    # 2) Pair the triple-backtick markers that are NOT part of a protected fence.
    #    Precompute the sorted protected index ONCE so the per-marker membership
    #    test is O(log n), not an O(n) scan over every protected span.
    protected = _sorted_spans(protected)
    prot_starts = [s for s, _ in protected]
    markers = [
        mm.start()
        for mm in re.finditer(r"```", text)
        if not _index_in_spans(mm.start(), protected, prot_starts)
    ]
    spans: list[tuple[int, int]] = []
    k = 0
    while k < len(markers):
        open_i = markers[k]
        if k + 1 < len(markers):
            spans.append((open_i, markers[k + 1] + 3))  # CLOSED pair
            k += 2
        else:
            spans.append((open_i, len(text)))  # unterminated trailing fence -> EOS
            k += 1
    return spans


def _inline_code_spans(text: str, fence_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Char spans of single-backtick ``...`` inline-code runs OUTSIDE the triple
    fences (HIGH-3 guard). Computed on the ORIGINAL text; any match whose start
    falls inside a ```...``` ``fence_spans`` range is dropped so the triple-backtick
    markers of a fence are never mistaken for inline code (mirrors the old behavior
    of stripping fences BEFORE inline spans, but without mutating the text).
    """
    # Precompute the sorted fence index ONCE: a degenerate backtick run yields
    # tens of thousands of inline matches, each previously doing an O(n) scan over
    # the fence spans (the O(n^2) hot path). bisect makes each check O(log n).
    fence_sorted = _sorted_spans(fence_spans)
    fence_starts = [s for s, _ in fence_sorted]
    spans: list[tuple[int, int]] = []
    for m in _INLINE_CODE_RE.finditer(text):
        if not _index_in_spans(m.start(), fence_sorted, fence_starts):
            spans.append((m.start(), m.end()))
    return spans


def _slice_balanced(s: str, open_ch: str, close_ch: str) -> str:
    """Return the first balanced ``open_ch``..``close_ch`` span in ``s``.

    String-aware (ignores delimiters inside JSON "..." strings, honoring \\
    escapes) so nested objects/arrays slice correctly. On an unbalanced run the
    remainder from the first opener is returned so the caller's JSON coercion can
    flag it as malformed rather than silently truncating.
    """
    start = s.find(open_ch)
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]


def _coerce_args(raw) -> tuple[dict, str | None]:
    """Coerce a raw arguments value to a dict. Returns ``(args, error_or_None)``.

    Mirrors the native path's contract (arguments is ALWAYS a dict; malformed ->
    {} + a parse-error string). A ``str`` is json.loads'd (and re-loaded once if
    that yields another JSON string — the args-as-string shape some Qwen builds
    emit). A non-object result is treated as malformed.
    """
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}, None
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError) as exc:
            return {}, str(exc)
        if isinstance(obj, str):  # double-encoded args: json.loads once more
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, ValueError) as exc:
                return {}, str(exc)
        return (obj, None) if isinstance(obj, dict) else ({}, "arguments not an object")
    return {}, "arguments not an object"


_PARAM_OPEN_RE = re.compile(r"<parameter\s*=\s*([^>]+?)\s*>")
_PARAM_CLOSE = "</parameter>"
_FUNCTION_CLOSE = "</function>"
_TOOLCALL_CLOSE = "</tool_call>"
# Every marker that ENDS a function body or a parameter value. qwen frequently
# omits </parameter> and </function> and closes only with </tool_call> (or with
# nothing — the next <tool_call>/<function= simply begins), so a value/body must
# stop at the NEAREST of any of these or it swallows the closer and the next call.
_XML_TERMINATORS = (_PARAM_CLOSE, _FUNCTION_CLOSE, _TOOLCALL_CLOSE,
                    "<tool_call", "<function=")


def _nearest_xml_boundary(body: str, start: int) -> int:
    """Index of the nearest call/closer marker at/after ``start`` (or len(body)).

    Used only for a NO-CLOSER or truncated parameter value — bounds it at the
    enclosing </function>/</tool_call> or the next call opener so it never spills
    into the following call's markup."""
    end = len(body)
    for marker in (_FUNCTION_CLOSE, _TOOLCALL_CLOSE, "<tool_call", "<function="):
        k = body.find(marker, start)
        if k != -1:
            end = min(end, k)
    return end


# Param keys that are genuinely scalar across llmc's tools (offsets, limits,
# timeouts, booleans). ONLY these are type-coerced from the XML param text; every
# other key — crucially the string BODIES path/content/old/new/command/pattern —
# stays a verbatim string so a one-line content of "42"/"true" is never turned
# into an int/bool that write_file/edit_file would then reject.
_SCALAR_PARAM_KEYS = frozenset({
    "offset", "limit", "timeout", "overwrite", "max_files", "top_k", "full",
})


def _coerce_param_value(raw: str, key: str):
    """Coerce one ``<parameter=KEY>VALUE</parameter>`` body to a Python value.

    Conservative + key-aware: a value is only ever turned into a bool/int/None
    when KEY is a known SCALAR argument (offset/limit/timeout/overwrite/...).
    Every other key — and any multi-line value (a file body / here-doc) — is kept
    verbatim as a string, since that is exactly the write_file/edit_file content
    and must never be mangled into another type.
    """
    s = raw.strip()
    if key not in _SCALAR_PARAM_KEYS:
        return s
    if "\n" in s or "\r" in s:
        return s
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":
        return None
    if re.fullmatch(r"-?\d+", s):
        try:
            return int(s)
        except ValueError:
            return s
    return s


# Bound the body scanned for params so a pathological run of many openers with no
# closers stays linear-ish (model output is token-bounded already; this just makes
# the worst case deterministic). ~1 MB is far above any real tool-call body.
_MAX_PARAM_BODY = 1_000_000


def _parse_xml_params(body: str) -> dict:
    """Parse a ``<function=NAME>`` body written as ``<parameter=KEY>VALUE</parameter>``
    tags (the Qwen3 / Hermes XML tool-call style) into an arguments dict.

    Robust to a TRUNCATED final value: each value runs to the nearest of its own
    ``</parameter>``, the next ``<parameter=``, the enclosing ``</function>``, or
    end-of-text — so a write_file whose huge ``content`` was cut off mid-stream
    still yields ``{path, content}`` (content = the partial body) rather than
    nothing. (The truncation-recovery retry then re-issues it uncapped.)
    """
    if len(body) > _MAX_PARAM_BODY:
        body = body[:_MAX_PARAM_BODY]
    args: dict = {}
    opens = list(_PARAM_OPEN_RE.finditer(body))
    # CLOSER form (`<parameter=K>V</parameter>`) vs NO-CLOSER form (qwen often
    # emits `<parameter=K> V <parameter=K2> ...` with no </parameter> at all). If
    # ANY </parameter> is present, the close tag is the authoritative value
    # boundary — so a write_file CONTENT value that itself contains "<parameter=",
    # "<function=" or "</tool_call>" is PRESERVED up to its own </parameter>. A
    # `cursor` then skips every <parameter= opener that sits INSIDE that consumed
    # value, so inner markup is not mis-parsed as a bogus extra arg. Only when
    # there is no </parameter> at all do we split on the next opener.
    has_closers = _PARAM_CLOSE in body
    cursor = 0
    for i, m in enumerate(opens):
        if m.start() < cursor:  # opener lies inside an already-consumed value
            continue
        key = m.group(1).strip()
        if not key:
            continue
        val_start = m.end()
        if has_closers:
            close = body.find(_PARAM_CLOSE, val_start)
            if close != -1:
                val_end = close
                cursor = close + len(_PARAM_CLOSE)
            else:  # a trailing/truncated param with no close of its own
                val_end = _nearest_xml_boundary(body, val_start)
                cursor = val_end
        else:
            val_end = len(body)
            if i + 1 < len(opens):
                val_end = min(val_end, opens[i + 1].start())
            val_end = min(val_end, _nearest_xml_boundary(body, val_start))
            cursor = val_end
        args[key] = _coerce_param_value(body[val_start:val_end], key)
    return args


def _name_args_from_object(raw: str, name_key: str, args_keys: tuple[str, ...]):
    """Parse a ``{"name":.., "<args_key>":..}`` object string.

    Returns ``(name, args, error_or_None)``. If the object JSON itself is
    malformed, the name is salvaged via a narrow regex so a broken call is still
    surfaced (args={} + the parse error) instead of being silently dropped.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
        return (nm.group(1) if nm else None), {}, str(exc)
    if not isinstance(data, dict):
        return None, {}, None
    name = data.get(name_key)
    raw_args: object = {}
    for k in args_keys:
        if k in data:
            raw_args = data[k]
            break
    args, err = _coerce_args(raw_args)
    return name, args, err


def _parse_deepseek(text: str, guard_spans: list[tuple[int, int]] | None = None) -> list[dict] | None:
    """Extract DeepSeek-V2/coder-v2 text tool calls anchored on the begin token.

    Per call: ``<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\\n```json\\n{ARGS}\\n
    ```<｜tool▁call▁end｜>``. The outer ``<｜tool▁calls▁end｜>`` and even a single
    call's ``call▁end`` may be absent, so each call's span runs to the next
    call-end / next call-begin / end-of-text. Anchoring on the begin token (and
    parsing the INNER ```json fence locally) is why DeepSeek runs BEFORE the
    global code-fence stripping — that stripping would otherwise eat its args.

    ``guard_spans`` are the shared example-guard ranges (closed/unterminated
    ```...``` fences, single-backtick inline code, and a leading <think> block):
    a begin token whose start falls INSIDE one is a fenced/inline EXAMPLE of the
    format, not a real call, and is skipped (MED-4 — the SAME positional guard the
    other three formats now use instead of mutating the text).
    """
    spans = _sorted_spans(guard_spans or [])
    starts = [s for s, _ in spans]
    calls: list[dict] = []
    pos = 0
    while True:
        i = text.find(_DS_CALL_BEGIN, pos)
        if i == -1:
            break
        if _index_in_spans(i, spans, starts):  # fenced example -> not a real call
            pos = i + len(_DS_CALL_BEGIN)
            continue
        seg_start = i + len(_DS_CALL_BEGIN)
        end_i = text.find(_DS_CALL_END, seg_start)
        next_i = text.find(_DS_CALL_BEGIN, seg_start)
        bounds = [x for x in (end_i, next_i) if x != -1]
        seg_end = min(bounds) if bounds else len(text)
        segment = text[seg_start:seg_end]
        pos = seg_end
        sep_i = segment.find(_DS_SEP)
        if sep_i == -1:  # no separator -> cannot trust a name; skip this block
            continue
        after_sep = segment[sep_i + len(_DS_SEP) :]
        nm = _DS_NAME_RE.match(after_sep)
        name = nm.group(1).strip() if nm else ""
        if not _valid_tool_name(name):  # empty or non-identifier name -> skip (LOW-6)
            continue
        # ARGS: prefer the inner ```json fence, else the first balanced { ... }.
        fm = _DS_ARGS_FENCE_RE.search(after_sep)
        raw_args = fm.group(1) if fm else _slice_balanced(after_sep, "{", "}")
        args, err = _coerce_args(raw_args)
        call: dict = {"name": name, "arguments": args}
        if err:
            call["_parse_error"] = err
        calls.append(call)
    return calls or None


def _parse_qwen(text: str, guard_spans: list[tuple[int, int]] | None = None) -> list[dict] | None:
    """Extract Qwen2.5/Qwen3 + Hermes-2-Pro/Hermes-3 ``<tool_call>`` JSON blocks.

    Each block is a ``{"name":.., "arguments":{..}}`` object; ``arguments`` may
    be a dict OR a JSON string (handled by _coerce_args). A trailing block whose
    ``</tool_call>`` is missing is still bounded by the next opener / EOS.

    Scans the ORIGINAL (never-mutated) text: an opener whose start falls inside a
    ``guard_spans`` example range (fence / inline code / <think>) is skipped, while
    the JSON body of a REAL call is sliced from the original so backticks/fences
    inside its argument values are preserved EXACTLY (HIGH-A).
    """
    if _QWEN_OPEN not in text:
        return None
    spans = _sorted_spans(guard_spans or [])
    starts = [s for s, _ in spans]
    calls: list[dict] = []
    pos = 0
    while True:
        i = text.find(_QWEN_OPEN, pos)
        if i == -1:
            break
        if _index_in_spans(i, spans, starts):  # fenced/inline example -> not a real call
            pos = i + len(_QWEN_OPEN)
            continue
        start = i + len(_QWEN_OPEN)
        close_i = text.find(_QWEN_CLOSE, start)
        next_i = text.find(_QWEN_OPEN, start)
        bounds = [x for x in (close_i, next_i) if x != -1]
        end = min(bounds) if bounds else len(text)
        pos = end
        seg = text[start:end]
        # A <tool_call> whose body is the Qwen XML-PARAM form (<function=NAME>
        # <parameter=…>) — i.e. NOT a JSON object — is owned by _parse_llama. Bail
        # so we never _slice_balanced a `{ … }` out of a write_file CONTENT value
        # and mis-fire it as a tool (content-to-execution hazard). Gated on the
        # body NOT starting with '{' so a legitimate JSON call whose argument
        # STRING happens to contain "<function="/"<parameter=" is still parsed
        # here (string-aware _slice_balanced keeps those tokens inside the value).
        if not seg.lstrip().startswith("{") and ("<function" in seg or "<parameter" in seg):
            continue
        raw = _slice_balanced(seg, "{", "}")
        if not raw:
            continue
        name, args, err = _name_args_from_object(raw, "name", ("arguments",))
        if not _valid_tool_name(name):  # missing / non-identifier name (LOW-6)
            continue
        call: dict = {"name": name, "arguments": args}
        if err:
            call["_parse_error"] = err
        calls.append(call)
    return calls or None


def _parse_mistral(text: str, guard_spans: list[tuple[int, int]] | None = None) -> list[dict] | None:
    """Extract Mistral/Mixtral/Nemo ``[TOOL_CALLS] [{...}, ...]`` array calls.

    Each array element is one call; a per-element ``id`` (if present) is ignored.
    Scans the ORIGINAL text: the first ``[TOOL_CALLS]`` tag NOT inside a
    ``guard_spans`` example range is used, and its array is sliced from the
    original so backticks/fences inside argument values are preserved (HIGH-A).
    """
    spans = _sorted_spans(guard_spans or [])
    starts = [s for s, _ in spans]
    pos = 0
    while True:
        i = text.find(_MISTRAL_TAG, pos)
        if i == -1:
            return None
        if not _index_in_spans(i, spans, starts):  # first non-fenced tag wins
            break
        pos = i + len(_MISTRAL_TAG)
    after = text[i + len(_MISTRAL_TAG) :]
    arr_raw = _slice_balanced(after, "[", "]")
    if not arr_raw:
        return None
    try:
        arr = json.loads(arr_raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(arr, list):
        return None
    calls: list[dict] = []
    for el in arr:
        if not isinstance(el, dict):
            continue
        name = el.get("name")
        if not _valid_tool_name(name):  # missing / non-identifier name (LOW-6)
            continue
        args, err = _coerce_args(el.get("arguments", {}))
        call: dict = {"name": name, "arguments": args}
        if err:
            call["_parse_error"] = err
        calls.append(call)
    return calls or None


def _parse_llama(text: str, guard_spans: list[tuple[int, int]] | None = None) -> list[dict] | None:
    """Extract Llama 3.1/3.2/3.3 text tool calls (two markup variants).

    A) ``<function=NAME>{JSON}</function>`` — name is in the tag attribute and
       the body is the BARE args object (no name key inside). Supports multiple.
       ALSO handles the Qwen3 / Hermes XML body ``<function=NAME>
       <parameter=KEY>VALUE</parameter>...</function>`` — the dominant qwen3.x
       shape (LM Studio often drops the param body server-side, yielding an empty
       native call, so parsing the text form here is what makes its writes land).
    B) ``<|python_tag|>{JSON}`` — JSON carries ``"name"`` + (``"parameters"`` OR
       ``"arguments"``).
    The pythonic ``[func(a='x')]`` bracket form is DELIBERATELY skipped: it is
    too ambiguous (ordinary prose lists look identical) and would false-fire.

    Scans the ORIGINAL text: a tag whose start falls inside a ``guard_spans``
    example range (fence / inline code / <think>) is skipped, while a real call's
    JSON body is sliced from the original so backticks/fences inside its argument
    values are preserved EXACTLY (HIGH-A).
    """
    spans = _sorted_spans(guard_spans or [])
    starts = [s for s, _ in spans]
    # Variant A: <function=NAME>{...}</function> (one or more).
    calls: list[dict] = []
    for m in _LLAMA_FN_RE.finditer(text):
        if _index_in_spans(m.start(), spans, starts):  # fenced/inline example -> skip
            continue
        name = m.group(1).strip()
        if not _valid_tool_name(name):  # non-identifier tag attribute (LOW-6)
            continue
        rest = text[m.end() :]
        # Bound this function's body. PREFER a closer (</function> or </tool_call>):
        # when one exists, the body runs to it, so inner <function=/<parameter=/
        # </tool_call> tokens inside a CONTENT value are preserved (not truncated).
        # Only when there is NO closer (qwen's no-closer run of
        # <tool_call><function=..><parameter=..></tool_call> blocks) do we split at
        # the next call opener so concatenated calls don't merge into the first.
        fn_close = rest.find(_LLAMA_FN_CLOSE)
        tc_close = rest.find(_TOOLCALL_CLOSE)
        closers = [x for x in (fn_close, tc_close) if x != -1]
        if closers:
            end = min(closers)
        else:
            end = len(rest)
            for marker in ("<function=", "<tool_call"):
                j = rest.find(marker)
                if j != -1:
                    end = min(end, j)
        body = rest[:end]
        err = None
        if body.lstrip().startswith("{"):
            # JSON body (Llama 3.x): <function=NAME>{ ...args... }</function>.
            args, err = _coerce_args(_slice_balanced(body, "{", "}"))
        elif "<parameter" in body:
            # XML-param body (Qwen3 / Hermes): <function=NAME>
            #   <parameter=path>...</parameter><parameter=content>...</parameter>
            args = _parse_xml_params(body)
            if not args:  # opener present but no usable params -> not a real call
                continue
        else:
            # HIGH-2: a bare <function=NAME> with no JSON object and no parameter
            # tags (or a body whose nearest '{' is distant prose) is NOT a call.
            continue
        call: dict = {"name": name, "arguments": args}
        if err:
            call["_parse_error"] = err
        calls.append(call)
    if calls:
        return calls
    # Variant B: <|python_tag|>{...} (name + parameters/arguments inside). Use the
    # first tag NOT inside a guard span.
    pos = 0
    while True:
        i = text.find(_LLAMA_PY_TAG, pos)
        if i == -1:
            break
        if _index_in_spans(i, spans, starts):  # fenced/inline example -> skip
            pos = i + len(_LLAMA_PY_TAG)
            continue
        after = text[i + len(_LLAMA_PY_TAG) :]
        raw = _slice_balanced(after, "{", "}")
        if raw:
            name, args, err = _name_args_from_object(raw, "name", ("parameters", "arguments"))
            if _valid_tool_name(name):  # LOW-6
                call = {"name": name, "arguments": args}
                if err:
                    call["_parse_error"] = err
                return [call]
        break
    return None


def extract_text_tool_calls(text: str) -> list[dict] | None:
    """Recognize a local model's OWN chat-template tool-call markup in plain text.

    Returns a list of ``{"name": str, "arguments": dict, ["_parse_error": str]}``
    for the FIRST matching high-confidence format (formats are never mixed), else
    None. Used by ``LocalProvider.stream_chat`` as the HIGH-confidence fallback
    when the server did not convert the markup to native ``tool_calls`` — runs
    BEFORE the strict generic ```json fence path so an in-template call is
    executed instead of being shown as prose (and its hallucinated trailing
    "output" + answer discarded).
    """
    if not text:
        return None
    # FAST PATH (the common turn is pure chat with NO tool-call markup): scan for
    # any known high-confidence tool-call SIGIL before doing ANY regex/full-text
    # work. These five literals are the ONLY triggers the four parsers below key on
    # (deepseek begin token / <tool_call> / [TOOL_CALLS] / <function= / <|python_tag|>),
    # so a real call ALWAYS carries one and the short-circuit can never drop a
    # genuine call. When none is present, return None immediately -- skipping
    # _outer_fence_spans / _inline_code_spans / _THINK_RE / the four parser passes
    # (notably _LLAMA_FN_RE.finditer over the full text), which on a large no-sigil
    # answer is ~3 full-text regex passes for nothing.
    #
    # A bare ``` fence with NO sigil is a normal prose answer that merely contains a
    # code block: it can never be one of these high-confidence formats, so we return
    # None here rather than running the expensive sweeps. The caller's SEPARATE
    # generic ```json fence path (parse_tool_block, which self-short-circuits on
    # "``` not in text") still handles a genuine fenced-json tool block, so no real
    # call is lost by leaving ``` out of this SIGIL set.
    if (
        _DS_CALL_BEGIN not in text
        and _QWEN_OPEN not in text
        and _MISTRAL_TAG not in text
        and "<function=" not in text
        and _LLAMA_PY_TAG not in text
    ):
        return None
    # Compute the example-GUARD spans ONCE over the ORIGINAL (never-mutated) text,
    # shared by ALL four parsers as a positional skip-list (no text is ever
    # mutated — that mutation deleted backticks/fences inside a real call's
    # argument values, corrupting args (HIGH-A), and re-formed a sigil split across
    # a stripped span into a live token (MED-B)). The guard set is:
    #   * CLOSED ```...``` fence pairs + an UNTERMINATED trailing ``` to EOS
    #     (HIGH-1), with a real DeepSeek call's OWN inner args fence excluded (MED-4)
    #   * single-backtick `...` inline-code spans OUTSIDE those fences (HIGH-3)
    #   * a leading <think>...</think> reasoning block (Qwen3)
    # A tool-call SIGIL whose START index falls inside any guard span is a fenced /
    # inline EXAMPLE (or reasoning), never a real call, and is skipped; the JSON
    # body of a REAL call is always sliced from the ORIGINAL text so backticks /
    # fences inside its argument values survive verbatim.
    #
    # LOW-B (known, safe-direction limitation): a real call that appears AFTER an
    # earlier UNTERMINATED / odd ``` falls inside the trailing-fence-to-EOS guard
    # span and is therefore DROPPED rather than executed. Dropping a possibly-real
    # call is the safe failure mode (the model can re-emit it); executing markup
    # hidden in an unterminated fence is not.
    fence_spans = _outer_fence_spans(text)
    guard_spans = fence_spans + _inline_code_spans(text, fence_spans)
    tm = _THINK_RE.match(text)
    if tm:
        guard_spans.append((tm.start(), tm.end()))
    # DeepSeek FIRST: anchor on its non-ASCII begin token and parse its inner
    # ```json args fence LOCALLY from the original text. Its control tokens are
    # unambiguous, EXCEPT for a begin token that sits inside a guard span (an
    # example). DELIBERATE: unfenced prose markup that is a syntactically valid
    # call is still treated as a REAL call (confirmation gating is the backstop for
    # dangerous tools; heuristics over free prose were rejected as unreliable).
    if _DS_CALL_BEGIN in text:
        ds = _parse_deepseek(text, guard_spans)
        if ds:
            return ds
    for parser in (_parse_qwen, _parse_mistral, _parse_llama):
        out = parser(text, guard_spans)
        if out:
            return out
    return None


_MODELS_TIMEOUT = 5  # seconds; bound the blocking model-list round-trip
# Generation timeout for chat/streaming. Must be GENEROUS: a local reasoning
# model processing a large context (e.g. after reading many files) can take well
# over a minute just to produce its first token. Reusing the tiny _MODELS_TIMEOUT
# here caused "[stream error: ReadTimeout]" on big-context turns.
_GEN_TIMEOUT = 600  # seconds


def list_local_models(base_url: str, api_key: str, private: bool = False) -> list[str]:
    """Return available model ids from an OpenAI-compatible server (e.g. LM Studio).

    Raises on connection/HTTP failure so the caller can decide how to treat an
    unverifiable server. openai is imported lazily to keep it optional. A short
    client timeout bounds the call so a hung server cannot freeze the REPL.

    In --private lockdown mode the client ignores proxy env vars (trust_env=False)
    and IP-pins the loopback host so a HTTP(S)_PROXY/ALL_PROXY cannot tunnel even
    this loopback metadata call to an external proxy. In the DEFAULT (network-on)
    mode the standard client is used (proxy env honored).
    """
    from openai import OpenAI  # noqa: PLC0415

    kwargs: dict = {"base_url": base_url, "api_key": api_key, "timeout": _MODELS_TIMEOUT}
    if private:
        import httpx  # noqa: PLC0415

        # IP-PINNING (DNS-rebinding defense, finding #1): resolve+validate the
        # host once and connect to that literal IP so httpx cannot re-resolve a
        # hostname to a public address between calls. The original Host header is
        # preserved so the local server still sees the intended vhost.
        pinned_url, host_header = _pin_loopback_base_url(base_url)
        kwargs["base_url"] = pinned_url
        headers = {"Host": host_header} if host_header else None
        kwargs["http_client"] = httpx.Client(
            trust_env=False, timeout=_MODELS_TIMEOUT, headers=headers
        )
    client = OpenAI(**kwargs)
    resp = client.models.list()
    return [m.id for m in (getattr(resp, "data", None) or [])]


def _pick_context_length(models, model: str) -> int | None:
    """From LM Studio /api/v0/models 'data', the context window for ``model``.

    Prefers an exact id match (handles multiple models loaded at once); else any
    loaded model. None when unknown. Pure (no network) so it is unit-testable.
    """
    if not isinstance(models, list):
        return None

    def ctx(m: dict):
        v = m.get("loaded_context_length") or m.get("max_context_length")
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None

    for m in models:  # exact, loaded model id wins
        if isinstance(m, dict) and m.get("id") == model and ctx(m):
            return ctx(m)
    for m in models:  # otherwise any loaded model's window
        if isinstance(m, dict) and m.get("state") == "loaded" and ctx(m):
            return ctx(m)
    return None


def detect_context_length(base_url: str, model: str, timeout: float = 3.0) -> int | None:
    """Best-effort: the loaded model's context window via LM Studio's native API.

    LM Studio exposes ``/api/v0/models`` (non-OpenAI) with per-model
    loaded_context_length/max_context_length. Returns the window for ``model``,
    else None on ANY failure (endpoint absent, server down, parse error) so the
    caller falls back to a conservative default. stdlib only.
    """
    import urllib.request  # noqa: PLC0415

    try:
        p = urllib.parse.urlparse(base_url)
        api = f"{p.scheme}://{p.netloc}/api/v0/models"
        with urllib.request.urlopen(api, timeout=timeout) as resp:  # noqa: S310 - loopback metadata
            data = json.loads(resp.read(1_000_000).decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - best-effort; any failure -> None
        return None
    return _pick_context_length(data.get("data") if isinstance(data, dict) else None, model)


def effort_extra_body(effort: str) -> dict:
    """Map an /effort level to LM Studio request-body extras (best-effort).

    low/medium/high -> the standard ``reasoning_effort`` param. ``off`` tries to
    disable thinking (``enable_thinking`` via chat_template_kwargs + minimal
    effort). "" -> nothing. Returned dict is passed as the OpenAI SDK
    ``extra_body`` so non-standard keys reach the server untouched. NOTE: honored
    only by models/servers that support these; some (e.g. qwen3.6 on LM Studio)
    ignore them entirely.
    """
    e = (effort or "").strip().lower()
    if e in ("low", "medium", "high"):
        return {"reasoning_effort": e}
    if e == "off":
        return {
            "reasoning_effort": "minimal",
            "chat_template_kwargs": {"enable_thinking": False},
        }
    return {}


# Reasoning-wrapper tags some models emit around their chain-of-thought even
# inside the reasoning_content channel. Stripped before the reasoning is ever
# surfaced as a fallback answer (see stream_chat) so a recovered answer is not
# cluttered with <think>…</think> scaffolding.
_THINK_TAG_RE = re.compile(r"</?(?:think|thinking|thought|reason(?:ing)?)\b[^>]*>", re.IGNORECASE)


def _clean_reasoning_text(text: str) -> str:
    """Strip ``<think>``/``</think>``-style wrapper tags and trim.

    Used ONLY when surfacing reasoning as a last-resort answer for an otherwise
    empty turn (reasoning model routed everything into reasoning_content). Removes
    the wrapper tags (open OR close, any of the common think/thinking/reason
    variants) and trims surrounding whitespace. Returns "" when nothing
    meaningful remains so the caller can decide not to emit it.
    """
    if not text:
        return ""
    return _THINK_TAG_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------

class Provider(ABC):
    name: str = "base"

    @abstractmethod
    def stream_chat(
        self, messages: list[dict], tools: list[dict] | None,
        tool_choice: str | None = None,
    ) -> Iterator[dict]:
        """Yield normalized text/tool_call/done events for one assistant turn.

        ``tool_choice`` (Feature 2): when set to the STRING "required" (the only
        form local engines support; the object form is not), the request forces a
        native tool call this turn. The agent uses it to retry a malformed tool
        call once. None = the server default (send nothing). Meaningless without
        ``tools``.
        """
        raise NotImplementedError


def effective_max_tokens(
    base: int | None, gentle_mode: bool, gentle_max_tokens: int | None,
) -> int | None:
    """The output-token cap actually sent to the server for one generation.

    ``base`` is the user/config cap (``None``/0/negative all mean "no cap").
    When ``gentle_mode`` is on, the effective cap is the MINIMUM of the base cap
    and ``gentle_max_tokens`` — so gentle only LOWERS an unset/larger cap and
    NEVER raises a smaller existing one. When gentle is off (or its cap is unset)
    the base cap is returned unchanged. Returns ``None`` when no cap applies, so
    callers omit ``max_tokens`` entirely (the LM Studio "unbounded" form).
    """
    base_cap = base if (isinstance(base, int) and not isinstance(base, bool) and base > 0) else None
    if not gentle_mode:
        return base_cap
    g = (
        gentle_max_tokens
        if (isinstance(gentle_max_tokens, int) and not isinstance(gentle_max_tokens, bool) and gentle_max_tokens > 0)
        else None
    )
    if g is None:
        return base_cap
    if base_cap is None:
        return g
    return min(base_cap, g)


# ---------------------------------------------------------------------------
# LocalProvider (LM Studio via OpenAI SDK) — openai imported lazily
# ---------------------------------------------------------------------------

class LocalProvider(Provider):
    name = "local"

    def __init__(
        self, model: str, base_url: str, api_key: str, effort: str = "",
        private: bool = False, cache_prompt: bool = False,
        max_output_tokens: int | None = None, embed_model: str | None = None,
        temperature: float = 0.2,
        gentle_mode: bool = False, gentle_max_tokens: int = 1024,
        seed: int | None = None,
        id_slot: int | None = None,
    ):
        """Deterministic scripted provider for offline tests + the smoke run.

        ``seed``: optional seed for deterministic scenario selection. When set,
        the mock provider cycles through scenarios based on (seed + message_count)
        so test runs are reproducible. When None, scenarios cycle 0, 1, 2, ...
        """
        self.seed = seed
        # ID_SLOT pin (Win 2): when set, every request carries extra_body.id_slot
        # so the llama.cpp server keeps the prefix KV cache on a fixed slot across
        # turns (re-prefill only the delta). None = do not pin (omit the key).
        self.id_slot = id_slot
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        # Sampling temperature sent on every request. LOW by default for
        # deterministic, well-formed code/tool-call output (Feature 1); the server
        # default (~0.7-0.8) is the root cause of malformed tool calls locally.
        self.temperature = temperature
        # Embedding model id for the /v1/embeddings endpoint (retrieval). None =
        # not configured: embeddings() raises a clear ValueError that callers
        # catch and fall back to BM25-only retrieval.
        self.embed_model = embed_model
        self.effort = effort  # reasoning-effort level (see effort_extra_body)
        # Opt-in per-request generation cap. When set, stream_chat sends
        # max_tokens=N so the server stops after N tokens. None = unbounded (no
        # max_tokens key is sent — omitting it is how LM Studio leaves it
        # uncapped; -1 is NOT honored as unbounded).
        self.max_output_tokens = max_output_tokens
        # GENTLE mode: when on, the effective per-request cap is lowered to
        # min(max_output_tokens, gentle_max_tokens) so generation bursts are
        # shorter (less average GPU load/heat). Computed at the single point in
        # stream_chat via effective_max_tokens(); never RAISES a smaller existing
        # cap. Sub-agents SHARE this provider instance, so they inherit the cap
        # automatically. Default off here (build_provider passes the config).
        self.gentle_mode = bool(gentle_mode)
        self.gentle_max_tokens = gentle_max_tokens
        # KV-cache reuse hint for llama.cpp/LM Studio (finding #12). Opt-in:
        # adds extra_body.cache_prompt=true so the server is explicitly told to
        # reuse the prompt-cache across turns. The OpenAI SDK passes extra_body
        # untouched; servers that don't recognize the key ignore it (same
        # backward-compatible pattern as reasoning_effort).
        self.cache_prompt = bool(cache_prompt)
        # PRIVATE mode: build the client with a proxy-env-ignoring httpx client
        # so HTTP(S)_PROXY/ALL_PROXY cannot tunnel even loopback traffic off-box.
        self.private = private
        # Native tool-call support is decided dynamically per turn: if the server
        # streams no tool_calls we fall back to parsing fenced text blocks. There
        # is no static capability flag.
        self._client = None  # built lazily

    def with_id_slot(self, id_slot: int | None) -> "LocalProvider":
        """Return a shallow clone with ``id_slot`` overridden, sharing this
        provider's client + config.

        Used to give a spawned sub-agent a provider that does NOT pin the
        orchestrator's slot 0. Sub-agents reuse the SAME LocalProvider instance,
        so a sub-agent's large, distinct prompt (also on slot 0) would evict the
        orchestrator's warm slot-0 KV prefix on the llama.cpp server, forcing a
        full re-prefill when control returns to the orchestrator. Passing
        ``id_slot=None`` omits the id_slot key for the sub-agent so its requests
        do not collide with (evict) the orchestrator's slot 0. copy.copy keeps the
        same ``_client`` reference, so no second connection/handshake is created.
        """
        clone = copy.copy(self)
        clone.id_slot = id_slot
        return clone

    def _get_client(self):
        if self._client is None:
            # LAZY IMPORT: keep openai out of the module import graph so the
            # rest of the app runs offline without openai installed.
            from openai import OpenAI  # noqa: PLC0415

            client_kwargs: dict = {
                "base_url": self.base_url,
                "api_key": self.api_key,
                # Generous timeout: this client drives streaming generation, which
                # can be slow for a reasoning model on a large context. (Metadata
                # calls like list_local_models use their own short-timeout client.)
                "timeout": _GEN_TIMEOUT,
            }
            if self.private:
                # PROXY-TUNNELING DEFENSE: openai's default httpx client uses
                # trust_env=True, so HTTP_PROXY/HTTPS_PROXY/ALL_PROXY would route
                # even loopback requests through an EXTERNAL proxy — exfiltrating
                # prompts + file contents. Build an explicit httpx client with
                # trust_env=False so proxy env vars are ignored entirely. httpx is
                # a transitive dependency of openai (not a new one).
                import httpx  # noqa: PLC0415

                # IP-PINNING (DNS-rebinding TOCTOU defense, finding #1): the
                # base_url host was validated as loopback at build_provider time,
                # but a bare hostname would be RE-RESOLVED by httpx on every
                # chat.completions.create()/models.list() — a low-TTL/split-horizon
                # attacker could then redirect the full conversation off-box. Pin
                # the connection to the literal loopback IP we just re-validated
                # (raises if it no longer resolves to loopback: fail-closed), and
                # carry the original Host header so the local vhost is preserved.
                pinned_url, host_header = _pin_loopback_base_url(self.base_url)
                client_kwargs["base_url"] = pinned_url
                headers = {"Host": host_header} if host_header else None
                client_kwargs["http_client"] = httpx.Client(
                    trust_env=False, timeout=_GEN_TIMEOUT, headers=headers
                )
            elif is_loopback_url(self.base_url):
                # DEFAULT (network-on) HARDENING (finding #1): the default
                # base_url targets a loopback LM Studio server. Even without
                # --private we IP-pin a loopback base_url to the literal address
                # we just validated so httpx cannot RE-RESOLVE a loopback
                # hostname (e.g. 'localhost') to a public address mid-session
                # (DNS-rebinding TOCTOU). External (non-loopback) base_urls are
                # left untouched — pointing at an external host is the documented
                # opt-in behavior of the default mode.
                #
                # PROXY-TUNNELING DEFENSE (finding #3): we ALSO set
                # trust_env=False for the loopback target. IP-pinning alone does
                # NOT stop proxying — with trust_env=True httpx still routes
                # through HTTP(S)_PROXY/ALL_PROXY regardless of the destination IP
                # (unless NO_PROXY happens to exclude loopback), so the full
                # conversation could be exfiltrated to a proxy. Loopback model
                # traffic must never be proxied. The genuinely external
                # (--allow-network) path above is untouched and still honors
                # proxy env vars.
                import httpx  # noqa: PLC0415

                pinned_url, host_header = _pin_loopback_base_url(self.base_url)
                client_kwargs["base_url"] = pinned_url
                headers = {"Host": host_header} if host_header else None
                client_kwargs["http_client"] = httpx.Client(
                    trust_env=False, timeout=_GEN_TIMEOUT, headers=headers
                )
            self._client = OpenAI(**client_kwargs)
        return self._client

    def stream_chat(
        self, messages: list[dict], tools: list[dict] | None,
        tool_choice: str | None = None,
    ) -> Iterator[dict]:
        try:
            client = self._get_client()
        except Exception as exc:  # noqa: BLE001
            # Surface only the exception TYPE, not str(exc): SDK auth/error reprs
            # can include request metadata, and this text is fed into the model's
            # message history.
            yield {"type": "text", "text": f"[provider error: {type(exc).__name__}]"}
            yield {"type": "done", "finish_reason": "error", "output_tokens": None}
            return

        kwargs: dict = {"model": self.model, "messages": messages, "stream": True}
        # Ask the server to stream a final usage chunk (completion_tokens) so we
        # can report an accurate tok/s footer. LM Studio / OpenAI emit this as a
        # trailing chunk with empty .choices and a populated .usage.
        kwargs["stream_options"] = {"include_usage": True}
        # Reasoning-effort extras (best-effort; ignored by servers that don't
        # support them). Passed via extra_body so non-standard keys reach LM Studio.
        extra_body = effort_extra_body(self.effort)
        # KV-cache reuse hint (finding #12): tell llama.cpp/LM Studio to reuse the
        # prompt cache across turns. The system prompt + history are an append-only
        # stable prefix, so an explicit cache_prompt:true locks in the multi-turn
        # prefill speedup instead of relying on the server's default heuristics.
        if self.cache_prompt:
            extra_body = {**extra_body, "cache_prompt": True}
        # ID_SLOT pin (Win 2): pin every request to a fixed llama.cpp slot so the
        # prefix KV cache stays warm across turns. Only sent when set.
        if self.id_slot is not None:
            extra_body = {**extra_body, "id_slot": int(self.id_slot)}
        if extra_body:
            kwargs["extra_body"] = extra_body
        # Opt-in generation cap. LM Studio documents `max_tokens` (NOT
        # max_completion_tokens). The EFFECTIVE cap is computed once here: when
        # gentle mode is on it is min(config cap, gentle cap) so bursts stay
        # short; otherwise it is exactly the config cap. Only sent when set; None
        # leaves it OFF so the default path is byte-identical to before (no
        # max_tokens key at all). This is the SINGLE point the cap is decided, so
        # it reaches sub-agents too (they share this provider instance).
        #
        # CRUCIAL: the gentle token cap is a heat lever for short PURE-CHAT
        # answers. It must NOT apply when `tools` are offered — a write_file/
        # edit_file call carries the ENTIRE file content inside its tool-call JSON
        # arguments, so a 1024-token cap truncates the call mid-stream and the
        # write never lands (the model "did it" but the file is empty / the call
        # arrives with no 'path'). On tool-capable turns we therefore use the
        # config cap ONLY (usually unbounded); between-turn gentle PACING still
        # provides the heat reduction. Pure-chat turns keep the gentle cap.
        # In agentic use the orchestrator + sub-agents almost always offer tools,
        # so the gentle TOKEN cap now guards mainly tool-less turns — PACING is the
        # interactive heat lever. A user's explicit /maxout (max_output_tokens) is
        # untouched and still caps every turn. Net heat DROPS: a write used to
        # truncate at 1024 then re-run uncapped (2 gens); now it's one.
        gentle_for_this_call = self.gentle_mode and not tools
        effective_cap = effective_max_tokens(
            self.max_output_tokens, gentle_for_this_call, self.gentle_max_tokens
        )
        if effective_cap:
            kwargs["max_tokens"] = effective_cap
        # Feature 1: send a LOW sampling temperature so code/tool turns are
        # deterministic and well-formed. Always sent (the server default is the
        # root cause of malformed tool calls), tuned via config.temperature / /temp.
        kwargs["temperature"] = self.temperature
        # SEED (Win 2): forward the configured seed so local turns are reproducible
        # when combined with temperature=0. Only sent when set to a valid value;
        # a negative seed means "no seed" (kept out of the request).
        if self.seed is not None and self.seed >= 0:
            kwargs["seed"] = self.seed
        # Only include tools when we actually have some, so tool-less models work.
        if tools:
            kwargs["tools"] = tools
            # Feature 2: constrained decode. The agent passes tool_choice="required"
            # to force a clean NATIVE tool call when retrying a malformed one. Only
            # the STRING form is supported by LM Studio/llama.cpp (the object form
            # is undefined); tool_choice is meaningless without tools.
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        # Accumulators for streamed native tool calls, keyed by .index.
        tool_buf: dict[int, dict] = {}
        text_acc: list[str] = []
        # Reasoning tokens streamed on the reasoning_content channel. Normally
        # DISCARDED (the agent never sees them as text). Captured here ONLY so an
        # otherwise-empty turn — a reasoning model that routed its whole answer
        # into reasoning_content and emitted no visible content — can surface the
        # reasoning as a last-resort answer at stream end (see below).
        reasoning_acc: list[str] = []
        finish_reason = "stop"
        usage_completion_tokens: int | None = None
        # Generation-speed timing. t_first marks the FIRST generated token of ANY
        # kind — including reasoning_content, which the agent never receives as a
        # text event. Measuring tok/s from here (not from the first VISIBLE token)
        # is essential for reasoning models: their completion_tokens count the
        # reasoning tokens, so the window must include the time spent generating
        # them. (first-token -> done) excludes prompt processing and matches what
        # the server / LM Studio report.
        t_start = time.perf_counter()
        t_first: float | None = None

        # Hold the stream OUTSIDE the loop so the finally can deterministically
        # close it (finding #3). The agent breaks out of consuming this generator
        # on the 'done' event and on Ctrl+C, which suspends us with the underlying
        # httpx stream still open; relying on GC leaks the loopback socket on
        # every interrupted turn. close() releases it as soon as we are resumed
        # (generator .close() runs our finally) or finish normally.
        stream = None
        try:
            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                # Capture usage BEFORE the empty-choices early-skip: the final
                # usage chunk typically has empty .choices but a populated
                # .usage. Reading it first avoids losing completion_tokens.
                u = getattr(chunk, "usage", None)
                if u is not None:
                    ct = getattr(u, "completion_tokens", None)
                    if ct is not None:
                        usage_completion_tokens = ct
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)

                if delta is not None:
                    # Mark the first generated token of ANY kind (visible content,
                    # reasoning, or a tool-call delta) for the tok/s window.
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if reasoning:
                        # Buffer reasoning for the empty-turn fallback. This does
                        # NOT yield it as text: visible content is still the only
                        # thing the agent renders on a normal turn.
                        reasoning_acc.append(reasoning)
                    if t_first is None and (
                        getattr(delta, "content", None)
                        or reasoning
                        or getattr(delta, "tool_calls", None)
                    ):
                        t_first = time.perf_counter()

                    content = getattr(delta, "content", None)
                    if content:
                        text_acc.append(content)
                        yield {"type": "text", "text": content}

                    dtcs = getattr(delta, "tool_calls", None)
                    if dtcs:
                        for tc in dtcs:
                            idx = getattr(tc, "index", 0) or 0
                            slot = tool_buf.setdefault(
                                idx, {"id": None, "name": None, "args": []}
                            )
                            if getattr(tc, "id", None):
                                slot["id"] = tc.id
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    slot["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    # Accumulate deltas in a LIST and join once at
                                    # consumption (below) -- streaming a large
                                    # write_file payload as many small deltas made
                                    # ``slot["args"] += fn.arguments`` O(n^2) on the
                                    # growing string (text_acc / reasoning_acc already
                                    # use list+join). The only behavioral change is
                                    # the accumulation strategy.
                                    slot["args"].append(fn.arguments)

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        except Exception as exc:  # noqa: BLE001
            # Type only (see _get_client): keep upstream error bodies out of the
            # model's context.
            yield {"type": "text", "text": f"[stream error: {type(exc).__name__}]"}
            yield {"type": "done", "finish_reason": "error", "output_tokens": None}
            return
        finally:
            # Deterministically release the HTTP stream/socket on normal exit,
            # break, exception, OR generator .close() (Ctrl+C). httpx streams and
            # the SDK wrapper expose .close(); guard for providers that don't.
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - teardown must never raise
                    pass

        # Generation window: first token -> now. Excludes prompt processing;
        # includes reasoning-token generation. Reported so the agent's tok/s
        # footer matches the server (no client-side undercount of the window).
        t_done = time.perf_counter()
        gen_elapsed = (t_done - t_first) if t_first is not None else (t_done - t_start)

        # Emit all assembled native tool calls BEFORE done.
        if tool_buf:
            # Recovery: LM Studio sometimes converts a Qwen XML tool call to a
            # NATIVE call but drops the <parameter=…> body, yielding empty args
            # ({}), while the SAME markup also survives in the text stream. When a
            # native call comes back arg-less, recover the real args by parsing the
            # text form and matching on tool name — so the write actually lands
            # instead of failing with "requires a string 'path'". Parsed lazily and
            # only when needed (the overwhelmingly common non-empty path is a no-op).
            _text_calls: list[dict] | None = None
            for idx in sorted(tool_buf):
                slot = tool_buf[idx]
                ev: dict = {
                    "type": "tool_call",
                    "id": slot["id"] or f"call_{idx}",
                    "name": slot["name"] or "",
                }
                raw = "".join(slot["args"]) or "{}"
                try:
                    parsed = json.loads(raw)
                    ev["arguments"] = parsed if isinstance(parsed, dict) else {}
                except (json.JSONDecodeError, ValueError) as exc:
                    ev["arguments"] = {}
                    ev["_parse_error"] = str(exc)
                if not ev["arguments"] and ev["name"]:
                    if _text_calls is None:
                        _text_calls = extract_text_tool_calls("".join(text_acc)) or []
                    # Match the FIRST not-yet-consumed text call of the same name,
                    # then CONSUME it (pop) so two arg-less native calls of the same
                    # tool don't both backfill from one text call (which would write
                    # the second file with the first's content).
                    match_i = next(
                        (j for j, c in enumerate(_text_calls)
                         if c.get("name") == ev["name"] and c.get("arguments")),
                        None,
                    )
                    if match_i is not None:
                        ev["arguments"] = _text_calls.pop(match_i)["arguments"]
                        ev.pop("_parse_error", None)
                yield ev
            yield {
                "type": "done",
                "finish_reason": "tool_calls",
                "output_tokens": usage_completion_tokens,
                "gen_elapsed": gen_elapsed,
            }
            return

        # No native tool calls: try the text-fallback parser on accumulated text.
        #
        # We only treat a fenced ```json block as a tool call when it is the SOLE
        # meaningful content of the turn. Otherwise a model that legitimately
        # answers in prose but includes a json EXAMPLE (e.g. documentation) would
        # be both shown AND executed. ``parse_tool_block`` already validates the
        # block shape; here we additionally require that stripping the fenced
        # block leaves no other substantive text, so a narrative answer that
        # merely *contains* a fence is never mis-fired as a tool call.
        full_text = "".join(text_acc)

        # HIGH-CONFIDENCE FALLBACK (runs BEFORE the generic ```json fence path):
        # the server may have failed to convert a model's OWN chat-template tool
        # markup (deepseek/qwen/hermes/mistral/llama control tokens) into native
        # tool_calls, leaving it as plain text. Those control tokens are an
        # unambiguous call signal, so extract the first tool-call group and
        # DISCARD any trailing text (models hallucinate a fake tool OUTPUT + prose
        # answer after the call when the server didn't stop them). Mirrors the
        # native emitted-event shape (id/name/arguments + _from_text_fence, plus
        # _parse_error/arguments={} on malformed args).
        text_calls = extract_text_tool_calls(full_text)
        if text_calls:
            for i, call in enumerate(text_calls):
                ev = {
                    "type": "tool_call",
                    "id": f"call_{i}",
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                    "_from_text_fence": True,
                }
                if call.get("_parse_error"):
                    ev["_parse_error"] = call["_parse_error"]
                yield ev
            yield {
                "type": "done",
                "finish_reason": "tool_calls",
                "output_tokens": usage_completion_tokens,
                "gen_elapsed": gen_elapsed,
            }
            return

        fallback = parse_tool_block(full_text)
        # Refuse ambiguous turns with MULTIPLE tool-shaped fences: firing only
        # the first would silently drop the rest. Treat as prose (no tool call)
        # so the model re-emits a single call.
        if (
            fallback is not None
            and _fence_is_sole_content(full_text)
            and count_tool_blocks(full_text) == 1
        ):
            yield {
                "type": "tool_call",
                "id": "call_0",
                "name": fallback["name"],
                "arguments": fallback["arguments"],
                # Mark this call as derived from a text fence so the agent can
                # avoid re-storing the consumed narration as assistant content.
                "_from_text_fence": True,
            }
            yield {
                "type": "done",
                "finish_reason": "tool_calls",
                "output_tokens": usage_completion_tokens,
                "gen_elapsed": gen_elapsed,
            }
            return

        # TRUNCATION NORMALIZATION (cap active): LM Studio sometimes reports
        # finish_reason="stop" even when it actually hit max_tokens. The agent's
        # "[output truncated at token limit]" marker only fires on "length", so
        # when a cap is set AND the server clearly consumed >= the cap, force
        # "length" here. A real "length" is left intact (never downgraded).
        if (
            effective_cap
            and usage_completion_tokens is not None
            and usage_completion_tokens >= effective_cap
        ):
            finish_reason = "length"

        # REASONING FALLBACK (root-cause fix for reasoning models, e.g. qwen3.6):
        # the model can route its ENTIRE turn into the reasoning_content channel
        # and emit no visible content, which would otherwise reach the agent as an
        # empty turn (-> the "[no answer produced …]" sentinel). When there is NO
        # visible text AND no tool call this turn, surface the buffered reasoning
        # as a normal text event so the agent treats it as the assistant answer.
        # Cleaned of <think>/<thinking> wrapper tags first; emitted only if
        # something meaningful survives. This ONLY triggers on an otherwise-empty
        # turn — a normal turn with visible content keeps reasoning discarded, so
        # the common path is byte-unchanged. The subtle, internal-only
        # ``_from_reasoning`` marker lets a caller tell it apart without cluttering
        # the answer text.
        if not full_text.strip() and reasoning_acc:
            recovered = _clean_reasoning_text("".join(reasoning_acc))
            if recovered:
                yield {"type": "text", "text": recovered, "_from_reasoning": True}

        # Preserve the server's real terminal signal (do not override 'stop').
        yield {
            "type": "done",
            "finish_reason": finish_reason,
            "output_tokens": usage_completion_tokens,
            "gen_elapsed": gen_elapsed,
        }

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` via the OpenAI-compatible ``/v1/embeddings`` endpoint.

        Reuses ``self._get_client()`` so it inherits ALL the loopback IP-pinning,
        private-mode trust_env hardening already built for chat. Raises a clear
        ValueError when no ``embed_model`` is configured (callers catch and fall
        back to BM25-only). Network/SDK errors are re-raised as a plain
        ``RuntimeError(type_name)`` so no SDK metadata (URLs, auth tokens, request
        bodies) leaks — same privacy discipline as stream_chat's
        ``[provider error: {type}]`` redaction. Returns one vector per input text.
        """
        if not self.embed_model:
            raise ValueError("no embed_model configured")
        client = self._get_client()
        try:
            resp = client.embeddings.create(model=self.embed_model, input=texts)
        except Exception as exc:  # noqa: BLE001 - redact SDK metadata to the type name
            raise RuntimeError(type(exc).__name__) from None
        return [list(d.embedding) for d in resp.data]


# ---------------------------------------------------------------------------
# MockProvider — deterministic, offline, no openai import
# ---------------------------------------------------------------------------

class MockProvider(Provider):
    """Deterministic scripted provider for offline tests + the smoke run.

    A "scenario" is a list of turns. Each turn is either:
      - {"text": <str>}  -> stream text then done(stop)
      - {"tool_calls": [{"id","name","arguments"}, ...]} -> emit calls then done(tool_calls)

    The provider is STATELESS across REPL turns: the current script step is
    derived from how many assistant turns have already happened in the message
    history passed into ``stream_chat`` (counted via assistant + tool messages),
    not from a per-process counter. The default 'hello' scenario writes hello.py
    then runs it, then gives a final text answer.
    """

    name = "mock"

    def __init__(self, scenario: str = "hello", temperature: float = 0.2, seed: int | None = None):
        """Deterministic scripted provider for offline tests + the smoke run.

        ``seed``: optional seed for deterministic scenario selection. When set,
        the mock provider cycles through scenarios based on (seed + message_count)
        so test runs are reproducible. When None, scenarios cycle 0, 1, 2, ...
        """
        self.scenario = scenario
        self.seed = seed
        # Accepted (and ignored) so construction stays uniform with LocalProvider
        # (Feature 1): build_provider passes temperature to every provider.
        self.temperature = temperature
        self._scripts = _MOCK_SCRIPTS

    def _select(self, messages: list[dict]) -> list[dict]:
        # Pick a script by explicit scenario, else infer from the last user msg.
        if self.scenario in self._scripts:
            return self._scripts[self.scenario]
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                # text_of() tolerates multimodal list content (extracts the text
                # part, ignores image parts) so a vision turn doesn't choke here.
                last_user = text_of(m.get("content")).lower()
                break
        if "hello" in last_user or "hi" in last_user:
            return self._scripts["hello"]
        return self._scripts["plain"]

    @staticmethod
    def _step_from_history(messages: list[dict]) -> int:
        """Derive the script step from the messages AFTER the last user turn.

        Each agent iteration appends exactly one assistant message (either a
        tool_calls turn or the final text turn). So the number of assistant
        messages appearing after the most recent user message is exactly the
        index of the script step we should emit next. This makes the provider
        stateless and correct across REPL turns (no per-process counter that
        desyncs between user inputs).
        """
        step = 0
        for m in reversed(messages):
            role = m.get("role")
            if role == "user":
                break
            if role == "assistant":
                step += 1
        return step

    def stream_chat(
        self, messages: list[dict], tools: list[dict] | None,
        tool_choice: str | None = None,
    ) -> Iterator[dict]:
        # tool_choice is accepted for signature parity (Feature 2); the scripted
        # provider ignores it. Tests that exercise the constrained retry subclass
        # and inspect it explicitly.
        script = self._select(messages)
        step = self._step_from_history(messages)
        if step >= len(script):
            # Safety: nothing scripted left -> terminal text.
            text = "Done."
            yield {"type": "text", "text": text}
            yield {
                "type": "done",
                "finish_reason": "stop",
                "output_tokens": max(1, len(text.split())),
            }
            return

        turn = script[step]

        if "tool_calls" in turn:
            for call in turn["tool_calls"]:
                yield {
                    "type": "tool_call",
                    "id": call["id"],
                    "name": call["name"],
                    "arguments": dict(call["arguments"]),
                }
            # Tool-call turns produce no visible text -> 0 tokens (no footer).
            yield {"type": "done", "finish_reason": "tool_calls", "output_tokens": 0}
            return

        # Text turn: stream in a couple of chunks to exercise delta handling.
        # Compute a DETERMINISTIC token count from the FULL text once (word
        # count), not per-chunk, so the count is stable and unit-testable.
        text = turn.get("text", "")
        mid = max(1, len(text) // 2)
        for piece in (text[:mid], text[mid:]):
            if piece:
                yield {"type": "text", "text": piece}
        yield {
            "type": "done",
            "finish_reason": "stop",
            "output_tokens": max(1, len(text.split())) if text else 0,
        }

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        """Deterministic, offline embeddings for retrieval tests (stdlib only).

        Each text -> a fixed 64-dim, L2-normalized bag-of-hashed-tokens vector:
        every token bumps the bucket ``sha256(token) % 64``. So the SAME text maps
        to the SAME vector and texts sharing tokens get a HIGHER cosine — enough to
        exercise the embedding path fully offline (no openai, no network). Mirrors
        ``LocalProvider.embeddings``' shape (one vector per input text).
        """
        dim = 64
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * dim
            for tok in _MOCK_TOKEN_RE.findall(str(text).lower()):
                # Stable cross-process hash (Python's hash() is salted per run).
                bucket = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:8], "big") % dim
                vec[bucket] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0.0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


# Tokenizer for MockProvider.embeddings — same lowercase [a-z0-9]+ shape the
# memory module uses, so mock vectors align with how docs/queries are tokenized.
_MOCK_TOKEN_RE = re.compile(r"[a-z0-9]+")


# Scripted scenarios. The 'hello' script is what the v1 smoke test drives.
_MOCK_SCRIPTS: dict[str, list[dict]] = {
    "hello": [
        {
            "tool_calls": [
                {
                    "id": "call_mock_0",
                    "name": "write_file",
                    "arguments": {
                        "path": "hello.py",
                        "content": "print('hi')\n",
                        "overwrite": True,
                    },
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "id": "call_mock_1",
                    "name": "run_bash",
                    "arguments": {"command": "python3 hello.py"},
                }
            ]
        },
        {"text": "Created hello.py and ran it; it printed 'hi'. Done."},
    ],
    "plain": [
        {"text": "Hello from the mock provider. No tools were needed."},
    ],
    "read": [
        {
            "tool_calls": [
                {
                    "id": "call_mock_0",
                    "name": "read_file",
                    "arguments": {"path": "hello.py"},
                }
            ]
        },
        {"text": "I read the file."},
    ],
}
