"""Gated LLM-judge reranker for weak-signal retrieval.

When lexical (BM25) and semantic (embeddings) retrieval both return a weak
result — the paraphrase case where the query shares no scoring token with any
record — the fused candidate order is little better than arbitrary. This module
asks the chat model to re-order a candidate pool by relevance, narrowing it to
the final ``top_k``.

It is pure-stdlib and duck-typed against any provider exposing
``stream_chat(messages, tools=None)`` (the same shape the rest of the codebase
uses; providers yield ``{"type": "text", "text": <str>}`` deltas followed by a
``done`` event). It is gated OFF by default (``config.rerank``) so the existing
retrieval path is byte-for-byte unchanged when the flag is unset, and EVERY
failure path falls back to the input order — retrieval never raises.
"""

from __future__ import annotations

import json
import re

# Default candidate-pool width callers pass when config doesn't override. The
# judge re-ranks this many fused candidates down to the final top_k.
RERANK_CAND_LIMIT = 20
# Per-snippet char cap in the rerank prompt — keeps one batched chat call small
# even when a candidate is a whole code chunk.
_SNIPPET_CHARS = 800


def rerank_enabled(provider, cfg) -> bool:
    """True iff the reranker should be *considered* for this turn.

    Callers additionally gate on the weak-signal condition (embeddings fired and
    the pool is wider than ``top_k``); this helper only checks the config flag +
    that the provider can actually chat, so callers don't repeat the duck-type
    check.
    """
    return (
        bool(getattr(cfg, "rerank", False))
        and provider is not None
        and hasattr(provider, "stream_chat")
    )


def _drain_stream(stream) -> str:
    """Collect assistant text deltas from a ``stream_chat`` iterator.

    providers.py yields event dicts; text deltas are
    ``{"type": "text", "text": <str>}``. Any other event type (``tool_call``,
    ``done``, ...) is ignored. Robust to a stream that yields non-dict values.
    """
    out: list[str] = []
    for ev in stream:
        if isinstance(ev, dict) and ev.get("type") == "text":
            t = ev.get("text")
            if isinstance(t, str):
                out.append(t)
    return "".join(out)


def _parse_ordering(text: str, n: int) -> list[int]:
    """Tolerantly parse the model's ranking into 0-based candidate positions.

    Accepts a JSON array of ints OR a comma/newline/space-separated run of
    ints. All ints are extracted in order of appearance; each is validated to be
    a 1-based candidate number (``1..n``) and mapped to 0-based. Duplicates are
    dropped. An empty/unparseable response yields ``[]`` so the caller falls
    back to input order.
    """
    if not text:
        return []
    stripped = text.strip()
    # Try JSON first: a model that returns a clean ``[3, 1, 2]`` array. A
    # JSON parse that doesn't yield a list (number/string/obj) is treated as no
    # ordering so we fall back, not as a partial ranking.
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
        except (ValueError, TypeError):
            arr = None
        if isinstance(arr, list):
            ints = [
                int(x) for x in arr
                if isinstance(x, (int, float)) and not isinstance(x, bool)
            ]
        else:
            ints = []
    else:
        ints = [int(m) for m in re.findall(r"\d+", stripped)]
    seen: set[int] = set()
    out: list[int] = []
    for v in ints:
        if 1 <= v <= n and (v - 1) not in seen:
            seen.add(v - 1)
            out.append(v - 1)
    return out


def rerank(
    provider, query: str, candidates: list[tuple[int, str]], top_k: int,
) -> list[int]:
    """Re-order ``candidates`` by LLM relevance judgment; never raises.

    ``candidates`` is ``[(orig_idx, text), ...]`` in the already-fused input
    order. Returns a list of ``orig_idx``: the judged-best first, then any
    unranked candidates in their original order, capped at ``len(candidates)``
    (never over-returns). On ANY failure — no provider, no stream, an
    unparseable response, or a raised exception — returns the input order
    unchanged. Retrieval never raises.
    """
    n = len(candidates)
    if n == 0:
        return []
    # The graceful fallback for every failure path: the input order, unchanged.
    input_order = [idx for idx, _ in candidates]
    if provider is None or not hasattr(provider, "stream_chat"):
        return input_order
    try:
        snippets = "\n".join(
            f"[{i + 1}] {text[:_SNIPPET_CHARS]}"
            for i, (_, text) in enumerate(candidates)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance judge. Rank the given snippets by "
                    "relevance to the user's query, from MOST to LEAST relevant. "
                    "Return ONLY a comma-separated list of the snippet numbers in "
                    "rank order (e.g. 3, 1, 2). No prose, no explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nSnippets:\n{snippets}",
            },
        ]
        stream = provider.stream_chat(messages=messages, tools=None)
        if stream is None:
            return input_order
        text = _drain_stream(stream)
    except Exception:  # noqa: BLE001 - retrieval NEVER raises
        return input_order
    order = _parse_ordering(text, n)
    if not order:
        return input_order
    # Judge's ranked orig_idx first; unranked appended in input order; cap at n.
    ranked = [candidates[pos][0] for pos in order]
    ranked_set = set(ranked)
    for idx in input_order:
        if idx not in ranked_set:
            ranked.append(idx)
            ranked_set.add(idx)
    return ranked[:n]