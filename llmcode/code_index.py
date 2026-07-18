"""Code-RAG retrieval core: hybrid BM25 ∪ embeddings over THIS project's source.

This is the model-callable ``code_search`` feature's engine. It chunks the
workspace's text files into line windows, ranks them against a query with the
SAME hybrid retrieval as conversation memory (``memory.py``), and returns the
most relevant CODE CHUNKS (path:lines + snippet) so the model FINDS where things
live without reading whole files (smaller context -> faster decode + accuracy).

It deliberately REUSES the hardened primitives instead of reinventing them:
  - ``memory._tokenize`` / ``memory._bm25_score_index`` / ``memory.cosine`` /
    ``memory.content_hash`` — the exact Okapi BM25 + cosine + hashing used by
    conversation memory, so lexical/semantic behaviour is identical.
  - ``tools._iter_source_files`` (prunes .git/.venv/node_modules/...),
    ``tools._looks_binary`` (NUL sniff), ``tools._truncate`` (byte budget).
  - ``session.session_id`` / ``session.sessions_dir`` / ``session._harden_state_dir``
    so the on-disk index sits beside the session + memory files and is LOCAL-ONLY.

Design (mirrors ``memory.MemoryStore`` exactly):
  - BM25 is ALWAYS available (no network, deterministic). Embeddings are OPTIONAL
    and FAILURE-SILENT: any embedding error degrades to BM25-only, never raises.
  - "auto" mode (the default) computes the embedding ranking WHENEVER a provider
    with usable embeddings is supplied — code_search is a deliberately-invoked
    tool, so it pays for semantic recall by default. Calibration proved BM25
    scores for lexical vs paraphrase queries OVERLAP, so there is NO "weak-score"
    gate that could detect the paraphrase case; the only safe signal is "do we
    have a provider". With no provider (or if embeddings fail/return []) it
    degrades SILENTLY to BM25-only. (This deliberately differs from memory.py,
    whose every-turn auto-gate stays cost-conservative.)
  - Fusion is EMBED-FIRST: when embeddings are computed they are the higher-value
    signal (we only embed when it matters), so the embedding ranking LEADS and
    BM25 is the secondary/tiebreaker.
  - Persistence is atomic (mkstemp + os.replace), best-effort (save never raises),
    and tolerant (load returns an EMPTY index on a missing/corrupt file). save()
    is a NO-OP when the index is not dirty.
  - Incremental: each refresh stats every file (size+mtime signature) vs the
    cached ``file_hashes`` and re-chunks only NEW/CHANGED files, DROPS chunks of
    files that disappeared. Total chunks are capped; past the cap we STOP adding
    and record it (so the tool can warn — no silent truncation).
"""

from __future__ import annotations

import ast
import json
import os
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from .memory import (
    _BM25Index,
    _QueryVecLRU,
    _atomic_write_json,
    _bm25_score_index,
    _build_bm25_index,
    _load_vectors,
    _save_vectors_sidecar,
    _tokenize,
    _unlink_sidecar,
    content_hash,
    cosine,
)
from .session import _harden_state_dir, session_id, sessions_dir
from .tools import _iter_source_files, _looks_binary, _truncate

# Chunking: ~60-line windows with ~15-line overlap, so a symbol that straddles a
# window boundary still appears whole in the neighbouring (overlapping) chunk.
CHUNK_LINES = 60
CHUNK_OVERLAP = 15
# A file larger than this is skipped entirely (generated/minified/vendored blobs
# would flood the index and embedding cost for ~no retrieval value).
MAX_FILE_BYTES = 1_000_000
# Hard cap on total chunks so a huge repo's index (and its per-search tokenization
# / embedding cost) stays bounded. Past this we STOP adding and set ``capped`` so
# the tool warns the model rather than silently dropping files.
MAX_CHUNKS = 4000
# Python files are chunked one-per-top-level-symbol (def/async def/class) so each
# embedding/BM25 chunk is a COHERENT unit instead of an arbitrary line window. A
# single symbol longer than this is too big to embed as one chunk, so its body is
# split back into the standard line windows (header line kept so it stays
# identifiable). Tuned a bit below 2x CHUNK_LINES so most real functions stay whole.
MAX_SYMBOL_LINES = 120
# Sub-batch size for the lazy first embedding: the missing chunk texts are embedded
# in fixed batches (not one giant multi-thousand-text request) so a big repo's
# initial embed stays bounded; a short/failed batch aborts to BM25 fallback.
EMBED_BATCH = 128
# code_search-specific output budget: deliberately SMALLER than tools._MAX_OUTPUT
# (the hard ceiling every tool obeys) so a single code_search call — which can be
# issued repeatedly — cannot dominate the model's context and slow local decode.
# Applied when formatting hits via _truncate; truncation stays graceful.
CODE_SEARCH_MAX_OUTPUT = 12_000


# ---------------------------------------------------------------------------
# CodeChunk
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """One indexed slice of a source file. ``path`` is relative to the workspace;
    ``start_line``/``end_line`` are 1-based INCLUSIVE line numbers; ``text`` is the
    raw chunk body; ``content_hash`` dedupes identical text and keys the embedding
    cache (so the same text is embedded at most once)."""

    path: str
    start_line: int
    end_line: int
    text: str
    content_hash: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CodeChunk:
        """Tolerant of missing/extra keys (like MemoryRecord.from_dict): every
        field has a safe default, line numbers coerce to int (bad values -> 0),
        and a missing/empty content_hash is recomputed from the text."""
        text = str(d.get("text", "") or "")
        ch = str(d.get("content_hash", "") or "") or content_hash(text)
        return cls(
            path=str(d.get("path", "") or ""),
            start_line=_as_int(d.get("start_line")),
            end_line=_as_int(d.get("end_line")),
            text=text,
            content_hash=ch,
        )


def _as_int(v) -> int:
    """Best-effort int coercion for a hand-edited/corrupt field (-> 0 on failure;
    bool is rejected so a stray ``true`` is not silently read as line 1)."""
    if isinstance(v, bool):
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _chunk_file(rel: str, text: str) -> list[CodeChunk]:
    """Split ``text`` into ~CHUNK_LINES windows overlapping by CHUNK_OVERLAP lines,
    each tagged with its relative path and 1-based inclusive line range. An empty
    file yields no chunks. The step (CHUNK_LINES - CHUNK_OVERLAP) guarantees forward
    progress (CHUNK_OVERLAP < CHUNK_LINES)."""
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    step = CHUNK_LINES - CHUNK_OVERLAP
    out: list[CodeChunk] = []
    start = 0  # 0-based line index
    while start < n:
        end = min(start + CHUNK_LINES, n)  # exclusive
        body = "\n".join(lines[start:end])
        out.append(CodeChunk(
            path=rel,
            start_line=start + 1,       # 1-based inclusive
            end_line=end,               # 1-based inclusive (end is exclusive 0-based == inclusive 1-based)
            text=body,
            content_hash=content_hash(body),
        ))
        if end >= n:
            break
        start += step
    return out


# Python source files get AST/symbol-aware chunking; everything else (and any
# Python file that fails to parse) falls back to the line-window ``_chunk_file``.
_PY_EXTS = (".py", ".pyi")


def _make_chunk(rel: str, lines: list[str], start: int, end: int,
                header: str = "") -> CodeChunk:
    """Build a CodeChunk from the 1-based INCLUSIVE line range ``start..end`` of
    ``lines``. ``header`` (if given) is prepended to the chunk TEXT only — the
    start/end line numbers stay the real source lines so ``path:start-end``
    citations remain correct."""
    body = "\n".join(lines[start - 1:end])
    text = f"{header}\n{body}" if header else body
    return CodeChunk(
        path=rel,
        start_line=start,
        end_line=end,
        text=text,
        content_hash=content_hash(text),
    )


def _window_symbol(rel: str, lines: list[str], start: int, end: int,
                   name: str) -> list[CodeChunk]:
    """Split an OVERSIZED symbol body (``start..end``, 1-based inclusive) back into
    CHUNK_LINES windows overlapping by CHUNK_OVERLAP, prepending ``# <name>`` to
    every window so each sub-chunk stays identifiable. Line numbers stay real."""
    step = CHUNK_LINES - CHUNK_OVERLAP
    out: list[CodeChunk] = []
    s = start
    while s <= end:
        e = min(s + CHUNK_LINES - 1, end)
        out.append(_make_chunk(rel, lines, s, e, header=f"# {name}"))
        if e >= end:
            break
        s += step
    return out


def _chunk_python(rel: str, text: str) -> list[CodeChunk]:
    """AST/symbol-aware chunking for Python: one chunk per top-level ``def`` /
    ``async def`` / ``class`` (decorators included, kept WHOLE), with the
    module-level code between/around symbols emitted as its own chunk(s). A symbol
    longer than MAX_SYMBOL_LINES is split back into line windows. On ANY ast
    failure (SyntaxError/ValueError) or when the file has no top-level symbols,
    falls back to the plain line-window ``_chunk_file`` so we never crash and a
    pure-script file keeps its previous behaviour."""
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):  # ValueError: source with NUL bytes etc.
        return _chunk_file(rel, text)
    symbols: list[tuple[int, int, str]] = []  # (start, end, name), 1-based incl.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            decos = getattr(node, "decorator_list", None) or []
            if decos:
                start = min(start, min(d.lineno for d in decos))
            end = getattr(node, "end_lineno", None) or node.lineno
            symbols.append((start, end, node.name))
    if not symbols:
        return _chunk_file(rel, text)  # pure module/script -> keep line windows
    symbols.sort()
    out: list[CodeChunk] = []
    cursor = 1  # next 1-based source line not yet emitted

    def _emit_module_gap(upto: int) -> None:
        """Emit lines cursor..upto (inclusive) as a module-level chunk, skipping a
        gap that is only blank lines."""
        if upto >= cursor and "\n".join(lines[cursor - 1:upto]).strip():
            out.append(_make_chunk(rel, lines, cursor, upto))

    for start, end, name in symbols:
        _emit_module_gap(start - 1)
        if (end - start + 1) > MAX_SYMBOL_LINES:
            out.extend(_window_symbol(rel, lines, start, end, name))
        else:
            out.append(_make_chunk(rel, lines, start, end))
        cursor = end + 1
    _emit_module_gap(n)
    return out


def _chunk_source(rel: str, text: str) -> list[CodeChunk]:
    """Dispatch chunking by file type: AST/symbol-aware for Python, line windows
    for everything else."""
    if rel.lower().endswith(_PY_EXTS):
        return _chunk_python(rel, text)
    return _chunk_file(rel, text)


# ---------------------------------------------------------------------------
# CodeIndex
# ---------------------------------------------------------------------------

@dataclass
class CodeIndex:
    """The chunked, searchable index for one workspace, with a content-hash-keyed
    embedding cache and a per-file signature map for incremental refresh.

    ``vectors`` maps content_hash -> embedding (same text embedded at most once).
    ``file_hashes`` maps relpath -> "size:mtime_ns" so a refresh re-chunks only the
    files that actually changed. ``capped`` records that the last scan hit
    MAX_CHUNKS (so the tool can warn instead of silently truncating)."""

    chunks: list[CodeChunk] = field(default_factory=list)
    # Values are ``array('f')`` (float32) at runtime, not boxed ``list[float]`` —
    # a ~85% RAM cut for a full corpus. On-disk JSON is unchanged (save() emits
    # ``list(v)``, load() rebuilds ``array('f')``).
    vectors: dict[str, list[float]] = field(default_factory=dict)
    file_hashes: dict[str, str] = field(default_factory=dict)
    capped: bool = False
    # Set by a scan that changed anything, cleared by a successful save(); makes
    # save() a no-op when the index matches disk. load() leaves it False.
    _dirty: bool = field(default=False, compare=False, repr=False)
    # Cached per-chunk tokenization for BM25, invalidated (set None) on any scan
    # change so search() doesn't re-tokenize every chunk each query. Never persisted.
    _bm25_tokens: list | None = field(default=None, compare=False, repr=False)
    # Cached BM25 corpus structures (per-doc tf, lengths, avgdl, df_map) built
    # from ``_bm25_tokens`` and invalidated alongside it. Saves a 4000-chunk
    # rebuild of tf dicts + df on every code_search. Never persisted.
    _bm25_index: _BM25Index | None = field(default=None, compare=False, repr=False)
    # Per-index LRU of query embedding vectors keyed by content_hash(query):
    # a repeated query skips the embed round-trip + model swap. Never persisted.
    _query_vecs: _QueryVecLRU = field(default_factory=_QueryVecLRU, compare=False, repr=False)

    # ----- building / incremental refresh ----------------------------------

    def build(self, workspace, embed_provider=None) -> None:
        """Initial build == a full incremental scan of ``workspace``.

        Embeddings are computed LAZILY by search() (so the embed model is never
        swapped in/out at index time on a shared GPU — mirroring memory.py's
        model-swap discipline); ``embed_provider`` is accepted for API symmetry
        and a future eager-embed hook but is intentionally not used to pre-embed.
        """
        del embed_provider  # lazy-embed by design; see docstring
        self._scan(workspace)

    def refresh(self, workspace) -> None:
        """Cheap incremental re-scan: stat every file and re-chunk only the
        new/changed ones, dropping chunks of files that disappeared."""
        self._scan(workspace)

    def _scan(self, workspace) -> None:
        """Walk ``workspace`` (pruning ignored dirs via _iter_source_files),
        re-chunking only changed/new files and dropping vanished files' chunks.
        Sets ``_dirty`` on any change; caps total chunks at MAX_CHUNKS."""
        root = Path(workspace)
        real_root = os.path.realpath(root)
        # Index existing chunks by file so an UNCHANGED file's chunks are reused
        # verbatim (same objects) rather than re-chunked.
        existing_by_file: dict[str, list[CodeChunk]] = {}
        for c in self.chunks:
            existing_by_file.setdefault(c.path, []).append(c)

        new_chunks: list[CodeChunk] = []
        new_hashes: dict[str, str] = {}
        capped = False
        changed = False
        for fp in _iter_source_files(root):
            # Defense-in-depth: never index (or persist) a symlink whose realpath
            # escapes the workspace — its content must not leak into the on-disk index.
            if fp.is_symlink():
                real = os.path.realpath(fp)
                if real != real_root and not real.startswith(real_root + os.sep):
                    continue
            try:
                st = fp.stat()
            except OSError:
                continue
            try:
                rel = os.path.relpath(str(fp), str(root))
            except ValueError:
                continue
            sig = f"{st.st_size}:{st.st_mtime_ns}"
            if self.file_hashes.get(rel) == sig and rel in existing_by_file:
                # Unchanged -> reuse: only a stat() above, never an open/read. An
                # unchanged file's size/binary-ness can't change without its sig
                # changing, so the size + binary checks live in the else branch.
                file_chunks = existing_by_file[rel]
            else:
                if st.st_size > MAX_FILE_BYTES:
                    continue
                if _looks_binary(fp):
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                file_chunks = _chunk_source(rel, text)
                changed = True  # new/changed file re-chunked
            # Enforce the global chunk cap: STOP adding once exceeded and record it.
            room = MAX_CHUNKS - len(new_chunks)
            if room <= 0:
                capped = True
                break
            if len(file_chunks) > room:
                # Partial: index what fits but do NOT record this file's signature, so
                # a later scan (e.g. after MAX_CHUNKS grows) re-evaluates it and chunks
                # the remainder instead of treating the truncated file as up to date.
                new_chunks.extend(file_chunks[:room])
                capped = True
                break
            new_chunks.extend(file_chunks)
            new_hashes[rel] = sig

        # A file that vanished (was in file_hashes, not seen now) -> its chunks are
        # already absent from new_chunks; flag the change so we re-save.
        if set(self.file_hashes) - set(new_hashes):
            changed = True
        if capped != self.capped:
            changed = True

        if not changed:
            return  # nothing to do: index already matches disk

        self.chunks = new_chunks
        self.file_hashes = new_hashes
        self.capped = capped
        # Drop any cached vector whose content_hash no longer belongs to a live
        # chunk so ``vectors`` can't outgrow ``chunks``.
        live = {c.content_hash for c in self.chunks}
        self.vectors = {h: v for h, v in self.vectors.items() if h in live}
        self._dirty = True
        self._bm25_tokens = None  # chunks changed -> drop the tokenization cache
        self._bm25_index = None

    # ----- search (identical gating to MemoryStore.retrieve) ---------------

    def search(self, query: str, top_k: int = 5, provider=None,
               mode: str = "auto", rerank: bool = False,
               rerank_candidates: int = 20) -> list[CodeChunk]:
        """Return up to ``top_k`` chunks most relevant to ``query``.

        BM25 is ALWAYS computed. The embeddings path (cosine, then fused
        EMBED-FIRST with BM25 as the secondary/tiebreaker) fires by mode:

          - "off" -> [].
          - "bm25" -> lexical only (NEVER calls embeddings; no provider needed).
          - "embed" -> force the hybrid path (compute embeddings, then fuse).
          - "auto" (the DEFAULT) -> hybrid WHENEVER a ``provider`` is supplied,
            else BM25-only. Unlike conversation memory, code_search is a
            deliberately-invoked tool, so it always pays for embeddings when a
            provider is available — calibration proved BM25 scores can't separate
            lexical from paraphrase queries, so there is no useful score gate.

        ANY embedding failure (or no provider) degrades silently to BM25-only —
        search never raises.
        """
        if mode == "off" or top_k <= 0 or not self.chunks:
            return []
        if self._bm25_tokens is None:
            self._bm25_tokens = [_tokenize(c.text) for c in self.chunks]
            self._bm25_index = _build_bm25_index(self._bm25_tokens)
        # When the LLM-judge reranker is opted in, widen the candidate pool so the
        # judge has more to rank than the final top_k. pool_k == top_k when rerank
        # is off, so the off-path is byte-for-byte unchanged.
        pool_k = max(top_k, rerank_candidates) if rerank else top_k
        bm25_full = _bm25_score_index(query, self._bm25_index)
        bm25_idx = [i for i, s in bm25_full[:pool_k] if s > 0.0]
        embed_idx: list[int] = []
        if provider is not None and mode in ("embed", "auto"):
            embed_idx = self._embed_rank(query, provider, pool_k)
        # EMBED-FIRST fusion: when embeddings are computed they LEAD (higher-value
        # signal — we only embed when it matters); BM25 is the secondary/tiebreaker.
        ordered = bm25_idx if not embed_idx else self._fuse(embed_idx, bm25_idx, pool_k)
        # GATED LLM-judge rerank: fires ONLY on the hybrid path (embeddings fired ->
        # embed_idx non-empty) AND only when the pool is wider than the final
        # top_k. Default rerank=False skips the branch entirely -> zero behavior
        # change, zero latency. Any failure falls back to the fused order; search
        # never raises.
        if (
            rerank and embed_idx and provider is not None
            and len(ordered) > top_k
        ):
            from .rerank import rerank as _rerank
            cands = ordered[:rerank_candidates]
            try:
                reord = _rerank(
                    provider, query,
                    [(i, self.chunks[i].text) for i in cands],
                    top_k,
                )
                ordered = reord + [i for i in ordered if i not in set(reord)]
            except Exception:  # noqa: BLE001 - keep fused order on any failure
                pass
        return [self.chunks[i] for i in ordered[:top_k]]

    def _embed_rank(self, query: str, provider, top_k: int) -> list[int]:
        """Cosine ranking via ``provider.embeddings``, or [] on ANY failure.

        DIMENSION GUARD (same as MemoryStore._embed_rank): embed the QUERY first to
        learn the target dim; any cached vector of the wrong length (stale embed
        model) is treated as MISSING and re-embedded to self-heal; a vector that
        STILL mismatches is SKIPPED rather than fed to cosine. Returns [] — so the
        caller falls back to BM25 — when the provider has no usable ``embeddings``,
        raises, or returns an empty/short result."""
        embed = getattr(provider, "embeddings", None)
        if not callable(embed):
            return []
        try:
            # Query-vector LRU: a repeated query skips the embed round-trip (and
            # the LM-Studio model swap). Keyed by content_hash(query).
            qh = content_hash(query)
            qvec = self._query_vecs.get(qh)
            if qvec is None:
                qres = embed([query])
                if not qres or not qres[0]:
                    return []
                qvec = [float(x) for x in qres[0]]
                self._query_vecs.put(qh, len(qvec), qvec)
            qdim = len(qvec)
            missing = [
                c for c in self.chunks
                if len(self.vectors.get(c.content_hash) or []) != qdim
            ]
            # Dedupe by content_hash so identical chunk text is embedded once.
            uniq: dict[str, CodeChunk] = {}
            for c in missing:
                uniq.setdefault(c.content_hash, c)
            if uniq:
                items = list(uniq.values())
                # Sub-batch the missing texts so the first embed of a big repo isn't a
                # single multi-thousand-text request; a short/failed batch aborts to
                # BM25 fallback exactly as a single failed call would.
                vecs: list = []
                for start in range(0, len(items), EMBED_BATCH):
                    batch = items[start:start + EMBED_BATCH]
                    bvecs = embed([c.text for c in batch])
                    if not bvecs or len(bvecs) != len(batch):
                        return []
                    vecs.extend(bvecs)
                for c, v in zip(items, vecs):
                    self.vectors[c.content_hash] = array('f', (float(x) for x in v))
                # Newly embedded vectors must survive across save/load (the documented
                # "vectors survive"): mark dirty so the caller's save() persists them.
                self._dirty = True
        except Exception:  # noqa: BLE001 - ANY embedding failure -> BM25 fallback
            return []
        scored: list[tuple[int, float]] = []
        for i, c in enumerate(self.chunks):
            v = self.vectors.get(c.content_hash)
            if not v or len(v) != qdim:
                continue
            scored.append((i, cosine(qvec, v)))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [i for i, _ in scored[:top_k]]

    @staticmethod
    def _fuse(primary_idx: list[int], secondary_idx: list[int], top_k: int) -> list[int]:
        """Concatenate two ranked index lists — ``primary_idx`` in full FIRST, then
        ``secondary_idx`` as backfill — deduped, capped at ``top_k``. NOT round-robin:
        search() passes the embedding ranking as primary, and a live LM-Studio
        benchmark showed round-robin INTERLEAVING demotes the embedding's #2/#3 hits
        below BM25's (paraphrase recall@3 0.5 vs 0.8). Backfill keeps the full
        embedding order (recall@3 -> 0.8, matching pure-embed) while still topping up
        from BM25 when the embedding ranking is short."""
        merged: list[int] = []
        seen: set[int] = set()
        for ranked in (primary_idx, secondary_idx):
            for idx in ranked:
                if idx not in seen:
                    seen.add(idx)
                    merged.append(idx)
                    if len(merged) >= top_k:
                        return merged
        return merged

    # ----- persistence (atomic; mirrors memory.MemoryStore) ----------------

    def save(self, path) -> None:
        """Atomically persist chunks + vectors + file_hashes. Best-effort: write a
        temp file in the same dir then os.replace; never raises. NO-OP when the
        index is not ``_dirty``.

        Vectors are written as a packed float32 SIDECAR (``<path>.vec``); the JSON
        keeps only slim metadata. If the sidecar can't be written (empty/ragged
        vectors, or ANY error) it FALLS BACK to the legacy inline-JSON format so
        data is never lost."""
        if not self._dirty:
            return
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _harden_state_dir()  # restrict ~/.llmcode to the owner (best-effort)
        except OSError:
            return  # can't prepare the dir — best-effort: silently skip
        base = {
            "chunks": [c.to_dict() for c in self.chunks],
            "file_hashes": dict(self.file_hashes),
            "capped": bool(self.capped),
        }
        # Preferred: packed-f32 vector sidecar + slim meta JSON (sidecar FIRST).
        if _save_vectors_sidecar(target, base, self.vectors):
            self._dirty = False
            return
        # Fallback: legacy inline-JSON (vectors embedded), best-effort. Drop any
        # stale sidecar first so an old `.vec` isn't orphaned beside inline meta.
        _unlink_sidecar(target)
        payload = dict(base)
        payload["vectors"] = {h: list(v) for h, v in self.vectors.items()}
        try:
            _atomic_write_json(target, payload)
            self._dirty = False  # on-disk copy now matches memory
        except OSError:
            return  # best-effort: disk full / permissions / etc. — silently skip

    @classmethod
    def load(cls, path) -> CodeIndex:
        """Read an index from ``path``; an EMPTY index on missing/corrupt/invalid
        input (never raises). Non-dict chunks and non-numeric vectors are dropped
        so a hand-edited file can't crash a consumer. Vectors load from the f32
        sidecar (new format) or inline JSON (old format); a missing/torn sidecar
        yields ``{}`` (lazy re-embed) rather than crashing."""
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError:
            return cls()
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        chunks = [CodeChunk.from_dict(cd) for cd in (data.get("chunks") or [])
                  if isinstance(cd, dict)]
        vectors = _load_vectors(path, data)
        file_hashes: dict[str, str] = {}
        raw_fh = data.get("file_hashes")
        if isinstance(raw_fh, dict):
            for k, val in raw_fh.items():
                if isinstance(k, str) and isinstance(val, str):
                    file_hashes[k] = val
        return cls(
            chunks=chunks,
            vectors=vectors,
            file_hashes=file_hashes,
            capped=bool(data.get("capped", False)),
        )


def index_path(cwd: str) -> Path:
    """Path to the per-project code index: sessions_dir()/<session_id>.code_index.json
    (sits alongside the session + memory files, keyed by the same workspace id;
    LOCAL-ONLY, never transmitted)."""
    return sessions_dir() / f"{session_id(cwd)}.code_index.json"


# ---------------------------------------------------------------------------
# make_code_search_tool — the injected, provider-bound code_search Tool
# ---------------------------------------------------------------------------

# Process-wide cache of the live CodeIndex per workspace so a tool call doesn't
# reload the whole index from disk every time (it still cheap-refreshes via stat
# on each call). Keyed by the absolute workspace path.
_INDEX_CACHE: dict[str, CodeIndex] = {}


def _chunk_source_lines(c: CodeChunk) -> dict[int, str] | None:
    """Map each 1-based source line number in ``c`` to its text, or ``None`` when
    the chunk text doesn't line up with its declared ``start_line..end_line`` range
    (corrupt/hand-edited chunk) so the caller renders it verbatim instead of merging.

    An oversized-symbol window (``_window_symbol``) prepends a single ``# name``
    header line to the text; it is detected (one extra line) and dropped so only the
    real source lines are mapped — the ``path:start-end`` citation stays accurate."""
    body_len = c.end_line - c.start_line + 1
    if body_len <= 0:
        return None
    text_lines = c.text.split("\n")
    if len(text_lines) == body_len:
        body = text_lines
    elif len(text_lines) == body_len + 1:
        body = text_lines[1:]  # drop the leading "# <name>" window header
    else:
        return None
    return {c.start_line + i: line for i, line in enumerate(body)}


def _merge_same_file_hits(hits: list[CodeChunk]) -> list[tuple[int, str, int, int, str]]:
    """Collapse OVERLAPPING or ADJACENT chunks from the SAME file into single blocks.

    Adjacent 60-line windows overlap by ~15 lines, so two hits from one file would
    otherwise print ~25% of their lines twice — wasted context. Returns render-ready
    ``(rank, path, start, end, text)`` tuples ordered by ``rank`` (position in the
    ranked ``hits``). A merged block spans the UNION line range with each source line
    appearing ONCE and takes the rank of its highest-ranked (earliest) constituent.
    Chunks from DIFFERENT files never merge; non-overlapping same-file chunks stay
    separate (keeping their order). A single un-merged chunk is rendered verbatim, so
    the non-overlapping path is byte-for-byte unchanged."""
    blocks: list[tuple[int, str, int, int, str]] = []
    # Group mappable chunks by file (insertion-ordered); keep the rank each carried.
    by_file: dict[str, list[tuple[int, CodeChunk, dict[int, str]]]] = {}
    for rank, c in enumerate(hits):
        lm = _chunk_source_lines(c)
        if lm is None:
            blocks.append((rank, c.path, c.start_line, c.end_line, c.text))
            continue
        by_file.setdefault(c.path, []).append((rank, c, lm))

    def _flush(path: str, rank: int, start: int, end: int,
               lines: dict[int, str], only: CodeChunk | None) -> None:
        if only is not None:  # single constituent -> preserve its original text
            text = only.text
        else:
            text = "\n".join(lines[ln] for ln in range(start, end + 1))
        blocks.append((rank, path, start, end, text))

    for path, items in by_file.items():
        # Sort by start line so interval-merging is order-robust (a hit that bridges
        # two earlier separate hits still chains them into one contiguous block).
        items.sort(key=lambda t: (t[1].start_line, t[1].end_line))
        cur_rank = cur_start = cur_end = 0
        cur_lines: dict[int, str] = {}
        cur_only: CodeChunk | None = None
        open_block = False
        for rank, c, lm in items:
            if open_block and c.start_line <= cur_end + 1:  # overlap OR adjacent
                cur_lines.update(lm)
                cur_end = max(cur_end, c.end_line)
                cur_rank = min(cur_rank, rank)
                cur_only = None  # 2+ constituents -> reconstruct from the line map
            else:
                if open_block:
                    _flush(path, cur_rank, cur_start, cur_end, cur_lines, cur_only)
                cur_rank, cur_start, cur_end = rank, c.start_line, c.end_line
                cur_lines = dict(lm)
                cur_only = c
                open_block = True
        if open_block:
            _flush(path, cur_rank, cur_start, cur_end, cur_lines, cur_only)

    blocks.sort(key=lambda b: b[0])  # restore ranked order across all files
    return blocks


def _format_hits(hits: list[CodeChunk], capped: bool) -> str:
    """Render the search hits as ``path:start-end`` followed by each snippet, with
    a warning line first when the index hit its chunk cap. Overlapping/adjacent hits
    from the same file are MERGED (union range, no duplicated lines) via
    ``_merge_same_file_hits``. The whole payload is byte-capped by the caller via
    tools._truncate (grep's discipline)."""
    if not hits:
        return "code_search: no matching code chunks found."
    parts: list[str] = []
    if capped:
        parts.append(
            f"[code index capped at {MAX_CHUNKS} chunks; results may be incomplete "
            "— narrow your query or use grep for exact matches]"
        )
    for _rank, path, start, end, text in _merge_same_file_hits(hits):
        parts.append(f"{path}:{start}-{end}")
        parts.append(text)
        parts.append("")  # blank line between hits
    return "\n".join(parts).rstrip() + "\n"


def make_code_search_tool(provider, workspace, private: bool = False,
                          top_k: int = 5, config=None):
    """Build the ``code_search`` :class:`~llmcode.tools.Tool` bound to a provider +
    workspace.

    On each call it gets-or-loads the workspace's CodeIndex (cached across calls),
    cheap-refreshes it (stat-based incremental), persists it if anything changed,
    runs the hybrid search, and returns ``{"ok": True, "result": <formatted>}``
    with each hit as ``path:start-end`` + snippet, byte-capped via _truncate.

    ``private`` is accepted for signature symmetry with the other tool builders
    but is a NO-OP for code_search: it only reads local workspace files (no egress)
    so it is SAFE in --private lockdown and is intentionally never dropped there.

    ``config`` is read LIVE on each call (not captured at build time) for the
    gated LLM-judge reranker: ``config.rerank`` / ``config.rerank_candidates``,
    and for ``config.code_search_recall`` (the search mode: "bm25" lexical-only by
    default, "auto"/"embed" to re-enable semantic recall). Reading it at call time
    means ``/rerank on`` / ``/codeembed on`` take effect this turn without
    rebuilding the agent. ``None`` (tests/sub-agents) leaves rerank off and the
    recall mode at the safe "bm25" default (no embedding-model GPU swap).
    """
    del private  # no-op: code_search is local-only and safe in private mode
    from .tools import Tool

    default_top_k = top_k if isinstance(top_k, int) and top_k > 0 else 5
    ws = os.path.abspath(str(workspace))

    def _code_search(args: dict) -> dict:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "code_search requires a non-empty 'query' string."}
        k = args.get("top_k")
        if not (isinstance(k, int) and not isinstance(k, bool) and k > 0):
            k = default_top_k

        idx = _INDEX_CACHE.get(ws)
        if idx is None:
            # First touch this process: load from disk (empty on missing/corrupt).
            idx = CodeIndex.load(index_path(ws))
            _INDEX_CACHE[ws] = idx
        idx.refresh(ws)          # cheap stat-based incremental re-chunk
        # Read the rerank flag LIVE so /rerank on|off applies this turn. Defaults
        # off when no config is supplied (tests, sub-agents built without one).
        rr = bool(getattr(config, "rerank", False)) if config is not None else False
        rc = getattr(config, "rerank_candidates", 20) if config is not None else 20
        # Read the recall mode LIVE so /codeembed on|off applies this turn. Defaults
        # to "bm25" (lexical-only, no embedding-model GPU swap) when no config is
        # supplied (tests, sub-agents built without one).
        mode = getattr(config, "code_search_recall", "bm25") if config is not None else "bm25"
        hits = idx.search(query, top_k=k, provider=provider, mode=mode, rerank=rr,
                          rerank_candidates=rc)
        # Persist AFTER search so query-time embedded vectors are saved too (otherwise a
        # static repo re-embeds from scratch every process); no-op when nothing changed.
        idx.save(index_path(ws))
        # Cap to the code_search-specific budget (< _MAX_OUTPUT, the hard ceiling)
        # so a single call can't dominate context; _truncate stays graceful.
        return {"ok": True,
                "result": _truncate(_format_hits(hits, idx.capped), CODE_SEARCH_MAX_OUTPUT)}

    return Tool(
        name="code_search",
        description=(
            "Search THIS project's code by meaning or keyword. Returns the most "
            "relevant code chunks as path:lines + snippet. Use it to FIND where "
            "something is implemented when you don't know the file — smarter/cheaper "
            "than reading whole files; then read_file that path with offset/limit "
            "for more."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to find (keyword or meaning)."},
                "top_k": {"type": "integer", "description": "Max chunks to return (default 5)."},
            },
            "required": ["query"],
        },
        fn=_code_search,
        requires_confirmation=False,
    )
