"""Conversation-memory retrieval core: hybrid BM25 ∪ embeddings, offline-testable.

STAGE 1 (this module): the retrieval engine ONLY — no agent-loop wiring. It is
pure stdlib (json, hashlib, math, re, os, tempfile, dataclasses) so it imports
and runs without numpy, openai, or a live LM Studio.

Design:
  - BM25 (hand-rolled Okapi, k1=1.5, b=0.75, +1-smoothed IDF) over record texts
    is ALWAYS available — it needs no network and is deterministic.
  - Embeddings are OPTIONAL: when a provider exposes ``embeddings()`` we add a
    cosine ranking and fuse it with BM25. ANY embedding failure (no provider, no
    endpoint, an exception, an empty/short result) silently falls back to
    BM25-only — retrieval never raises.
  - One shared tokenizer (lowercase ``[a-z0-9]+``) is applied to BOTH docs and
    queries so ``/mcp`` -> ``mcp`` and ``read_file`` -> ``read``,``file``.

Persistence mirrors session.py's atomic mkstemp+os.replace pattern: a save NEVER
raises and a load returns an EMPTY store on a missing/corrupt file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .session import _harden_state_dir, session_id, sessions_dir

# Shared tokenizer: lowercase runs of [a-z0-9]. Applied IDENTICALLY to docs and
# queries so lexical matching is symmetric (e.g. 'read_file' and '/mcp').
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English function words dropped from BM25 tokens. They carry no topical
# signal but, in a SMALL store, get inflated IDF and produce spurious matches —
# and, critically, a stopword-only overlap between a paraphrased query and a
# record would leave a nonzero BM25 score, so "auto" mode would NOT escalate to
# the embedding path that exists to handle exactly that paraphrase case. Dropping
# them makes lexical overlap mean CONTENT overlap, so the auto-gate is crisp.
# Deliberately EXCLUDES domain-meaningful short words this CLI uses as values or
# negations ("on"/"off" toggles like '/mcp on|off', "no"/"not"), so lexical
# precision on commands is preserved — only topically-empty function words go.
_STOPWORDS = frozenset(
    "a an and are as at be but by do does for from has have how i if in into is "
    "it its me my of or our so that the their them then there these "
    "this to up us was we were what when where which who will with you your".split()
)


def _tokenize(s: str) -> list[str]:
    """Lowercase ``s`` and return its content [a-z0-9]+ tokens (stopwords dropped).

    Empty input -> []. An all-stopword string also yields [] (a query that is
    pure function words has no lexical signal, which is the correct trigger for
    the embedding escalation in MemoryStore.retrieve).
    """
    if not s:
        return []
    return [t for t in _TOKEN_RE.findall(s.lower()) if t not in _STOPWORDS]


def content_hash(text: str) -> str:
    """Stable 16-hex content id = first 16 chars of sha256(text). Used to dedupe
    identical records AND key the embedding cache so the same text is embedded
    only once."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """One remembered item. ``text`` is the raw, lossless content; ``summary`` is
    an optional short gist; ``content_hash`` dedupes/keys the vector cache; ``ts``
    is an optional ISO timestamp."""

    id: str
    text: str
    summary: str
    content_hash: str
    ts: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "summary": self.summary,
            "content_hash": self.content_hash,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MemoryRecord:
        """Tolerant of missing/extra keys: every field has a safe default and a
        missing/empty content_hash is recomputed from the text."""
        text = str(d.get("text", "") or "")
        ch = str(d.get("content_hash", "") or "") or content_hash(text)
        return cls(
            id=str(d.get("id", "") or ""),
            text=text,
            summary=str(d.get("summary", "") or ""),
            content_hash=ch,
            ts=str(d.get("ts", "") or ""),
        )


# ---------------------------------------------------------------------------
# BM25 (hand-rolled Okapi) + cosine — pure, deterministic, unit-testable
# ---------------------------------------------------------------------------

def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """Rank ``docs`` against ``query`` with Okapi BM25; return (doc_index, score)
    sorted by score desc (ties broken by ascending index for determinism).

    Empty corpus -> []. Empty query (no tokens) -> every doc at score 0.0. IDF is
    the +1-smoothed form ``ln(1 + (N - df + 0.5)/(df + 0.5))`` so it is always
    positive (no negative-IDF surprises on common terms)."""
    return _bm25_over_tokenized(query, [_tokenize(d) for d in docs], k1, b)


def _bm25_over_tokenized(query: str, tokenized: list[list[str]], k1: float = 1.5,
                         b: float = 0.75) -> list[tuple[int, float]]:
    """BM25 core over ALREADY-tokenized docs so a caller (MemoryStore.retrieve)
    can cache per-record tokenization across turns instead of re-tokenizing every
    doc on each query. Identical scoring to bm25_rank (which tokenizes, then calls
    this)."""
    n = len(tokenized)
    if n == 0:
        return []
    lengths = [len(t) for t in tokenized]
    avgdl = (sum(lengths) / n) if n else 0.0
    tfs: list[dict[str, int]] = []
    for toks in tokenized:
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)

    q_terms = set(_tokenize(query))  # unique terms: a repeated query term scores once
    if not q_terms:
        return [(i, 0.0) for i in range(n)]

    df = {term: sum(1 for tf in tfs if term in tf) for term in q_terms}
    scored: list[tuple[int, float]] = []
    for i in range(n):
        tf = tfs[i]
        dl = lengths[i]
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            n_q = df[term]
            idf = math.log(1 + (n - n_q + 0.5) / (n_q + 0.5))
            denom = f + k1 * (1 - b + b * (dl / avgdl if avgdl else 0.0))
            s += idf * (f * (k1 + 1)) / denom
        scored.append((i, s))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


# ---------------------------------------------------------------------------
# Cached BM25 corpus index + query-vector LRU — pure caching wins, identical
# scoring/behaviour. Built once from tokenized docs and reused across queries
# so a 4000-chunk repo doesn't rebuild 4000 per-doc tf dicts + a df map on
# every search; the query LRU skips a redundant embed round-trip (+ LM-Studio
# model swap) for a repeated query.
# ---------------------------------------------------------------------------


@dataclass
class _BM25Index:
    """Cached Okapi BM25 corpus structures. Built once from tokenized docs;
    dropped (set None) whenever the corpus changes and rebuilt lazily on the
    next query. Mirrors the ``_bm25_tokens`` cache sentinel's lifecycle so the
    two are invalidated together."""
    tfs: list[dict[str, int]]
    lengths: list[int]
    avgdl: float
    df_map: dict[str, int]
    n: int


def _build_bm25_index(tokenized: list[list[str]]) -> _BM25Index:
    """Build the cached BM25 structures (per-doc tf, doc lengths, avgdl, and a
    global term->df map) from already-tokenized docs. O(total tokens); called
    once per corpus change instead of once per query."""
    n = len(tokenized)
    tfs: list[dict[str, int]] = []
    lengths = [0] * n
    df_map: dict[str, int] = {}
    for i, toks in enumerate(tokenized):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)
        lengths[i] = len(toks)
        for term in tf:  # each distinct term in this doc bumps its document frequency
            df_map[term] = df_map.get(term, 0) + 1
    avgdl = (sum(lengths) / n) if n else 0.0
    return _BM25Index(tfs=tfs, lengths=lengths, avgdl=avgdl, df_map=df_map, n=n)


def _bm25_score_index(query: str, idx: _BM25Index, k1: float = 1.5,
                      b: float = 0.75) -> list[tuple[int, float]]:
    """Score ``idx`` against ``query`` with Okapi BM25. IDENTICAL scoring to
    ``_bm25_over_tokenized`` (same k1/b, same +1-smoothed IDF, same sort) but
    reuses the cached tf/lengths/avgdl/df so only the query terms are iterated.
    Empty corpus -> []. Empty query (no tokens) -> every doc at 0.0."""
    n = idx.n
    if n == 0:
        return []
    q_terms = set(_tokenize(query))  # unique terms: a repeated query term scores once
    if not q_terms:
        return [(i, 0.0) for i in range(n)]
    scored: list[tuple[int, float]] = []
    for i in range(n):
        tf = idx.tfs[i]
        dl = idx.lengths[i]
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            n_q = idx.df_map.get(term, 0)
            idf = math.log(1 + (n - n_q + 0.5) / (n_q + 0.5))
            denom = f + k1 * (1 - b + b * (dl / idx.avgdl if idx.avgdl else 0.0))
            s += idf * (f * (k1 + 1)) / denom
        scored.append((i, s))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


# Process-wide default size for the per-store query-vector LRU. Small on
# purpose: query vectors are cheap to recompute and the win is the common
# "same query twice in a row" case, not long-term memoization.
_QUERY_VEC_LRU_SIZE = 32


class _QueryVecLRU:
    """Small LRU of query embedding vectors keyed by ``content_hash(query)``.

    A repeated/near-identical query skips the embed HTTP round-trip (and the
    LM-Studio model swap on a shared single-GPU setup). Each entry stores its
    embedding dim so a model swap (dim change) is detected and the WHOLE cache
    flushed rather than serving a stale-dimension vector — consistent with the
    record-vector dim guard in MemoryStore._embed_rank / CodeIndex._embed_rank.
    """

    __slots__ = ("_size", "_d")

    def __init__(self, size: int = _QUERY_VEC_LRU_SIZE):
        self._size = size
        self._d: "OrderedDict[str, tuple[int, list[float]]]" = OrderedDict()

    def get(self, key: str) -> list[float] | None:
        """Return the cached query vector for ``key`` (moving it to MRU), or
        ``None`` on a miss."""
        ent = self._d.get(key)
        if ent is None:
            return None
        self._d.move_to_end(key)
        return ent[1]

    def put(self, key: str, dim: int, vec: list[float]) -> None:
        """Cache ``vec`` (with its dim) under ``key``. If the dim differs from
        any existing entry's dim the embed model changed mid-session -> flush
        the whole cache before storing, so a stale-dim vector is never served."""
        if self._d:
            existing_dim = next(iter(self._d.values()))[0]
            if existing_dim != dim:
                self._d.clear()
        self._d[key] = (dim, vec)
        self._d.move_to_end(key)
        while len(self._d) > self._size:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors. Returns 0.0 on a length
    mismatch, an empty vector, or a zero-norm vector (never raises/NaNs)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


# ---------------------------------------------------------------------------
# Vector persistence: packed-float32 sidecar (shared by MemoryStore + CodeIndex)
# ---------------------------------------------------------------------------
#
# Corpus vectors are already ``array('f')`` (float32) in RAM. Writing them INLINE
# in the store JSON quadruples on-disk size and spikes peak-RAM on load (parse the
# whole JSON string, then materialize every float). Instead we persist them as a
# packed float32 blob in a SIDECAR file next to the JSON, and the JSON keeps only
# slim metadata (dim + hash order). load() mmap-free reads the exact byte count.
#
# SAFETY (this is a shipped on-disk format change):
#   - Backward compatible: an OLD-format JSON (no ``vector_format`` key, vectors
#     inline) still loads via the legacy branch.
#   - Never lose data: if ANYTHING about the sidecar fails on save, the caller
#     writes the legacy inline JSON instead. If the sidecar is missing/short/torn
#     on load, vectors are treated as ABSENT ({}) — they lazily re-embed via the
#     existing content-hash guard. Never crash, never load wrong vectors.
#   - Atomic: same mkstemp + os.replace discipline as the JSON write.
_VECTOR_FORMAT = "f32-sidecar-v1"


def _sidecar_path(path) -> Path:
    """The packed-float32 vector sidecar next to the store JSON: ``<path>.vec``."""
    return Path(str(path) + ".vec")


def _unlink_sidecar(path) -> None:
    """Best-effort remove a stale ``<path>.vec`` sidecar (e.g. after an inline
    fallback save, so an orphaned old sidecar isn't left next to fresh inline
    meta). Never raises."""
    try:
        os.unlink(_sidecar_path(path))
    except OSError:
        pass


def _atomic_write_bytes(path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: a temp file in the SAME directory
    (so ``os.replace`` is atomic on one filesystem) then replace. The parent dir
    must already exist. Raises ``OSError`` on failure after removing the temp
    file (mirrors session.py's mkstemp+os.replace pattern)."""
    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, str(target))
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path, payload) -> None:
    """Serialize ``payload`` to UTF-8 JSON and write it to ``path`` atomically."""
    _atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _pack_vectors(vectors) -> tuple[list[str], int, bytes] | None:
    """Pack ``{hash: array('f')}`` into ``(vhashes, dim, blob)`` for the f32
    sidecar, where ``blob`` concatenates each vector's raw float32 bytes in
    ``vhashes`` order. Return ``None`` — signalling the caller to write the legacy
    inline format — when the set can't be packed UNIFORMLY: empty, zero-dim, or
    any vector whose length differs from the first."""
    vhashes = [h for h in vectors]
    if not vhashes:
        return None
    dim = len(vectors[vhashes[0]])
    if dim <= 0:
        return None
    buf = bytearray()
    for h in vhashes:
        v = vectors[h]
        if len(v) != dim:
            return None
        buf += array('f', v).tobytes()  # already float32; copy is cheap + safe
    return vhashes, dim, bytes(buf)


def _save_vectors_sidecar(path, base_meta: dict, vectors) -> bool:
    """Write the packed-f32 sidecar ``<path>.vec`` PLUS the slim meta JSON
    (``base_meta`` augmented with ``vector_format``/``vector_dim``/
    ``vector_hashes``) to ``path`` — both atomically, SIDECAR FIRST so the meta
    that references it is only committed once the bytes exist.

    Return ``True`` on success. Return ``False`` — signalling the caller to fall
    back to the legacy inline-JSON write — when the vectors can't be packed
    (empty/ragged) OR on ANY error, so a failed/torn sidecar never loses data."""
    try:
        packed = _pack_vectors(vectors)
        if packed is None:
            return False
        vhashes, dim, blob = packed
        _atomic_write_bytes(_sidecar_path(path), blob)  # sidecar FIRST
        meta = dict(base_meta)
        meta["vector_format"] = _VECTOR_FORMAT
        meta["vector_dim"] = dim
        meta["vector_hashes"] = vhashes
        # Fingerprint the exact sidecar bytes so load() can reject a torn write
        # (NEW sidecar + STALE meta) that the size guard alone would pass — same
        # count+dim means identical byte length, so bytes-identity is the only
        # cross-check that catches positionally-mispaired vectors.
        meta["vector_sig"] = hashlib.sha1(blob).hexdigest()[:16]
        meta["vector_byteorder"] = sys.byteorder  # self-describing (native f32)
        _atomic_write_json(path, meta)  # then the meta that references it
        return True
    except Exception:
        return False


def _read_vector_sidecar(path, dim, vhashes, vector_sig=None, byteorder="little") -> dict:
    """Read the f32 sidecar ``<path>.vec`` and rebuild ``{hash: array('f')}``.
    Return ``{}`` (vectors lazily re-embed; never crash) when the file is missing,
    the metadata is malformed, the byte size != ``dim*4*len(vhashes)``, or the
    content fingerprint doesn't match ``vector_sig`` (a torn write left a NEW
    sidecar paired with a STALE meta — same size, wrong bytes)."""
    if not isinstance(dim, int) or dim <= 0 or not isinstance(vhashes, list):
        return {}
    try:
        raw = _sidecar_path(path).read_bytes()
    except OSError:
        return {}
    stride = dim * 4  # float32 == 4 bytes
    if len(raw) != stride * len(vhashes):
        return {}
    # Content fingerprint cross-check (defense beyond the size guard): rejects a
    # count-preserving, same-dim torn write whose STALE meta would otherwise pair
    # vhashes against unrelated new bytes. Absent sig -> size-check-only (no regress).
    if vector_sig is not None and hashlib.sha1(raw).hexdigest()[:16] != vector_sig:
        return {}
    swap = byteorder != sys.byteorder  # correct a foreign-endian .vec on read
    out: dict = {}
    for i, h in enumerate(vhashes):
        if not isinstance(h, str):
            return {}
        a = array('f')
        a.frombytes(raw[i * stride:(i + 1) * stride])
        if swap:
            a.byteswap()  # fingerprint is over on-disk bytes, so it still matches
        out[h] = a
    return out


def _load_vectors(path, data: dict) -> dict:
    """Reconstruct ``{hash: array('f')}`` from a parsed store JSON ``data``.

    New format (``vector_format == 'f32-sidecar-v1'``): read the packed-f32
    sidecar next to ``path`` (absent/short/torn -> ``{}``). Old format (no
    ``vector_format`` key): parse the inline ``data['vectors']`` exactly as the
    legacy code did (backward compatible). Never raises."""
    if data.get("vector_format") == _VECTOR_FORMAT:
        return _read_vector_sidecar(
            path,
            data.get("vector_dim"),
            data.get("vector_hashes"),
            data.get("vector_sig"),
            data.get("vector_byteorder", "little"),
        )
    vectors: dict = {}
    raw_vecs = data.get("vectors")
    if isinstance(raw_vecs, dict):
        for h, v in raw_vecs.items():
            if isinstance(h, str) and isinstance(v, list):
                try:
                    vectors[h] = array('f', (float(x) for x in v))
                except (TypeError, ValueError):
                    continue
    return vectors


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

# Hard cap on stored records so a long-lived project's memory file (and its
# per-retrieve tokenization/embedding cost) stays bounded. Past this the OLDEST
# records are evicted on add(); their orphaned vectors are dropped too so records
# + vectors stay O(MAX_RECORDS) rather than growing without limit.
MAX_RECORDS = 500


@dataclass
class MemoryStore:
    """Holds the remembered records plus a content-hash-keyed embedding cache.

    ``vectors`` maps content_hash -> embedding so the SAME text is embedded at
    most once across a session (and across save/load)."""

    records: list[MemoryRecord] = field(default_factory=list)
    # Values are ``array('f')`` (float32) at runtime, not boxed ``list[float]`` —
    # a ~85% RAM cut for a full corpus. On-disk JSON is unchanged (save() emits
    # ``list(v)``, load() rebuilds ``array('f')``).
    vectors: dict[str, list[float]] = field(default_factory=dict)
    # Set by add()/eviction, cleared by a successful save(); makes save() a no-op
    # when nothing changed so the REPL's per-turn save doesn't re-serialize an
    # unchanged store. load() leaves it False (a fresh load matches disk).
    _dirty: bool = field(default=False, compare=False, repr=False)
    # Cached per-record tokenization for BM25, invalidated (set None) on add/evict
    # so retrieve doesn't re-tokenize every doc on each query. Never persisted.
    _bm25_tokens: list | None = field(default=None, compare=False, repr=False)
    # Cached BM25 corpus structures (per-doc tf, lengths, avgdl, df_map) built
    # from ``_bm25_tokens`` and invalidated alongside it. Saves a 4000-doc
    # rebuild of tf dicts + df on every retrieve. Never persisted.
    _bm25_index: _BM25Index | None = field(default=None, compare=False, repr=False)
    # Per-store LRU of query embedding vectors keyed by content_hash(query):
    # a repeated query skips the embed round-trip + model swap. Never persisted.
    _query_vecs: _QueryVecLRU = field(default_factory=_QueryVecLRU, compare=False, repr=False)

    def add(self, text: str, summary: str = "", ts: str = "") -> MemoryRecord:
        """Append a record for ``text`` (id = ``r{index}``) and return it. Dedupes
        by content_hash: re-adding identical text returns the EXISTING record
        rather than double-storing it. Past MAX_RECORDS the oldest records (and
        their orphaned vectors) are evicted so the store stays bounded."""
        ch = content_hash(text)
        for r in self.records:
            if r.content_hash == ch:
                return r  # dedupe: existing record, store unchanged (not dirtied)
        rec = MemoryRecord(id=f"r{len(self.records)}", text=text, summary=summary,
                           content_hash=ch, ts=ts)
        self.records.append(rec)
        self._dirty = True
        self._bm25_tokens = None  # records changed -> drop the tokenization cache
        self._bm25_index = None
        self._evict_overflow()
        return rec

    def _evict_overflow(self) -> None:
        """Cap records at MAX_RECORDS by dropping the OLDEST, and drop any cached
        vector whose content_hash no longer belongs to a surviving record so
        ``vectors`` can't outgrow ``records``. No-op under the cap."""
        if len(self.records) <= MAX_RECORDS:
            return
        self.records = self.records[len(self.records) - MAX_RECORDS:]
        live = {r.content_hash for r in self.records}
        self.vectors = {h: v for h, v in self.vectors.items() if h in live}
        self._dirty = True
        self._bm25_tokens = None
        self._bm25_index = None

    def retrieve(self, query: str, provider=None, mode: str = "auto",
                 top_k: int = 3, rerank: bool = False,
                 rerank_candidates: int = 20) -> list[MemoryRecord]:
        """Return up to ``top_k`` records most relevant to ``query``.

        BM25 is ALWAYS computed first. The embeddings path (cosine fused with
        BM25, round-robin BM25-first so lexical exactness wins ties) is gated:

          - "bm25" -> lexical only;  "off" -> [].
          - "embed" -> ALWAYS hybrid (the opt-in for a resident/separate embed
            endpoint): embeddings are attempted every retrieve.
          - "auto" (the DEFAULT) -> embeddings are attempted ONLY when BM25 found
            NO lexical signal (its best score over the corpus is <= 0, i.e. the
            query shares no scoring token with any record — the paraphrase case
            embeddings exist to cover). When BM25 has a real top hit we return
            BM25-only and DO NOT embed.

        Rationale: on a shared single-GPU LM Studio, embedding the query swaps the
        embed model in/out, so "auto" pays that cost only when lexical search has
        clearly failed. ANY embedding failure degrades silently to BM25-only —
        retrieval never raises."""
        if mode == "off" or top_k <= 0 or not self.records:
            return []
        if self._bm25_tokens is None:
            self._bm25_tokens = [_tokenize(r.text) for r in self.records]
            self._bm25_index = _build_bm25_index(self._bm25_tokens)
        # When the LLM-judge reranker is opted in, widen the candidate pool so the
        # judge has more to rank than the final top_k (a rerank that only sees
        # top_k candidates cannot narrow anything). pool_k stays == top_k when
        # rerank is off, so the off-path is byte-for-byte unchanged.
        pool_k = max(top_k, rerank_candidates) if rerank else top_k
        bm25_full = _bm25_score_index(query, self._bm25_index)
        bm25_idx = [i for i, s in bm25_full[:pool_k] if s > 0.0]
        best_bm25 = bm25_full[0][1] if bm25_full else 0.0
        embed_idx: list[int] = []
        if provider is not None and (
            mode == "embed" or (mode == "auto" and best_bm25 <= 0.0)
        ):
            embed_idx = self._embed_rank(query, provider, pool_k)
        ordered = bm25_idx if not embed_idx else self._fuse(bm25_idx, embed_idx, pool_k)
        # GATED LLM-judge rerank: fires ONLY on the weak-signal path (embeddings
        # were attempted -> embed_idx non-empty == BM25 found no lexical hit) AND
        # only when the pool is wider than the final top_k. Default rerank=False
        # skips the branch entirely -> zero behavior change, zero latency. Any
        # failure falls back to the fused order; retrieval never raises.
        if (
            rerank and embed_idx and provider is not None
            and len(ordered) > top_k
        ):
            from .rerank import rerank as _rerank
            cands = ordered[:rerank_candidates]
            try:
                reord = _rerank(
                    provider, query,
                    [(i, self.records[i].text) for i in cands],
                    top_k,
                )
                ordered = reord + [i for i in ordered if i not in set(reord)]
            except Exception:  # noqa: BLE001 - keep fused order on any failure
                pass
        return [self.records[i] for i in ordered[:top_k]]

    def _embed_rank(self, query: str, provider, top_k: int) -> list[int]:
        """Cosine ranking via ``provider.embeddings``, or [] on ANY failure.

        DIMENSION GUARD: the QUERY is embedded FIRST so we know the target dim.
        Any record whose cached vector length != that dim (e.g. the embed model
        changed between sessions) is treated as MISSING and re-embedded — this
        self-heals a stale cache. A vector that STILL mismatches afterwards is
        SKIPPED rather than fed to cosine, so a dim mismatch can never corrupt the
        ranking. Returns [] — so the caller falls back to BM25 — when the provider
        has no usable ``embeddings``, raises, or returns an empty/short result
        (including the query failing to embed)."""
        embed = getattr(provider, "embeddings", None)
        if not callable(embed):
            return []
        try:
            # Query-vector LRU: a repeated query skips the embed round-trip (and
            # the LM-Studio model swap). Keyed by content_hash so text identity
            # — not object identity — is what's cached.
            qh = content_hash(query)
            qvec = self._query_vecs.get(qh)
            if qvec is None:
                # Embed the query FIRST to learn the target dimension.
                qres = embed([query])
                if not qres or not qres[0]:
                    return []
                qvec = [float(x) for x in qres[0]]
                self._query_vecs.put(qh, len(qvec), qvec)
            qdim = len(qvec)
            # MISSING = no cached vector OR a cached vector of the wrong dim
            # (stale cache from a different embed model) -> re-embed to self-heal.
            missing = [
                r for r in self.records
                if len(self.vectors.get(r.content_hash) or []) != qdim
            ]
            if missing:
                vecs = embed([r.text for r in missing])
                if not vecs or len(vecs) != len(missing):
                    return []
                for r, v in zip(missing, vecs):
                    self.vectors[r.content_hash] = array('f', (float(x) for x in v))
                # Newly embedded vectors must survive save/load — otherwise a turn
                # that escalates to embeddings but doesn't also add() a record
                # (truncated/error turn) silently drops them and re-embeds next
                # time (full GPU embed + model swap). Mirrors CodeIndex._embed_rank.
                self._dirty = True
        except Exception:  # noqa: BLE001 - ANY embedding failure -> BM25 fallback
            return []
        scored: list[tuple[int, float]] = []
        for i, r in enumerate(self.records):
            v = self.vectors.get(r.content_hash)
            # Skip a missing or still-dim-mismatched vector (never a garbage cosine).
            if not v or len(v) != qdim:
                continue
            scored.append((i, cosine(qvec, v)))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [i for i, _ in scored[:top_k]]

    @staticmethod
    def _fuse(bm25_idx: list[int], embed_idx: list[int], top_k: int) -> list[int]:
        """Round-robin merge of the two ranked index lists, BM25 first, deduped,
        capped at ``top_k``. Taking bm25[0] before embed[0] keeps lexical hits in
        front on ties."""
        merged: list[int] = []
        seen: set[int] = set()
        for i in range(max(len(bm25_idx), len(embed_idx))):
            for ranked in (bm25_idx, embed_idx):
                if i < len(ranked):
                    idx = ranked[i]
                    if idx not in seen:
                        seen.add(idx)
                        merged.append(idx)
                        if len(merged) >= top_k:
                            return merged
        return merged

    # ----- persistence (atomic; mirrors session.py:107-111) ----------------

    def purge(self, ids: list[str] | None = None) -> int:
        """Purge records by ID (or all records if ``ids`` is ``None``).

        Returns the count of records removed. After purging, ``_dirty`` is set
        so the next save() persists the reduced store.

        Use ``/memory purge all`` to clear everything, or ``/memory purge r0 r2``
        to selectively remove records.
        """
        if ids is None:
            # Clear all records and vectors.
            removed = len(self.records)
            if removed > 0:
                self.records.clear()
                self.vectors.clear()
                self._bm25_tokens = None
                self._bm25_index = None
                self._dirty = True
            return removed
        # Selective purge: remove matching IDs, drop their vectors too.
        before = len(self.records)
        before_ids = {r.id for r in self.records}
        # Record hashes of records being removed (before filtering)
        removed_hashes = set()
        for r in list(self.records):
            if r.id in ids:
                removed_hashes.add(r.content_hash)
        self.records = [r for r in self.records if r.id not in ids]
        # Drop vectors for removed records
        for h in removed_hashes:
            self.vectors.pop(h, None)
        self._bm25_tokens = None
        self._bm25_index = None
        removed = before - len(self.records)
        if removed:
            self._dirty = True
        return removed

    def compact(self, max_records: int | None = None) -> int:
        """Compact the store to ``max_records`` (default: MAX_RECORDS).

        Drops the OLDEST records past the limit, along with their orphaned vectors.
        Returns the count of records removed.

        Use ``/memory compact <N>`` to reduce the store to N records, or
        ``/memory compact`` (no arg) to use the default MAX_RECORDS limit.
        """
        limit = max_records if max_records is not None else MAX_RECORDS
        if limit >= len(self.records):
            return 0  # nothing to compact
        before = len(self.records)
        # Drop oldest records past the limit.
        self.records = self.records[before - limit:]
        live = {r.content_hash for r in self.records}
        self.vectors = {h: v for h, v in self.vectors.items() if h in live}
        self._bm25_tokens = None
        self._bm25_index = None
        removed = before - len(self.records)
        if removed:
            self._dirty = True
        return removed

    # ----- persistence (atomic; mirrors session.py:107-111) ----------------

    def save(self, path) -> None:
        """Atomically persist records + vectors. Best-effort: a write temp file in
        the same dir then os.replace; never raises.

        Vectors are written as a packed float32 SIDECAR (``<path>.vec``) with the
        JSON holding only slim metadata (``vector_format``/``vector_dim``/
        ``vector_hashes``). If the sidecar can't be written (empty/ragged vectors,
        or ANY error) it FALLS BACK to the legacy inline-JSON format so data is
        never lost.

        NO-OP when the store is not ``_dirty`` (nothing added/evicted since the
        last save/load), so the REPL's per-turn save never re-serializes an
        unchanged store."""
        if not self._dirty:
            return
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _harden_state_dir()  # restrict ~/.llmcode to the owner (best-effort)
        except OSError:
            return  # can't prepare the dir — best-effort: silently skip
        records = [r.to_dict() for r in self.records]
        # Preferred: packed-f32 vector sidecar + slim meta JSON (sidecar FIRST).
        if _save_vectors_sidecar(target, {"records": records}, self.vectors):
            self._dirty = False
            return
        # Fallback: legacy inline-JSON (vectors embedded), best-effort. Drop any
        # stale sidecar first so an old `.vec` isn't orphaned beside inline meta.
        _unlink_sidecar(target)
        payload = {"records": records,
                   "vectors": {h: list(v) for h, v in self.vectors.items()}}
        try:
            _atomic_write_json(target, payload)
            self._dirty = False  # on-disk copy now matches memory
        except OSError:
            return  # best-effort: disk full / permissions / etc. — silently skip

    @classmethod
    def load(cls, path) -> MemoryStore:
        """Read a store from ``path``; an EMPTY store on missing/corrupt/invalid
        input (never raises). Non-dict records and non-numeric vectors are
        dropped so a hand-edited file can't crash a consumer. Vectors load from
        the f32 sidecar (new format) or inline JSON (old format); a missing/torn
        sidecar yields ``{}`` (lazy re-embed) rather than crashing."""
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
        records = [MemoryRecord.from_dict(rd) for rd in (data.get("records") or [])
                   if isinstance(rd, dict)]
        return cls(records=records, vectors=_load_vectors(path, data))


def store_path(cwd: str) -> Path:
    """Path to the per-project memory file: sessions_dir()/<session_id>.memory.json
    (sits alongside the conversation session, keyed by the same workspace id)."""
    return sessions_dir() / f"{session_id(cwd)}.memory.json"
