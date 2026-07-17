"""Conversation-memory retrieval tests. All offline/deterministic (MockProvider).

Covers the pure primitives (_tokenize, bm25_rank, cosine), MemoryRecord
round-trip, MemoryStore.add dedupe, retrieve in every mode (bm25/off/auto/embed)
with graceful BM25 fallback, incremental embedding caching, and save/load
round-trip including missing/corrupt files.
"""

from __future__ import annotations

import math

from llmcli.memory import (
    MemoryRecord,
    MemoryStore,
    bm25_rank,
    content_hash,
    cosine,
    store_path,
    _tokenize,
)
from llmcli.providers import MockProvider
from llmcli.session import session_id


# --------------------------------------------------------------------------- #
# _tokenize
# --------------------------------------------------------------------------- #

def test_tokenize_lowercases_and_splits_on_non_alnum():
    assert _tokenize("/mcp on|off") == ["mcp", "on", "off"]
    assert _tokenize("read_file(offset, limit)") == ["read", "file", "offset", "limit"]
    assert _tokenize("MixedCASE 123abc") == ["mixedcase", "123abc"]


def test_tokenize_empty_is_empty_list():
    assert _tokenize("") == []
    assert _tokenize("   ---  ") == []


# --------------------------------------------------------------------------- #
# bm25_rank
# --------------------------------------------------------------------------- #

_DOCS = [
    "the mcp toggle command turns servers on or off",
    "read_file supports offset and limit for big files",
    "git rebase keeps a clean linear history",
]


def test_bm25_lexical_hit_ranks_right_doc_first():
    ranked = bm25_rank("mcp", _DOCS)
    assert ranked[0][0] == 0  # the mcp doc wins
    assert ranked[0][1] > 0.0
    # 'read_file' tokenizes to read,file -> querying 'file' hits doc 1.
    ranked2 = bm25_rank("file", _DOCS)
    assert ranked2[0][0] == 1


def test_bm25_returns_all_docs_sorted_desc():
    ranked = bm25_rank("history", _DOCS)
    assert len(ranked) == len(_DOCS)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0][0] == 2  # only doc 2 mentions history


def test_bm25_empty_corpus_safe():
    assert bm25_rank("anything", []) == []


def test_bm25_empty_query_scores_all_zero():
    ranked = bm25_rank("", _DOCS)
    assert len(ranked) == len(_DOCS)
    assert all(s == 0.0 for _, s in ranked)
    # Deterministic order: ascending index when all scores tie.
    assert [i for i, _ in ranked] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# cosine
# --------------------------------------------------------------------------- #

def test_cosine_identical_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_zero_norm_is_zero():
    assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0


def test_cosine_length_mismatch_is_zero():
    assert cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert cosine([], [1.0]) == 0.0


# --------------------------------------------------------------------------- #
# MemoryRecord round-trip
# --------------------------------------------------------------------------- #

def test_memory_record_round_trip():
    r = MemoryRecord(id="r0", text="hello world", summary="greeting",
                     content_hash=content_hash("hello world"), ts="2026-06-27")
    assert MemoryRecord.from_dict(r.to_dict()) == r


def test_memory_record_from_dict_tolerates_missing_and_extra_keys():
    # Missing content_hash is recomputed from text; extra keys are ignored.
    r = MemoryRecord.from_dict({"text": "yo", "bogus": 1})
    assert r.text == "yo"
    assert r.content_hash == content_hash("yo")
    assert r.id == "" and r.summary == "" and r.ts == ""
    # Wholly empty dict is safe too.
    empty = MemoryRecord.from_dict({})
    assert empty.text == ""
    assert empty.content_hash == content_hash("")


# --------------------------------------------------------------------------- #
# MemoryStore.add dedupe
# --------------------------------------------------------------------------- #

def test_add_dedupes_by_content_hash():
    store = MemoryStore()
    a = store.add("same text", summary="first")
    b = store.add("same text", summary="ignored duplicate")
    assert len(store.records) == 1
    assert a is b  # the existing record is returned, not a new one
    assert a.summary == "first"
    c = store.add("a different text")
    assert len(store.records) == 2
    assert a.id == "r0" and c.id == "r1"


# --------------------------------------------------------------------------- #
# retrieve: modes + graceful fallback
# --------------------------------------------------------------------------- #

def _seeded_store() -> MemoryStore:
    store = MemoryStore()
    for d in _DOCS:
        store.add(d)
    return store


def test_retrieve_off_returns_empty():
    assert _seeded_store().retrieve("mcp", provider=MockProvider(), mode="off") == []


def test_retrieve_bm25_only_finds_lexical_hit():
    res = _seeded_store().retrieve("mcp", mode="bm25", top_k=1)
    assert len(res) == 1
    assert res[0].text == _DOCS[0]


def test_retrieve_auto_returns_results_with_mock_provider():
    res = _seeded_store().retrieve("mcp", provider=MockProvider(), mode="auto", top_k=2)
    assert res
    assert any(r.text == _DOCS[0] for r in res)  # the lexical hit is present


def test_retrieve_embed_mode_returns_results_and_populates_cache():
    store = _seeded_store()
    res = store.retrieve("mcp", provider=MockProvider(), mode="embed", top_k=2)
    assert res
    # The embedding path ran -> the corpus vectors were cached for every record.
    assert len(store.vectors) == len(_DOCS)


def test_retrieve_falls_back_when_provider_has_no_embeddings():
    class _NoEmbed:
        pass  # no embeddings attribute at all

    res = _seeded_store().retrieve("mcp", provider=_NoEmbed(), mode="auto", top_k=1)
    assert len(res) == 1
    assert res[0].text == _DOCS[0]  # clean BM25 fallback


def test_retrieve_falls_back_when_embeddings_raise():
    class _BoomProvider(MockProvider):
        def embeddings(self, texts):
            raise RuntimeError("embed endpoint down")

    store = _seeded_store()
    res = store.retrieve("mcp", provider=_BoomProvider(), mode="auto", top_k=1)
    assert len(res) == 1
    assert res[0].text == _DOCS[0]  # never raised; BM25 still answered
    assert store.vectors == {}  # nothing got cached on failure


def test_retrieve_empty_store_is_safe():
    assert MemoryStore().retrieve("anything", provider=MockProvider()) == []


# --------------------------------------------------------------------------- #
# incremental embedding: cache only embeds NEW content hashes
# --------------------------------------------------------------------------- #

class _CountingProvider(MockProvider):
    """Records each embeddings() call's input list so tests can prove the corpus
    is embedded once and only NEW records are embedded on later retrieves."""

    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def embeddings(self, texts):
        self.calls.append(list(texts))
        return super().embeddings(texts)


def test_incremental_embedding_only_embeds_new_hashes():
    # mode="embed" (always-hybrid) so the embedding path runs every retrieve —
    # mode="auto" now SKIPS embeddings when BM25 has a lexical hit (these queries
    # all do), which is covered by the auto-gating tests below.
    store = _seeded_store()
    p = _CountingProvider()

    store.retrieve("mcp", provider=p, mode="embed", top_k=2)
    # First retrieve: all 3 corpus docs + the query are embedded.
    first_total = sum(len(c) for c in p.calls)
    assert first_total == len(_DOCS) + 1

    p.calls.clear()
    store.retrieve("file", provider=p, mode="embed", top_k=2)
    # Corpus is fully cached now -> only the query (1 text) is embedded.
    second_total = sum(len(c) for c in p.calls)
    assert second_total == 1
    assert all("read_file" not in t and _DOCS[0] not in t for c in p.calls for t in c)

    # Adding a NEW record embeds only that one new doc (+ the query).
    store.add("a brand new note about caching")
    p.calls.clear()
    store.retrieve("caching", provider=p, mode="embed", top_k=2)
    third_total = sum(len(c) for c in p.calls)
    assert third_total == 2  # the one new corpus doc + the query


# --------------------------------------------------------------------------- #
# save / load round-trip + missing/corrupt -> empty store
# --------------------------------------------------------------------------- #

def test_save_load_round_trip(tmp_path):
    store = _seeded_store()
    # Populate the vector cache so it is exercised by the round-trip.
    store.retrieve("mcp", provider=MockProvider(), mode="auto", top_k=2)
    p = tmp_path / "mem.json"
    store.save(p)

    loaded = MemoryStore.load(p)
    assert [r.to_dict() for r in loaded.records] == [r.to_dict() for r in store.records]
    assert set(loaded.vectors) == set(store.vectors)
    for h, v in store.vectors.items():
        assert loaded.vectors[h] == v
        assert len(v) == 64


def test_save_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "mem.json"
    store = _seeded_store()
    store.save(nested)
    assert nested.exists()


def test_load_missing_returns_empty_store(tmp_path):
    loaded = MemoryStore.load(tmp_path / "does-not-exist.json")
    assert loaded.records == []
    assert loaded.vectors == {}


def test_load_corrupt_returns_empty_store(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    loaded = MemoryStore.load(bad)
    assert loaded.records == []
    assert loaded.vectors == {}


def test_load_non_object_returns_empty_store(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text('["a", "b"]', encoding="utf-8")
    assert MemoryStore.load(p).records == []


def test_save_never_raises_on_oserror(tmp_path, monkeypatch):
    import llmcli.memory as m

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(m.tempfile, "mkstemp", _boom)
    # Must NOT raise.
    _seeded_store().save(tmp_path / "mem.json")


# --------------------------------------------------------------------------- #
# store_path: alongside the session, keyed by the same workspace id
# --------------------------------------------------------------------------- #

def test_store_path_shape(tmp_path):
    cwd = str(tmp_path)
    sp = store_path(cwd)
    assert sp.name == f"{session_id(cwd)}.memory.json"
    assert sp.parent.name == "sessions"


# --------------------------------------------------------------------------- #
# auto-gating: "auto" embeds ONLY when BM25 found no lexical signal
# --------------------------------------------------------------------------- #

def test_auto_skips_embeddings_on_strong_lexical_hit():
    store = _seeded_store()
    p = _CountingProvider()
    res = store.retrieve("mcp", provider=p, mode="auto", top_k=2)
    assert res
    assert res[0].text == _DOCS[0]  # BM25 found the lexical hit
    # BM25 had a real top hit, so embeddings were NEVER called (no model swap).
    assert p.calls == []


def test_auto_escalates_to_embeddings_on_no_lexical_overlap():
    store = _seeded_store()
    p = _CountingProvider()
    # A query sharing NO scoring token with any record -> best BM25 score is 0 ->
    # "auto" escalates to the embeddings path (the paraphrase case).
    res = store.retrieve("zzzzz qqqqq", provider=p, mode="auto", top_k=2)
    assert res  # still returns records (fused embeddings ranking)
    assert p.calls  # embeddings WERE attempted (query + corpus)


def test_embed_mode_always_embeds_even_with_lexical_hit():
    store = _seeded_store()
    p = _CountingProvider()
    # "embed" is the always-hybrid opt-in: embeddings run even on a lexical hit.
    store.retrieve("mcp", provider=p, mode="embed", top_k=2)
    assert p.calls  # embeddings attempted despite the strong BM25 hit


def test_retrieve_bm25_excludes_zero_score_records():
    """top_k > actual matching docs: zero-score records must NOT pad the result."""
    store = _seeded_store()
    # "mcp" lexically matches only _DOCS[0] (score > 0); _DOCS[1] and _DOCS[2] get
    # BM25 score 0.  Requesting top_k=3 with only 1 real match must return exactly 1.
    res = store.retrieve("mcp", mode="bm25", top_k=3)
    assert len(res) == 1
    assert res[0].text == _DOCS[0]


def test_auto_escalation_top_result_is_best_semantic_hit():
    """On auto-escalation (BM25 zero), res[0] must be the best embedding hit."""
    store = _seeded_store()
    p = MockProvider()
    query = "zzzzz qqqqq"  # no lexical overlap -> best_bm25=0 -> auto-escalates to embed
    res = store.retrieve(query, provider=p, mode="auto", top_k=3)
    assert res  # still returns records
    # After the fix: bm25_idx=[] so ordered=embed_idx directly (no BM25 padding).
    # Independently compute the embed ranking; res[0] must match its top hit.
    embed_order = store._embed_rank(query, p, 3)
    assert embed_order, "embed ranking must succeed with MockProvider"
    assert res[0] is store.records[embed_order[0]]


# --------------------------------------------------------------------------- #
# dimension guard: a wrong-length cached vector is self-healed / skipped
# --------------------------------------------------------------------------- #

def test_retrieve_self_heals_dim_mismatched_cached_vector():
    store = _seeded_store()
    # Poison one cached vector with a WRONG length (e.g. a different embed model).
    store.vectors[store.records[0].content_hash] = [1.0, 2.0]  # len 2, not 64
    res = store.retrieve("mcp", provider=MockProvider(), mode="embed", top_k=3)
    assert res  # no crash, no garbage cosine
    # The poisoned vector was treated as missing and re-embedded to the right dim.
    assert len(store.vectors[store.records[0].content_hash]) == 64
    # BM25-consistent: the lexical "mcp" hit still leads (fusion is BM25-first).
    assert res[0].text == _DOCS[0]


def test_retrieve_dim_mismatch_query_unembeddable_falls_back_to_bm25():
    class _QueryFailsEmbed(MockProvider):
        def embeddings(self, texts):
            # Query can't be embedded -> _embed_rank returns [] -> BM25 fallback.
            raise RuntimeError("embed endpoint down")

    store = _seeded_store()
    store.vectors[store.records[0].content_hash] = [9.9]  # bogus cached vector
    res = store.retrieve("zzzzz qqqqq", provider=_QueryFailsEmbed(), mode="embed", top_k=3)
    # No crash; BM25 over a no-overlap query still returns deterministic order.
    assert isinstance(res, list)


# --------------------------------------------------------------------------- #
# store cap + dirty flag (bounded cost)
# --------------------------------------------------------------------------- #

def test_store_caps_records_and_drops_orphan_vectors(monkeypatch):
    import llmcli.memory as m

    monkeypatch.setattr(m, "MAX_RECORDS", 3)
    store = MemoryStore()
    for i in range(5):
        store.add(f"note number {i}")
        # Attach a fake vector for the just-added record (simulating an embed).
        store.vectors[store.records[-1].content_hash] = [float(i)]

    # Capped at the (monkeypatched) MAX_RECORDS, oldest evicted.
    assert len(store.records) == 3
    assert [r.text for r in store.records] == [
        "note number 2", "note number 3", "note number 4"
    ]
    # No orphan vectors: every cached vector belongs to a surviving record.
    live = {r.content_hash for r in store.records}
    assert set(store.vectors) <= live
    assert len(store.vectors) == 3


def test_save_is_noop_when_not_dirty(tmp_path, monkeypatch):
    import llmcli.memory as m

    store = _seeded_store()  # 3 adds -> dirty
    p = tmp_path / "mem.json"
    store.save(p)
    assert p.exists()
    assert store._dirty is False  # cleared by the successful save

    # Spy on serialization: a clean store must NOT reach mkstemp again.
    calls = []
    orig = m.tempfile.mkstemp

    def _spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(m.tempfile, "mkstemp", _spy)
    store.save(p)
    assert calls == []  # short-circuited: nothing changed, no re-serialize

    store.add("a new note")  # re-dirties the store
    assert store._dirty is True
    store.save(p)
    assert calls == [1]  # now it serialized exactly once


def test_load_leaves_store_not_dirty(tmp_path):
    store = _seeded_store()
    p = tmp_path / "mem.json"
    store.save(p)
    loaded = MemoryStore.load(p)
    assert loaded._dirty is False  # a fresh load matches disk -> save() is a no-op


# --------------------------------------------------------------------------- #
# BM25 index cache + query-vector LRU (perf: pure caching wins, no behavior change)
# --------------------------------------------------------------------------- #

def test_bm25_cached_index_matches_fresh_recompute():
    """Regression: the cached _BM25Index must produce IDENTICAL scores to a
    fresh recompute via _bm25_over_tokenized for any query (including empty
    and no-overlap queries) — pure caching, no scoring change."""
    from llmcli.memory import (
        _bm25_over_tokenized,
        _build_bm25_index,
        _bm25_score_index,
    )
    docs = [d.split() for d in _DOCS]
    idx = _build_bm25_index(docs)
    for q in ["mcp", "file", "history", "zzz qqq no overlap", "", "the on off rebase"]:
        fresh = _bm25_over_tokenized(q, docs)
        cached = _bm25_score_index(q, idx)
        assert fresh == cached, f"score mismatch for query {q!r}: {fresh} vs {cached}"


def test_bm25_index_invalidated_on_add():
    """Regression: adding a record drops the cached BM25 index so the next
    retrieve rebuilds against the new corpus (no stale tf/df)."""
    store = _seeded_store()
    store.retrieve("mcp", mode="bm25", top_k=1)  # populates the cache
    assert store._bm25_index is not None
    store.add("a brand new note about caching")
    assert store._bm25_index is None
    assert store._bm25_tokens is None


def test_query_vector_lru_skips_second_embed_for_same_query():
    """Regression: a repeated query is served from the query-vector LRU, so the
    embedding endpoint is NOT called a second time for that query (the corpus
    is already cached too -> zero embed calls on the second retrieve)."""
    store = _seeded_store()
    p = _CountingProvider()
    store.retrieve("mcp", provider=p, mode="embed", top_k=2)
    # corpus now cached + query "mcp" in the LRU
    p.calls.clear()
    store.retrieve("mcp", provider=p, mode="embed", top_k=2)
    assert sum(len(c) for c in p.calls) == 0  # LRU hit -> no embed round-trip


def test_query_vector_lru_misses_for_different_query():
    """Regression: a DIFFERENT query is an LRU miss -> the query is embedded
    (corpus stays cached), so exactly one embed call of size 1 happens."""
    store = _seeded_store()
    p = _CountingProvider()
    store.retrieve("mcp", provider=p, mode="embed", top_k=2)
    p.calls.clear()
    store.retrieve("rebase", provider=p, mode="embed", top_k=2)
    assert sum(len(c) for c in p.calls) == 1
