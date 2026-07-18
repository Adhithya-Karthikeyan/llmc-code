"""Code-RAG (code_search) tests. All offline/deterministic (MockProvider).

Covers chunking (line windows + overlap, 1-based line numbers); incremental
refresh (unchanged not re-chunked, changed re-chunked, deleted dropped); BM25
ranking; the embeddings path via MockProvider (auto-escalation, dimension
self-heal, silent fallback); save/load round-trip + corrupt tolerance; the
make_code_search_tool Tool (well-formed {ok, result} with path:lines, byte cap,
provider=None BM25-only); and the chunk-cap warning. Orchestration threading of
code_search into a spawned sub-agent lives in test_orchestration.py.
"""

from __future__ import annotations

import llmcode.code_index as ci
from llmcode.code_index import (
    CodeChunk,
    CodeIndex,
    _chunk_file,
    _chunk_source,
    index_path,
    make_code_search_tool,
)
from llmcode.config import Config
from llmcode.providers import MockProvider
from llmcode.session import session_id


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _write(root, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _numbered(n: int) -> str:
    """A file body of n lines: 'line1\\nline2\\n...'."""
    return "\n".join(f"line{i}" for i in range(1, n + 1)) + "\n"


# --------------------------------------------------------------------------- #
# chunking: line windows + overlap + 1-based line numbers
# --------------------------------------------------------------------------- #

def test_chunk_file_windows_and_overlap_line_numbers():
    chunks = _chunk_file("a.py", _numbered(130))
    # 130 lines, 60-line windows, 15-line overlap -> step 45 -> starts 0,45,90.
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (46, 105), (91, 130)]
    # Windows actually overlap by 15 lines (46..60 appear in both 1-60 and 46-105).
    assert "line46" in chunks[0].text and "line46" in chunks[1].text
    assert "line60" in chunks[0].text and "line60" in chunks[1].text
    assert chunks[0].path == "a.py"


def test_chunk_file_single_window_when_small():
    chunks = _chunk_file("s.py", _numbered(10))
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 10)


def test_chunk_file_empty_is_no_chunks():
    assert _chunk_file("e.py", "") == []


# --------------------------------------------------------------------------- #
# AST/symbol-aware chunking for Python (.py/.pyi) via _chunk_source
# --------------------------------------------------------------------------- #

def test_chunk_python_one_chunk_per_top_level_symbol():
    src = (
        "import os\n"              # 1  module-level leading code
        "\n"                       # 2
        "def first():\n"          # 3
        "    return 1\n"          # 4
        "\n"                       # 5
        "def second():\n"         # 6
        "    return 2\n"          # 7
        "\n"                       # 8
        "class Widget:\n"         # 9
        "    def method(self):\n" # 10
        "        return 3\n"      # 11
    )
    chunks = _chunk_source("m.py", src)
    ranges = [(c.start_line, c.end_line) for c in chunks]
    # One chunk per top-level symbol on its REAL line boundaries (not 60-line
    # windows), plus the module-level import block as its own chunk.
    assert ranges == [(1, 2), (3, 4), (6, 7), (9, 11)]
    first = next(c for c in chunks if c.start_line == 3)
    assert "def first" in first.text and "def second" not in first.text  # whole + isolated
    cls = next(c for c in chunks if c.start_line == 9)
    assert "class Widget" in cls.text and "def method" in cls.text  # class kept whole


def test_chunk_python_oversized_symbol_falls_back_to_windows():
    # A single function whose body is far longer than MAX_SYMBOL_LINES (120).
    body = "\n".join(f"    x{i} = {i}" for i in range(200))  # lines 2..201
    src = "def huge():\n" + body + "\n"
    chunks = _chunk_source("big.py", src)
    assert len(chunks) > 1                              # split into windows, not one chunk
    assert all("# huge" in c.text for c in chunks)      # qualified-name header kept per window
    assert chunks[0].start_line == 1                    # first window starts at the def
    assert max(c.end_line for c in chunks) == 201       # windows cover the whole symbol


def test_chunk_python_syntax_error_falls_back_to_line_windows():
    # ast.parse raises SyntaxError -> identical to the plain line-window chunker,
    # and crucially never raises.
    src = "def broken(:\n    not valid python !!!\n" + _numbered(130)
    chunks = _chunk_source("bad.py", src)
    assert chunks == _chunk_file("bad.py", src)
    assert len(chunks) > 1


def test_chunk_source_non_python_uses_line_windows():
    src = _numbered(130)
    chunks = _chunk_source("notes.txt", src)
    # Non-Python keeps the fixed 60/15 line windows unchanged.
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (46, 105), (91, 130)]
    assert chunks == _chunk_file("notes.txt", src)


def test_build_indexes_files_with_relative_paths(tmp_path):
    _write(tmp_path, "pkg/foo.py", "def find_widget():\n    return 1\n")
    _write(tmp_path, "bar.py", "x = 2\n")
    idx = CodeIndex()
    idx.build(tmp_path)
    rels = {c.path for c in idx.chunks}
    assert rels == {"pkg/foo.py", "bar.py"}
    assert set(idx.file_hashes) == {"pkg/foo.py", "bar.py"}


def test_build_skips_binary_and_oversized(tmp_path, monkeypatch):
    _write(tmp_path, "ok.py", "x=1\n")  # 4 bytes -> kept
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02NUL")  # binary -> skipped
    monkeypatch.setattr(ci, "MAX_FILE_BYTES", 10)
    _write(tmp_path, "big.py", "this file is definitely larger than ten bytes\n")  # > 10 -> skipped
    idx = CodeIndex()
    idx.build(tmp_path)
    assert {c.path for c in idx.chunks} == {"ok.py"}


# --------------------------------------------------------------------------- #
# incremental refresh
# --------------------------------------------------------------------------- #

def test_refresh_unchanged_file_not_rechunked(tmp_path):
    _write(tmp_path, "a.py", _numbered(5))
    idx = CodeIndex()
    idx.build(tmp_path)
    before = idx.chunks
    before_obj = before[0]
    idx.refresh(tmp_path)
    # No file changed -> _scan returns early, chunk list + objects are identical.
    assert idx.chunks is before
    assert idx.chunks[0] is before_obj


def test_refresh_changed_file_is_rechunked(tmp_path):
    _write(tmp_path, "a.py", _numbered(5))
    idx = CodeIndex()
    idx.build(tmp_path)
    old_obj = idx.chunks[0]
    old_hash = old_obj.content_hash
    _write(tmp_path, "a.py", _numbered(9))  # size + content change
    idx.refresh(tmp_path)
    assert idx.chunks[0] is not old_obj          # re-chunked (new object)
    assert idx.chunks[0].content_hash != old_hash  # new content -> new hash
    assert idx.chunks[0].end_line == 9


def test_refresh_deleted_file_chunks_dropped(tmp_path):
    _write(tmp_path, "a.py", "a = 1\n")
    _write(tmp_path, "b.py", "b = 2\n")
    idx = CodeIndex()
    idx.build(tmp_path)
    assert {c.path for c in idx.chunks} == {"a.py", "b.py"}
    (tmp_path / "b.py").unlink()
    idx.refresh(tmp_path)
    assert {c.path for c in idx.chunks} == {"a.py"}
    assert "b.py" not in idx.file_hashes


def test_refresh_drops_orphan_vectors_of_deleted_file(tmp_path):
    _write(tmp_path, "a.py", "alpha = 1\n")
    _write(tmp_path, "b.py", "beta = 2\n")
    idx = CodeIndex()
    idx.build(tmp_path)
    # Embed everything so the vector cache is populated, then delete a file.
    idx.search("alpha beta", provider=MockProvider(), mode="embed", top_k=5)
    assert idx.vectors
    (tmp_path / "b.py").unlink()
    idx.refresh(tmp_path)
    live = {c.content_hash for c in idx.chunks}
    assert set(idx.vectors) <= live  # orphan vector of b.py dropped


# --------------------------------------------------------------------------- #
# BM25 ranking
# --------------------------------------------------------------------------- #

def test_bm25_search_returns_relevant_chunk(tmp_path):
    _write(tmp_path, "math.py", "def multiply(x, y):\n    return x * y\n")
    _write(tmp_path, "io.py", "def read_widget(path):\n    return open(path).read()\n")
    idx = CodeIndex()
    idx.build(tmp_path)
    res = idx.search("multiply", top_k=1)  # no provider -> pure BM25
    assert len(res) == 1
    assert res[0].path == "math.py"


def test_search_off_and_empty_are_safe(tmp_path):
    _write(tmp_path, "a.py", "x = 1\n")
    idx = CodeIndex()
    idx.build(tmp_path)
    assert idx.search("x", mode="off") == []
    assert idx.search("x", top_k=0) == []
    assert CodeIndex().search("anything") == []  # empty index


# --------------------------------------------------------------------------- #
# embeddings path: provider-gated auto-embed, embed-first fusion, dimension
# self-heal, silent fallback
# --------------------------------------------------------------------------- #

class _CountingProvider(MockProvider):
    """Records embeddings() calls so a test can prove when embeddings ran."""

    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def embeddings(self, texts):
        self.calls.append(list(texts))
        return super().embeddings(texts)


def _seeded_index(tmp_path) -> CodeIndex:
    _write(tmp_path, "toggle.py", "def mcp_toggle():\n    # turn servers on or off\n    return True\n")
    _write(tmp_path, "files.py", "def read_file(path, offset, limit):\n    return slice\n")
    _write(tmp_path, "git.py", "def rebase():\n    # clean linear history\n    return True\n")
    idx = CodeIndex()
    idx.build(tmp_path)
    return idx


def test_auto_embeds_even_with_strong_lexical_hit(tmp_path):
    """NEW CONTRACT (BUG 1 fix): auto + a provider with usable embeddings ALWAYS
    computes the embedding ranking — even when BM25 has a strong (nonzero) top
    hit. Calibration proved BM25 scores can't detect the paraphrase case, so the
    only safe gate is "is a provider present", not the BM25 score."""
    idx = _seeded_index(tmp_path)
    p = _CountingProvider()
    res = idx.search("mcp_toggle", provider=p, mode="auto", top_k=2)
    assert res and res[0].path == "toggle.py"  # correct hit still returned
    assert p.calls  # embeddings WERE computed despite the nonzero BM25 top hit


def test_auto_no_provider_is_bm25_only_no_error(tmp_path):
    """NEW CONTRACT (BUG 1 fix): auto with NO provider degrades silently to
    BM25-only — pure-offline use (no embed model) still works, never raises."""
    idx = _seeded_index(tmp_path)
    res = idx.search("mcp_toggle", provider=None, mode="auto", top_k=2)
    assert res and res[0].path == "toggle.py"


def test_auto_returns_embedding_results_on_no_lexical_overlap(tmp_path):
    idx = _seeded_index(tmp_path)
    p = _CountingProvider()
    res = idx.search("zzzzz qqqqq", provider=p, mode="auto", top_k=2)
    assert res            # still returns chunks (embedding ranking, no BM25 hits)
    assert p.calls        # embeddings WERE computed


def test_bm25_mode_never_calls_embeddings(tmp_path):
    """NEW CONTRACT: "bm25" mode is lexical-only and must NEVER touch the provider,
    even when one with usable embeddings is supplied."""
    idx = _seeded_index(tmp_path)
    p = _CountingProvider()
    res = idx.search("mcp_toggle", provider=p, mode="bm25", top_k=2)
    assert res and res[0].path == "toggle.py"
    assert p.calls == []  # lexical only -> no model swap


def test_fusion_is_embed_first_when_rankings_disagree(tmp_path):
    """NEW CONTRACT (BUG 2 fix): when the embedding ranking and the BM25 ranking
    DISAGREE, the embedding top result LEADS the fused output (BM25 is the
    secondary/tiebreaker)."""
    _write(tmp_path, "alpha.py", "def alpha():\n    return widget\n")
    _write(tmp_path, "beta.py", "def beta():\n    return gadget\n")
    idx = CodeIndex()
    idx.build(tmp_path)

    # Sanity: pure BM25 for "alpha" puts alpha.py first (only it matches the token).
    bm25 = idx.search("alpha", mode="bm25", top_k=2)
    assert bm25 and bm25[0].path == "alpha.py"

    class _OrderProvider(MockProvider):
        """Deterministic 2-d embeddings that rank beta.py ABOVE alpha.py for the
        query 'alpha' — the OPPOSITE of BM25 — so we can prove embed-first wins."""

        def embeddings(self, texts):
            out = []
            for t in texts:
                if t.strip() == "alpha":      # the QUERY -> point it at beta.py
                    out.append([1.0, 0.0])
                elif "beta" in t:             # beta.py chunk
                    out.append([1.0, 0.0])
                else:                          # alpha.py chunk (orthogonal)
                    out.append([0.0, 1.0])
            return out

    res = idx.search("alpha", provider=_OrderProvider(), mode="auto", top_k=2)
    # Embed-first: embedding's top (beta.py) leads, BM25's top (alpha.py) follows.
    assert [c.path for c in res] == ["beta.py", "alpha.py"]


def test_embed_mode_always_embeds_even_with_lexical_hit(tmp_path):
    idx = _seeded_index(tmp_path)
    p = _CountingProvider()
    idx.search("mcp_toggle", provider=p, mode="embed", top_k=2)
    assert p.calls  # forced even with a lexical hit


def test_embed_caches_vectors_only_once(tmp_path):
    idx = _seeded_index(tmp_path)
    p = _CountingProvider()
    idx.search("mcp_toggle", provider=p, mode="embed", top_k=3)
    first = sum(len(c) for c in p.calls)
    assert first == len(idx.chunks) + 1  # all chunks + the query
    p.calls.clear()
    idx.search("rebase", provider=p, mode="embed", top_k=3)
    assert sum(len(c) for c in p.calls) == 1  # corpus cached -> only the query


def test_embed_self_heals_dim_mismatched_cached_vector(tmp_path):
    idx = _seeded_index(tmp_path)
    idx.vectors[idx.chunks[0].content_hash] = [1.0, 2.0]  # wrong length (not 64)
    res = idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)
    assert res  # no crash, no garbage cosine
    assert len(idx.vectors[idx.chunks[0].content_hash]) == 64  # re-embedded


def test_search_silent_fallback_when_embeddings_raise(tmp_path):
    class _Boom(MockProvider):
        def embeddings(self, texts):
            raise RuntimeError("embed endpoint down")

    idx = _seeded_index(tmp_path)
    res = idx.search("mcp_toggle", provider=_Boom(), mode="auto", top_k=1)
    assert res and res[0].path == "toggle.py"  # BM25 still answered, never raised


def test_search_no_embeddings_attr_falls_back(tmp_path):
    class _NoEmbed:
        pass

    idx = _seeded_index(tmp_path)
    res = idx.search("mcp_toggle", provider=_NoEmbed(), mode="auto", top_k=1)
    assert res and res[0].path == "toggle.py"


# --------------------------------------------------------------------------- #
# save / load round-trip + corrupt tolerance
# --------------------------------------------------------------------------- #

def test_save_load_round_trip(tmp_path):
    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)  # fill vectors
    p = tmp_path / "store" / "code_index.json"
    idx.save(p)
    assert p.exists()

    loaded = CodeIndex.load(p)
    assert [c.to_dict() for c in loaded.chunks] == [c.to_dict() for c in idx.chunks]
    assert loaded.file_hashes == idx.file_hashes
    assert set(loaded.vectors) == set(idx.vectors)
    for h, v in idx.vectors.items():
        assert loaded.vectors[h] == v


def test_save_is_noop_when_not_dirty(tmp_path, monkeypatch):
    # The atomic writer now lives in the shared memory module (mkstemp+os.replace).
    import llmcode.memory as mem

    idx = _seeded_index(tmp_path)
    p = tmp_path / "ci.json"
    idx.save(p)
    assert idx._dirty is False

    calls = []
    orig = mem.tempfile.mkstemp

    def _spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(mem.tempfile, "mkstemp", _spy)
    idx.save(p)  # clean -> must NOT serialize
    assert calls == []


def test_save_never_raises_on_oserror(tmp_path, monkeypatch):
    # The atomic writer now lives in the shared memory module (mkstemp+os.replace).
    import llmcode.memory as mem

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(mem.tempfile, "mkstemp", _boom)
    _seeded_index(tmp_path).save(tmp_path / "ci.json")  # must NOT raise


def test_load_missing_returns_empty(tmp_path):
    loaded = CodeIndex.load(tmp_path / "nope.json")
    assert loaded.chunks == [] and loaded.vectors == {} and loaded.file_hashes == {}


def test_load_corrupt_returns_empty(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    loaded = CodeIndex.load(bad)
    assert loaded.chunks == [] and loaded.file_hashes == {}


def test_load_non_object_returns_empty(tmp_path):
    p = tmp_path / "arr.json"
    p.write_text('["a", "b"]', encoding="utf-8")
    assert CodeIndex.load(p).chunks == []


def test_save_writes_f32_sidecar_not_inline_json(tmp_path):
    """NEW format: vectors go to a packed-f32 `<path>.vec` sidecar; the JSON keeps
    only slim metadata (vector_format/vector_dim/vector_hashes), NOT inline
    "vectors" — chunks/file_hashes/capped stay in the JSON as before."""
    import json

    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)  # fill vectors
    p = tmp_path / "store" / "code_index.json"
    idx.save(p)

    assert (p.parent / "code_index.json.vec").exists()  # sidecar written
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "vectors" not in data  # no inline float bloat
    assert data["vector_format"] == "f32-sidecar-v1"
    assert data["vector_dim"] == 64
    assert set(data["vector_hashes"]) == set(idx.vectors)
    assert "chunks" in data and "file_hashes" in data  # non-vector state stays inline


def test_load_old_inline_format_still_works(tmp_path):
    """BACKWARD COMPAT: an OLD-format index (vectors inline, no vector_format /
    no sidecar) still loads its vectors + chunks + file_hashes correctly."""
    import json

    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "chunks": [{"path": "a.py", "start_line": 1, "end_line": 2,
                    "text": "x", "content_hash": "abc"}],
        "vectors": {"abc": [0.5, 1.5, 2.5]},
        "file_hashes": {"a.py": "sig"},
        "capped": False,
    }), encoding="utf-8")
    loaded = CodeIndex.load(p)
    assert [c.content_hash for c in loaded.chunks] == ["abc"]
    assert list(loaded.vectors["abc"]) == [0.5, 1.5, 2.5]
    assert loaded.file_hashes == {"a.py": "sig"}


def test_load_missing_sidecar_yields_empty_vectors(tmp_path):
    """SAFETY: a new-format index whose `.vec` sidecar is gone loads with
    vectors == {} (they lazily re-embed) — NO crash, chunks/file_hashes intact."""
    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)
    p = tmp_path / "ci.json"
    idx.save(p)
    (tmp_path / "ci.json.vec").unlink()  # sidecar vanishes

    loaded = CodeIndex.load(p)
    assert loaded.vectors == {}  # absent, not wrong
    assert len(loaded.chunks) == len(idx.chunks)
    assert loaded.file_hashes == idx.file_hashes


def test_load_truncated_sidecar_yields_empty_vectors(tmp_path):
    """SAFETY: a short/torn `.vec` (size != dim*4*count) is rejected wholesale ->
    vectors == {}, no exception, chunks intact."""
    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)
    p = tmp_path / "ci.json"
    idx.save(p)
    (tmp_path / "ci.json.vec").write_bytes(b"\x00\x01\x02")  # truncate

    loaded = CodeIndex.load(p)
    assert loaded.vectors == {}
    assert len(loaded.chunks) == len(idx.chunks)


def test_save_falls_back_to_inline_when_sidecar_write_fails(tmp_path, monkeypatch):
    """SAFETY: if the sidecar write raises, save() falls back to the legacy
    inline-JSON format (vectors embedded) — data is never lost — and load()
    recovers those vectors exactly."""
    import json
    import llmcode.memory as mem

    real = mem._atomic_write_bytes

    def _fail_on_vec(path, data):
        if str(path).endswith(".vec"):
            raise OSError("sidecar boom")
        return real(path, data)

    monkeypatch.setattr(mem, "_atomic_write_bytes", _fail_on_vec)
    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)
    p = tmp_path / "ci.json"
    idx.save(p)

    assert not (tmp_path / "ci.json.vec").exists()  # sidecar never landed
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "vectors" in data and "vector_format" not in data  # legacy inline
    loaded = CodeIndex.load(p)
    assert set(loaded.vectors) == set(idx.vectors)
    for h, v in idx.vectors.items():
        assert loaded.vectors[h] == v


def test_load_torn_write_sig_mismatch_yields_empty_vectors(tmp_path):
    """SAFETY (torn-write guard): a NEW-format index whose `.vec` was overwritten
    with DIFFERENT bytes of the SAME length — the crash-between-writes case (new
    sidecar + STALE meta) the byte-size guard alone would MISS — is rejected by the
    content fingerprint -> vectors == {} (lazy re-embed), chunks intact. The old
    size-only guard would have loaded positionally-mispaired vectors."""
    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)
    p = tmp_path / "ci.json"
    idx.save(p)
    vec = tmp_path / "ci.json.vec"
    raw = vec.read_bytes()
    tampered = bytes((b ^ 0xFF) for b in raw)  # same length, every byte differs
    assert len(tampered) == len(raw)
    vec.write_bytes(tampered)

    loaded = CodeIndex.load(p)
    assert loaded.vectors == {}  # mispaired bytes refused, not loaded
    assert len(loaded.chunks) == len(idx.chunks)  # chunks intact
    assert loaded.file_hashes == idx.file_hashes


def test_save_records_sig_and_byteorder_in_meta(tmp_path):
    """The f32 sidecar meta is self-describing: it carries a content fingerprint
    (vector_sig) and the native byte order (vector_byteorder)."""
    import json
    import sys

    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)
    p = tmp_path / "ci.json"
    idx.save(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(data["vector_sig"], str) and len(data["vector_sig"]) == 16
    assert data["vector_byteorder"] == sys.byteorder


def test_inline_fallback_removes_stale_sidecar(tmp_path, monkeypatch):
    """When save() falls back to inline-JSON (sidecar writer raises), a stale
    `.vec` from a prior successful save is removed so it can't linger next to the
    fresh inline meta — and vectors still recover from the inline copy."""
    import json
    import llmcode.memory as mem

    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", provider=MockProvider(), mode="embed", top_k=3)
    p = tmp_path / "ci.json"
    idx.save(p)  # first save writes a real sidecar
    assert (tmp_path / "ci.json.vec").exists()

    real = mem._atomic_write_bytes

    def _fail_on_vec(path, data):
        if str(path).endswith(".vec"):
            raise OSError("sidecar boom")
        return real(path, data)

    monkeypatch.setattr(mem, "_atomic_write_bytes", _fail_on_vec)
    idx._dirty = True  # re-arm the no-op guard so save() actually runs
    idx.save(p)

    assert not (tmp_path / "ci.json.vec").exists()  # stale sidecar cleaned up
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "vectors" in data and "vector_format" not in data  # legacy inline
    loaded = CodeIndex.load(p)
    assert set(loaded.vectors) == set(idx.vectors)


def test_chunk_from_dict_tolerant_of_missing_keys():
    c = CodeChunk.from_dict({"text": "hi"})
    assert c.path == "" and c.start_line == 0 and c.end_line == 0
    assert c.content_hash  # recomputed from text
    bad = CodeChunk.from_dict({"text": "x", "start_line": "oops", "end_line": True})
    assert bad.start_line == 0 and bad.end_line == 0


# --------------------------------------------------------------------------- #
# cap / warn behavior
# --------------------------------------------------------------------------- #

def test_cap_stops_adding_and_flags_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "MAX_CHUNKS", 2)
    _write(tmp_path, "a.py", _numbered(130))  # 3 chunks at 60/15
    idx = CodeIndex()
    idx.build(tmp_path)
    assert len(idx.chunks) == 2
    assert idx.capped is True


# --------------------------------------------------------------------------- #
# index_path: alongside the session, keyed by the same workspace id
# --------------------------------------------------------------------------- #

def test_index_path_shape(tmp_path):
    cwd = str(tmp_path)
    ip = index_path(cwd)
    assert ip.name == f"{session_id(cwd)}.code_index.json"
    assert ip.parent.name == "sessions"


# --------------------------------------------------------------------------- #
# make_code_search_tool: well-formed {ok, result}, byte cap, provider=None
# --------------------------------------------------------------------------- #

def _pin_home(tmp_path, monkeypatch):
    """Point Path.home() at tmp so the tool's index save lands under tmp, not the
    real ~/.llmcode (mirrors test_session)."""
    import llmcode.session as s
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(s.Path, "home", classmethod(lambda cls: home))


def test_tool_returns_pathlines_and_snippet(tmp_path, monkeypatch):
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "math.py", "def multiply(x, y):\n    return x * y\n")
    ci._INDEX_CACHE.clear()
    tool = make_code_search_tool(provider=MockProvider(), workspace=str(ws))
    assert tool.name == "code_search" and tool.requires_confirmation is False
    out = tool.fn({"query": "multiply"})
    assert out["ok"] is True
    assert "math.py:1-2" in out["result"]
    assert "def multiply" in out["result"]


def test_tool_works_offline_with_provider_none(tmp_path, monkeypatch):
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "g.py", "def rebase_history():\n    return 1\n")
    ci._INDEX_CACHE.clear()
    tool = make_code_search_tool(provider=None, workspace=str(ws))
    out = tool.fn({"query": "rebase_history"})
    assert out["ok"] is True
    assert "g.py:1-2" in out["result"]


def test_tool_no_match_message(tmp_path, monkeypatch):
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "g.py", "x = 1\n")
    ci._INDEX_CACHE.clear()
    tool = make_code_search_tool(provider=None, workspace=str(ws))
    out = tool.fn({"query": "nonexistentsymbol"})
    assert out["ok"] is True
    assert "no matching" in out["result"].lower()


def test_tool_rejects_empty_query(tmp_path, monkeypatch):
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "g.py", "x = 1\n")
    ci._INDEX_CACHE.clear()
    tool = make_code_search_tool(provider=None, workspace=str(ws))
    out = tool.fn({"query": "   "})
    assert out["ok"] is False


def test_tool_result_is_byte_capped(tmp_path, monkeypatch):
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    # One huge matching file so the formatted payload would exceed _MAX_OUTPUT.
    body = "\n".join(f"def fn_match_{i}(): return {i}" for i in range(20000))
    _write(ws, "big.py", body + "\n")
    ci._INDEX_CACHE.clear()
    tool = make_code_search_tool(provider=None, workspace=str(ws))
    out = tool.fn({"query": "fn_match_0", "top_k": 50})
    from llmcode.tools import _MAX_OUTPUT
    assert len(out["result"].encode("utf-8")) <= _MAX_OUTPUT + 32  # cap + marker slack


def test_tool_persists_index_across_instances(tmp_path, monkeypatch):
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "math.py", "def multiply(x, y):\n    return x * y\n")
    ci._INDEX_CACHE.clear()
    make_code_search_tool(provider=None, workspace=str(ws)).fn({"query": "multiply"})
    # The on-disk index now exists and a fresh load sees the chunks.
    from llmcode.code_index import CodeIndex as _CI
    loaded = _CI.load(index_path(str(ws)))
    assert any(c.path == "math.py" for c in loaded.chunks)


# --------------------------------------------------------------------------- #
# review-found fixes: incremental open-avoidance, vector persistence, symlink
# safety, embed sub-batching, partial-cap signatures, tool cap warning
# --------------------------------------------------------------------------- #

def test_refresh_unchanged_does_not_open_files(tmp_path, monkeypatch):
    """MEDIUM-1: a no-change refresh must NOT open/read unchanged files — the
    size + binary checks (and the read) live behind the size:mtime reuse gate,
    so an unchanged file is reached with only a stat()."""
    _write(tmp_path, "a.py", _numbered(5))
    _write(tmp_path, "b.py", _numbered(5))
    idx = CodeIndex()
    idx.build(tmp_path)  # first scan opens + sniffs each file
    binary_calls: list = []
    real_binary = ci._looks_binary
    monkeypatch.setattr(
        ci, "_looks_binary",
        lambda p: binary_calls.append(p) or real_binary(p),
    )
    read_calls: list = []
    real_read = ci.Path.read_text

    def _counting_read(self, *a, **k):
        read_calls.append(self)
        return real_read(self, *a, **k)

    monkeypatch.setattr(ci.Path, "read_text", _counting_read)
    idx.refresh(tmp_path)  # nothing changed
    assert binary_calls == []  # no binary sniff (no open) on unchanged files
    assert read_calls == []    # no content read on unchanged files


def test_symlink_escaping_workspace_not_indexed(tmp_path):
    """LOW-a: an in-workspace symlink whose realpath escapes the workspace is
    skipped, and its content never lands in the index."""
    ws = tmp_path / "proj"
    ws.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("secret_password = 'hunter2'\n", encoding="utf-8")
    (ws / "link.py").symlink_to(outside)  # escapes ws
    _write(ws, "real.py", "def real_widget():\n    return 1\n")
    idx = CodeIndex()
    idx.build(ws)
    paths = {c.path for c in idx.chunks}
    assert "real.py" in paths
    assert "link.py" not in paths                      # escaping symlink skipped
    assert "hunter2" not in "\n".join(c.text for c in idx.chunks)  # no content leak


def test_embed_sub_batches_large_corpus(tmp_path):
    """LOW-b: the first embedding of a >1-batch corpus is split into fixed
    EMBED_BATCH-sized calls (not one giant request), and the lexically-correct
    chunk is still retrieved via the fused output."""
    body = "\n".join(f"unique_token_{i} value_{i}" for i in range(6000)) + "\n"
    _write(tmp_path, "big.py", body)
    idx = CodeIndex()
    idx.build(tmp_path)
    uniq = len({c.content_hash for c in idx.chunks})
    assert uniq > ci.EMBED_BATCH  # forces more than one batch
    p = _CountingProvider()
    # top_k=2: embed-first fusion leads with the embedding rank; the BM25 hit for
    # the rare "0" token (the unique_token_0 chunk) surfaces as the secondary.
    res = idx.search("unique_token_0", provider=p, mode="embed", top_k=2)
    corpus = p.calls[1:]  # calls[0] is the single-text QUERY embed
    expected = (uniq + ci.EMBED_BATCH - 1) // ci.EMBED_BATCH
    assert expected >= 2
    assert len(corpus) == expected                       # batched into multiple calls
    assert all(len(c) <= ci.EMBED_BATCH for c in corpus)  # never exceeds batch size
    assert sum(len(c) for c in corpus) == uniq            # every chunk embedded once
    # Embed-first concatenation fills the top slots from the embedding ranking; with
    # MockProvider's content-independent vectors the order is arbitrary, so this test
    # only asserts batching + that search returns (capped) results. Fusion ORDER is
    # covered by test_fusion_is_embed_first_when_rankings_disagree.
    assert res and len(res) <= 2


def test_embed_short_batch_aborts_to_bm25(tmp_path):
    """LOW-b: a short/failed batch still degrades to BM25-only (search never raises)."""
    class _ShortBatch(MockProvider):
        def embeddings(self, texts):
            vecs = super().embeddings(texts)
            return vecs[:-1] if len(vecs) > 1 else vecs  # drop one -> length mismatch

    body = "\n".join(f"unique_token_{i} value_{i}" for i in range(6000)) + "\n"
    _write(tmp_path, "big.py", body)
    idx = CodeIndex()
    idx.build(tmp_path)
    res = idx.search("unique_token_0", provider=_ShortBatch(), mode="embed", top_k=1)
    assert res and "unique_token_0" in res[0].text  # BM25 answered, no crash


def test_query_time_vectors_persist_across_processes(tmp_path, monkeypatch):
    """MEDIUM-2: vectors embedded during a (auto-escalating) search are persisted,
    so a fresh CodeIndex.load from the same path has them cached (no re-embed)."""
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "a.py", "def alpha_widget():\n    return 1\n")
    ci._INDEX_CACHE.clear()
    # This test exercises the embed path via the TOOL, so opt in explicitly
    # (the tool now defaults to bm25-only to avoid mid-turn embedding GPU swaps).
    tool = make_code_search_tool(
        provider=MockProvider(), workspace=str(ws),
        config=Config(code_search_recall="auto"),
    )
    # No lexical overlap -> auto escalates to embeddings -> vectors cached + saved.
    out = tool.fn({"query": "zzzzz qqqqq paraphrase"})
    assert out["ok"] is True

    loaded = CodeIndex.load(index_path(str(ws)))
    assert loaded.vectors  # query-time vectors survived to disk
    live = {c.content_hash for c in loaded.chunks}
    assert set(loaded.vectors) & live
    # Fresh process: an embed-mode search re-embeds ONLY the query (corpus cached).
    p = _CountingProvider()
    loaded.search("alpha", provider=p, mode="embed", top_k=1)
    assert sum(len(c) for c in p.calls) == 1  # no corpus re-embed


def test_capped_file_sig_not_recorded_so_grows_later(tmp_path, monkeypatch):
    """LOW-c: a cap-truncated file does NOT record its signature, so a later
    MAX_CHUNKS increase re-chunks the remainder."""
    monkeypatch.setattr(ci, "MAX_CHUNKS", 2)
    _write(tmp_path, "a.py", _numbered(130))  # 3 chunks at 60/15
    idx = CodeIndex()
    idx.build(tmp_path)
    assert len(idx.chunks) == 2 and idx.capped is True
    assert "a.py" not in idx.file_hashes  # partial -> signature withheld

    monkeypatch.setattr(ci, "MAX_CHUNKS", 10)
    idx.refresh(tmp_path)
    assert len(idx.chunks) == 3 and idx.capped is False  # remainder now indexed
    assert "a.py" in idx.file_hashes


def test_tool_emits_chunk_cap_warning(tmp_path, monkeypatch):
    """LOW-d: the tool surfaces the chunk-cap warning string in its result."""
    _pin_home(tmp_path, monkeypatch)
    monkeypatch.setattr(ci, "MAX_CHUNKS", 1)
    ws = tmp_path / "proj"
    body = "\n".join(f"def fn_{i}(): return widget_{i}" for i in range(200)) + "\n"
    _write(ws, "big.py", body)
    ci._INDEX_CACHE.clear()
    tool = make_code_search_tool(provider=None, workspace=str(ws))
    out = tool.fn({"query": "widget_0"})  # matches the one indexed chunk
    assert out["ok"] is True
    assert "code index capped" in out["result"]
    assert "narrow your query" in out["result"]


# --------------------------------------------------------------------------- #
# BM25 index cache + query-vector LRU (perf: pure caching wins, no behavior change)
# --------------------------------------------------------------------------- #

def test_bm25_cached_index_matches_fresh_recompute(tmp_path):
    """Regression: the cached _BM25Index produces IDENTICAL scores to a fresh
    recompute via _bm25_over_tokenized — pure caching, no scoring change."""
    from llmcode.memory import (
        _bm25_over_tokenized,
        _build_bm25_index,
        _bm25_score_index,
        _tokenize,
    )
    idx = _seeded_index(tmp_path)
    toks = [_tokenize(c.text) for c in idx.chunks]
    bm25_idx = _build_bm25_index(toks)
    for q in ["mcp", "rebase", "zzz no overlap", "", "toggle off"]:
        fresh = _bm25_over_tokenized(q, toks)
        cached = _bm25_score_index(q, bm25_idx)
        assert fresh == cached, f"score mismatch for query {q!r}: {fresh} vs {cached}"


def test_bm25_index_invalidated_on_scan_change(tmp_path):
    """Regression: a scan that changes chunks drops the cached BM25 index so
    the next search rebuilds against the new corpus (no stale tf/df)."""
    idx = _seeded_index(tmp_path)
    idx.search("mcp_toggle", mode="bm25", top_k=1)  # populates the cache
    assert idx._bm25_index is not None
    # Change a file so the next refresh re-chunks it.
    _write(tmp_path, "toggle.py", "def mcp_toggle():\n    return 'changed'\n")
    idx.refresh(tmp_path)
    assert idx._bm25_index is None
    assert idx._bm25_tokens is None


def test_query_vector_lru_skips_second_embed_for_same_query(tmp_path):
    """Regression: a repeated query is served from the query-vector LRU, so the
    embedding endpoint is NOT called a second time for that query (corpus is
    already cached too -> zero embed calls on the second search)."""
    idx = _seeded_index(tmp_path)
    p = _CountingProvider()
    idx.search("mcp_toggle", provider=p, mode="embed", top_k=3)
    p.calls.clear()
    idx.search("mcp_toggle", provider=p, mode="embed", top_k=3)
    assert sum(len(c) for c in p.calls) == 0  # LRU hit -> no embed round-trip


def test_query_vector_lru_misses_for_different_query(tmp_path):
    """Regression: a DIFFERENT query is an LRU miss -> the query is embedded
    (corpus stays cached), so exactly one embed call of size 1 happens."""
    idx = _seeded_index(tmp_path)
    p = _CountingProvider()
    idx.search("mcp_toggle", provider=p, mode="embed", top_k=3)
    p.calls.clear()
    idx.search("rebase", provider=p, mode="embed", top_k=3)
    assert sum(len(c) for c in p.calls) == 1


# --------------------------------------------------------------------------- #
# overlap dedup / merge (same-file overlapping-or-adjacent chunks -> one block)
# + code_search-specific output budget
# --------------------------------------------------------------------------- #

def _chunk(path: str, start: int, end: int) -> CodeChunk:
    """A CodeChunk whose text is exactly source lines start..end ('lineN' bodies),
    so its declared range lines up with its text (mergeable)."""
    text = "\n".join(f"line{i}" for i in range(start, end + 1))
    return CodeChunk(path=path, start_line=start, end_line=end, text=text, content_hash="h")


def test_overlapping_same_file_chunks_merge_no_duplicate_lines(tmp_path):
    """(a) Two overlapping same-file chunks merge into ONE block with the union
    line range and each source line printed exactly once (no repeated overlap)."""
    hits = _chunk_file("a.py", _numbered(105))
    # Real 60/15 windows overlap by 15 lines: (1,60) and (46,105).
    assert [(c.start_line, c.end_line) for c in hits] == [(1, 60), (46, 105)]
    out = ci._format_hits(hits, capped=False)
    assert "a.py:1-105" in out                                  # union range, one block
    assert "a.py:1-60" not in out and "a.py:46-105" not in out  # constituents gone
    for ln in range(1, 106):
        assert out.count(f"line{ln}\n") == 1, f"line{ln} duplicated/missing"


def test_non_overlapping_same_file_chunks_stay_separate():
    """(b) Two same-file chunks that neither overlap nor touch stay as two blocks."""
    hits = [_chunk("a.py", 1, 10), _chunk("a.py", 50, 60)]  # gap 11..49
    blocks = ci._merge_same_file_hits(hits)
    assert [(b[1], b[2], b[3]) for b in blocks] == [("a.py", 1, 10), ("a.py", 50, 60)]
    out = ci._format_hits(hits, capped=False)
    assert "a.py:1-10" in out and "a.py:50-60" in out


def test_chunks_from_different_files_never_merge():
    """(c) Identical (overlapping) line ranges in DIFFERENT files never merge."""
    hits = [_chunk("a.py", 1, 60), _chunk("b.py", 1, 60)]
    blocks = ci._merge_same_file_hits(hits)
    assert len(blocks) == 2
    assert {(b[1], b[2], b[3]) for b in blocks} == {("a.py", 1, 60), ("b.py", 1, 60)}


def test_merge_keeps_rank_of_highest_ranked_constituent():
    """A merged block takes the position/score of its highest-ranked (earliest)
    constituent; different-file hits keep their ranked order."""
    hits = [
        _chunk("a.py", 46, 105),  # rank 0 -> merges with rank 2 below
        _chunk("z.py", 1, 5),     # rank 1
        _chunk("a.py", 1, 60),    # rank 2 -> overlaps rank 0
    ]
    blocks = ci._merge_same_file_hits(hits)
    # a.py merged block (union 1-105) inherits rank 0, so it leads z.py.
    assert [(b[1], b[2], b[3]) for b in blocks] == [("a.py", 1, 105), ("z.py", 1, 5)]


def test_tool_result_capped_to_code_search_budget(tmp_path, monkeypatch):
    """(d) The tool caps its output to the code_search-specific budget (smaller than
    _MAX_OUTPUT), truncating a very large result set gracefully below it while
    staying under the hard _MAX_OUTPUT ceiling."""
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    pad = "x" * 60
    body = "\n".join(f"def fn_match_{i}(): return {i}  # {pad}" for i in range(2000))
    _write(ws, "big.py", body + "\n")
    ci._INDEX_CACHE.clear()
    tool = make_code_search_tool(provider=None, workspace=str(ws))
    out = tool.fn({"query": "fn_match_0", "top_k": 2000})
    from llmcode.tools import _MAX_OUTPUT
    size = len(out["result"].encode("utf-8"))
    assert ci.CODE_SEARCH_MAX_OUTPUT < _MAX_OUTPUT             # budget is the tighter one
    assert size <= ci.CODE_SEARCH_MAX_OUTPUT + 32             # truncated below the budget
    assert size < _MAX_OUTPUT                                  # and under the hard ceiling
    assert "...[truncated]" in out["result"]                   # graceful, marked omission
