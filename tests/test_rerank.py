"""Gated LLM-judge reranker tests (offline/deterministic — MockProvider + fakes).

Covers:
  - rerank() module: parse tolerance (comma / JSON / garbage), graceful fallback
    (no provider, raising provider, unparseable response), rerank_enabled gate.
  - MemoryStore.retrieve: default-off no-op (stream_chat NOT invoked), gating
    (BM25 real hit -> rerank NOT called), weak-signal (embed_idx non-empty ->
    rerank called once, results still returned).
  - CodeIndex.search: default-off no-op parity + weak-signal rerank fires once.
  - config load_config: rerank loads bool, rerank_candidates loads int>=1, junk
    keeps the safe defaults; round-trips via save_config.
  - /rerank slash command (on/off/status) mirrors /seed's dispatch shape.
"""

from __future__ import annotations

import json

import pytest

from llmcli.code_index import CodeIndex
from llmcli.config import Config, load_config, save_config
from llmcli.memory import MemoryStore
from llmcli.providers import MockProvider
from llmcli.rerank import rerank, rerank_enabled


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #

class _RankProvider(MockProvider):
    """A MockProvider whose ``stream_chat`` yields a FIXED ranking string and
    counts the calls, so the reranker's parse path is deterministic. Inherits
    ``embeddings`` from MockProvider so the weak-signal embed path still works
    (the reranker is gated on embed_idx being non-empty)."""

    def __init__(self, rank_text: str = ""):
        super().__init__()
        self.rank_text = rank_text
        self.stream_calls = 0

    def stream_chat(self, messages, tools=None, tool_choice=None):
        self.stream_calls += 1
        yield {"type": "text", "text": self.rank_text}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


class _BoomChatProvider(MockProvider):
    """stream_chat raises; embeddings inherited (so the embed path runs first
    and the rerank branch then blows up -> must fall back)."""

    def stream_chat(self, messages, tools=None, tool_choice=None):
        raise RuntimeError("chat endpoint down")
        yield {}  # pragma: no cover - generator marker


class _NoChatProvider:
    """Has embeddings but NO stream_chat -> rerank_enabled False, rerank falls
    back to input order without ever attempting a chat call."""

    def embeddings(self, texts):
        return MockProvider().embeddings(texts)


# --------------------------------------------------------------------------- #
# rerank() unit: parse tolerance + graceful fallback
# --------------------------------------------------------------------------- #

def test_rerank_empty_candidates_returns_empty():
    assert rerank(MockProvider(), "q", [], top_k=3) == []


def test_rerank_comma_separated_reorders():
    p = _RankProvider(rank_text="3, 1, 2")
    cands = [(10, "a"), (20, "b"), (30, "c")]
    out = rerank(p, "q", cands, top_k=2)
    # Parsed order 3,1,2 -> orig_idx 30,10,20; no unranked remain.
    assert out == [30, 10, 20]
    assert p.stream_calls == 1


def test_rerank_json_array_reorders():
    p = _RankProvider(rank_text="[3, 1, 2]")
    cands = [(10, "a"), (20, "b"), (30, "c")]
    assert rerank(p, "q", cands, top_k=3) == [30, 10, 20]


def test_rerank_partial_order_appends_unranked_in_input_order():
    p = _RankProvider(rank_text="2")  # only ranks snippet 2
    cands = [(10, "a"), (20, "b"), (30, "c"), (40, "d")]
    out = rerank(p, "q", cands, top_k=2)
    # 20 first (the judge's pick), then the rest in input order.
    assert out == [20, 10, 30, 40]


def test_rerank_garbage_falls_back_to_input_order():
    p = _RankProvider(rank_text="I cannot rank these snippets.")
    cands = [(10, "a"), (20, "b"), (30, "c")]
    assert rerank(p, "q", cands, top_k=2) == [10, 20, 30]


def test_rerank_out_of_range_ints_ignored():
    p = _RankProvider(rank_text="9, 0, 1")  # only 1 is valid (1..3)
    cands = [(10, "a"), (20, "b"), (30, "c")]
    assert rerank(p, "q", cands, top_k=3) == [10, 20, 30]


def test_rerank_no_provider_returns_input_order():
    cands = [(10, "a"), (20, "b")]
    assert rerank(None, "q", cands, top_k=2) == [10, 20]


def test_rerank_provider_without_stream_chat_returns_input_order():
    cands = [(10, "a"), (20, "b")]
    assert rerank(_NoChatProvider(), "q", cands, top_k=2) == [10, 20]


def test_rerank_raising_provider_falls_back():
    p = _BoomChatProvider()
    cands = [(10, "a"), (20, "b"), (30, "c")]
    # Must not raise; returns the input order.
    assert rerank(p, "q", cands, top_k=2) == [10, 20, 30]


def test_rerank_truncates_to_candidate_count():
    # A ranking that (somehow) repeats is still capped + deduped to len(cands).
    p = _RankProvider(rank_text="1, 1, 2, 2, 3")
    cands = [(10, "a"), (20, "b"), (30, "c")]
    out = rerank(p, "q", cands, top_k=3)
    assert out == [10, 20, 30]
    assert len(out) == len(cands)


def test_rerank_enabled_gate():
    assert rerank_enabled(MockProvider(), Config(rerank=True)) is True
    assert rerank_enabled(MockProvider(), Config(rerank=False)) is False
    assert rerank_enabled(None, Config(rerank=True)) is False  # no provider
    assert rerank_enabled(_NoChatProvider(), Config(rerank=True)) is False  # no chat


# --------------------------------------------------------------------------- #
# MemoryStore.retrieve gating
# --------------------------------------------------------------------------- #

def _weak_store() -> MemoryStore:
    """5 records with NO token overlap to the query 'zephyr quartz', so BM25
    finds no lexical hit and the auto-mode embed path fires (embed_idx non-empty
    == the weak-signal case the reranker is gated on)."""
    s = MemoryStore()
    for text in ("alpha bravo", "charlie delta", "echo foxtrot",
                 "golf hotel", "india juliet"):
        s.add(text)
    return s


def test_memory_default_off_does_not_call_stream_chat():
    """rerank=False -> the rerank branch is skipped -> stream_chat never invoked,
    even on the weak-signal path where embeddings fired."""
    p = _RankProvider(rank_text="3,1,2")
    store = _weak_store()
    records = store.retrieve("zephyr quartz", provider=p, mode="auto",
                             top_k=2, rerank=False)
    assert len(records) == 2          # results still returned
    assert p.stream_calls == 0        # rerank never ran


def test_memory_strong_bm25_hit_skips_rerank():
    """rerank=True but the query shares tokens with a record -> BM25 has a real
    hit -> auto mode does NOT embed -> embed_idx empty -> rerank NOT called."""
    p = _RankProvider(rank_text="1,2")
    store = MemoryStore()
    store.add("the mcp toggle command turns servers on or off")
    store.add("an unrelated note about kittens and yarn")
    records = store.retrieve("mcp toggle", provider=p, mode="auto",
                             top_k=2, rerank=True)
    assert records and "mcp toggle" in records[0].text
    assert p.stream_calls == 0        # embed_idx empty -> gate closed


def test_memory_weak_signal_rerank_called_once():
    """rerank=True AND weak signal (embed_idx non-empty) -> rerank fires exactly
    once and results come back without raising."""
    p = _RankProvider(rank_text="3, 1, 2, 5, 4")
    store = _weak_store()
    records = store.retrieve("zephyr quartz", provider=p, mode="auto",
                             top_k=2, rerank=True, rerank_candidates=20)
    assert len(records) == 2
    assert p.stream_calls == 1         # rerank ran exactly once
    # The judge's #1 (snippet 3 -> record "echo foxtrot") now leads.
    assert "echo foxtrot" in records[0].text


def test_memory_rerank_failure_falls_back_to_fused_order():
    """A raising stream_chat keeps the fused (embed) order — retrieval never
    raises."""
    p = _BoomChatProvider()
    store = _weak_store()
    records = store.retrieve("zephyr quartz", provider=p, mode="auto",
                             top_k=2, rerank=True)
    assert len(records) == 2           # fused order survived the rerank failure


# --------------------------------------------------------------------------- #
# CodeIndex.search gating parity
# --------------------------------------------------------------------------- #

def _seeded_index(tmp_path) -> CodeIndex:
    from llmcli.code_index import CodeIndex as _CI
    for rel, body in [
        ("toggle.py", "def mcp_toggle():\n    return True\n"),
        ("files.py", "def read_file(path, offset, limit):\n    return slice\n"),
        ("git.py", "def rebase():\n    return True\n"),
        ("utils.py", "def helper():\n    return 0\n"),
        ("io.py", "def load():\n    return None\n"),
    ]:
        p = tmp_path / rel
        p.write_text(body, encoding="utf-8")
    idx = _CI()
    idx.build(str(tmp_path))
    return idx


def test_code_index_default_off_no_provider_chat(tmp_path):
    """rerank=False -> search never calls stream_chat, results identical to the
    pre-feature path (embeddings still fire in auto+provider mode)."""
    idx = _seeded_index(tmp_path)
    p = _RankProvider(rank_text="3,1,2")
    res = idx.search("zzzzz qqqqq", provider=p, mode="auto", top_k=2, rerank=False)
    assert len(res) == 2
    assert p.stream_calls == 0


def test_code_index_weak_signal_rerank_fires(tmp_path):
    """rerank=True + a no-lexical-overlap query (auto always embeds when a
    provider is given) -> embed_idx non-empty and the pool (5 chunks) exceeds
    top_k (2) -> rerank fires once."""
    idx = _seeded_index(tmp_path)
    p = _RankProvider(rank_text="4, 2")
    res = idx.search("zzzzz qqqqq", provider=p, mode="auto", top_k=2,
                     rerank=True, rerank_candidates=20)
    assert len(res) == 2
    assert p.stream_calls == 1


def test_code_index_rerank_failure_falls_back(tmp_path):
    """A raising stream_chat keeps the fused order; search never raises."""
    idx = _seeded_index(tmp_path)
    p = _BoomChatProvider()
    res = idx.search("zzzzz qqqqq", provider=p, mode="auto", top_k=2,
                     rerank=True)
    assert len(res) == 2


# --------------------------------------------------------------------------- #
# config load_config + save_config round-trip
# --------------------------------------------------------------------------- #

def test_rerank_defaults():
    cfg = Config()
    assert cfg.rerank is False
    assert cfg.rerank_candidates == 20


def test_rerank_bool_loads(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"rerank": True}), encoding="utf-8")
    assert load_config(path=p).rerank is True
    p.write_text(json.dumps({"rerank": "yes"}), encoding="utf-8")
    assert load_config(path=p).rerank is False  # junk -> safe default kept


def test_rerank_candidates_int_loads_and_rejects_junk(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"rerank_candidates": 50}), encoding="utf-8")
    assert load_config(path=p).rerank_candidates == 50
    # bool is an int subclass -> rejected; <1 -> rejected; string -> rejected.
    for bad in [True, 0, -3, "20", 1.5]:
        p.write_text(json.dumps({"rerank_candidates": bad}), encoding="utf-8")
        assert load_config(path=p).rerank_candidates == 20


def test_rerank_round_trips(tmp_path):
    p = tmp_path / "config.json"
    save_config(Config(rerank=True, rerank_candidates=7), path=p)
    back = load_config(path=p)
    assert back.rerank is True
    assert back.rerank_candidates == 7


# --------------------------------------------------------------------------- #
# /rerank slash command
# --------------------------------------------------------------------------- #

@pytest.fixture
def repl(monkeypatch):
    import llmcli.repl as r
    cfg = Config(
        provider="mock", private=True,
        base_url="http://127.0.0.1:1234/v1", model="m", seed=None,
    )
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_rerank_shows_current_when_no_arg(repl, capsys):
    assert repl._dispatch_slash("/rerank") is True
    out = capsys.readouterr().out
    assert "rerank: off" in out


def test_rerank_on_sets_flag_and_mirrors_to_agent(repl, capsys):
    assert repl._dispatch_slash("/rerank on") is True
    assert repl.config.rerank is True
    assert repl.agent.rerank is True            # live agent updated
    out = capsys.readouterr().out
    assert "rerank -> on" in out


def test_rerank_off_clears_flag(repl, capsys):
    repl.config.rerank = True
    repl.agent.rerank = True
    assert repl._dispatch_slash("/rerank off") is True
    assert repl.config.rerank is False
    assert repl.agent.rerank is False
    out = capsys.readouterr().out
    assert "rerank -> off" in out


def test_rerank_rejects_garbage(repl, capsys):
    assert repl._dispatch_slash("/rerank banana") is True
    out = capsys.readouterr().out
    assert "Usage:" in out
    assert repl.config.rerank is False           # unchanged