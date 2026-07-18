"""code_search recall-mode tests (offline/deterministic — MockProvider + fakes).

The default is BM25-only so an interactive turn NEVER triggers a mid-session
embedding-model GPU swap (on a single-GPU / max-1-model server one embed call
evicts the chat model -> full weight reload + cold re-prefill -> tok/s cliff).
Semantic recall stays available as an opt-in via /codeembed.

Covers:
  - Config default is "bm25"; load_config accepts each valid mode and rejects
    junk (keeps the safe default); round-trips via save_config.
  - make_code_search_tool TOOL path: a "bm25" config with a provider NEVER calls
    embeddings; an "auto" config DOES attempt the embed path; config=None
    (sub-agents/tests) defaults to "bm25" -> no embed call.
  - /codeembed slash command mirrors /rerank's dispatch shape (no-arg status,
    on -> "auto", off -> "bm25", garbage -> Usage).
"""

from __future__ import annotations

import json

import pytest

import llmcode.code_index as ci
from llmcode.code_index import make_code_search_tool
from llmcode.config import CODE_SEARCH_RECALL_MODES, Config, load_config, save_config
from llmcode.providers import MockProvider


# --------------------------------------------------------------------------- #
# helpers / fakes
# --------------------------------------------------------------------------- #

def _write(root, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _pin_home(tmp_path, monkeypatch):
    """Point Path.home() at tmp so the tool's index save lands under tmp, not the
    real ~/.llmcode (mirrors test_code_index._pin_home)."""
    import llmcode.session as s
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(s.Path, "home", classmethod(lambda cls: home))


class _CountingProvider(MockProvider):
    """Records embeddings() calls so a test can prove whether embeddings ran."""

    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def embeddings(self, texts):
        self.calls.append(list(texts))
        return super().embeddings(texts)


# --------------------------------------------------------------------------- #
# config: default + load_config validation (mirrors recall_mode's guard)
# --------------------------------------------------------------------------- #

def test_code_search_recall_default_is_bm25():
    # BM25-only default = no embedding-model GPU swap on an interactive turn.
    assert Config().code_search_recall == "bm25"


def test_code_search_recall_accepts_each_valid_mode(tmp_path):
    p = tmp_path / "config.json"
    for mode in CODE_SEARCH_RECALL_MODES:
        p.write_text(json.dumps({"code_search_recall": mode}), encoding="utf-8")
        assert load_config(path=p).code_search_recall == mode


def test_code_search_recall_rejects_junk_keeps_default(tmp_path):
    p = tmp_path / "config.json"
    # Unknown string / wrong case / wrong type all keep the safe "bm25" default.
    for bad in ["semantic", "AUTO", "", 1, True, None, ["bm25"]]:
        p.write_text(json.dumps({"code_search_recall": bad}), encoding="utf-8")
        assert load_config(path=p).code_search_recall == "bm25"


def test_code_search_recall_round_trips(tmp_path):
    p = tmp_path / "config.json"
    save_config(Config(code_search_recall="auto"), path=p)
    assert load_config(path=p).code_search_recall == "auto"


# --------------------------------------------------------------------------- #
# make_code_search_tool: recall mode is read LIVE and gates the embed path
# --------------------------------------------------------------------------- #

def test_tool_bm25_config_never_calls_embeddings(tmp_path, monkeypatch):
    """The whole point: bm25 config + a provider -> lexical only, no model swap."""
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "math.py", "def multiply(x, y):\n    return x * y\n")
    ci._INDEX_CACHE.clear()
    p = _CountingProvider()
    tool = make_code_search_tool(
        provider=p, workspace=str(ws), config=Config(code_search_recall="bm25")
    )
    out = tool.fn({"query": "multiply"})
    assert out["ok"] is True
    assert "math.py:1-2" in out["result"]  # BM25 still answers
    assert p.calls == []                   # no embedding-model GPU swap


def test_tool_auto_config_attempts_embed_path(tmp_path, monkeypatch):
    """Opt-in: auto config + a provider re-enables the semantic embed path."""
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "math.py", "def multiply(x, y):\n    return x * y\n")
    ci._INDEX_CACHE.clear()
    p = _CountingProvider()
    tool = make_code_search_tool(
        provider=p, workspace=str(ws), config=Config(code_search_recall="auto")
    )
    out = tool.fn({"query": "multiply"})
    assert out["ok"] is True
    assert p.calls  # embeddings WERE attempted


def test_tool_config_none_defaults_to_bm25(tmp_path, monkeypatch):
    """Sub-agents/tests build the tool with config=None -> mode defaults to bm25,
    so a provider is NEVER embedded (no chat-model-evicting GPU swap)."""
    _pin_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj"
    _write(ws, "math.py", "def multiply(x, y):\n    return x * y\n")
    ci._INDEX_CACHE.clear()
    p = _CountingProvider()
    tool = make_code_search_tool(provider=p, workspace=str(ws))  # config=None
    out = tool.fn({"query": "multiply"})
    assert out["ok"] is True
    assert "math.py:1-2" in out["result"]
    assert p.calls == []


# --------------------------------------------------------------------------- #
# /codeembed slash command (mirrors /rerank's dispatch shape)
# --------------------------------------------------------------------------- #

@pytest.fixture
def repl(monkeypatch):
    import llmcode.repl as r
    # /codeembed persists config; stub the disk write so tests never clobber the
    # real ~/.llmcode/config.json (there is no HOME pinning in conftest).
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    cfg = Config(
        provider="mock", private=True,
        base_url="http://127.0.0.1:1234/v1", model="m", seed=None,
    )
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_codeembed_shows_current_when_no_arg(repl, capsys):
    assert repl._dispatch_slash("/codeembed") is True
    out = capsys.readouterr().out
    assert "code_search embeddings: off" in out  # bm25 default reads as "off"


def test_codeembed_on_sets_auto(repl, capsys):
    assert repl._dispatch_slash("/codeembed on") is True
    assert repl.config.code_search_recall == "auto"
    out = capsys.readouterr().out
    assert "codeembed -> on" in out


def test_codeembed_off_sets_bm25(repl, capsys):
    repl.config.code_search_recall = "auto"
    assert repl._dispatch_slash("/codeembed off") is True
    assert repl.config.code_search_recall == "bm25"
    out = capsys.readouterr().out
    assert "codeembed -> off" in out


def test_codeembed_rejects_garbage(repl, capsys):
    assert repl._dispatch_slash("/codeembed banana") is True
    out = capsys.readouterr().out
    assert "Usage:" in out
    assert repl.config.code_search_recall == "bm25"  # unchanged
