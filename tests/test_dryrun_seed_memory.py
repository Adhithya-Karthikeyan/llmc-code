"""--dry-run and /seed slash command tests.

Tests:
  - CLI --dry-run flag in __main__.py (show config, never call provider)
  - REPL /seed command
  - REPL /memory command (purge/compact)
"""

from __future__ import annotations

import json

import pytest

from llmcli.__main__ import _parse_args, main
from llmcli.config import Config
from llmcli.memory import MemoryStore
from llmcli.providers import MockProvider
from llmcli.repl import Repl

# Import the repl module to reference its classes
import llmcli.repl as r


# --------------------------------------------------------------------------- #
# --dry-run CLI flag
# --------------------------------------------------------------------------- #


def test_parse_args_has_dry_run():
    args = _parse_args(["--dry-run"])
    assert args.dry_run is True

    args2 = _parse_args([])
    assert args2.dry_run is False


def test_dry_run_shows_config_and_returns_zero(tmp_workspace, capsys):
    """--dry-run parses everything but never calls the provider."""
    result = main(["--dry-run", "--provider", "mock", "-y"])
    assert result == 0
    err = capsys.readouterr().err
    assert "[dry-run]" in err
    assert "provider=mock" in err


def test_dry_run_with_seed(tmp_workspace, capsys):
    """--dry-run shows the seed in the config output."""
    result = main(["--dry-run", "--seed", "42", "--provider", "mock"])
    assert result == 0
    err = capsys.readouterr().err
    assert "seed=42" in err


def test_dry_run_without_seed_shows_none(tmp_workspace, monkeypatch, capsys):
    """--dry-run with no --seed shows the DEFAULT seed (None).

    Must be isolated from the user's real ~/.llm-cli/config.json (which may
    persist a seed): main() calls load_config() with the bound-default
    CONFIG_PATH pointing at Path.home(), so chdir alone does not isolate it.
    Patch the load_config reference in __main__ to return a fresh default
    Config so this test asserts the default, not whatever the user saved.
    """
    import llmcli.__main__ as _main
    monkeypatch.setattr(_main, "load_config", lambda *a, **k: Config())
    result = main(["--dry-run"])
    assert result == 0
    err = capsys.readouterr().err
    assert "seed=None" in err


# --------------------------------------------------------------------------- #
# /seed REPL slash command (via _dispatch_slash, matching test_repl_slash.py)
# --------------------------------------------------------------------------- #


@pytest.fixture
def repl(monkeypatch):
    """Build a REPL with mock provider, no real MCP, seed-capable config."""
    cfg = Config(
        provider="mock", private=True, base_url="http://127.0.0.1:1234/v1",
        model="m", seed=None,
    )
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_seed_shows_current_when_no_arg(repl, capsys):
    """/seed with no argument shows the current seed."""
    assert repl._dispatch_slash("/seed") is True
    out = capsys.readouterr().out
    assert "seed: None" in out


def test_seed_sets_value(repl, capsys):
    """/seed with a value sets the config seed."""
    assert repl._dispatch_slash("/seed 42") is True
    assert repl.config.seed == 42
    out = capsys.readouterr().out
    assert "seed -> 42" in out


def test_seed_rejects_negative(repl, capsys):
    """/seed with a negative value is rejected."""
    repl.config.seed = 10  # start with 10
    assert repl._dispatch_slash("/seed -1") is True
    assert repl.config.seed == 10  # unchanged
    out = capsys.readouterr().out
    assert "must be >= 0" in out


def test_seed_rejects_non_int(repl, capsys):
    """/seed with a non-integer is rejected."""
    assert repl._dispatch_slash("/seed hot") is True
    out = capsys.readouterr().out
    assert "Usage:" in out


def test_seed_zero_is_valid(repl, capsys):
    """/seed with 0 is valid."""
    assert repl._dispatch_slash("/seed 0") is True
    assert repl.config.seed == 0


# --------------------------------------------------------------------------- #
# /memory REPL slash command (via _dispatch_slash)
# --------------------------------------------------------------------------- #


def test_memory_shows_usage_when_no_arg(repl, capsys):
    """/memory with no argument shows usage."""
    assert repl._dispatch_slash("/memory") is True
    out = capsys.readouterr().out
    assert "Usage:" in out


def test_memory_purge_clears_on_disk_memory_file(tmp_workspace, capsys):
    """/memory purge clears the on-disk memory file."""
    memory_path = r.memory.store_path(str(tmp_workspace))
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps({
        "records": [{"id": "r0", "text": "test", "summary": "t", "content_hash": "abc123"}],
        "vectors": {"abc123": [0.1] * 768},
    }), encoding="utf-8")
    assert memory_path.exists()

    cfg = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)
    repl.agent.memory = None  # no live memory

    assert repl._dispatch_slash("/memory purge all") is True
    assert not memory_path.exists()


def test_memory_purge_clears_live_memory_store(tmp_workspace, capsys):
    """/memory purge clears the live MemoryStore."""
    memory_path = r.memory.store_path(str(tmp_workspace))
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps({
        "records": [{"id": "r0", "text": "a", "summary": "a", "content_hash": "abc1"}],
        "vectors": {"abc1": [0.1] * 768},
    }), encoding="utf-8")

    cfg = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    # Give the agent a live MemoryStore
    store = MemoryStore()
    store.add("doc", summary="d")
    repl.agent.memory = store

    assert repl._dispatch_slash("/memory purge all") is True
    assert len(repl.agent.memory.records) == 0
    out = capsys.readouterr().out
    assert "purged" in out


def test_memory_compact_no_arg(repl, capsys):
    """/memory compact without arg shows nothing to compact when store is small."""
    store = MemoryStore()
    store.add("short doc", summary="s")
    repl.agent.memory = store

    assert repl._dispatch_slash("/memory compact") is True
    out = capsys.readouterr().out
    assert "nothing to compact" in out


def test_memory_compact_shrinks_store(repl, capsys):
    """/memory compact with a limit shrinks the store."""
    store = MemoryStore()
    for i in range(20):
        store.add(f"doc {i}", summary=f"d{i}")
    repl.agent.memory = store

    assert repl._dispatch_slash("/memory compact 5") is True
    assert len(store.records) == 5
    out = capsys.readouterr().out
    assert "compacted" in out
    assert "15" in out


def test_memory_compact_no_memory_store(repl, capsys):
    """/memory compact with no memory store prints a message."""
    repl.agent.memory = None

    assert repl._dispatch_slash("/memory compact") is True
    out = capsys.readouterr().out
    assert "no memory store" in out


def test_memory_unknown_action(repl, capsys):
    """/memory with an unknown action shows error."""
    assert repl._dispatch_slash("/memory delete all") is True
    out = capsys.readouterr().out
    assert "Unknown memory action" in out
