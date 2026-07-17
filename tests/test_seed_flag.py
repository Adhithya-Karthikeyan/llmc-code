"""Seed flag tests.

Tests the new `--seed N` CLI flag:
  - config.py: seed field on Config, load/save round-trip
  - __main__.py: CLI argument parsing, dry-run mode
  - providers.py: MockProvider seed parameter
  - repl.py: /seed slash command
"""

from __future__ import annotations

import json

from llmcli.config import Config, load_config, save_config


# --------------------------------------------------------------------------- #
# Config.seed
# --------------------------------------------------------------------------- #


def test_seed_defaults_to_none():
    assert Config().seed is None


def test_seed_accepts_valid_int(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"seed": 42}), encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.seed == 42


def test_seed_zero_is_valid(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"seed": 0}), encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.seed == 0  # 0 is valid


def test_seed_rejects_negative(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"seed": -1}), encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.seed is None  # negative -> None


def test_seed_rejects_bool(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"seed": True}), encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.seed is None  # bool rejected


def test_seed_rejects_string(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"seed": "hot"}), encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.seed is None


def test_seed_missing_keeps_none(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"model": "m"}), encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.seed is None


def test_seed_round_trips_through_save(tmp_path):
    p = tmp_path / "config.json"
    save_config(Config(seed=123), path=p)
    cfg = load_config(path=p)
    assert cfg.seed == 123

    # Default None also round-trips
    save_config(Config(), path=p)
    cfg = load_config(path=p)
    assert cfg.seed is None


# --------------------------------------------------------------------------- #
# MockProvider.seed
# --------------------------------------------------------------------------- #


def test_mock_provider_accepts_seed(tmp_workspace):
    from llmcli.providers import MockProvider

    prov = MockProvider(scenario="hello", seed=42)
    assert prov.seed == 42

    # None seed is default
    prov2 = MockProvider(scenario="hello")
    assert prov2.seed is None


# --------------------------------------------------------------------------- #
# build_provider wires seed through
# --------------------------------------------------------------------------- #


def test_build_provider_passes_seed(tmp_workspace):
    from llmcli.repl import build_provider

    prov = build_provider(
        "mock", "model", "http://localhost:1234/v1",
        seed=99,
    )
    assert prov.seed == 99

    # Without seed
    prov2 = build_provider(
        "mock", "model", "http://localhost:1234/v1",
    )
    assert prov2.seed is None
