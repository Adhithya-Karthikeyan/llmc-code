"""CLI flag surface: --auto-pilot rename (with --yes/-y aliases) and the
foundation-wave flags (--mode, --output-format, --no-checkpoints,
--git-autocommit) that carry values onto the session config.

The parse-layer tests use ``_parse_args`` directly. The wiring tests drive
``main`` with the provider/session side effects monkeypatched out, capturing the
resolved config via a stubbed ``save_config`` (triggered by ``--save``), so an
un-flagged run is proven byte-identical to today's defaults.
"""

from __future__ import annotations

import json

from llmcli import __main__ as cli
from llmcli.config import Config


# ----- --auto-pilot rename + back-compat aliases --------------------------

def test_auto_pilot_sets_yes_dest():
    assert cli._parse_args(["--auto-pilot"]).yes is True


def test_yes_and_y_aliases_still_set_same_dest():
    # Back-compat: the old spellings map onto the SAME dest ("yes").
    assert cli._parse_args(["--yes"]).yes is True
    assert cli._parse_args(["-y"]).yes is True


def test_auto_pilot_defaults_off():
    assert cli._parse_args([]).yes is False


# ----- parse layer for the foundation flags -------------------------------

def test_mode_and_output_format_parse():
    args = cli._parse_args(["--mode", "plan", "--output-format", "json"])
    assert args.mode == "plan"
    assert args.output_format == "json"


def test_checkpoints_and_git_autocommit_parse():
    # Absent => None (so the config default is left untouched).
    a = cli._parse_args([])
    assert a.checkpoints is None
    assert a.git_autocommit is None
    # Present => the store_false / store_true values.
    b = cli._parse_args(["--no-checkpoints", "--git-autocommit"])
    assert b.checkpoints is False
    assert b.git_autocommit is True


# ----- config wiring (main applies overrides onto the session config) -----

def _run_capturing_config(monkeypatch, argv):
    """Drive main() with side effects stubbed; return the config main resolved."""
    captured = {}

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: Config())
    monkeypatch.setattr(cli, "save_config", lambda cfg, *a, **k: captured.setdefault("cfg", cfg))
    monkeypatch.setattr(cli, "set_private", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_provider", lambda *a, **k: object())
    monkeypatch.setattr(cli, "run_once", lambda *a, **k: "")
    # --save routes the resolved config through the stubbed save_config; -p mock
    # keeps it a one-shot so no REPL/stdin is touched.
    rc = cli.main(["-p", "hi", "--provider", "mock", "--save", *argv])
    assert rc == 0
    return captured["cfg"]


def test_flags_map_onto_config(monkeypatch):
    cfg = _run_capturing_config(
        monkeypatch,
        ["--mode", "read-only", "--output-format", "json",
         "--no-checkpoints", "--git-autocommit"],
    )
    assert cfg.permission_mode == "read-only"
    assert cfg.output_format == "json"
    assert cfg.checkpoints_enabled is False
    assert cfg.git_autocommit is True


def test_unflagged_run_leaves_defaults_identical(monkeypatch):
    cfg = _run_capturing_config(monkeypatch, [])
    d = Config()
    assert cfg.permission_mode == d.permission_mode == "default"
    assert cfg.output_format == d.output_format == "text"
    assert cfg.checkpoints_enabled == d.checkpoints_enabled is True
    assert cfg.git_autocommit == d.git_autocommit is False
    assert cfg.diff_preview == d.diff_preview is True
    assert cfg.hooks_enabled == d.hooks_enabled is False
    assert cfg.rules_file_enabled == d.rules_file_enabled is True


# ----- JSON one-shot: main() must NOT append a stray trailing blank line ---

def _stub_main_side_effects(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: Config())
    monkeypatch.setattr(cli, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(cli, "set_private", lambda *a, **k: None)
    monkeypatch.setattr(cli, "build_provider", lambda *a, **k: object())


def test_main_json_mode_single_object_no_blank_line(monkeypatch, capsys):
    def fake_run_once(provider, config, prompt, auto_confirm, resume=False):
        # run_once emits the one object itself; returns the answer WITHOUT a
        # trailing newline (which in text mode triggers the tidy-up print()).
        print(json.dumps({"ok": True, "answer": "hi", "model": "m"}))
        return "hi"

    _stub_main_side_effects(monkeypatch)
    monkeypatch.setattr(cli, "run_once", fake_run_once)
    rc = cli.main(["-p", "hi", "--provider", "mock", "--output-format", "json"])
    assert rc == 0
    out = capsys.readouterr().out
    # Stdout parses as exactly ONE json object and has no extra blank line.
    assert json.loads(out.strip())["ok"] is True
    assert out.count("\n") == 1
    assert out.endswith("}\n")


def test_main_text_mode_still_appends_trailing_newline(monkeypatch, capsys):
    def fake_run_once(provider, config, prompt, auto_confirm, resume=False):
        return "answer-no-newline"  # simulate streamed text with no final newline

    _stub_main_side_effects(monkeypatch)
    monkeypatch.setattr(cli, "run_once", fake_run_once)
    rc = cli.main(["-p", "hi", "--provider", "mock"])
    assert rc == 0
    # Text mode is unchanged: the tidy-up print() still emits a trailing newline.
    assert capsys.readouterr().out == "\n"
