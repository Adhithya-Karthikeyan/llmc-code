"""load_config robustness: corrupt JSON warns and falls back to defaults."""

from __future__ import annotations

from llmcli.config import Config, load_config


def test_corrupt_config_warns_and_uses_defaults(tmp_path, capsys):
    bad = tmp_path / "config.json"
    bad.write_text("{ not valid json ", encoding="utf-8")
    cfg = load_config(path=bad)
    assert isinstance(cfg, Config)
    assert cfg.provider == "local"  # defaults
    err = capsys.readouterr().err
    assert "invalid JSON" in err


def test_mcp_enabled_defaults_on_and_round_trips(tmp_path):
    import json
    from llmcli.config import save_config

    assert Config().mcp_enabled is True  # back-compat: on by default
    p = tmp_path / "config.json"
    # an explicit false persists + loads back as false.
    save_config(Config(mcp_enabled=False), path=p)
    assert load_config(path=p).mcp_enabled is False
    # a non-bool value is ignored -> safe default kept.
    p.write_text(json.dumps({"mcp_enabled": "nope"}), encoding="utf-8")
    assert load_config(path=p).mcp_enabled is True


def test_missing_config_returns_defaults_silently(tmp_path, capsys):
    cfg = load_config(path=tmp_path / "nope.json")
    assert cfg.provider == "local"
    assert capsys.readouterr().err == ""


def test_valid_config_loads_fields(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"provider": "mock", "model": "m", "max_iterations": 5, "effort": "high"}',
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.provider == "mock"
    assert cfg.model == "m"
    assert cfg.max_iterations == 5
    assert cfg.effort == "high"


def test_theme_defaults_to_clean():
    # Fresh installs default to the minimal dark "clean" look.
    assert Config().theme == "clean"


def test_valid_theme_loads(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"theme": "ansi"}', encoding="utf-8")
    assert load_config(path=p).theme == "ansi"


def test_bad_theme_falls_back_to_default(tmp_path):
    # An unknown / typo'd theme must not be honored; keep the safe "clean" default.
    p = tmp_path / "config.json"
    p.write_text('{"theme": "neon"}', encoding="utf-8")
    assert load_config(path=p).theme == "clean"
    # Wrong type is also ignored.
    p.write_text('{"theme": 123}', encoding="utf-8")
    assert load_config(path=p).theme == "clean"


def test_theme_round_trips_through_save(tmp_path):
    from llmcli.config import save_config

    p = tmp_path / "config.json"
    save_config(Config(theme="ansi"), path=p)
    assert load_config(path=p).theme == "ansi"


def test_max_output_tokens_defaults_to_none():
    assert Config().max_output_tokens is None


def test_max_output_tokens_loads_valid_int(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"max_output_tokens": 4096}', encoding="utf-8")
    assert load_config(path=p).max_output_tokens == 4096


def test_max_output_tokens_rejects_bool_and_non_positive(tmp_path):
    # bool is an int subclass in Python — it must be rejected, not stored as 1.
    p = tmp_path / "config.json"
    p.write_text('{"max_output_tokens": true}', encoding="utf-8")
    assert load_config(path=p).max_output_tokens is None
    # 0 and -1 are the "unbounded" sentinels: they stay None (never sent).
    p.write_text('{"max_output_tokens": 0}', encoding="utf-8")
    assert load_config(path=p).max_output_tokens is None
    p.write_text('{"max_output_tokens": -1}', encoding="utf-8")
    assert load_config(path=p).max_output_tokens is None
    # Wrong type is ignored too.
    p.write_text('{"max_output_tokens": "lots"}', encoding="utf-8")
    assert load_config(path=p).max_output_tokens is None


def test_max_output_tokens_missing_stays_none(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"model": "m"}', encoding="utf-8")
    assert load_config(path=p).max_output_tokens is None


def test_max_output_tokens_round_trips_through_save(tmp_path):
    from llmcli.config import save_config

    p = tmp_path / "config.json"
    save_config(Config(max_output_tokens=2048), path=p)
    assert load_config(path=p).max_output_tokens == 2048
    # The default None round-trips as None (no cap persisted).
    save_config(Config(), path=p)
    assert load_config(path=p).max_output_tokens is None


# ----- conversation-memory fields -----------------------------------------

def test_memory_fields_defaults():
    c = Config()
    assert c.embed_model == "text-embedding-nomic-embed-text-v1.5"
    assert c.recall_mode == "auto"
    assert c.memory_enabled is True
    assert c.memory_top_k == 3


def test_memory_fields_load_valid(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"embed_model": "custom-embed", "recall_mode": "bm25", '
        '"memory_enabled": false, "memory_top_k": 7}',
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.embed_model == "custom-embed"
    assert cfg.recall_mode == "bm25"
    assert cfg.memory_enabled is False
    assert cfg.memory_top_k == 7


def test_recall_mode_rejects_unknown_value(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"recall_mode": "telepathy"}', encoding="utf-8")
    assert load_config(path=p).recall_mode == "auto"  # unknown kept at default
    p.write_text('{"recall_mode": 5}', encoding="utf-8")
    assert load_config(path=p).recall_mode == "auto"  # wrong type ignored


def test_memory_top_k_rejects_bool_and_non_positive(tmp_path):
    p = tmp_path / "config.json"
    # bool is an int subclass — must be rejected, not stored as 1.
    p.write_text('{"memory_top_k": true}', encoding="utf-8")
    assert load_config(path=p).memory_top_k == 3
    p.write_text('{"memory_top_k": 0}', encoding="utf-8")
    assert load_config(path=p).memory_top_k == 3
    p.write_text('{"memory_top_k": -2}', encoding="utf-8")
    assert load_config(path=p).memory_top_k == 3
    p.write_text('{"memory_top_k": "lots"}', encoding="utf-8")
    assert load_config(path=p).memory_top_k == 3


def test_empty_embed_model_ignored(tmp_path):
    # An empty string must not blank out the default model.
    p = tmp_path / "config.json"
    p.write_text('{"embed_model": ""}', encoding="utf-8")
    assert load_config(path=p).embed_model == "text-embedding-nomic-embed-text-v1.5"


def test_memory_fields_round_trip_through_save(tmp_path):
    from llmcli.config import save_config

    p = tmp_path / "config.json"
    save_config(
        Config(embed_model="e", recall_mode="off", memory_enabled=False, memory_top_k=5),
        path=p,
    )
    cfg = load_config(path=p)
    assert cfg.embed_model == "e"
    assert cfg.recall_mode == "off"
    assert cfg.memory_enabled is False
    assert cfg.memory_top_k == 5


# ----- Feature 1: temperature ---------------------------------------------

def test_temperature_defaults_to_low():
    # Feature 1: a LOW default temperature for deterministic code/tool turns.
    assert Config().temperature == 0.2


def test_temperature_loads_valid_float_and_int(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"temperature": 0.7}', encoding="utf-8")
    assert load_config(path=p).temperature == 0.7
    # An int is accepted and coerced to float.
    p.write_text('{"temperature": 1}', encoding="utf-8")
    assert load_config(path=p).temperature == 1.0


def test_temperature_rejects_out_of_range_and_bool(tmp_path):
    # Out of the 0.0-2.0 range keeps the safe default.
    p = tmp_path / "config.json"
    p.write_text('{"temperature": 2.5}', encoding="utf-8")
    assert load_config(path=p).temperature == 0.2
    p.write_text('{"temperature": -0.1}', encoding="utf-8")
    assert load_config(path=p).temperature == 0.2
    # bool is an int subclass — must NOT be read as 0.0/1.0.
    p.write_text('{"temperature": true}', encoding="utf-8")
    assert load_config(path=p).temperature == 0.2
    # Wrong type ignored.
    p.write_text('{"temperature": "hot"}', encoding="utf-8")
    assert load_config(path=p).temperature == 0.2
    # The boundary values ARE valid.
    p.write_text('{"temperature": 0.0}', encoding="utf-8")
    assert load_config(path=p).temperature == 0.0
    p.write_text('{"temperature": 2.0}', encoding="utf-8")
    assert load_config(path=p).temperature == 2.0


def test_temperature_round_trips_through_save(tmp_path):
    from llmcli.config import save_config

    p = tmp_path / "config.json"
    save_config(Config(temperature=0.5), path=p)
    assert load_config(path=p).temperature == 0.5


# ----- Features 2/3/4: quality flags --------------------------------------

def test_quality_flag_defaults():
    c = Config()
    assert c.constrained_retry is True   # Feature 2
    assert c.verify_cmd == ""            # Feature 3 (disabled = opt-in)
    assert c.review_writes is False      # Feature 4 (now opt-in / default OFF)


def test_quality_flags_load_valid(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"constrained_retry": false, "verify_cmd": "python -m pytest -q", '
        '"review_writes": false}',
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.constrained_retry is False
    assert cfg.verify_cmd == "python -m pytest -q"
    assert cfg.review_writes is False


def test_quality_flags_reject_wrong_types(tmp_path):
    p = tmp_path / "config.json"
    # non-bool constrained_retry / review_writes ignored -> defaults kept.
    p.write_text('{"constrained_retry": "yes", "review_writes": 1}', encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.constrained_retry is True
    assert cfg.review_writes is False
    # non-str verify_cmd ignored -> default empty.
    p.write_text('{"verify_cmd": 123}', encoding="utf-8")
    assert load_config(path=p).verify_cmd == ""


def test_quality_flags_round_trip_through_save(tmp_path):
    from llmcli.config import save_config

    p = tmp_path / "config.json"
    save_config(
        Config(constrained_retry=False, verify_cmd="make test", review_writes=False),
        path=p,
    )
    cfg = load_config(path=p)
    assert cfg.constrained_retry is False
    assert cfg.verify_cmd == "make test"
    assert cfg.review_writes is False


# ----- Foundation wave: permission/output/feature-toggle fields -----------

def test_foundation_field_defaults_preserve_current_behavior():
    # Un-flagged defaults must behave exactly as before this wave.
    c = Config()
    assert c.permission_mode == "default"
    assert c.git_autocommit is False
    assert c.checkpoints_enabled is True
    assert c.hooks_enabled is False
    assert c.rules_file_enabled is True
    assert c.output_format == "text"
    assert c.diff_preview is True


def test_permission_mode_loads_valid(tmp_path):
    from llmcli.config import PERMISSION_MODES

    assert PERMISSION_MODES == ("default", "auto-edit", "read-only", "plan", "full-auto")
    p = tmp_path / "config.json"
    for mode in PERMISSION_MODES:
        p.write_text(f'{{"permission_mode": "{mode}"}}', encoding="utf-8")
        assert load_config(path=p).permission_mode == mode


def test_permission_mode_rejects_unknown_and_wrong_type(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"permission_mode": "yolo"}', encoding="utf-8")
    assert load_config(path=p).permission_mode == "default"
    p.write_text('{"permission_mode": 7}', encoding="utf-8")
    assert load_config(path=p).permission_mode == "default"


def test_output_format_loads_valid_and_rejects_bad(tmp_path):
    from llmcli.config import OUTPUT_FORMATS

    assert OUTPUT_FORMATS == ("text", "json")
    p = tmp_path / "config.json"
    p.write_text('{"output_format": "json"}', encoding="utf-8")
    assert load_config(path=p).output_format == "json"
    # Unknown / wrong type keeps the safe "text" default.
    p.write_text('{"output_format": "xml"}', encoding="utf-8")
    assert load_config(path=p).output_format == "text"
    p.write_text('{"output_format": 1}', encoding="utf-8")
    assert load_config(path=p).output_format == "text"


def test_foundation_bool_fields_reject_wrong_types(tmp_path):
    # Non-bool values are ignored -> safe defaults kept.
    p = tmp_path / "config.json"
    p.write_text(
        '{"git_autocommit": "yes", "checkpoints_enabled": 0, '
        '"hooks_enabled": 1, "rules_file_enabled": "no", "diff_preview": "x"}',
        encoding="utf-8",
    )
    c = load_config(path=p)
    assert c.git_autocommit is False
    assert c.checkpoints_enabled is True
    assert c.hooks_enabled is False
    assert c.rules_file_enabled is True
    assert c.diff_preview is True


def test_foundation_bool_fields_load_valid(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"git_autocommit": true, "checkpoints_enabled": false, '
        '"hooks_enabled": true, "rules_file_enabled": false, "diff_preview": false}',
        encoding="utf-8",
    )
    c = load_config(path=p)
    assert c.git_autocommit is True
    assert c.checkpoints_enabled is False
    assert c.hooks_enabled is True
    assert c.rules_file_enabled is False
    assert c.diff_preview is False


def test_foundation_fields_round_trip_through_save(tmp_path):
    from llmcli.config import save_config

    p = tmp_path / "config.json"
    save_config(
        Config(
            permission_mode="full-auto",
            git_autocommit=True,
            checkpoints_enabled=False,
            hooks_enabled=True,
            rules_file_enabled=False,
            output_format="json",
            diff_preview=False,
        ),
        path=p,
    )
    c = load_config(path=p)
    assert c.permission_mode == "full-auto"
    assert c.git_autocommit is True
    assert c.checkpoints_enabled is False
    assert c.hooks_enabled is True
    assert c.rules_file_enabled is False
    assert c.output_format == "json"
    assert c.diff_preview is False
