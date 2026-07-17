"""Tests for llmcli.rules — project conventions/rules auto-loading."""

from __future__ import annotations

from llmcli import rules


def test_rules_filenames_priority_order():
    assert rules.RULES_FILENAMES == (
        "AGENTS.md",
        "LLMCLI.md",
        ".llmclirules",
        "CONVENTIONS.md",
    )


def test_find_rules_file_none_present(tmp_path):
    assert rules.find_rules_file(tmp_path) is None


def test_find_rules_file_single(tmp_path):
    f = tmp_path / "CONVENTIONS.md"
    f.write_text("use tabs", encoding="utf-8")
    assert rules.find_rules_file(tmp_path) == f


def test_find_rules_file_priority_agents_wins(tmp_path):
    # Both present: AGENTS.md (priority 1) must beat CONVENTIONS.md (priority 4).
    (tmp_path / "AGENTS.md").write_text("agents rules", encoding="utf-8")
    (tmp_path / "CONVENTIONS.md").write_text("conventions rules", encoding="utf-8")
    found = rules.find_rules_file(tmp_path)
    assert found is not None
    assert found.name == "AGENTS.md"


def test_find_rules_file_full_priority_chain(tmp_path):
    # Create ALL of them, then remove in priority order and confirm the next wins.
    for name in rules.RULES_FILENAMES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    for name in rules.RULES_FILENAMES:
        found = rules.find_rules_file(tmp_path)
        assert found is not None and found.name == name
        found.unlink()
    assert rules.find_rules_file(tmp_path) is None


def test_find_rules_file_ignores_directory(tmp_path):
    # A directory named like a rules file must NOT be treated as the rules file.
    (tmp_path / "AGENTS.md").mkdir()
    assert rules.find_rules_file(tmp_path) is None


def test_load_rules_none_present(tmp_path):
    assert rules.load_rules(tmp_path) == ""


def test_load_rules_reads_and_strips(tmp_path):
    (tmp_path / "AGENTS.md").write_text("\n\n  hello rules  \n\n", encoding="utf-8")
    assert rules.load_rules(tmp_path) == "hello rules"


def test_load_rules_empty_file(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n\t\n  ", encoding="utf-8")
    assert rules.load_rules(tmp_path) == ""


def test_load_rules_ignores_bad_bytes(tmp_path):
    f = tmp_path / "AGENTS.md"
    f.write_bytes(b"good\xffcontent")
    out = rules.load_rules(tmp_path)
    assert "good" in out and "content" in out


def test_load_rules_oversized_truncated_with_marker(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 20_000, encoding="utf-8")
    out = rules.load_rules(tmp_path, max_bytes=8000)
    assert out.endswith("…[rules truncated]")
    assert out.endswith(rules._TRUNCATION_MARKER)
    body = out[: -len(rules._TRUNCATION_MARKER)]
    # Byte payload of the kept portion stays within the cap.
    assert len(body.encode("utf-8")) <= 8000
    assert len(out) < 20_000


def test_load_rules_under_cap_not_truncated(tmp_path):
    (tmp_path / "AGENTS.md").write_text("short and sweet", encoding="utf-8")
    out = rules.load_rules(tmp_path, max_bytes=8000)
    assert out == "short and sweet"
    assert "truncated" not in out


def test_load_rules_unreadable_returns_empty(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("secret rules", encoding="utf-8")

    def boom(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.read_text", boom)
    assert rules.load_rules(tmp_path) == ""


def test_rules_prompt_block_absent_is_empty(tmp_path):
    assert rules.rules_prompt_block(tmp_path) == ""


def test_rules_prompt_block_empty_file_is_empty(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   ", encoding="utf-8")
    assert rules.rules_prompt_block(tmp_path) == ""


def test_rules_prompt_block_includes_filename_and_contents(tmp_path):
    (tmp_path / "CONVENTIONS.md").write_text(
        "Always write typed Python.", encoding="utf-8"
    )
    block = rules.rules_prompt_block(tmp_path)
    assert "CONVENTIONS.md" in block
    assert "Always write typed Python." in block
    assert block.startswith("# Project rules")


def test_rules_prompt_block_uses_priority_filename(tmp_path):
    (tmp_path / "AGENTS.md").write_text("agents win", encoding="utf-8")
    (tmp_path / "CONVENTIONS.md").write_text("zzz_lowprio_marker", encoding="utf-8")
    block = rules.rules_prompt_block(tmp_path)
    assert "from AGENTS.md" in block
    assert "agents win" in block
    assert "zzz_lowprio_marker" not in block


def test_default_template_is_nonempty_markdown():
    tpl = rules.default_template()
    assert isinstance(tpl, str)
    assert tpl.strip()
    assert tpl.lstrip().startswith("#")
    for heading in ("Project overview", "Conventions", "Do", "Don't"):
        assert heading in tpl


def test_accepts_str_path(tmp_path):
    (tmp_path / "AGENTS.md").write_text("str path ok", encoding="utf-8")
    assert rules.load_rules(str(tmp_path)) == "str path ok"
