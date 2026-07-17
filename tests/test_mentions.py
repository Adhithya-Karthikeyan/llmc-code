"""Tests for llmcli.mentions — @-mention expansion into injected context."""

from __future__ import annotations

import os

import pytest

from llmcli import mentions


# ---------------------------------------------------------------------------
# project_files
# ---------------------------------------------------------------------------

def test_project_files_lists_source_and_sorts(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("x = 1\n")
    files = mentions.project_files(tmp_path)
    assert files == ["a.py", "pkg/b.py"]


def test_project_files_excludes_git_dir(tmp_path):
    (tmp_path / "keep.py").write_text("ok\n")
    gitdir = tmp_path / ".git"
    gitdir.mkdir()
    (gitdir / "config").write_text("[core]\n")
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    files = mentions.project_files(tmp_path)
    assert files == ["keep.py"]
    assert not any(f.startswith(".git") for f in files)


def test_project_files_excludes_huge_file(tmp_path):
    (tmp_path / "small.py").write_text("ok\n")
    huge = tmp_path / "huge.py"
    huge.write_text("# pad\n" + ("x" * (mentions._MAX_LIST_FILE_BYTES + 10)))
    files = mentions.project_files(tmp_path)
    assert "small.py" in files
    assert "huge.py" not in files


def test_project_files_excludes_binary_and_minified(tmp_path):
    (tmp_path / "real.py").write_text("ok\n")
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "app.min.js").write_text("var a=1;\n")
    files = mentions.project_files(tmp_path)
    assert files == ["real.py"]


def test_project_files_respects_limit(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x\n")
    files = mentions.project_files(tmp_path, limit=3)
    assert len(files) == 3


# ---------------------------------------------------------------------------
# expand_mentions — files
# ---------------------------------------------------------------------------

def test_at_file_injects_contents(tmp_path):
    (tmp_path / "hello.py").write_text("print('hi there')\n")
    text = "please look at @hello.py and explain"
    out, blocks = mentions.expand_mentions(text, tmp_path)
    assert out == text  # text is left intact
    assert len(blocks) == 1
    b = blocks[0]
    assert b["kind"] == "file"
    assert b["ref"] == "hello.py"
    assert "print('hi there')" in b["content"]


def test_at_file_is_size_capped(tmp_path):
    (tmp_path / "big.py").write_text("A" * 5000)
    _, blocks = mentions.expand_mentions("@big.py", tmp_path, max_file_bytes=100)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "file"
    assert "...[truncated]" in blocks[0]["content"]
    # capped content is far smaller than the original 5000 bytes
    assert len(blocks[0]["content"].encode("utf-8")) < 400


def test_at_missing_file_yields_notice_not_exception(tmp_path):
    _, blocks = mentions.expand_mentions("check @nope.py", tmp_path)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "notice"
    assert "not found" in blocks[0]["content"]


def test_path_outside_root_is_refused(tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("SECRET")
    root = tmp_path / "proj"
    root.mkdir()
    _, blocks = mentions.expand_mentions("read @../secret.txt", root)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "notice"
    assert "outside workspace" in blocks[0]["content"]
    assert "SECRET" not in blocks[0]["content"]


def test_absolute_path_is_refused(tmp_path):
    secret = tmp_path / "abs.txt"
    secret.write_text("TOPSECRET")
    _, blocks = mentions.expand_mentions(f"@{secret}", tmp_path)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "notice"
    assert "TOPSECRET" not in blocks[0]["content"]


def test_injected_read_file_callable_is_used(tmp_path):
    calls = []

    def fake_read(relpath):
        calls.append(relpath)
        return "FAKE CONTENT for " + relpath

    _, blocks = mentions.expand_mentions("@x.py", tmp_path, read_file=fake_read)
    assert calls == ["x.py"]
    assert blocks[0]["kind"] == "file"
    assert "FAKE CONTENT for x.py" in blocks[0]["content"]


# ---------------------------------------------------------------------------
# expand_mentions — directories
# ---------------------------------------------------------------------------

def test_at_dir_lists_files(tmp_path):
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "one.py").write_text("1\n")
    (sub / "two.py").write_text("2\n")
    _, blocks = mentions.expand_mentions("look in @src/", tmp_path)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["kind"] == "dir"
    assert "src/one.py" in b["content"]
    assert "src/two.py" in b["content"]


def test_at_dir_missing_yields_notice(tmp_path):
    _, blocks = mentions.expand_mentions("@nodir/", tmp_path)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "notice"
    assert "not a directory" in blocks[0]["content"]


# ---------------------------------------------------------------------------
# expand_mentions — diff and url (dependency-injected)
# ---------------------------------------------------------------------------

def test_at_diff_uses_injected_callable(tmp_path):
    def fake_diff():
        return "diff --git a/x b/x\n+added line\n"

    _, blocks = mentions.expand_mentions("review @diff", tmp_path, git_diff=fake_diff)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "diff"
    assert "+added line" in blocks[0]["content"]


def test_at_diff_without_callable_yields_notice(tmp_path):
    _, blocks = mentions.expand_mentions("@diff", tmp_path)
    assert blocks[0]["kind"] == "notice"
    assert "no git_diff" in blocks[0]["content"]


def test_at_url_prefix_uses_web_fetch(tmp_path):
    seen = []

    def fake_fetch(url):
        seen.append(url)
        return "PAGE BODY"

    _, blocks = mentions.expand_mentions(
        "see @url:https://example.com/x", tmp_path, web_fetch=fake_fetch
    )
    assert seen == ["https://example.com/x"]
    assert blocks[0]["kind"] == "url"
    assert "PAGE BODY" in blocks[0]["content"]


def test_bare_http_url_uses_web_fetch(tmp_path):
    def fake_fetch(url):
        return f"FETCHED:{url}"

    _, blocks = mentions.expand_mentions(
        "@https://foo.test/page", tmp_path, web_fetch=fake_fetch
    )
    assert blocks[0]["kind"] == "url"
    assert "FETCHED:https://foo.test/page" in blocks[0]["content"]


def test_url_without_fetcher_yields_notice(tmp_path):
    _, blocks = mentions.expand_mentions("@https://foo.test/x", tmp_path)
    assert blocks[0]["kind"] == "notice"
    assert "no web_fetch" in blocks[0]["content"]


# ---------------------------------------------------------------------------
# non-misfire cases
# ---------------------------------------------------------------------------

def test_email_does_not_trigger(tmp_path):
    text = "email me at a@b.com please"
    out, blocks = mentions.expand_mentions(text, tmp_path)
    assert out == text
    assert blocks == []


def test_no_at_symbol_returns_empty(tmp_path):
    out, blocks = mentions.expand_mentions("just plain text", tmp_path)
    assert blocks == []
    assert out == "just plain text"


def test_at_at_start_of_line_triggers(tmp_path):
    (tmp_path / "top.py").write_text("hi\n")
    _, blocks = mentions.expand_mentions("@top.py", tmp_path)
    assert blocks[0]["kind"] == "file"
    assert blocks[0]["ref"] == "top.py"


def test_trailing_punctuation_stripped(tmp_path):
    (tmp_path / "foo.py").write_text("body\n")
    _, blocks = mentions.expand_mentions("open @foo.py.", tmp_path)
    assert blocks[0]["kind"] == "file"
    assert blocks[0]["ref"] == "foo.py"
    assert "body" in blocks[0]["content"]


def test_duplicate_mentions_deduped(tmp_path):
    (tmp_path / "d.py").write_text("dup\n")
    _, blocks = mentions.expand_mentions("@d.py and again @d.py", tmp_path)
    assert len(blocks) == 1


# ---------------------------------------------------------------------------
# render_blocks
# ---------------------------------------------------------------------------

def test_render_blocks_formats_delimited(tmp_path):
    (tmp_path / "r.py").write_text("hello render\n")
    _, blocks = mentions.expand_mentions("@r.py", tmp_path)
    rendered = mentions.render_blocks(blocks)
    assert "# Attached context" in rendered
    assert "## @file: r.py" in rendered
    assert "hello render" in rendered
    assert "```" in rendered


def test_render_blocks_empty_is_empty_string():
    assert mentions.render_blocks([]) == ""


def test_render_blocks_notice_shape(tmp_path):
    _, blocks = mentions.expand_mentions("@missing.py", tmp_path)
    rendered = mentions.render_blocks(blocks)
    assert "## notice:" in rendered
    assert "not found" in rendered
