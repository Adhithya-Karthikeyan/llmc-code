"""Path-reliability tests: workspace-relative paths land correctly, out-of-
workspace absolute paths are refused with ACTIONABLE guidance, and run_bash
uses a shell that honors `echo -n` (the macOS /bin/sh mangling bug)."""

from __future__ import annotations

from llmcli.tools import _edit_file, _read_file, _run_bash, _write_file


# ----- relative paths inside the workspace succeed -------------------------

def test_write_file_relative_path_lands_in_workspace(tmp_workspace):
    r = _write_file({"path": "subdir/note.txt", "content": "hello"})
    assert r["ok"] is True
    landed = tmp_workspace / "subdir" / "note.txt"
    assert landed.is_file()
    assert landed.read_text(encoding="utf-8") == "hello"


def test_edit_file_relative_path_succeeds(tmp_workspace):
    (tmp_workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    r = _edit_file({"path": "app.py", "old": "x = 1", "new": "x = 2"})
    assert r["ok"] is True
    assert (tmp_workspace / "app.py").read_text(encoding="utf-8") == "x = 2\n"


# ----- out-of-workspace absolute paths refused WITH actionable guidance ----

def test_write_file_absolute_outside_refused_with_relative_hint(tmp_workspace):
    r = _write_file({"path": "/etc/definitely_outside.txt", "content": "x"})
    assert r["ok"] is False
    assert "Refusing to write outside the workspace root" in r["error"]
    assert "relative" in r["error"].lower()


def test_edit_file_absolute_outside_refused_with_relative_hint(tmp_workspace):
    r = _edit_file({"path": "/etc/hosts_outside", "old": "a", "new": "b"})
    assert r["ok"] is False
    assert "Refusing to edit outside the workspace root" in r["error"]
    assert "relative" in r["error"].lower()


def test_read_file_absolute_outside_refused_with_relative_hint(tmp_workspace):
    r = _read_file({"path": "/etc/passwd_outside"})
    assert r["ok"] is False
    assert "Refusing to read outside the workspace root" in r["error"]
    assert "relative" in r["error"].lower()


# ----- run_bash: `echo -n` must NOT emit a literal "-n" --------------------

def test_run_bash_echo_n_not_corrupted(tmp_workspace, monkeypatch):
    """The shell fix: `echo -n hi > f` must write exactly `hi`, never `-n hi`.
    macOS /bin/sh (POSIX bash) prints a literal "-n"; /bin/bash honors it."""
    monkeypatch.setattr("llmcli.tools._PRIVATE", False)
    r = _run_bash({"command": "echo -n hi > f.txt"})
    assert r["ok"] is True
    assert (tmp_workspace / "f.txt").read_text(encoding="utf-8") == "hi"


# ----- edit_file tolerant fallback: fewer wasted retry rounds --------------

def test_edit_file_exact_match_unchanged_semantics(tmp_workspace):
    """(a) An EXACT match keeps the original behavior: replacements=1, no
    'note' key (the fallback path is never taken)."""
    (tmp_workspace / "app.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    r = _edit_file({"path": "app.py", "old": "a = 1", "new": "a = 9"})
    assert r["ok"] is True
    assert r["result"]["replacements"] == 1
    assert "note" not in r["result"]  # exact path, untouched
    assert (tmp_workspace / "app.py").read_text(encoding="utf-8") == "a = 9\nb = 2\n"


def test_edit_file_crlf_vs_lf_mismatch_succeeds_via_fallback(tmp_workspace):
    """(b) The model supplies a CRLF 'old' against an LF file -> exact match
    fails, tolerant fallback succeeds. (File-side CRLF is already LF-normalized
    by read_file's universal-newline read, so CRLF only reaches the fallback
    when it is present in the model's 'old'.)"""
    (tmp_workspace / "win.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    r = _edit_file({"path": "win.txt", "old": "one\r\ntwo", "new": "one\nTWO"})
    assert r["ok"] is True
    assert r["result"]["replacements"] == 1
    assert "normaliz" in r["result"]["note"].lower()
    # Only the matched region changed; every other byte survives.
    assert (tmp_workspace / "win.txt").read_text(encoding="utf-8") == "one\nTWO\nthree\n"


def test_edit_file_multiline_crlf_block_replaced_exactly(tmp_workspace):
    """(b') A multi-line CRLF 'old' matches an LF block; the real span is
    replaced and the rest is preserved exactly."""
    (tmp_workspace / "m.txt").write_text("head\nfoo\nbar\ntail\n", encoding="utf-8")
    r = _edit_file({"path": "m.txt", "old": "foo\r\nbar", "new": "X\nY"})
    assert r["ok"] is True
    assert (tmp_workspace / "m.txt").read_text(encoding="utf-8") == "head\nX\nY\ntail\n"


def test_edit_file_trailing_whitespace_mismatch_succeeds(tmp_workspace):
    """(c) File line has trailing spaces the model omitted -> tolerant match
    succeeds; the trailing whitespace inside the matched span is consumed."""
    (tmp_workspace / "t.py").write_text("x = 1   \ny = 2\n", encoding="utf-8")
    r = _edit_file({"path": "t.py", "old": "x = 1\ny = 2", "new": "x = 1\ny = 3"})
    assert r["ok"] is True
    assert "normaliz" in r["result"]["note"].lower()
    assert (tmp_workspace / "t.py").read_text(encoding="utf-8") == "x = 1\ny = 3\n"


def test_edit_file_ambiguous_normalized_match_refused(tmp_workspace):
    """(d) 'old' matches TWO places after normalization -> REFUSE, file
    unchanged, guidance tells the model to add surrounding context."""
    # Both blocks carry trailing whitespace so NEITHER matches exactly, but
    # both match after normalization -> ambiguous.
    original = "foo \nbar\nfoo\t\nbar\n"
    (tmp_workspace / "dup.txt").write_text(original, encoding="utf-8")
    r = _edit_file({"path": "dup.txt", "old": "foo\nbar", "new": "ZZZ"})
    assert r["ok"] is False
    assert "2 places" in r["error"] or "ambiguous" in r["error"].lower()
    assert "context" in r["error"].lower()
    assert (tmp_workspace / "dup.txt").read_text(encoding="utf-8") == original


def test_edit_file_genuine_no_match_refused_with_read_first(tmp_workspace):
    """(e) Genuinely absent text -> REFUSE with read-first guidance, file
    unchanged."""
    original = "alpha\nbeta\n"
    (tmp_workspace / "n.txt").write_text(original, encoding="utf-8")
    r = _edit_file({"path": "n.txt", "old": "gamma", "new": "delta"})
    assert r["ok"] is False
    assert "read_file" in r["error"]
    assert "not found" in r["error"].lower()
    assert (tmp_workspace / "n.txt").read_text(encoding="utf-8") == original


def test_edit_file_leading_indentation_diff_not_silently_matched(tmp_workspace):
    """(f) Leading indentation differs -> must REFUSE (never silently match),
    because indentation carries meaning and matching it would risk editing the
    wrong location. File unchanged."""
    original = "def f():\n\t\treturn 1\n"  # body indented with TABS
    (tmp_workspace / "ind.py").write_text(original, encoding="utf-8")
    # Model supplies SPACE indentation -> exact fails and only leading
    # indentation differs; normalization must NOT bridge it.
    r = _edit_file({"path": "ind.py", "old": "    return 1", "new": "    return 2"})
    assert r["ok"] is False
    assert "read_file" in r["error"]
    assert (tmp_workspace / "ind.py").read_text(encoding="utf-8") == original
