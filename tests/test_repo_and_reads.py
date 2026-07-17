"""Tests for context-shrinking features: read_file line slices + repo_map."""

from __future__ import annotations

from llmcli.tools import _edit_file, _read_file, _repo_map


# ----- read_file offset/limit (sliced reads) -------------------------------

def test_read_file_slice_returns_only_range(tmp_workspace):
    p = tmp_workspace / "f.txt"
    p.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n", encoding="utf-8")
    r = _read_file({"path": "f.txt", "offset": 5, "limit": 3})
    assert r["ok"]
    out = r["result"]
    assert "[lines 5-7 of 20]" in out
    assert "line5" in out and "line6" in out and "line7" in out
    assert "line4" not in out and "line8" not in out


def test_read_file_offset_to_end_when_no_limit(tmp_workspace):
    p = tmp_workspace / "f.txt"
    p.write_text("\n".join(f"l{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    r = _read_file({"path": "f.txt", "offset": 9})
    assert "[lines 9-10 of 10]" in r["result"]
    assert "l9" in r["result"] and "l10" in r["result"]


def test_read_file_whole_file_unchanged_without_slice(tmp_workspace):
    p = tmp_workspace / "f.txt"
    p.write_text("alpha\nbeta\n", encoding="utf-8")
    r = _read_file({"path": "f.txt"})
    assert r["result"] == "alpha\nbeta\n"  # no header, byte-identical


def test_read_file_slice_ignores_garbage_offset(tmp_workspace):
    p = tmp_workspace / "f.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    # bool/str offsets are ignored -> treated as start=1
    r = _read_file({"path": "f.txt", "offset": True, "limit": 2})
    assert "[lines 1-2 of 3]" in r["result"]


def test_read_file_refuses_binary(tmp_workspace):
    """TOOLS-2: a file with a NUL byte is refused instead of returning garbage."""
    (tmp_workspace / "blob.bin").write_bytes(b"PK\x03\x04\x00\x01garbage")
    r = _read_file({"path": "blob.bin"})
    assert r["ok"] is False
    assert "binary" in r["error"].lower()


def test_read_file_whole_file_default_line_cap(tmp_workspace):
    """PERF-3: a whole-file read without offset/limit caps lines and notes it."""
    import llmcli.tools as t

    p = tmp_workspace / "many.txt"
    p.write_text("".join(f"line{i}\n" for i in range(t._READ_FILE_MAX_LINES + 50)),
                 encoding="utf-8")
    out = _read_file({"path": "many.txt"})["result"]
    assert "use offset/limit to read a specific slice" in out
    body = out.split("[truncated", 1)[0]
    assert body.count("\n") == t._READ_FILE_MAX_LINES  # only the first N lines


def test_read_file_refuses_oversized_whole_read(tmp_workspace, monkeypatch):
    """TOOLS-1: a whole-file read refuses files larger than the byte guard."""
    import llmcli.tools as t

    monkeypatch.setattr(t, "_READ_FILE_MAX_BYTES", 50)
    (tmp_workspace / "huge.txt").write_text("z" * 200, encoding="utf-8")
    r = _read_file({"path": "huge.txt"})
    assert r["ok"] is False
    assert "too large" in r["error"].lower()


def test_read_file_slice_works_on_oversized_file(tmp_workspace, monkeypatch):
    """TOOLS-1: an offset/limit slice still works (streamed via islice) on a file
    over the byte guard. The exact line TOTAL is intentionally omitted for an
    oversized file — computing it streams the whole (multi-GB) file, defeating the
    cheap-slice intent — so the header shows the slice range without 'of N'."""
    import llmcli.tools as t

    monkeypatch.setattr(t, "_READ_FILE_MAX_BYTES", 50)
    (tmp_workspace / "huge.txt").write_text("".join(f"L{i}\n" for i in range(1, 101)),
                                            encoding="utf-8")
    r = _read_file({"path": "huge.txt", "offset": 3, "limit": 2})
    assert r["ok"] is True
    assert "[lines 3-4]" in r["result"]          # slice works; total omitted (no full scan)
    assert "of 100" not in r["result"]
    assert "L3" in r["result"] and "L4" in r["result"]


def test_edit_file_refuses_oversized(tmp_workspace, monkeypatch):
    """TOOLS-1: edit_file refuses to load a file larger than the byte guard."""
    import llmcli.tools as t

    monkeypatch.setattr(t, "_READ_FILE_MAX_BYTES", 50)
    (tmp_workspace / "huge.txt").write_text("a" * 200, encoding="utf-8")
    r = _edit_file({"path": "huge.txt", "old": "a", "new": "b"})
    assert r["ok"] is False
    assert "too large" in r["error"].lower()


# ----- repo_map (compact structural index) ---------------------------------

def test_repo_map_lists_files_with_lines_and_symbols(tmp_workspace):
    (tmp_workspace / "mod.py").write_text(
        "import os\n\n\ndef foo():\n    pass\n\n\nclass Bar:\n    def m(self):\n        pass\n",
        encoding="utf-8",
    )
    (tmp_workspace / "app.js").write_text(
        "export function hello() {}\nconst x = 1\n", encoding="utf-8"
    )
    out = _repo_map({"path": "."})["result"]
    assert "mod.py" in out and "foo" in out and "Bar" in out
    assert "app.js" in out and "hello" in out
    assert "lines" in out  # per-file line count


def test_repo_map_prunes_vcs_and_skips_binary(tmp_workspace):
    (tmp_workspace / "keep.py").write_text("def k():\n    pass\n", encoding="utf-8")
    gitdir = tmp_workspace / ".git"
    gitdir.mkdir()
    (gitdir / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_workspace / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01")
    out = _repo_map({"path": "."})["result"]
    assert "keep.py" in out
    assert ".git" not in out       # pruned VCS dir
    assert "pic.png" not in out    # skipped binary asset


def test_repo_map_skips_minified_js(tmp_workspace):
    (tmp_workspace / "app.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "jquery.min.js").write_text("!function(){}()\n", encoding="utf-8")
    out = _repo_map({"path": "."})["result"]
    assert "app.py" in out
    assert "jquery.min.js" not in out  # multi-dot suffix matched by name


def test_repo_map_big_file_counts_lines_but_skips_symbols(tmp_workspace, monkeypatch):
    import llmcli.tools as t

    monkeypatch.setattr(t, "_REPO_MAP_MAX_FILE_BYTES", 10)  # force the big-file path
    big = "\n".join(f"def f{i}():" for i in range(20)) + "\n"
    (tmp_workspace / "big.py").write_text(big, encoding="utf-8")
    out = _repo_map({"path": "."})["result"]
    assert "big.py (20 lines)" in out
    # symbols skipped for the over-limit file => no "def" symbol line beneath it
    lines = out.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("big.py "))
    assert idx + 1 >= len(lines) or not lines[idx + 1].startswith("  f0")


def test_repo_map_respects_max_files(tmp_workspace):
    for i in range(10):
        (tmp_workspace / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    out = _repo_map({"path": ".", "max_files": 3})["result"]
    assert "truncated" in out
    listed = [ln for ln in out.splitlines() if ln.endswith("lines)")]
    assert len(listed) == 3
