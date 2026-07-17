"""Tests for the read_file skip-list guard (llmcli.tools._read_file).

A lockfile/minified/generated file read WHOLE (no offset/limit) and larger than
``_SKIP_FILE_WARN_BYTES`` is refused with a hint; the same file WITH offset/
limit is allowed (targeted slice); a small lockfile (< threshold) is allowed;
a normal source file is always allowed.
"""

from __future__ import annotations

from llmcli.tools import _read_file, is_context_bloat_file


def test_large_lockfile_whole_read_refused_with_hint(tmp_workspace):
    (tmp_workspace / "package-lock.json").write_text(
        "{" + " " * 60_000 + "}", encoding="utf-8"
    )
    r = _read_file({"path": "package-lock.json"})
    assert r["ok"] is False
    assert "lockfile" in r["error"].lower() or "generated" in r["error"].lower()
    assert "offset+limit" in r["error"]   # the actionable hint
    assert "grep" in r["error"].lower() or "code_search" in r["error"].lower() \
        or "repo_map" in r["error"].lower()


def test_large_lockfile_with_offset_limit_allowed(tmp_workspace):
    (tmp_workspace / "package-lock.json").write_text(
        "\n".join(f"line{i}" for i in range(2000)) + "\n", encoding="utf-8"
    )
    r = _read_file({"path": "package-lock.json", "offset": 1, "limit": 5})
    assert r["ok"] is True
    assert "[lines 1-5 of 2000]" in r["result"]
    assert "line1" in r["result"]


def test_small_lockfile_allowed(tmp_workspace):
    # Under the 50KB threshold -> allowed even though it matches the skip-list.
    (tmp_workspace / "yarn.lock").write_text("# small lockfile\n", encoding="utf-8")
    r = _read_file({"path": "yarn.lock"})
    assert r["ok"] is True
    assert r["result"] == "# small lockfile\n"


def test_large_minified_js_whole_read_refused(tmp_workspace):
    (tmp_workspace / "bundle.min.js").write_text(
        "!" + "x" * 60_000 + "()\n", encoding="utf-8"
    )
    r = _read_file({"path": "bundle.min.js"})
    assert r["ok"] is False
    assert "minified" in r["error"].lower() or "lockfile" in r["error"].lower() \
        or "generated" in r["error"].lower()


def test_normal_source_file_allowed(tmp_workspace):
    (tmp_workspace / "app.py").write_text("def main():\n    pass\n", encoding="utf-8")
    r = _read_file({"path": "app.py"})
    assert r["ok"] is True
    assert "def main" in r["result"]


def test_is_context_bloat_file_predicate():
    assert is_context_bloat_file("package-lock.json") is True
    assert is_context_bloat_file("x/yarn.lock") is True
    assert is_context_bloat_file("go.sum") is True
    assert is_context_bloat_file("jquery.min.js") is True
    assert is_context_bloat_file("style.min.css") is True
    assert is_context_bloat_file("out.map") is True
    assert is_context_bloat_file("app.py") is False
    assert is_context_bloat_file("src/index.ts") is False
    assert is_context_bloat_file("README.md") is False