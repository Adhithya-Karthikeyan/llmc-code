"""Tests for the ranked, budget-aware repo_map (llmcli.tools._repo_map).

Covers: ranked output stays within the token budget (15% tolerance), the
`query` arg boosts matching files to the top, an empty query uses the 8x
budget expansion, the skip-list still excludes .lock/.min.js/.map, and the
header shows the `~N tok` estimate.
"""

from __future__ import annotations

from llmcli import repo_graph
from llmcli.tools import _repo_map
from llmcli.tokens import estimate_text_tokens


def _setup_repo(tmp_workspace, n_files: int = 6) -> None:
    # hub.py is imported by several others -> ranks high; the rest are leaves.
    (tmp_workspace / "hub.py").write_text("def hub():\n    pass\n", encoding="utf-8")
    for i in range(n_files):
        (tmp_workspace / f"u{i}.py").write_text("import hub\n", encoding="utf-8")
    (tmp_workspace / "leaf.py").write_text("x = 1\n", encoding="utf-8")


def test_ranked_output_within_token_budget(tmp_workspace):
    # A large repo; a tight max_map_tokens must keep the rendered map within
    # budget * 1.15 (the 15% tolerance).
    _setup_repo(tmp_workspace, n_files=20)
    repo_graph.clear_cache()
    out = _repo_map({"path": ".", "max_map_tokens": 200})["result"]
    header = out.splitlines()[0]
    assert "~" in header and "tok" in header
    # The rendered body is everything after the header line.
    body = out.split("\n", 1)[1]
    # Allow the 15% tolerance on the budget (200 * 8 = 1600 expanded target for
    # an empty query -> tolerance 1840). The body should fit well under that.
    assert estimate_text_tokens(body) <= int(200 * 8 * 1.15) + 1


def test_query_keeps_tight_budget(tmp_workspace):
    # With a query, the budget is NOT expanded (target = max_map_tokens), so a
    # small max_map_tokens yields a small map.
    _setup_repo(tmp_workspace, n_files=10)
    repo_graph.clear_cache()
    out = _repo_map({"path": ".", "query": "hub", "max_map_tokens": 60})["result"]
    body = out.split("\n", 1)[1]
    assert estimate_text_tokens(body) <= int(60 * 1.15) + 1


def test_query_boosts_matching_file_to_top(tmp_workspace):
    (tmp_workspace / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "billing.py").write_text("def charge():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "main.py").write_text("import auth\nimport billing\n", encoding="utf-8")
    repo_graph.clear_cache()
    out = _repo_map({"path": ".", "query": "login auth"})["result"]
    files = [ln.split(" (")[0] for ln in out.splitlines() if ln.endswith("lines)")
             and not ln.startswith("repo map")]
    assert files[0] == "auth.py"  # query-matching file surfaced first


def test_empty_query_uses_8x_expansion(tmp_workspace):
    # Empty query: target = max_map_tokens * 8. With many small files and a
    # generous max_map_tokens, all files should fit (no token truncation).
    _setup_repo(tmp_workspace, n_files=4)
    repo_graph.clear_cache()
    out = _repo_map({"path": ".", "max_map_tokens": 500})["result"]
    # All 6 files present (hub + 4 u's + leaf).
    assert "hub.py" in out and "leaf.py" in out
    assert "truncated" not in out  # comfortably within the 8x budget


def test_skip_list_excludes_lock_and_min_and_map(tmp_workspace):
    (tmp_workspace / "app.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "package-lock.json").write_text("{" + " " * 60000 + "}",
                                                      encoding="utf-8")
    (tmp_workspace / "jquery.min.js").write_text("!function(){}()\n", encoding="utf-8")
    (tmp_workspace / "style.min.css").write_text("*{}/*" + "x" * 200 + "*/\n",
                                                  encoding="utf-8")
    (tmp_workspace / "out.map").write_text("{}\n", encoding="utf-8")
    repo_graph.clear_cache()
    out = _repo_map({"path": "."})["result"]
    assert "app.py" in out
    assert "package-lock.json" not in out
    assert "jquery.min.js" not in out
    assert "style.min.css" not in out
    assert "out.map" not in out


def test_header_shows_token_estimate(tmp_workspace):
    (tmp_workspace / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
    repo_graph.clear_cache()
    out = _repo_map({"path": "."})["result"]
    header = out.splitlines()[0]
    assert "tok" in header
    assert "~" in header  # "~N tok" form
    # The N is a non-negative integer.
    tok_part = header.split("~")[1].split(" tok")[0]
    assert tok_part.isdigit() and int(tok_part) >= 0


def test_max_files_hard_ceiling_honored(tmp_workspace):
    for i in range(10):
        (tmp_workspace / f"f{i}.py").write_text("x = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    out = _repo_map({"path": ".", "max_files": 3})["result"]
    listed = [ln for ln in out.splitlines() if ln.endswith("lines)")
              and not ln.startswith("repo map")]
    assert len(listed) == 3
    assert "truncated" in out