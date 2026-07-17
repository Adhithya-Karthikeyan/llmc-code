"""Tests for the dependency-graph PageRank ranker (llmcli.repo_graph).

Covers per-language import extraction, PageRank ranking sanity, the small/
edgeless fallback to flat order, the cache key changing when a file is added,
and the per-file parse-error robustness (a bad file never raises).
"""

from __future__ import annotations

from llmcli import repo_graph


# ----- Python import extraction (relative + absolute) ----------------------

def test_python_absolute_import_creates_edge(tmp_workspace):
    (tmp_workspace / "a.py").write_text("import b\n", encoding="utf-8")
    (tmp_workspace / "b.py").write_text("def hello():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "c.py").write_text("x = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.py"), str(tmp_workspace / "b.py"),
         str(tmp_workspace / "c.py")],
        "", str(tmp_workspace),
    )
    # b.py is imported by a.py -> it should rank above the unreferenced c.py.
    assert ranked.index("b.py") < ranked.index("c.py")


def test_python_relative_import_resolves(tmp_workspace):
    pkg = tmp_workspace / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("def thing():\n    pass\n", encoding="utf-8")
    (pkg / "use.py").write_text("from . import mod\n", encoding="utf-8")
    (tmp_workspace / "other.py").write_text("y = 2\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(pkg / "__init__.py"), str(pkg / "mod.py"), str(pkg / "use.py"),
         str(tmp_workspace / "other.py")],
        "", str(tmp_workspace),
    )
    # pkg/mod.py is referenced by pkg/use.py -> ranks above the unreferenced
    # other.py.
    assert ranked.index("pkg/mod.py") < ranked.index("other.py")


# ----- JS/TS import + require ----------------------------------------------

def test_js_import_from_creates_edge(tmp_workspace):
    (tmp_workspace / "a.js").write_text(
        "import { hello } from './b.js'\nhello()\n", encoding="utf-8"
    )
    (tmp_workspace / "b.js").write_text("export function hello() {}\n", encoding="utf-8")
    (tmp_workspace / "c.js").write_text("const z = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.js"), str(tmp_workspace / "b.js"),
         str(tmp_workspace / "c.js")],
        "", str(tmp_workspace),
    )
    assert ranked.index("b.js") < ranked.index("c.js")


def test_js_require_creates_edge(tmp_workspace):
    (tmp_workspace / "a.js").write_text("const x = require('./b')\n", encoding="utf-8")
    (tmp_workspace / "b.js").write_text("module.exports = {}\n", encoding="utf-8")
    (tmp_workspace / "c.js").write_text("const z = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.js"), str(tmp_workspace / "b.js"),
         str(tmp_workspace / "c.js")],
        "", str(tmp_workspace),
    )
    assert ranked.index("b.js") < ranked.index("c.js")


# ----- Go import block -----------------------------------------------------

def test_go_import_block_creates_edge(tmp_workspace):
    (tmp_workspace / "a.go").write_text(
        "package main\n\nimport (\n\t\"fmt\"\n\t\"example.com/m/b\"\n)\n\nfunc main() {}\n",
        encoding="utf-8",
    )
    (tmp_workspace / "b.go").write_text("package b\n\nfunc F() {}\n", encoding="utf-8")
    (tmp_workspace / "c.go").write_text("package c\n\nvar X = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.go"), str(tmp_workspace / "b.go"),
         str(tmp_workspace / "c.go")],
        "", str(tmp_workspace),
    )
    assert ranked.index("b.go") < ranked.index("c.go")


# ----- Rust use ------------------------------------------------------------

def test_rust_use_crate_resolves(tmp_workspace):
    (tmp_workspace / "a.rs").write_text("use crate::b::thing;\n", encoding="utf-8")
    (tmp_workspace / "b.rs").write_text("pub fn thing() {}\n", encoding="utf-8")
    (tmp_workspace / "c.rs").write_text("pub static X: i32 = 1;\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.rs"), str(tmp_workspace / "b.rs"),
         str(tmp_workspace / "c.rs")],
        "", str(tmp_workspace),
    )
    assert ranked.index("b.rs") < ranked.index("c.rs")


# ----- PageRank sanity / fallback / cache / robustness ---------------------

def test_pagerank_ranks_referenced_above_unreferenced(tmp_workspace):
    # hub.py is imported by two files; leaf.py by none.
    (tmp_workspace / "hub.py").write_text("def hub():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "u1.py").write_text("import hub\n", encoding="utf-8")
    (tmp_workspace / "u2.py").write_text("import hub\n", encoding="utf-8")
    (tmp_workspace / "leaf.py").write_text("v = 0\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "hub.py"), str(tmp_workspace / "u1.py"),
         str(tmp_workspace / "u2.py"), str(tmp_workspace / "leaf.py")],
        "", str(tmp_workspace),
    )
    assert ranked[0] == "hub.py"   # most-referenced file ranks first
    assert ranked.index("hub.py") < ranked.index("leaf.py")


def test_fallback_flat_when_too_few_nodes(tmp_workspace):
    (tmp_workspace / "a.py").write_text("import b\n", encoding="utf-8")
    (tmp_workspace / "b.py").write_text("x = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.py"), str(tmp_workspace / "b.py")],
        "", str(tmp_workspace),
    )
    # <3 nodes -> flat path-sorted order.
    assert ranked == sorted(["a.py", "b.py"])


def test_fallback_flat_when_zero_edges(tmp_workspace):
    for n in ("a.py", "b.py", "c.py"):
        (tmp_workspace / n).write_text("x = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.py"), str(tmp_workspace / "b.py"),
         str(tmp_workspace / "c.py")],
        "", str(tmp_workspace),
    )
    # 3 nodes but zero edges -> flat path-sorted order.
    assert ranked == ["a.py", "b.py", "c.py"]


def test_cache_key_changes_when_file_added(tmp_workspace):
    (tmp_workspace / "a.py").write_text("import b\n", encoding="utf-8")
    (tmp_workspace / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
    repo_graph.clear_cache()
    files1 = [str(tmp_workspace / "a.py"), str(tmp_workspace / "b.py")]
    r1 = repo_graph.rank(files1, "", str(tmp_workspace))
    # Adding a file changes the repo signature (file count) -> cache miss and
    # re-rank, so the new file appears in the result.
    (tmp_workspace / "c.py").write_text("import b\n", encoding="utf-8")
    files2 = files1 + [str(tmp_workspace / "c.py")]
    r2 = repo_graph.rank(files2, "", str(tmp_workspace))
    assert "c.py" in r2
    assert "c.py" not in r1


def test_parse_error_file_skipped_without_raising(tmp_workspace):
    # b.py is syntactically broken; a.py imports it. Extraction must skip b's
    # edges gracefully (no exception), and ranking still returns all files.
    (tmp_workspace / "a.py").write_text("import b\n", encoding="utf-8")
    (tmp_workspace / "b.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_workspace / "c.py").write_text("x = 1\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "a.py"), str(tmp_workspace / "b.py"),
         str(tmp_workspace / "c.py")],
        "", str(tmp_workspace),
    )
    assert set(ranked) == {"a.py", "b.py", "c.py"}


def test_query_boosts_matching_file_to_top(tmp_workspace):
    (tmp_workspace / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "billing.py").write_text("def charge():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "main.py").write_text("import auth\nimport billing\n", encoding="utf-8")
    repo_graph.clear_cache()
    ranked = repo_graph.rank(
        [str(tmp_workspace / "auth.py"), str(tmp_workspace / "billing.py"),
         str(tmp_workspace / "main.py")],
        "login auth",
        str(tmp_workspace),
    )
    # auth.py matches query tokens -> ranked first.
    assert ranked[0] == "auth.py"


# ----- PageRank fast-path equivalence (reverse-adjacency rewrite) ----------

def _pagerank_bruteforce(nodes, edges, personal, damping=0.85,
                         max_iter=40, tol=1e-6):
    """Reference implementation matching the pre-optimization O(N*E) loop.

    Kept here verbatim to prove the reverse-adjacency rewrite produces
    byte-identical scores (same summation order, same dangling handling).
    """
    n = len(nodes)
    if n == 0:
        return {}
    out_deg = {node: len(edges.get(node, [])) for node in nodes}
    if not personal or sum(personal.values()) <= 0:
        personal = {node: 1.0 / n for node in nodes}
    else:
        s = sum(personal.values())
        personal = {k: v / s for k, v in personal.items()}
    rank = {node: 1.0 / n for node in nodes}
    for _ in range(max_iter):
        new_rank = {}
        dangling = sum(rank[node] for node in nodes if out_deg[node] == 0)
        for node in nodes:
            incoming = damping * (dangling / n)
            for src in edges:
                if node in edges[src] and out_deg[src] > 0:
                    incoming += damping * (rank[src] / out_deg[src])
            new_rank[node] = incoming + (1 - damping) * personal[node]
        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < tol:
            break
    return rank


def test_pagerank_matches_bruteforce_reference():
    # A small known graph with a hub (d), a dangling node (e), and a cycle.
    nodes = ["a", "b", "c", "d", "e"]
    edges = {
        "a": ["b", "c"],   # a -> b, a -> c
        "b": ["d"],        # b -> d
        "c": ["d"],        # c -> d
        "d": ["a"],        # d -> a  (cycle a->b/c->d->a)
        # "e" is dangling (no out-edges)
    }
    personal = {}
    fast = repo_graph._pagerank(nodes, edges, personal)
    ref = _pagerank_bruteforce(nodes, edges, personal)
    assert set(fast) == set(ref)
    for node in nodes:
        # Byte-identical: same operations in the same order.
        assert fast[node] == ref[node]
    # Hub d (2 incoming) ranks above the dangling, unreferenced e.
    fast_order = sorted(nodes, key=lambda r: (-fast[r], r))
    ref_order = sorted(nodes, key=lambda r: (-ref[r], r))
    assert fast_order == ref_order
    assert fast_order.index("d") < fast_order.index("e")


def test_pagerank_matches_bruteforce_with_personalization():
    nodes = ["a", "b", "c", "d"]
    edges = {"a": ["b"], "b": ["c"], "c": ["a"]}  # d dangling
    personal = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    fast = repo_graph._pagerank(nodes, edges, personal)
    ref = _pagerank_bruteforce(nodes, edges, personal)
    for node in nodes:
        assert fast[node] == ref[node]


# ----- Cache is bounded (LRU eviction to _CACHE_MAX) -----------------------

def test_cache_bounded_to_max_entries(tmp_workspace):
    # Build a real graph once so rank() populates the cache, then force many
    # distinct signatures and assert the cache never exceeds _CACHE_MAX.
    repo_graph.clear_cache()
    (tmp_workspace / "auth.py").write_text("def login():\n    pass\n", encoding="utf-8")
    (tmp_workspace / "billing.py").write_text("def charge():\n    pass\n", encoding="utf-8")
    files = [str(tmp_workspace / "auth.py"), str(tmp_workspace / "billing.py")]

    for i in range(50):
        # Each new file changes the repo signature (file count) -> new key.
        extra = tmp_workspace / f"mod_{i}.py"
        extra.write_text(f"import auth\nx_{i} = {i}\n", encoding="utf-8")
        files.append(str(extra))
        repo_graph.rank(files, "", str(tmp_workspace))
        assert len(repo_graph._CACHE) <= repo_graph._CACHE_MAX

    assert len(repo_graph._CACHE) == repo_graph._CACHE_MAX
    repo_graph.clear_cache()
    assert len(repo_graph._CACHE) == 0