"""Dependency-graph repo ranking (aider-style) — pure stdlib.

Builds a file->file reference graph by extracting imports per language
(Python via ``ast``; JS/TS/Go/Rust via regex), then runs pure-Python PageRank
with an optional personalization vector derived from a ``query`` string. The
ranked relpath list is consumed by ``tools._repo_map`` to render the most
relevant files first under a token budget.

No third-party deps (no numpy): PageRank is a small power iteration. Results
are cached per process keyed by a cheap repo signature (file count + per
top-level-dir max mtime), mirroring ``code_index._INDEX_CACHE``.

Robustness: any per-file parse error skips that file's edges (never raises).
If the graph has fewer than 3 nodes or zero edges, ranking is meaningless, so
we fall back to flat path-sorted order (the prior repo_map behaviour).
"""

from __future__ import annotations

import ast
import os
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

__all__ = ["rank", "clear_cache"]

# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

# JS/TS: `import ... from 'x'` and `require('x')`. Only relative specifiers
# (./ or ../) resolve to a repo file; bare specifiers ('react', 'lodash')
# are external packages and produce no edge.
_JS_IMPORT_FROM = re.compile(r"""^\s*import\b.*\bfrom\s*['"]([^'"]+)['"]""")
_JS_REQUIRE = re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)""")
# TS `import type` and side-effect imports already covered by _JS_IMPORT_FROM
# (the `.*` swallows `type`/names); pure side-effect `import 'x'` is handled
# by the `from`-less branch below.
_JS_SIDE_EFFECT = re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""")

# Go: single-line `import "pkg"` and block lines `alias "pkg"`. Go imports are
# always package paths; only intra-repo ones resolve (a local module's packages
# share the repo root prefix). We treat any import path as a potential repo
# module name and let the module->relpath index resolve it (no edge if unknown).
_GO_IMPORT_SINGLE = re.compile(r"""^\s*import\s+"([^"]+)""")
_GO_IMPORT_BLOCK_LINE = re.compile(r"""^\s*(?:[\w.]+\s+)?"([^"]+)""")

# Rust: `use crate::...` (internal) or `use <crate>::...`. The first path segment
# is the crate name; `crate` / `self` / `super` are internal-relative. We only
# use the first segment to map to a repo crate name; `crate::` resolves to the
# current file's crate (handled via the module->relpath index).
_RUST_USE = re.compile(r"""^\s*use\s+([\w:]+)""")

# Python file extensions we attempt ast.parse on.
_PY_EXTS = frozenset({".py", ".pyi"})
# JS/TS family.
_JS_EXTS = frozenset({".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"})
_GO_EXTS = frozenset({".go"})
_RUST_EXTS = frozenset({".rs"})


def _module_to_relpath(repo_root: str, files: list[str]) -> dict[str, str]:
    """Build a ``dotted.module.name -> repo relpath`` index over collected files.

    Python: a file ``pkg/sub/mod.py`` maps to module ``pkg.sub.mod`` (and its
    package dir ``pkg/sub`` to ``pkg.sub``). ``__init__.py`` maps to the bare
    package name. JS/TS/Go/Rust: the file's stem (without extension) maps to a
    module name, plus the path-with-slashes form so relative resolutions can
    find them. We populate several keys per file to maximize match chances.
    """
    root = os.path.abspath(repo_root)
    index: dict[str, str] = {}
    for f in files:
        rel = os.path.relpath(f, root)
        rel_norm = rel.replace(os.sep, "/")
        ext = os.path.splitext(rel)[1].lower()
        stem, _ = os.path.splitext(rel)
        stem_norm = stem.replace(os.sep, "/")
        if ext in _PY_EXTS:
            if rel_norm.endswith("/__init__.py"):
                mod = stem_norm.replace("/", ".")
            else:
                mod = stem_norm.replace("/", ".")
            if mod:
                index.setdefault(mod, rel)
        else:
            # JS/TS/Go/Rust: index by stem and by relpath so both the dotted
            # form and the path form can resolve.
            if stem_norm:
                index.setdefault(stem_norm, rel)
            index.setdefault(rel_norm, rel)
            index.setdefault(stem_norm.replace("/", "."), rel)
    return index


def _resolve_py_imports(node: ast.AST, file_pkg: str) -> list[str]:
    """Collect module names referenced by a Python file.

    ``ast.Import`` -> each alias's ``name``. ``ast.ImportFrom`` -> ``module``
    (with relative ``level`` resolved against the file's package). Returns the
    list of dotted module names to look up in the module->relpath index.
    """
    out: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name:
                    out.append(alias.name)
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            resolved_base = ""
            if n.level and n.level > 0:
                # Relative import: resolve against the file's package. The file
                # ``pkg/use.py`` has module name ``pkg.use`` and lives in package
                # ``pkg`` (file_pkg minus its last segment). level=1 (``.``)
                # refers to that package; level=2 (``..``) to its parent; etc. So
                # we drop ``level`` components from the file's own dotted name.
                parts = file_pkg.split(".") if file_pkg else []
                drop = n.level
                base_parts = parts[: max(0, len(parts) - drop)]
                if base:
                    base_parts.append(base)
                resolved_base = ".".join(p for p in base_parts if p)
            elif base:
                resolved_base = base
            if resolved_base:
                # `from pkg import name` may import a submodule named `name`
                # (i.e. pkg.name) OR a name re-exported by pkg/__init__.py; add
                # both candidates and let the module->relpath index decide.
                out.append(resolved_base)
                for alias in n.names:
                    if alias.name and alias.name != "*":
                        out.append(resolved_base + "." + alias.name)
    return out


def _resolve_relative_spec(spec: str, file_dir_rel: str) -> str | None:
    """Resolve a JS/TS relative specifier (./ or ../) to a repo relpath stem.

    Returns the resolved path-with-slashes stem (no extension) or None for
    bare/external specifiers.
    """
    if not spec.startswith("."):
        return None
    base = file_dir_rel.replace(os.sep, "/")
    parts = base.split("/") if base else []
    for seg in spec.split("/"):
        if seg == "" or seg == ".":
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(p for p in parts if p)


def _extract_edges(rel: str, text: str, file_pkg: str,
                   module_index: dict[str, str]) -> list[str]:
    """Return the list of repo relpaths this file references (edges A -> B).

    Unknown externals produce no edge. Any parse error is caught by the caller.
    """
    ext = os.path.splitext(rel)[1].lower()
    targets: list[str] = []
    if ext in _PY_EXTS:
        try:
            tree = ast.parse(text, filename=rel)
        except (SyntaxError, ValueError, TypeError):
            return []
        for mod in _resolve_py_imports(tree, file_pkg):
            if not mod:
                continue
            tgt = module_index.get(mod)
            if tgt and tgt != rel:
                targets.append(tgt)
        return targets
    if ext in _JS_EXTS:
        file_dir_rel = os.path.dirname(rel)
        specs: list[str] = []
        for line in text.splitlines():
            m = _JS_IMPORT_FROM.match(line)
            if not m:
                m = _JS_SIDE_EFFECT.match(line)
            if m:
                specs.append(m.group(1))
            for rm in _JS_REQUIRE.finditer(line):
                specs.append(rm.group(1))
        for spec in specs:
            resolved = _resolve_relative_spec(spec, file_dir_rel)
            if resolved is None:
                # bare specifier: try the module index directly (a workspace
                # alias import like `from 'components/Foo'` may still resolve).
                tgt = module_index.get(spec) or module_index.get(spec.replace("/", "."))
            else:
                tgt = (module_index.get(resolved) or module_index.get(resolved + "/index")
                       or module_index.get(resolved.replace("/", ".")))
            if tgt and tgt != rel:
                targets.append(tgt)
        return targets
    if ext in _GO_EXTS:
        # Go: build the full import block (the parser gathers lines between
        # `import (` and `)`). Single-line imports too.
        in_block = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ("):
                in_block = True
                continue
            if in_block:
                if stripped.startswith(")"):
                    in_block = False
                    continue
                m = _GO_IMPORT_BLOCK_LINE.match(line)
                if m:
                    path = m.group(1)
                    # Only the last segment is a usable module name locally.
                    seg = path.split("/")[-1]
                    tgt = module_index.get(seg) or module_index.get(path)
                    if tgt and tgt != rel:
                        targets.append(tgt)
                continue
            m = _GO_IMPORT_SINGLE.match(line)
            if m:
                path = m.group(1)
                seg = path.split("/")[-1]
                tgt = module_index.get(seg) or module_index.get(path)
                if tgt and tgt != rel:
                    targets.append(tgt)
        return targets
    if ext in _RUST_EXTS:
        for line in text.splitlines():
            m = _RUST_USE.match(line)
            if not m:
                continue
            path = m.group(1)
            first = path.split("::")[0]
            if first in ("self", "super", "crate"):
                # internal-relative: resolve to current file's crate via index.
                # `crate::foo` maps to this file's stem's first segment.
                parts = rel.replace(os.sep, "/").split("/")
                seg = os.path.splitext(parts[-1])[0]
                tgt = module_index.get(seg)
            else:
                tgt = module_index.get(first) or module_index.get(path)
            if tgt and tgt != rel:
                targets.append(tgt)
        return targets
    return targets


# ---------------------------------------------------------------------------
# Repo signature + cache
# ---------------------------------------------------------------------------

# In-process cache of the built graph snapshot per repo root, mirroring
# code_index._INDEX_CACHE. Keyed by (repo_root_abs, signature) so distinct
# temp workspaces never collide in tests. Bounded to the last _CACHE_MAX
# insertions (LRU by insertion order) so a long-running process editing files
# does not leak one full (nodes, edges, syms) snapshot per distinct signature.
_CACHE_MAX = 8
_CACHE: "OrderedDict[tuple[str, tuple], tuple]" = OrderedDict()


def clear_cache() -> None:
    """Drop the in-process ranking cache (used by tests for isolation)."""
    _CACHE.clear()


def _repo_signature(repo_root: str, files: list[str]) -> tuple:
    """Cheap repo signature: file count + per top-level dir (name, max mtime).

    A file added/removed changes file_count; an edit bumps a top-dir mtime. This
    avoids re-walking+re-parsing on every call within a process while still
    invalidating when the repo shape changes.
    """
    root = os.path.abspath(repo_root)
    top_dirs: dict[str, float] = {}
    for f in files:
        rel = os.path.relpath(f, root)
        top = rel.split(os.sep)[0]
        try:
            mt = os.path.getmtime(f)
        except OSError:
            mt = 0.0
        if mt > top_dirs.get(top, 0.0):
            top_dirs[top] = mt
    return (len(files), tuple(sorted(top_dirs.items())))


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------

def _pagerank(nodes: list[str], edges: dict[str, list[str]],
              personal: dict[str, float], damping: float = 0.85,
              max_iter: int = 40, tol: float = 1e-6) -> dict[str, float]:
    """Pure-Python PageRank via power iteration. No numpy.

    ``edges[A] = [B, ...]`` means A links to B. ``personal`` is the
    personalization vector (sums to 1). Returns {node: score}.
    """
    n = len(nodes)
    if n == 0:
        return {}
    out_deg = {node: len(edges.get(node, [])) for node in nodes}
    # Sum of personalization must be 1; if empty, uniform.
    if not personal or sum(personal.values()) <= 0:
        personal = {node: 1.0 / n for node in nodes}
    else:
        s = sum(personal.values())
        personal = {k: v / s for k, v in personal.items()}
    # Precompute reverse adjacency ONCE so each iteration is O(N+E) instead of
    # O(N*E). incoming[dst] lists (src, out_deg[src]) for every src that links
    # to dst. We dedupe destinations per src (via ``set``) to match the old
    # ``node in edges[src]`` membership test, which counted each src's
    # contribution to a given node exactly once regardless of edge multiplicity.
    # Srcs are visited in ``edges`` iteration order and appended in that order,
    # so per-node summation order (and thus the float result) is identical.
    incoming: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for src in edges:
        od = out_deg.get(src, len(edges[src]))
        if od <= 0:
            continue
        for dst in set(edges[src]):
            incoming[dst].append((src, od))
    rank = {node: 1.0 / n for node in nodes}
    for _ in range(max_iter):
        new_rank: dict[str, float] = {}
        dangling = sum(rank[node] for node in nodes if out_deg[node] == 0)
        dangling_share = damping * (dangling / n)
        for node in nodes:
            acc = dangling_share
            for src, od in incoming.get(node, ()):  # empty tuple -> no incoming
                acc += damping * (rank[src] / od)
            # teleport via personalization
            new_rank[node] = acc + (1 - damping) * personal[node]
        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < tol:
            break
    return rank


def _query_tokens(query: str) -> list[str]:
    """Tokenize a query: lowercase, split on non-alphanumeric. Empty if no query."""
    if not query:
        return []
    return [t for t in re.split(r"[^0-9a-z]+", query.lower()) if t]


def _build_graph(root: str, files: list[str]) -> tuple[list[str], dict[str, list[str]], dict[str, set[str]]]:
    """Read every source file once, extract import edges + symbol sets.

    Returns (nodes, edges, syms_by_file). Per-file parse errors skip that
    file's edges (never raise). ``nodes`` are repo-relative paths.
    """
    module_index = _module_to_relpath(root, files)
    edges: dict[str, list[str]] = {}
    syms_by_file: dict[str, set[str]] = {}
    for f in files:
        rel = os.path.relpath(f, root)
        ext = os.path.splitext(rel)[1].lower()
        if ext not in (_PY_EXTS | _JS_EXTS | _GO_EXTS | _RUST_EXTS):
            continue
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        file_pkg = rel.replace(os.sep, "/")[:-len(ext)].replace("/", ".") \
            if rel.endswith(ext) else rel.replace(os.sep, "/").replace("/", ".")
        try:
            tgts = _extract_edges(rel, text, file_pkg, module_index)
        except Exception:
            tgts = []
        seen: set[str] = set()
        for t in tgts:
            if t != rel and t not in seen:
                seen.add(t)
                edges.setdefault(rel, []).append(t)
        syms_by_file[rel] = _file_symbols(rel, text)
    nodes = [os.path.relpath(f, root) for f in files]
    return nodes, edges, syms_by_file


def rank(files: list[str], query: str, repo_root: str) -> list[str]:
    """Return ``files`` relpaths in ranked order (all files, ranked).

    Side-effect free apart from the in-process cache. The graph (edges +
    symbols) is cached per repo signature; PageRank is re-run per call with a
    personalization vector derived from ``query`` (matching files get x10
    weight, default uniform 1/N). Falls back to flat path-sorted order when the
    graph has <3 nodes or zero edges.
    """
    if not files:
        return []
    root = os.path.abspath(repo_root)
    sig = _repo_signature(root, files)
    key = (root, sig)
    cached = _CACHE.get(key)
    if cached is None:
        nodes, edges, syms = _build_graph(root, files)
        cached = (nodes, edges, syms)
        _CACHE[key] = cached
        # Bound the cache to the last _CACHE_MAX insertions (evict oldest).
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    nodes, edges, syms = cached

    if len(nodes) < 3 or not edges:
        return sorted(nodes)

    personal = _personalization(nodes, syms, query)
    pr = _pagerank(nodes, edges, personal)
    return sorted(nodes, key=lambda r: (-pr.get(r, 0.0), r))


def _personalization(nodes: list[str], syms_by_file: dict[str, set[str]],
                     query: str) -> dict[str, float]:
    """Build the PageRank personalization vector from the query.

    Tokenize the query (lowercase, split on non-alnum). Files whose relpath or
    extracted symbols contain a query token get weight x10; the rest get 1.
    Normalized to sum=1 inside ``_pagerank``. Empty query -> uniform (caller
    passes {} which _pagerank treats as uniform 1/N).
    """
    toks = _query_tokens(query)
    if not toks:
        return {}
    weights: dict[str, float] = {}
    for rel in nodes:
        hay = rel.lower()
        sym_set = syms_by_file.get(rel, set())
        w = 1.0
        for tk in toks:
            if tk in hay or any(tk in s.lower() for s in sym_set):
                w = 10.0
                break
        weights[rel] = w
    return weights


def _file_symbols(rel: str, text: str) -> set[str]:
    """Lightweight top-level symbol set for personalization. Reuses the same
    regex idea as tools._repo_symbols but returns a set (no cap) for matching."""
    ext = os.path.splitext(rel)[1].lower()
    out: set[str] = set()
    if ext == ".py":
        try:
            tree = ast.parse(text, filename=rel)
        except Exception:
            tree = None
        if tree is not None:
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out.add(n.name)
        return out
    # generic line-start regex (good enough for query boosting)
    rx = {
        ".go": re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)|^type\s+([A-Za-z_]\w*)"),
        ".rs": re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_]\w*)"),
    }.get(ext)
    if rx is None and ext in _JS_EXTS:
        rx = re.compile(
            r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)"
            r"|^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="
        )
    if rx is None:
        return out
    for line in text.splitlines():
        m = rx.match(line)
        if m:
            name = next((g for g in m.groups() if g), None)
            if name:
                out.add(name)
    return out


