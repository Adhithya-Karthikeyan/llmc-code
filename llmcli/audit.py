"""Map-reduce project audit.

A "full audit" is the worst case for context size: reviewing the whole project
in ONE agent loop pulls every file into a single context, which balloons and
slows every generated token (decode cost scales with context length).

This module does it the scalable way instead:
  MAP    — split the repo into small chunks; review each chunk in its OWN
           throwaway sub-agent context (it reads only its chunk, then that
           context is discarded — it never touches the main conversation).
  REDUCE — feed only the short per-chunk findings (not file contents) to one
           synthesis pass that merges + dedupes them into a single report.

So each worker stays small and fast regardless of project size, and the main
REPL conversation is never bloated by the audit at all.
"""

from __future__ import annotations

import os
from pathlib import Path

from .agent import Agent
from .config import Config
from .providers import Provider
from .prompts import role_prompt
from .tools import (
    READ_ONLY,
    _count_lines,
    _is_skip_path,
    _iter_source_files,
    _looks_binary,
    _within_workspace,
    tool_subset,
)

# Bounds so an audit can never run away on a huge repo.
_AUDIT_MAX_FILES = 250        # total files considered
_AUDIT_CHUNK_LINES = 1500     # ~source lines per worker chunk (keeps each small)
_AUDIT_MAX_CHUNKS = 24        # cap workers (sequential on one local model)

# Audit workers never need the network — drop web_fetch so there is no egress
# surface during an audit (in any mode), regardless of the global tool set.
AUDIT_TOOLS = [n for n in READ_ONLY if n != "web_fetch"]


def discover_files(root: Path) -> list[tuple[str, int]]:
    """``(cwd-relative path, line_count)`` for auditable text files under ``root``.

    Paths are relative to the CURRENT WORKING DIRECTORY (not ``root``), because
    the worker sub-agents' ``read_file`` resolves paths against cwd and confines
    to the workspace — so a subdir audit (``/audit llmcli``) must hand workers
    ``llmcli/x.py``, not the root-relative ``x.py`` (which would be unreadable).
    Prunes VCS/vendor/build dirs (walk-time), binary/minified files, and any file
    that sniffs binary or sits outside the workspace.
    """
    cwd = os.getcwd()
    out: list[tuple[str, int]] = []
    items = [root] if root.is_file() else _iter_source_files(root)
    for p in items:
        if not _within_workspace(p) or _is_skip_path(p) or _looks_binary(p):
            continue
        out.append((os.path.relpath(str(p), cwd), _count_lines(p)))
    return out


def chunk_files(
    files: list[tuple[str, int]],
    chunk_lines: int = _AUDIT_CHUNK_LINES,
    max_chunks: int = _AUDIT_MAX_CHUNKS,
) -> list[list[tuple[str, int]]]:
    """Greedily pack files into chunks under a per-chunk line budget.

    A single file larger than the budget becomes its own chunk (the reviewer
    reads it in slices). Returns at most ``max_chunks`` chunks.
    """
    chunks: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_lines = 0
    for rel, n in files:
        if cur and cur_lines + n > chunk_lines:
            chunks.append(cur)
            cur, cur_lines = [], 0
        cur.append((rel, n))
        cur_lines += n
    if cur:
        chunks.append(cur)
    return chunks[:max_chunks]


def _chunk_task(files: list[tuple[str, int]], focus: str) -> str:
    names = ", ".join(rel for rel, _ in files)
    focus_line = f" Focus especially on: {focus}." if focus else ""
    return (
        "Audit ONLY these files for REAL issues (bugs, broken logic, security "
        f"risks, resource leaks, clear smells).{focus_line} Read each with "
        "read_file (use offset/limit on big files). Files: " + names + ".\n"
        "Return a SHORT prioritized list — for each issue: `file:line` — "
        "severity (high/med/low) — one-line problem — one-line fix. "
        "If you find nothing real, reply exactly: No issues found."
    )


def run_audit(
    provider: Provider,
    config: Config,
    console,
    *,
    accent: str | None = None,
    code_theme: str = "monokai",
    path: str = ".",
    focus: str = "",
) -> str:
    """Run the map-reduce audit and return the merged report (also printed).

    Worker sub-agents run with ``console`` (so the spinner + dim progress show)
    but a nested ``line_prefix`` (so their answers render plain, not as the main
    "▌ Answer" block). The final synthesis prints as the real answer.
    """
    root = Path(path).expanduser()
    if not root.exists():
        console.print(f"[audit] path not found: {path}", style="red")
        return ""
    if not _within_workspace(root):
        console.print(
            f"[audit] refusing to audit outside the workspace: {path}", style="red"
        )
        return ""
    all_files = discover_files(root)
    if not all_files:
        console.print("[audit] no auditable source files found.", style="dim")
        return ""
    dropped_files = max(0, len(all_files) - _AUDIT_MAX_FILES)
    files = all_files[:_AUDIT_MAX_FILES]
    chunks = chunk_files(files)
    covered = sum(len(c) for c in chunks)
    dropped_chunks = len(files) - covered
    total_lines = sum(n for _, n in files)
    console.print(
        f"[audit] {len(files)} files (~{total_lines} lines) → {len(chunks)} "
        f"chunk(s), each reviewed in its own small context.",
        style=accent or "dim",
    )
    # Be HONEST about dropped coverage instead of silently reporting "clean".
    if dropped_files:
        console.print(
            f"[audit] WARNING: {dropped_files} more file(s) exceeded the "
            f"{_AUDIT_MAX_FILES}-file cap and were NOT reviewed — narrow with "
            f"/audit <subdir>.",
            style="red",
        )
    if dropped_chunks:
        console.print(
            f"[audit] WARNING: {dropped_chunks} file(s) beyond the "
            f"{_AUDIT_MAX_CHUNKS}-chunk cap were NOT reviewed — narrow with "
            f"/audit <subdir>.",
            style="red",
        )

    workspace = os.getcwd()
    registry = tool_subset(AUDIT_TOOLS)
    findings: list[tuple[int, list[str], str]] = []
    failed_chunks: list[int] = []
    for i, chunk in enumerate(chunks, 1):
        rels = [rel for rel, _ in chunk]
        console.print(
            f"[audit] chunk {i}/{len(chunks)} — {len(chunk)} file(s)", style="dim"
        )
        worker = Agent(
            provider=provider,
            system_prompt=role_prompt("reviewer", workspace=workspace),
            tool_names=AUDIT_TOOLS,
            registry=registry,
            console=console,
            auto_confirm=True,
            max_iterations=config.max_iterations,
            code_theme=code_theme,
            # Nested marker => plain (non-gutter) output; its big read context is
            # this worker's alone and is discarded when it returns.
            line_prefix="  ",
        )
        try:
            summary = worker.run(_chunk_task(chunk, focus))
        except KeyboardInterrupt:
            console.print("\n[audit] interrupted.", style="red")
            break
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not abort the audit
            failed_chunks.append(i)
            console.print(
                f"[audit] chunk {i} error: {type(exc).__name__}: {exc}", style="dim"
            )
            continue
        findings.append((i, rels, summary))

    if failed_chunks:
        console.print(
            f"[audit] WARNING: {len(failed_chunks)} chunk(s) failed and were NOT reviewed",
            style="red",
        )

    if not findings:
        return ""

    # Single chunk: the worker summary IS the report — no synthesis needed.
    if len(findings) == 1:
        _, _, summary = findings[0]
        console.print(summary, style=accent or "dim")
        return summary

    console.print(
        f"[audit] merging findings from {len(findings)} chunk(s)…", style="dim"
    )
    merged_input = "\n\n".join(
        f"## Chunk {i} — {', '.join(rels)}\n{summary}"
        for i, rels, summary in findings
    )
    synth = Agent(
        provider=provider,
        system_prompt=role_prompt("reviewer", workspace=workspace),
        tool_names=[],          # pure synthesis over the short summaries — no tools
        registry={},
        console=console,
        auto_confirm=True,
        # 2 (not 1) so the empty-turn nudge can land for a reasoning-only model
        # and still produce the final report (no tools => it cannot loop).
        max_iterations=2,
        code_theme=code_theme,
        accent=accent,          # final report renders as the real "▌ Answer" block
    )
    report = synth.run(
        "Merge these per-chunk audit findings into ONE deduplicated, prioritized "
        "report (highest severity first). Drop duplicates and noise; keep concrete "
        "`file:line` references and the one-line fixes. If everything was clean, "
        "say so plainly.\n\n" + merged_input
    )
    return report
