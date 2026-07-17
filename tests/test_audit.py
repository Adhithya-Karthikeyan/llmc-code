"""Tests for the map-reduce /audit (chunking + an end-to-end smoke run)."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from llmcli.audit import chunk_files, discover_files, run_audit
from llmcli.config import Config
from llmcli.providers import MockProvider


class _TextOnly(MockProvider):
    """A provider that returns one fixed text answer and calls no tools — so the
    audit loop runs deterministically without a real model."""

    def __init__(self, text="No issues found."):
        super().__init__()
        self._text = text

    def stream_chat(self, messages, tools):
        yield {"type": "text", "text": self._text}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}


# ----- pure helpers --------------------------------------------------------

def test_discover_files_prunes_and_counts(tmp_workspace):
    (tmp_workspace / "a.py").write_text("x\ny\nz\n", encoding="utf-8")
    (tmp_workspace / "b.txt").write_text("one\n", encoding="utf-8")
    venv = tmp_workspace / ".venv"
    venv.mkdir()
    (venv / "junk.py").write_text("nope\n", encoding="utf-8")
    (tmp_workspace / "img.png").write_bytes(b"\x89PNG")
    found = dict(discover_files(Path(".")))
    assert found.get("a.py") == 3
    assert found.get("b.txt") == 1
    assert "img.png" not in found  # binary skipped
    assert all(".venv" not in k for k in found)  # vendor pruned


def test_chunk_files_packs_under_line_budget():
    files = [("a", 600), ("b", 600), ("c", 600), ("d", 600)]
    chunks = chunk_files(files, chunk_lines=1500, max_chunks=10)
    # 600+600=1200 ok; +600 would be 1800>1500 -> new chunk. So [a,b][c,d].
    assert [len(c) for c in chunks] == [2, 2]


def test_chunk_files_oversized_single_file_is_own_chunk():
    files = [("big", 5000), ("small", 10)]
    chunks = chunk_files(files, chunk_lines=1500, max_chunks=10)
    assert chunks[0] == [("big", 5000)]


def test_chunk_files_caps_chunks():
    files = [(f"f{i}", 2000) for i in range(30)]  # each its own chunk
    chunks = chunk_files(files, chunk_lines=1500, max_chunks=5)
    assert len(chunks) == 5


# ----- end-to-end smoke ----------------------------------------------------

def test_run_audit_smoke_returns_report(tmp_workspace):
    (tmp_workspace / "x.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_workspace / "y.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, markup=False, highlight=False)
    report = run_audit(
        _TextOnly("No issues found."), Config(provider="mock"), console, path="."
    )
    assert isinstance(report, str) and report.strip()
    out = buf.getvalue()
    assert "[audit]" in out          # progress shown
    assert "chunk" in out.lower()


def test_discover_subdir_paths_are_cwd_relative_and_readable(tmp_workspace):
    """Regression for the /audit <subdir> bug: discovered paths must be
    cwd-relative so the worker's read_file (resolves against cwd) can open them."""
    from llmcli.tools import _read_file

    sub = tmp_workspace / "pkg"
    sub.mkdir()
    (sub / "x.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    found = discover_files(Path("pkg"))
    assert found == [("pkg/x.py", 2)]
    # the exact path handed to a worker MUST be readable from cwd
    rel = found[0][0]
    assert _read_file({"path": rel})["ok"] is True


def test_audit_tools_exclude_web_fetch():
    from llmcli.audit import AUDIT_TOOLS

    assert "web_fetch" not in AUDIT_TOOLS
    assert "read_file" in AUDIT_TOOLS and "repo_map" in AUDIT_TOOLS


def test_run_audit_refuses_outside_workspace(tmp_workspace):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, markup=False, highlight=False)
    report = run_audit(_TextOnly(), Config(provider="mock"), console, path="/etc")
    assert report == ""
    assert "outside the workspace" in buf.getvalue()


def test_run_audit_warns_when_file_cap_drops_coverage(tmp_workspace, monkeypatch):
    import llmcli.audit as a

    monkeypatch.setattr(a, "_AUDIT_MAX_FILES", 1)
    for i in range(3):
        (tmp_workspace / f"m{i}.py").write_text("x = 1\n", encoding="utf-8")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, markup=False, highlight=False)
    run_audit(_TextOnly(), Config(provider="mock"), console, path=".")
    out = buf.getvalue()
    assert "WARNING" in out and "NOT reviewed" in out


def test_run_audit_no_files_is_graceful(tmp_workspace):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, markup=False, highlight=False)
    # only a pruned dir + binary -> nothing auditable
    (tmp_workspace / "logo.png").write_bytes(b"\x89PNG")
    report = run_audit(_TextOnly(), Config(provider="mock"), console, path=".")
    assert report == ""
    assert "no auditable" in buf.getvalue().lower()


def test_run_audit_single_chunk_skips_synthesis(tmp_workspace, monkeypatch):
    """ORCH-1: one chunk -> worker summary returned directly, no second Agent call."""
    import llmcli.audit as a

    call_count = 0

    class _CountingProvider(MockProvider):
        def stream_chat(self, messages, tools):
            nonlocal call_count
            call_count += 1
            yield {"type": "text", "text": "Only finding."}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}

    # Force at most 1 chunk so the single-chunk path is exercised
    monkeypatch.setattr(a, "_AUDIT_MAX_CHUNKS", 1)
    (tmp_workspace / "z.py").write_text("x = 1\n", encoding="utf-8")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, markup=False, highlight=False)
    report = run_audit(_CountingProvider(), Config(provider="mock"), console, path=".")
    assert report == "Only finding."
    # Only the single worker call — no second synthesis Agent call.
    assert call_count == 1


def test_run_audit_failed_chunk_warns_and_excluded(tmp_workspace, monkeypatch):
    """ORCH-4: failed chunk prints WARNING and is not passed to synthesis."""
    import llmcli.audit as a

    class _RaisingProvider(MockProvider):
        def stream_chat(self, messages, tools):
            raise RuntimeError("boom")

    # Force 1 chunk so the single failing chunk covers all files
    monkeypatch.setattr(a, "_AUDIT_MAX_CHUNKS", 1)
    (tmp_workspace / "z.py").write_text("x = 1\n", encoding="utf-8")
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, markup=False, highlight=False)
    report = run_audit(_RaisingProvider(), Config(provider="mock"), console, path=".")
    out = buf.getvalue()
    assert "WARNING" in out
    assert "NOT reviewed" in out
    # No successful findings -> nothing fed to synth -> empty report
    assert report == ""
