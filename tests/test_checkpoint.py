"""Tests for llmcli/checkpoint.py — the git-free file-snapshot / undo safety net.

All offline/deterministic: a temp HOME (so we never touch the real
~/.llm-cli/checkpoints) and tmp_path project roots. Covers snapshot+undo of a
modified file, of a newly-created file (undo deletes it), LIFO undo ordering,
the MAX_CHECKPOINTS history bound (oldest evicted), binary safety, the no-op
"nothing to undo" message, path confinement (never escapes root), and the
list/clear helpers.
"""

from __future__ import annotations

import pytest

import llmcli.checkpoint as c
import llmcli.session as s


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    """Point Path.home() at a temp dir so checkpoints land under tmp, not real HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(c.Path, "home", classmethod(lambda cls: home))
    # session.session_id reads Path.home() indirectly only via session module's
    # Path; point it at the same temp home for consistency.
    monkeypatch.setattr(s.Path, "home", classmethod(lambda cls: home))
    return home


@pytest.fixture()
def root(tmp_path):
    """A project root directory used for snapshot/undo."""
    r = tmp_path / "proj"
    r.mkdir()
    return r


# --------------------------------------------------------------------------- #
# Storage location + project keying
# --------------------------------------------------------------------------- #

def test_project_dir_keyed_by_session_id(root):
    assert c.project_dir(str(root)).name == s.session_id(str(root))
    assert c.checkpoints_dir() in c.project_dir(str(root)).parents


def test_snapshot_returns_id_and_creates_storage(root):
    f = root / "a.txt"
    f.write_text("hello")
    ck = c.snapshot([str(f)], root=str(root), label="edit a")
    assert isinstance(ck, str) and ck.startswith("ck-")
    assert c._index_path(str(root)).exists()


# --------------------------------------------------------------------------- #
# Modified file: undo restores the old content
# --------------------------------------------------------------------------- #

def test_undo_restores_modified_file(root):
    f = root / "a.txt"
    f.write_text("original")
    c.snapshot([str(f)], root=str(root))
    f.write_text("MUTATED")  # simulate the tool overwriting it

    result = c.undo(str(root))

    assert result["undone"] is True
    assert "a.txt" in result["restored"]
    assert f.read_text() == "original"
    # checkpoint was popped
    assert c.list_checkpoints(str(root)) == []


# --------------------------------------------------------------------------- #
# Newly-created file: undo deletes it
# --------------------------------------------------------------------------- #

def test_undo_deletes_newly_created_file(root):
    f = root / "new.txt"
    # snapshot BEFORE the file exists -> existed=False
    c.snapshot([str(f)], root=str(root))
    assert not f.exists()
    f.write_text("brand new")  # simulate write_file creating it

    result = c.undo(str(root))

    assert result["undone"] is True
    assert "new.txt" in result["deleted"]
    assert not f.exists()


def test_undo_absent_file_already_gone_is_noop_not_error(root):
    f = root / "ghost.txt"
    c.snapshot([str(f)], root=str(root))  # existed=False, file never created
    result = c.undo(str(root))
    assert result["undone"] is True
    assert result["errors"] == []
    assert not f.exists()


# --------------------------------------------------------------------------- #
# LIFO: multiple checkpoints undo most-recent first
# --------------------------------------------------------------------------- #

def test_multiple_checkpoints_undo_in_lifo_order(root):
    f = root / "a.txt"
    f.write_text("v0")
    c.snapshot([str(f)], root=str(root), label="cp0", timestamp=1.0)
    f.write_text("v1")
    c.snapshot([str(f)], root=str(root), label="cp1", timestamp=2.0)
    f.write_text("v2")
    c.snapshot([str(f)], root=str(root), label="cp2", timestamp=3.0)
    f.write_text("v3-current")

    # newest first in listing
    listed = c.list_checkpoints(str(root))
    assert [x["label"] for x in listed] == ["cp2", "cp1", "cp0"]

    r1 = c.undo(str(root))          # undo cp2 -> back to v2
    assert r1["label"] == "cp2"
    assert f.read_text() == "v2"

    r2 = c.undo(str(root))          # undo cp1 -> back to v1
    assert r2["label"] == "cp1"
    assert f.read_text() == "v1"

    r3 = c.undo(str(root))          # undo cp0 -> back to v0
    assert r3["label"] == "cp0"
    assert f.read_text() == "v0"

    assert c.undo(str(root))["undone"] is False


# --------------------------------------------------------------------------- #
# History bound: oldest evicted beyond MAX_CHECKPOINTS
# --------------------------------------------------------------------------- #

def test_history_bound_evicts_oldest(root, monkeypatch):
    monkeypatch.setattr(c, "MAX_CHECKPOINTS", 3)
    f = root / "a.txt"
    ids = []
    for i in range(5):
        f.write_text(f"v{i}")
        ids.append(c.snapshot([str(f)], root=str(root), label=f"cp{i}", timestamp=float(i)))

    listed = c.list_checkpoints(str(root))
    assert len(listed) == 3
    kept_ids = {x["id"] for x in listed}
    assert kept_ids == set(ids[2:])          # only the last 3 survive
    assert ids[0] not in kept_ids            # oldest evicted

    # evicted blobs are gone; kept blobs remain
    blobs = list(c._blobs_dir(str(root)).glob("*.blob"))
    assert all(not b.name.startswith(ids[0]) for b in blobs)
    assert len(blobs) == 3


# --------------------------------------------------------------------------- #
# Binary safety
# --------------------------------------------------------------------------- #

def test_binary_file_round_trips(root):
    blob = bytes(range(256)) * 4  # non-UTF-8, includes NULs
    f = root / "img.bin"
    f.write_bytes(blob)
    c.snapshot([str(f)], root=str(root))
    f.write_bytes(b"\x00corrupted\x00")

    c.undo(str(root))
    assert f.read_bytes() == blob


# --------------------------------------------------------------------------- #
# No-op undo message
# --------------------------------------------------------------------------- #

def test_undo_nothing_is_actionable(root):
    result = c.undo(str(root))
    assert result["undone"] is False
    assert result["id"] is None
    assert "nothing to undo" in result["message"].lower()


# --------------------------------------------------------------------------- #
# Confinement: never escapes root
# --------------------------------------------------------------------------- #

def test_snapshot_rejects_path_outside_root(root, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("do not touch")
    with pytest.raises(ValueError):
        c.snapshot([str(outside)], root=str(root))
    with pytest.raises(ValueError):
        c.snapshot(["../secret.txt"], root=str(root))
    # nothing was written
    assert c.list_checkpoints(str(root)) == []


def test_snapshot_rejects_traversal_and_leaves_no_partial(root):
    good = root / "a.txt"
    good.write_text("keep")
    # a batch containing an escaping path must fail atomically (no partial cp)
    with pytest.raises(ValueError):
        c.snapshot([str(good), "../../etc/passwd"], root=str(root))
    assert c.list_checkpoints(str(root)) == []


def test_snapshot_rejects_directory(root):
    d = root / "subdir"
    d.mkdir()
    with pytest.raises(ValueError):
        c.snapshot([str(d)], root=str(root))


def test_undo_ignores_tampered_out_of_root_entry(root, tmp_path, monkeypatch):
    # Simulate a tampered index whose entry points outside root; undo must skip
    # it (record an error) and NEVER touch the outside file.
    victim = tmp_path / "victim.txt"
    victim.write_text("safe")
    f = root / "a.txt"
    f.write_text("orig")
    c.snapshot([str(f)], root=str(root))

    idx = c._load_index(str(root))
    idx[-1]["files"].append({"path": "../victim.txt", "existed": False, "blob": None})
    c._save_index(str(root), idx)

    result = c.undo(str(root))
    assert victim.exists() and victim.read_text() == "safe"
    assert any("out-of-root" in e for e in result["errors"])


# --------------------------------------------------------------------------- #
# Relative-path input + list/clear helpers
# --------------------------------------------------------------------------- #

def test_relative_path_input_resolved_against_root(root):
    (root / "rel.txt").write_text("R0")
    c.snapshot(["rel.txt"], root=str(root))   # relative to root
    (root / "rel.txt").write_text("R1")
    c.undo(str(root))
    assert (root / "rel.txt").read_text() == "R0"


def test_clear_removes_all_checkpoints(root):
    f = root / "a.txt"
    f.write_text("x")
    c.snapshot([str(f)], root=str(root))
    c.snapshot([str(f)], root=str(root))
    assert c.list_checkpoints(str(root))
    c.clear(str(root))
    assert c.list_checkpoints(str(root)) == []
    assert not c.project_dir(str(root)).exists()


# --------------------------------------------------------------------------- #
# Session scoping: per-session isolation (a fresh session sees nothing)
# --------------------------------------------------------------------------- #

def test_session_subdir_under_project_dir(root):
    base = c.project_dir(str(root))
    scoped = c.project_dir(str(root), "A")
    assert scoped == base / "sessions" / "A"
    assert base in scoped.parents


def test_session_scoped_snapshots_are_isolated(root):
    f = root / "a.txt"
    f.write_text("orig")
    # a snapshot recorded under session "A"
    c.snapshot([str(f)], root=str(root), session="A", label="in-A")

    # session "B" must NOT see A's checkpoint
    assert c.list_checkpoints(str(root), session="B") == []
    res_b = c.undo(str(root), session="B")
    assert res_b["undone"] is False
    assert "nothing to undo" in res_b["message"].lower()

    # session "A" still sees its own and can undo it
    assert [x["label"] for x in c.list_checkpoints(str(root), session="A")] == ["in-A"]
    f.write_text("MUTATED")
    res_a = c.undo(str(root), session="A")
    assert res_a["undone"] is True
    assert f.read_text() == "orig"


def test_fresh_session_reports_nothing_to_undo(root):
    f = root / "a.txt"
    f.write_text("x")
    c.snapshot([str(f)], root=str(root), session="old-session")
    # a brand-new session token starts empty
    res = c.undo(str(root), session="new-session")
    assert res["undone"] is False
    assert res["id"] is None


def test_session_scoped_does_not_leak_into_default_layout(root):
    f = root / "a.txt"
    f.write_text("x")
    c.snapshot([str(f)], root=str(root), session="A")
    # session-less (legacy) view sees nothing recorded by session "A"
    assert c.list_checkpoints(str(root)) == []
    assert c.undo(str(root))["undone"] is False


# --------------------------------------------------------------------------- #
# Regression guard: None-session paths are byte-identical to the legacy layout
# --------------------------------------------------------------------------- #

def test_none_session_paths_unchanged(root):
    # explicit None must equal the no-arg call, with NO "sessions" component
    assert c.project_dir(str(root), None) == c.project_dir(str(root))
    assert c.project_dir(str(root)) == c.checkpoints_dir() / s.session_id(str(root))
    assert "sessions" not in c.project_dir(str(root)).parts
    assert c._index_path(str(root), None) == c._index_path(str(root))
    assert c._blobs_dir(str(root), None) == c._blobs_dir(str(root))


def test_none_session_snapshot_writes_legacy_location(root):
    f = root / "a.txt"
    f.write_text("hi")
    c.snapshot([str(f)], root=str(root))  # no session
    # index + blobs live directly under the project dir (no sessions/ level)
    assert (c.project_dir(str(root)) / "index.json").exists()
    assert not (c.project_dir(str(root)) / "sessions").exists()


# --------------------------------------------------------------------------- #
# Orphan-blob cleanup: a failed _save_index leaves no blobs behind
# --------------------------------------------------------------------------- #

def test_snapshot_cleans_orphan_blobs_when_index_save_fails(root, monkeypatch):
    f = root / "a.txt"
    f.write_text("content")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(c, "_save_index", _boom)
    ck = c.snapshot([str(f)], root=str(root))  # must not raise

    # no blob for this checkpoint should be left orphaned on disk
    blobs = c._blobs_dir(str(root))
    leftover = list(blobs.glob(f"{ck}-*.blob")) if blobs.exists() else []
    assert leftover == []


def test_snapshot_cleans_orphan_blobs_when_index_save_returns_false(root, monkeypatch):
    # _save_index signalling failure (return False, no raise) also triggers cleanup
    f = root / "a.txt"
    f.write_text("content")
    monkeypatch.setattr(c, "_save_index", lambda *a, **k: False)
    ck = c.snapshot([str(f)], root=str(root))
    blobs = c._blobs_dir(str(root))
    leftover = list(blobs.glob(f"{ck}-*.blob")) if blobs.exists() else []
    assert leftover == []


def test_multi_file_checkpoint(root):
    a = root / "a.txt"
    b = root / "b.txt"
    a.write_text("A0")
    # b does not exist yet
    c.snapshot([str(a), str(b)], root=str(root))
    a.write_text("A1")
    b.write_text("B-new")

    result = c.undo(str(root))
    assert a.read_text() == "A0"
    assert not b.exists()
    assert "a.txt" in result["restored"]
    assert "b.txt" in result["deleted"]
