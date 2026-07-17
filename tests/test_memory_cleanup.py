"""MemoryStore cleanup (purge/compact) tests.

Tests the new `purge()` and `compact()` methods added to MemoryStore:
  - purge() with no args: removes ALL records + vectors
  - purge(ids): selectively removes specific records by ID
  - compact(): reduces store to a specified max (default MAX_RECORDS)
  - Both set _dirty so the next save() persists changes
  - Both drop orphaned vectors
  - Both are no-ops when nothing to do
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llmcli.memory import MemoryStore, MAX_RECORDS, _tokenize


# --------------------------------------------------------------------------- #
# purge()
# --------------------------------------------------------------------------- #


def test_purge_all_clears_everything():
    store = MemoryStore()
    store.add("doc one", summary="one")
    store.add("doc two", summary="two")
    store.add("doc three", summary="three")
    assert len(store.records) == 3

    removed = store.purge()
    assert removed == 3
    assert len(store.records) == 0
    assert len(store.vectors) == 0
    assert store._bm25_tokens is None
    assert store._dirty is True


def test_purge_selective_removes_matching_ids():
    store = MemoryStore()
    r0 = store.add("alpha", summary="a")
    r1 = store.add("beta", summary="b")
    r2 = store.add("gamma", summary="c")

    removed = store.purge([r0.id, r2.id])
    assert removed == 2
    assert len(store.records) == 1
    assert store.records[0].id == r1.id


def test_purge_nonexistent_ids_is_noop():
    store = MemoryStore()
    store.add("only doc", summary="o")
    # add() sets _dirty=True. We need to reset it to verify purge doesn't
    # re-dirty when nothing is removed.
    store._dirty = False

    removed = store.purge(["r99"])
    assert removed == 0
    assert len(store.records) == 1
    assert not store._dirty  # no change -> not dirtied


def test_purge_mixed_ids():
    store = MemoryStore()
    r0 = store.add("alpha", summary="a")
    r1 = store.add("beta", summary="b")
    # After adding 2 records, _dirty is True. Reset to verify behavior.
    store._dirty = False

    removed = store.purge([r0.id, "r99"])  # one real, one fake
    assert removed == 1
    assert len(store.records) == 1
    assert store.records[0].id == r1.id


# --------------------------------------------------------------------------- #
# compact()
# --------------------------------------------------------------------------- #


def test_compact_default_max_records():
    store = MemoryStore()
    # MAX_RECORDS = 500, but add() also evicts at MAX_RECORDS.
    for i in range(MAX_RECORDS + 10):
        store.add(f"doc {i}", summary=f"d{i}")
    # After add(), store has exactly MAX_RECORDS (500) due to _evict_overflow
    assert len(store.records) == MAX_RECORDS
    # Now compact with a tighter limit
    removed = store.compact(max_records=490)
    assert removed == 10
    assert len(store.records) == 490
    assert store._dirty is True


def test_compact_below_limit_is_noop():
    store = MemoryStore()
    for i in range(5):
        store.add(f"doc {i}", summary=f"d{i}")
    # add() sets _dirty=True; reset to verify compact doesn't re-dirty
    store._dirty = False

    removed = store.compact(max_records=10)
    assert removed == 0
    assert len(store.records) == 5
    assert not store._dirty


def test_compact_sets_dirty_only_on_change():
    store = MemoryStore()
    store.add("doc", summary="d")
    # add() sets _dirty=True; reset to verify compact behavior
    store._dirty = False

    # No change -> not dirtied
    removed = store.compact()
    assert removed == 0
    assert not store._dirty

    # Force a change: compact to 499 (currently 500 after re-add)
    for i in range(MAX_RECORDS + 5):
        store.add(f"extra {i}", summary=f"e{i}")
    # Now we have 500 records; compact to 499
    removed = store.compact(max_records=499)
    assert removed == 1
    assert store._dirty


def test_compact_empty_store_is_safe():
    store = MemoryStore()
    removed = store.compact()
    assert removed == 0
    assert len(store.records) == 0


def test_compact_drops_orphan_vectors():
    store = MemoryStore()
    for i in range(10):
        store.add(f"doc {i}", summary=f"d{i}")
    # Manually add some vectors
    for r in store.records:
        store.vectors[r.content_hash] = [0.1] * 768

    removed = store.compact(max_records=3)
    assert removed == 7
    # Only the last 3 records' vectors remain
    live_hashes = {r.content_hash for r in store.records}
    for h in store.vectors:
        assert h in live_hashes


# --------------------------------------------------------------------------- #
# save/load after cleanup
# --------------------------------------------------------------------------- #


def test_save_load_after_purge():
    tmp = Path(tempfile.mkdtemp())
    store = MemoryStore()
    store.add("doc one", summary="one")
    store.add("doc two", summary="two")
    store.add("doc three", summary="three")
    store.purge()  # remove all
    store.save(tmp / "mem.json")

    loaded = MemoryStore.load(tmp / "mem.json")
    assert len(loaded.records) == 0


def test_save_load_after_compact():
    tmp = Path(tempfile.mkdtemp())
    store = MemoryStore()
    for i in range(15):
        store.add(f"doc {i}", summary=f"d{i}")
    store.compact(max_records=5)
    store.save(tmp / "mem.json")

    loaded = MemoryStore.load(tmp / "mem.json")
    assert len(loaded.records) == 5
    assert loaded.records[0].id == "r10"
    assert loaded.records[-1].id == "r14"
