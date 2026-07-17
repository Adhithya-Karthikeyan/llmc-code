"""Shared pytest fixtures. All offline: no network, no openai import."""

from __future__ import annotations

import os

import pytest

from llmcli.providers import MockProvider


@pytest.fixture(autouse=True)
def reset_global_state(monkeypatch):
    """Reset all module-level mutable global state before every test.

    This ensures test isolation when tests run in a shared process.
    Without this, a test that modifies global state (e.g. populates
    CodeIndex's in-process cache) can affect a later test that assumes
    a fresh state.
    """
    # Reset the code index process-wide cache (per-workspace dict).
    from llmcli import code_index
    code_index._INDEX_CACHE.clear()


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """Provide a temp dir and chdir into it for the duration of the test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def mock_provider():
    """Factory-ish fixture returning a fresh MockProvider (default 'hello')."""
    def _make(scenario: str = "hello") -> MockProvider:
        return MockProvider(scenario=scenario)
    return _make
