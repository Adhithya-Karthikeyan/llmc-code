"""/model verifies the name against the server's model list.

Regression guard: a typo like `/model fkbf` must NOT switch to a non-existent
model. No network: list_local_models is monkeypatched.
"""

from __future__ import annotations

import llmcli.repl as repl
from llmcli.config import Config
from llmcli.providers import MockProvider


def _repl(monkeypatch, provider="local"):
    monkeypatch.setattr(repl, "save_config", lambda *a, **k: None)
    cfg = Config(provider=provider, model="qwen/x", base_url="http://localhost:1234/v1")
    return repl.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_model_rejects_unknown(monkeypatch):
    monkeypatch.setattr(repl, "list_local_models", lambda *a, **k: ["good", "qwen/x"])
    r = _repl(monkeypatch)
    r._dispatch_slash("/model fkbf")
    assert r.config.model == "qwen/x"  # unchanged — rejected


def test_model_accepts_known(monkeypatch):
    monkeypatch.setattr(repl, "list_local_models", lambda *a, **k: ["good", "qwen/x"])
    r = _repl(monkeypatch)
    r._dispatch_slash("/model good")
    assert r.config.model == "good"


def test_model_allows_when_server_unreachable(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("server down")

    monkeypatch.setattr(repl, "list_local_models", boom)
    r = _repl(monkeypatch)
    r._dispatch_slash("/model whatever")
    assert r.config.model == "whatever"  # can't verify -> allowed with a warning


def test_model_mock_provider_skips_server_check(monkeypatch):
    calls = []
    monkeypatch.setattr(repl, "list_local_models", lambda *a, **k: calls.append(1) or [])
    r = _repl(monkeypatch, provider="mock")
    r._dispatch_slash("/model anything")
    assert r.config.model == "anything"
    assert calls == []  # mock ignores the model; never queried the server


def test_effort_sets_config_and_rebuilds_provider(monkeypatch):
    r = _repl(monkeypatch, provider="mock")
    r._dispatch_slash("/effort high")
    assert r.config.effort == "high"

    # An invalid level is rejected and leaves the config unchanged.
    r._dispatch_slash("/effort bogus")
    assert r.config.effort == "high"


def test_effort_rebuilds_local_provider_with_level(monkeypatch):
    monkeypatch.setattr(repl, "list_local_models", lambda *a, **k: ["qwen/x"])
    r = _repl(monkeypatch, provider="local")
    r._dispatch_slash("/effort low")
    assert r.config.effort == "low"
    # The rebuilt local provider carries the new effort level.
    assert r.provider.effort == "low"
