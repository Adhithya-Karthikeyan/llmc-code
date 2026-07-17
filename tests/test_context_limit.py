"""Auto-compaction budget tracks the loaded model's real context window.

Regression for: compaction firing at 24k while qwen has a 262,144 (256k) window.
"""

from __future__ import annotations

from llmcli.config import Config
from llmcli.providers import _pick_context_length
from llmcli.repl import _effective_soft_limit


class _Prov:
    def __init__(self, base_url, model, ctx="unset"):
        self.base_url = base_url
        self.model = model
        if ctx != "unset":
            self._ctx_len = ctx


def test_pick_context_length():
    models = [
        {"id": "qwen/qwen3.6-35b-a3b", "state": "loaded", "loaded_context_length": 262144},
        {"id": "google/gemma-3-1b", "state": "loaded", "loaded_context_length": 32768},
    ]
    assert _pick_context_length(models, "qwen/qwen3.6-35b-a3b") == 262144  # exact id
    assert _pick_context_length(models, "google/gemma-3-1b") == 32768
    assert _pick_context_length(models, "not-loaded") == 262144  # falls back to a loaded one
    assert _pick_context_length([], "x") is None
    assert _pick_context_length(None, "x") is None
    # bool must not be treated as an int context length
    assert _pick_context_length([{"id": "x", "state": "loaded", "max_context_length": True}], "x") is None


def test_effective_soft_limit_raises_for_big_context():
    cfg = Config()
    cfg.context_soft_limit = 24000
    # qwen 256k -> budget ~80% of it (cached ctx avoids any network call)
    assert _effective_soft_limit(_Prov("http://h/v1", "qwen", 262144), cfg) == max(24000, int(262144 * 0.8))
    # gemma 32k -> just above the floor
    assert _effective_soft_limit(_Prov("http://h/v1", "gemma", 32768), cfg) == max(24000, int(32768 * 0.8))


def test_effective_soft_limit_falls_back_to_floor():
    cfg = Config()
    cfg.context_soft_limit = 24000
    assert _effective_soft_limit(_Prov("http://h/v1", "m", None), cfg) == 24000   # detection failed
    assert _effective_soft_limit(_Prov(None, None), cfg) == 24000                  # mock / no endpoint
