"""Win 2 — id_slot pinning + dead-seed fix.

Captures the kwargs sent to the OpenAI client's ``chat.completions.create``
via a fake client (no network). Verifies:
  - ``id_slot`` lands in ``extra_body`` only when set (None omits it).
  - ``seed`` lands in ``kwargs`` only when set to a valid value (None/-1 omit).
  - ``build_provider("local", seed=..., id_slot=...)`` forwards both to
    LocalProvider (the dead-seed fix: previously seed was dropped).
  - ``build_provider("mock", seed=...)`` still forwards seed to MockProvider.
"""

from __future__ import annotations

from llmcli.providers import LocalProvider
from llmcli.repl import build_provider


class _FakeStream:
    """Minimal iterable that yields a single stop chunk then exhausts."""

    def __init__(self) -> None:
        self._sent = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._sent:
            raise StopIteration
        self._sent = True
        # A stop chunk is enough; stream_chat only reads .choices[].delta and
        # .usage off the streamed chunks — an empty one ends the loop cleanly.
        return type("C", (), {"choices": [], "usage": None})()


def _local_capturing(**provider_kwargs) -> tuple[LocalProvider, dict]:
    """Build a LocalProvider with a fake client that records create() kwargs."""
    lp = LocalProvider(model="m", base_url="http://127.0.0.1:1234/v1", api_key="k",
                       **provider_kwargs)
    captured: dict = {}
    stream = _FakeStream()

    class _FakeCompletions:
        def create(_self, **kwargs):
            captured.update(kwargs)
            return stream

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    lp._client = _FakeClient()
    return lp, captured


def _drain(lp: LocalProvider) -> None:
    """Consume the stream so create() is actually invoked once."""
    list(lp.stream_chat(messages=[{"role": "user", "content": "hi"}], tools=None))


def test_id_slot_pinned_and_seed_sent():
    """id_slot + cache_prompt land in extra_body; seed lands in kwargs."""
    lp, cap = _local_capturing(cache_prompt=True, id_slot=0, seed=7)
    _drain(lp)
    assert cap["extra_body"]["id_slot"] == 0
    assert cap["extra_body"]["cache_prompt"] is True
    assert cap["seed"] == 7


def test_id_slot_none_omitted():
    """id_slot=None must not add the key to extra_body."""
    lp, cap = _local_capturing(id_slot=None, seed=1)
    _drain(lp)
    assert "id_slot" not in cap.get("extra_body", {})


def test_seed_none_omitted():
    """seed=None must not add the key to kwargs."""
    lp, cap = _local_capturing(id_slot=None, seed=None)
    _drain(lp)
    assert "seed" not in cap


def test_seed_negative_omitted():
    """A negative seed means 'no seed' and must be kept out of the request."""
    lp, cap = _local_capturing(id_slot=None, seed=-1)
    _drain(lp)
    assert "seed" not in cap


def test_build_provider_local_forwards_seed_and_id_slot():
    """Dead-seed fix: build_provider('local', ...) forwards seed AND id_slot."""
    p = build_provider(
        "local", model="m", base_url="http://127.0.0.1:1234/v1",
        seed=5, id_slot=0,
    )
    assert isinstance(p, LocalProvider)
    assert p.seed == 5
    assert p.id_slot == 0


def test_build_provider_mock_forwards_seed_unchanged():
    """MockProvider still receives seed (unchanged behavior)."""
    from llmcli.providers import MockProvider

    p = build_provider("mock", model="m", base_url="http://127.0.0.1:1234/v1",
                       seed=5)
    assert isinstance(p, MockProvider)
    assert p.seed == 5