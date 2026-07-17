"""Auto-compaction must not THRASH: when a compaction can't meaningfully shrink
history (recent tail near budget, or a weak model writes a bloated summary), it
must roll back (keep the KV-cache prefix stable), stay silent, and not re-fire
every turn. Regression for the "~24557 -> ~24543" near-no-op the user hit.
"""

from __future__ import annotations

import io

from rich.console import Console

from llmcli.agent import Agent


class _FakeProv:
    model = "fake"

    def __init__(self, summary_text: str):
        self.summary_text = summary_text
        self.calls = 0

    def stream_chat(self, messages, tools):
        self.calls += 1
        yield {"type": "text", "text": self.summary_text}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


def _agent(summary_text: str, soft_limit: int, console=None) -> Agent:
    a = Agent(_FakeProv(summary_text), "system prompt", [], console=console,
              context_soft_limit=soft_limit)
    big = "x" * 4000  # ~1000 est tok per assistant message
    for i in range(6):
        a.messages.append({"role": "user", "content": f"question {i}"})
        a.messages.append({"role": "assistant", "content": big})
    return a


def test_trivial_compaction_is_rolled_back_and_silent():
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, markup=False, width=100)
    # Summary BIGGER than what it replaces => no meaningful reduction.
    a = _agent("y" * 20000, soft_limit=2000, console=con)
    before_msgs = list(a.messages)
    a._maybe_auto_compact()
    assert a.messages == before_msgs              # rolled back -> KV-cache prefix intact
    assert "auto-compacted" not in buf.getvalue()  # no noisy line
    assert a._compact_floor > 0                    # remembered: don't retry yet


def test_does_not_refire_when_below_floor():
    a = _agent("y" * 20000, soft_limit=2000)
    a._maybe_auto_compact()
    assert a.provider.calls == 1 and a._compact_floor > 0
    a._maybe_auto_compact()                        # same size, below the floor
    assert a.provider.calls == 1                   # provider NOT called again (no thrash)


def test_meaningful_compaction_applies_and_announces():
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=False, markup=False, width=100)
    a = _agent("brief", soft_limit=2000, console=con)  # tiny summary => big reduction
    before = a._estimate_tokens(a.messages)
    a._maybe_auto_compact()
    assert a._estimate_tokens(a.messages) < before  # actually shrank
    assert "auto-compacted" in buf.getvalue()        # announced
    assert a._compact_floor == 0
