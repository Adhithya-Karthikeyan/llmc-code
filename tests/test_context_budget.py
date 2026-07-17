"""Adaptive working-context budget + aggressive /compact (keep_turns)."""

from __future__ import annotations

from llmcli.agent import Agent, request_weight
from llmcli.config import Config, load_config
from llmcli.providers import MockProvider


class _Text(MockProvider):
    def __init__(self, text="ok"):
        super().__init__()
        self._text = text

    def stream_chat(self, messages, tools):
        yield {"type": "text", "text": self._text}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


# ----- request_weight heuristic --------------------------------------------

def test_request_weight_small_vs_big():
    assert request_weight("hi") == 1.0
    assert request_weight("what is x") == 1.0
    assert request_weight("audit the whole project") == 2.5  # big keyword(s)
    assert request_weight("refactor everything across the codebase") == 2.5
    assert request_weight("x" * 500) == 1.5      # long prompt bonus
    assert request_weight("x" * 1300) == 2.0     # longer prompt bonus
    assert request_weight("audit " + "x" * 1300) == 3.0  # capped at 3.0


# ----- _compute_turn_budget -------------------------------------------------

def _agent(**kw):
    return Agent(provider=MockProvider(), system_prompt="s", tool_names=[], **kw)


def test_compute_turn_budget_adaptive_and_floor_and_ceiling():
    a = _agent(context_budget=12_000, context_ceiling=100_000, context_adaptive=True)
    # "hi" is TRIVIAL/meta -> intent shrinks it to half (pull-on-demand); a real
    # task or broad request keeps the full length+keyword weight.
    assert a._compute_turn_budget("hi") == 6_000                # trivial -> base//2
    assert a._compute_turn_budget("add retry to providers.py") == 12_000  # task, weight 1.0
    assert a._compute_turn_budget("audit the whole project") == 30_000  # 2.5x
    # fixed (non-adaptive) ignores the request weight
    b = _agent(context_budget=12_000, context_ceiling=100_000, context_adaptive=False)
    assert b._compute_turn_budget("audit the whole project") == 12_000
    # floor: tiny base is raised to the minimum
    c = _agent(context_budget=1_000, context_ceiling=100_000, context_adaptive=True)
    assert c._compute_turn_budget("hi") == 4_000
    # ceiling caps a flexed budget
    d = _agent(context_budget=12_000, context_ceiling=20_000, context_adaptive=True)
    assert d._compute_turn_budget("audit the whole project") == 20_000


def test_run_sets_adaptive_budget_per_turn():
    a = Agent(
        provider=_Text(), system_prompt="s", tool_names=[], console=None,
        context_budget=10_000, context_ceiling=500_000, context_adaptive=True,
    )
    a.run("audit the entire codebase end-to-end")  # big -> 2.5x
    assert a.context_soft_limit == 25_000
    a.run("hi")  # trivial/meta -> half base (load only what's needed)
    assert a.context_soft_limit == 5_000


def test_budget_change_resets_compact_floor():
    """A big turn then a small turn must NOT skip a needed compaction because of a
    stale anti-thrash floor recorded under the bigger budget (finding #1)."""
    a = Agent(
        provider=_Text(), system_prompt="s", tool_names=[], console=None,
        context_budget=10_000, context_ceiling=500_000, context_adaptive=True,
    )
    a.run("audit the whole project")     # big -> budget 25_000
    a._compact_floor = 33_000            # simulate a floor recorded under the big budget
    a.run("hi")                          # trivial -> budget 5_000, must reset the floor
    assert a.context_soft_limit == 5_000
    assert a._compact_floor == 0


def test_off_mode_keeps_near_window_valve():
    """/context off (budget 0) must still trim near the window when a ceiling is
    set — the documented safety valve (finding #2)."""
    a = Agent(
        provider=_Text(), system_prompt="s", tool_names=[], console=None,
        context_budget=0, context_ceiling=24_000, context_adaptive=True,
    )
    a.run("hi")
    assert a.context_soft_limit == 24_000  # falls back to the ceiling, not 0


def test_nudge_message_not_counted_as_user_turn():
    """A synthetic _nudge message must not count as a real turn in compact()
    (finding #3)."""
    a = Agent(provider=_Summarizer(), system_prompt="SYS", tool_names=[])
    a.messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "real task one"},
        {"role": "assistant", "content": "did one"},
        {"role": "user", "content": "real task two"},
        {"role": "user", "_nudge": True, "content": "write your answer"},
        {"role": "assistant", "content": "did two"},
    ]
    a.compact(keep_turns=1)  # keep the last REAL turn (task two), not the nudge
    tail = " ".join(str(m.get("content")) for m in a.messages)
    assert "real task two" in tail        # the real last turn is kept
    assert "real task one" not in tail     # the earlier turn is summarized


def test_no_budget_keeps_static_soft_limit():
    # context_budget == 0 => run() must NOT touch context_soft_limit (old behaviour)
    a = Agent(
        provider=_Text(), system_prompt="s", tool_names=[], console=None,
        context_soft_limit=24_000,
    )
    a.run("audit the whole project")
    assert a.context_soft_limit == 24_000


# ----- aggressive compact (keep_turns) -------------------------------------

class _Summarizer(MockProvider):
    def stream_chat(self, messages, tools):
        yield {"type": "text", "text": "- prior work summary"}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}


def _history_agent():
    a = Agent(provider=_Summarizer(), system_prompt="SYS", tool_names=[])
    a.messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "did first"},
        {"role": "user", "content": "second task"},
        {"role": "assistant", "content": "did second"},
        {"role": "user", "content": "third task"},
        {"role": "assistant", "content": "did third"},
    ]
    return a


def test_compact_aggressive_keeps_only_last_exchange():
    a = _history_agent()
    a.compact(keep_turns=1)
    tail = " ".join(str(m.get("content")) for m in a.messages)
    assert "third task" in tail
    assert "second task" not in tail and "first task" not in tail
    assert any("summary" in str(m.get("content")).lower() for m in a.messages)


def test_compact_default_keeps_last_two_turns():
    a = _history_agent()
    a.compact()  # keep_turns=2 (auto-compaction's safe default)
    tail = " ".join(str(m.get("content")) for m in a.messages)
    assert "second task" in tail and "third task" in tail
    assert "first task" not in tail


# ----- config defaults + load ----------------------------------------------

def test_context_config_defaults_and_load(tmp_path):
    import json

    assert Config().context_budget == 12_000
    assert Config().context_adaptive is True
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"context_budget": 8000, "context_adaptive": False}), encoding="utf-8")
    cfg = load_config(path=p)
    assert cfg.context_budget == 8000 and cfg.context_adaptive is False
    # 0 disables (allowed); a negative/garbage value is ignored -> default kept
    p.write_text(json.dumps({"context_budget": 0}), encoding="utf-8")
    assert load_config(path=p).context_budget == 0
    p.write_text(json.dumps({"context_budget": -5}), encoding="utf-8")
    assert load_config(path=p).context_budget == 12_000
