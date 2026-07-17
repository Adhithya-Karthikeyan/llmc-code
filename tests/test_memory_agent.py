"""STAGE 2: passive conversation-memory wiring in the agent loop.

All offline/deterministic (MockProvider, BM25 mode — no embeddings, no network).

NEW DESIGN (PERF-1 + AGENT-3): the recalled-memory note is NEVER inserted into
``agent.messages``. It is built ONCE per turn and appended only to the per-REQUEST
payload as the FINAL element, so:
  - relevant past records still reach the model (proven via a recording provider
    that captures the messages passed to ``stream_chat``),
  - ``agent.messages`` stays append-only -> its prefix is byte-stable turn over
    turn (cache_prompt KV reuse keeps paying off),
  - the note is NEVER persisted (not in history -> session-save can't leak it) and
    ``_maybe_auto_compact()`` can never strip it,
  - a completed Q/A turn is still recorded for later recall,
  - a provider/stream FAILURE is NOT persisted as a real answer (AGENT-1),
  - memory is a strict no-op on the no-memory defaults (sub-agent/back-compat
    path unchanged),
  - the REPL one-shot path persists a .memory.json next to the session file.
"""

from __future__ import annotations

import re

import pytest

from llmcli.agent import Agent
from llmcli.memory import MemoryStore
from llmcli.providers import MockProvider
from llmcli import repl, session as session_mod, memory as memory_mod
from llmcli.config import Config


_MCP_DOC = "the mcp toggle command turns servers on or off"
_GIT_DOC = "git rebase keeps a clean linear history"


def _seeded_store() -> MemoryStore:
    store = MemoryStore()
    store.add(_MCP_DOC)
    store.add(_GIT_DOC)
    return store


class _RecordingProvider(MockProvider):
    """A MockProvider that captures the message list passed to each stream_chat
    call, so a test can assert what the model actually SAW (incl. the per-request
    memory note that never enters agent.messages)."""

    def __init__(self, scenario: str = "plain"):
        super().__init__(scenario=scenario)
        self.seen: list[list[dict]] = []

    def stream_chat(self, messages, tools):
        self.seen.append(list(messages))
        yield from super().stream_chat(messages, tools)


def _memory_msgs(agent: Agent) -> list[dict]:
    """Messages in HISTORY that look like an injected memory note — by the old
    `_memory` tag OR by the rendered header. New design: this must ALWAYS be empty
    (the note rides the request tail, never history)."""
    return [
        m for m in agent.messages
        if m.get("_memory") or "RELEVANT MEMORY" in str(m.get("content", ""))
    ]


def _req_mem_block(req: list[dict]) -> dict | None:
    """The memory note in a captured request payload, if present."""
    for m in req:
        if "RELEVANT MEMORY" in str(m.get("content", "")):
            return m
    return None


# --------------------------------------------------------------------------- #
# injection: a matching record reaches the PROVIDER (request tail), not history
# --------------------------------------------------------------------------- #

def test_memory_block_injected_into_request_not_history(tmp_workspace):
    store = _seeded_store()
    prov = _RecordingProvider(scenario="plain")
    agent = Agent(
        provider=prov,
        system_prompt="sys",
        tool_names=[],
        memory=store,
        recall_mode="bm25",
        memory_top_k=2,
        memory_enabled=True,
    )
    agent.run("how does the mcp toggle work")

    # The recalled memory was passed to the provider as the FINAL request element,
    # carrying role "user" (a trailing user-role context block is portable across
    # strict chat templates in any position — see Agent.run's req_messages build).
    assert prov.seen
    req = prov.seen[0]
    last = req[-1]
    assert last["role"] == "user"
    assert "RELEVANT MEMORY" in last["content"]
    # The lexically-matched record's content rode along in the note.
    assert "mcp toggle" in last["content"]
    # It sits AFTER the user message (the model sees it right before answering).
    assert req[-2] == {"role": "user", "content": "how does the mcp toggle work"}
    # CRITICAL: it never entered history (cache-stable prefix + never persisted).
    assert _memory_msgs(agent) == []


# --------------------------------------------------------------------------- #
# iteration-0-only: a TOOL-CALL turn injects memory on the FIRST request only,
# never AFTER the assistant(tool_calls)+tool messages (portability + tokens)
# --------------------------------------------------------------------------- #

def test_memory_only_on_first_iteration_of_tool_turn(tmp_workspace):
    """A multi-iteration TOOL-CALL turn injects the recalled note ONLY on the
    FIRST provider call (iteration 0, as a trailing role-"user" block) and OMITS
    it from the post-tool-result iteration (which ends with the tool message, not
    a memory block). This locks the iteration-0-only + portable-role design: the
    note can never land AFTER assistant(tool_calls)+tool messages where a strict
    chat template might reject a trailing system block."""
    # Make read_file succeed so the loop reaches the second (post-result) call.
    (tmp_workspace / "hello.py").write_text("print('hi')\n")
    store = _seeded_store()
    prov = _RecordingProvider(scenario="read")  # read_file -> tool result -> text
    agent = Agent(
        provider=prov,
        system_prompt="sys",
        tool_names=["read_file"],
        memory=store,
        recall_mode="bm25",
        memory_top_k=2,
        memory_enabled=True,
    )
    agent.run("how does the mcp toggle work")

    # The tool turn drove TWO provider calls: iteration 0 + the post-result one.
    assert len(prov.seen) == 2
    first, second = prov.seen[0], prov.seen[1]

    # Iteration 0: the memory note is the FINAL element and carries role "user".
    assert first[-1]["role"] == "user"
    assert "RELEVANT MEMORY" in first[-1]["content"]
    assert "mcp toggle" in first[-1]["content"]

    # Post-tool-result iteration (1+): NO memory block anywhere, and the request
    # ENDS with the tool result (never a memory block after assistant/tool turns).
    assert _req_mem_block(second) is None
    assert second[-1].get("role") == "tool"

    # It never entered history on either iteration.
    assert _memory_msgs(agent) == []


# --------------------------------------------------------------------------- #
# record creation: a successful answer is remembered (Q + A, lossless)
# --------------------------------------------------------------------------- #

def test_successful_answer_creates_record(tmp_workspace):
    store = _seeded_store()
    agent = Agent(
        provider=MockProvider(scenario="plain"),
        system_prompt="sys",
        tool_names=[],
        memory=store,
        recall_mode="bm25",
        memory_top_k=2,
        memory_enabled=True,
    )
    n_before = len(store.records)
    final = agent.run("tell me about mcp toggling")

    assert len(store.records) == n_before + 1  # the store grew by one
    rec = store.records[-1]
    # The record text is lossless: it combines the user's Q and the assistant's A.
    assert rec.text.startswith("Q: tell me about mcp toggling")
    assert "A: " in rec.text
    assert final in rec.text
    # A cheap heuristic summary was attached (no extra LLM call).
    assert rec.summary
    assert rec.summary in final


# --------------------------------------------------------------------------- #
# back-compat: defaults OFF -> no injection, no record (sub-agent path intact)
# --------------------------------------------------------------------------- #

def test_no_memory_on_defaults(tmp_workspace):
    """The no-memory defaults (memory=None) inject nothing and record nothing."""
    agent = Agent(
        provider=MockProvider(scenario="plain"),
        system_prompt="sys",
        tool_names=[],
    )
    agent.run("anything at all")
    assert _memory_msgs(agent) == []


def test_no_memory_when_disabled(tmp_workspace):
    """A store present but memory_enabled=False is a strict no-op (no inject, no
    record) — mirrors a sub-agent that is never handed an active store."""
    store = _seeded_store()
    n_before = len(store.records)
    prov = _RecordingProvider(scenario="plain")
    agent = Agent(
        provider=prov,
        system_prompt="sys",
        tool_names=[],
        memory=store,
        recall_mode="bm25",   # mode set, but the master switch is OFF
        memory_top_k=2,
        memory_enabled=False,
    )
    agent.run("how does the mcp toggle work")
    assert _memory_msgs(agent) == []
    # Nothing was injected into the REQUEST either.
    assert all(_req_mem_block(req) is None for req in prov.seen)
    assert len(store.records) == n_before  # nothing recorded


# --------------------------------------------------------------------------- #
# ephemeral + cache-safe: history is an append-only PREFIX across turns
# --------------------------------------------------------------------------- #

def test_history_is_append_only_prefix_across_turns(tmp_workspace):
    """The recalled note rides the REQUEST tail every turn but never enters
    history, so self.messages is byte-stable as a PREFIX across turns (PERF-1:
    a mid-history insert/strip would backfill a slot and break KV-cache reuse)."""
    store = _seeded_store()
    prov = _RecordingProvider(scenario="plain")
    agent = Agent(
        provider=prov,
        system_prompt="sys",
        tool_names=[],
        memory=store,
        recall_mode="bm25",
        memory_top_k=2,
        memory_enabled=True,
    )
    agent.run("first about mcp toggle")
    snapshot1 = [dict(m) for m in agent.messages]   # history after turn 1
    agent.run("second about mcp toggle")
    snapshot2 = [dict(m) for m in agent.messages]   # history after turn 2

    # No memory note ever leaked into history (across BOTH turns).
    assert _memory_msgs(agent) == []
    # Turn 1's history is an EXACT prefix of turn 2's history (append-only).
    assert snapshot2[: len(snapshot1)] == snapshot1
    assert len(snapshot2) > len(snapshot1)  # turn 2 really did append
    # Yet the recalled memory reached the provider on BOTH turns, as the final
    # element of each request.
    assert len(prov.seen) == 2
    for req in prov.seen:
        assert "RELEVANT MEMORY" in str(req[-1].get("content", ""))


# --------------------------------------------------------------------------- #
# never persisted: the recalled note is absent from a saved session
# --------------------------------------------------------------------------- #

def test_recalled_memory_never_persisted_to_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    store = _seeded_store()
    prov = _RecordingProvider(scenario="plain")
    agent = Agent(
        provider=prov,
        system_prompt="sys",
        tool_names=[],
        memory=store,
        recall_mode="bm25",
        memory_top_k=2,
        memory_enabled=True,
    )
    agent.run("how does the mcp toggle work")

    # It reached the provider...
    assert any("RELEVANT MEMORY" in str(req[-1].get("content", "")) for req in prov.seen)
    # ...and is absent from history (the source of the saved file).
    assert _memory_msgs(agent) == []

    # Persisting the conversation must not leak the recalled note.
    session_mod.save_session(str(tmp_path), agent.messages, model="mock", title="t")
    blob = "".join(p.read_text() for p in session_mod.sessions_dir().glob("*.json"))
    assert blob  # a session file was actually written
    assert "RELEVANT MEMORY" not in blob


# --------------------------------------------------------------------------- #
# compaction safety: a `_memory` block is NEVER summarized or leaked
# (defense-in-depth — the new design keeps `_memory` out of history entirely, but
# compact() still guards it should one ever appear)
# --------------------------------------------------------------------------- #

def test_compact_never_summarizes_memory_block(tmp_workspace):
    """A `_memory` block in history must be DROPPED from the to-summarize range,
    so its content never reaches the summarizer nor the resulting summary note."""

    class _CapturingSummarizer(MockProvider):
        def __init__(self):
            super().__init__()
            self.seen: list[str] = []

        def stream_chat(self, messages, tools):
            # messages[1] is the transcript compact() built for the summarizer.
            self.seen.append(str(messages[1]["content"]))
            yield {"type": "text", "text": "- short compact summary"}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 4}

    prov = _CapturingSummarizer()
    agent = Agent(provider=prov, system_prompt="SYS", tool_names=[])
    agent.messages += [
        {"role": "user", "content": "first task " * 30},
        {"role": "assistant", "content": "first reply " * 30},
        # An ephemeral memory block embedded in earlier history (the to-summarize
        # range for keep_turns=1) carrying a unique marker.
        {"role": "system", "_memory": True,
         "content": "RELEVANT MEMORY (from earlier in this project):\n1. SECRET_MEMORY_MARKER"},
        {"role": "user", "content": "second task " * 30},
        {"role": "assistant", "content": "second reply " * 30},
        {"role": "user", "content": "third task " * 30},
        {"role": "assistant", "content": "third reply " * 30},
    ]
    before, after = agent.compact(keep_turns=1)

    # The summarizer ran (the normal user-boundary summarize path).
    assert prov.seen
    transcript = prov.seen[0]
    # The `_memory` content was EXCLUDED from what got summarized.
    assert "SECRET_MEMORY_MARKER" not in transcript
    assert "RELEVANT MEMORY" not in transcript

    # The resulting summary note exists and carries no memory content.
    summary_note = next(
        m["content"] for m in agent.messages
        if m["role"] == "system" and "Summary of earlier conversation" in str(m["content"])
    )
    assert "SECRET_MEMORY_MARKER" not in summary_note
    # No `_memory` block survives the compaction (regenerated per turn).
    assert not _memory_msgs(agent)
    # Sane, shrinking return.
    assert isinstance(before, int) and isinstance(after, int)
    assert after < before


# --------------------------------------------------------------------------- #
# AGENT-1: a provider/stream FAILURE is not persisted as a real answer
# --------------------------------------------------------------------------- #

class _StreamErrorSentinel(MockProvider):
    """Emits the providers.py stream-error sentinel as the answer text with a
    NORMAL finish_reason, so only the sentinel string can trigger the bail-out."""

    def stream_chat(self, messages, tools):
        yield {"type": "text", "text": "[stream error: ReadTimeout]"}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 0}


class _FinishReasonError(MockProvider):
    """Non-sentinel text but finish_reason=='error' — the explicit failure signal."""

    def stream_chat(self, messages, tools):
        yield {"type": "text", "text": "half a sentence before the socket died"}
        yield {"type": "done", "finish_reason": "error", "output_tokens": 3}


@pytest.mark.parametrize(
    "prov_cls, expected",
    [
        (_StreamErrorSentinel, "[stream error: ReadTimeout]"),
        (_FinishReasonError, "half a sentence before the socket died"),
    ],
)
def test_provider_error_not_persisted_or_recorded(tmp_workspace, prov_cls, expected):
    store = MemoryStore()
    add_calls = {"n": 0}
    _orig_add = store.add

    def _counting_add(*a, **k):  # prove memory.add is NEVER reached on failure
        add_calls["n"] += 1
        return _orig_add(*a, **k)

    store.add = _counting_add  # type: ignore[assignment]

    agent = Agent(
        provider=prov_cls(),
        system_prompt="sys",
        tool_names=[],
        memory=store,
        recall_mode="bm25",
        memory_top_k=2,
        memory_enabled=True,
    )
    out = agent.run("do the thing that errors out")

    # The raw text is RETURNED (so one-shot/spawn callers can see the failure)...
    assert out == expected
    # ...but NO assistant turn was appended to history (state not polluted)...
    assert not any(m.get("role") == "assistant" for m in agent.messages)
    # ...and the failure was NOT recorded as a "fact".
    assert add_calls["n"] == 0
    assert len(store.records) == 0


# --------------------------------------------------------------------------- #
# REPL one-shot: a completed turn persists a .memory.json beside the session
# --------------------------------------------------------------------------- #

def test_run_once_persists_memory_store(tmp_path, monkeypatch):
    # Pin HOME so sessions_dir()/store_path() write into the tmp tree, not the
    # real ~/.llm-cli. chdir so the cwd (== session/store key) is the tmp project.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # mcp_enabled=False keeps the one-shot fully offline (no MCP subprocesses);
    # memory defaults (enabled, mode "auto") drive a record + a persisted store.
    cfg = Config(provider="mock", mcp_enabled=False)
    repl.run_once(
        MockProvider(scenario="plain"), cfg, "remember this mcp fact",
        auto_confirm=True,
    )

    sp = memory_mod.store_path(str(tmp_path))
    assert sp.exists()  # a .memory.json was written next to the session file
    reloaded = MemoryStore.load(sp)
    assert len(reloaded.records) == 1
    assert reloaded.records[0].text.startswith("Q: remember this mcp fact")


# --------------------------------------------------------------------------- #
# end-to-end: a recalled record actually CHANGES the produced answer
# --------------------------------------------------------------------------- #

class _RecallEchoProvider(MockProvider):
    """Answers with the SECRET token it finds inside the injected memory note,
    proving the recalled context truly influences the output. Detects the note by
    its rendered header (the note no longer carries a `_memory` tag)."""

    def stream_chat(self, messages, tools):
        secret = ""
        for m in messages:
            content = str(m.get("content", ""))
            if "RELEVANT MEMORY" in content:
                match = re.search(r"SECRET=\d+", content)
                if match:
                    secret = match.group(0)
                    break
        text = f"recalled {secret}" if secret else "no memory found"
        yield {"type": "text", "text": text}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}


def test_recall_actually_influences_answer(tmp_workspace):
    store = MemoryStore()
    store.add("the deploy key note SECRET=42 lives here")
    agent = Agent(
        provider=_RecallEchoProvider(),
        system_prompt="sys",
        tool_names=[],
        memory=store,
        recall_mode="bm25",
        memory_top_k=2,
        memory_enabled=True,
    )
    final = agent.run("what is the deploy key note")
    # The injected record's token reached the model and shaped the answer.
    assert "SECRET=42" in final


# --------------------------------------------------------------------------- #
# resume/load: a record A persisted is recalled+injected by a SECOND agent B
# --------------------------------------------------------------------------- #

def test_persisted_record_recalled_by_second_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cwd = str(tmp_path)

    store_a = MemoryStore()
    agent_a = Agent(
        provider=MockProvider(scenario="plain"),
        system_prompt="sys", tool_names=[],
        memory=store_a, recall_mode="bm25", memory_top_k=2, memory_enabled=True,
    )
    agent_a.run("remember the mcp toggle command")  # records this Q/A turn
    store_a.save(memory_mod.store_path(cwd))  # persist beside the session
    assert memory_mod.store_path(cwd).exists()

    # A SECOND agent loads the persisted store and recalls it on its own turn.
    store_b = MemoryStore.load(memory_mod.store_path(cwd))
    assert store_b.records  # the record survived the save/load round-trip
    recalled = store_b.retrieve("hello mock provider note", mode="bm25", top_k=2)
    assert any("Hello from the mock provider" in r.text for r in recalled)

    prov_b = _RecordingProvider(scenario="plain")
    agent_b = Agent(
        provider=prov_b,
        system_prompt="sys", tool_names=[],
        memory=store_b, recall_mode="bm25", memory_top_k=2, memory_enabled=True,
    )
    agent_b.run("hello mock provider note")
    # The recalled record reached the provider on agent B's turn (request tail)...
    assert prov_b.seen
    assert any(
        "Hello from the mock provider" in str((_req_mem_block(req) or {}).get("content", ""))
        for req in prov_b.seen
    )
    # ...but never entered agent B's history.
    assert _memory_msgs(agent_b) == []


# --------------------------------------------------------------------------- #
# embed-failure survival: a raising embeddings() never breaks a turn
# --------------------------------------------------------------------------- #

class _BoomEmbedProvider(MockProvider):
    def embeddings(self, texts):
        raise RuntimeError("embed endpoint down")


@pytest.mark.parametrize("mode", ["auto", "embed"])
def test_embed_failure_does_not_break_turn(tmp_workspace, mode):
    store = MemoryStore()
    # No lexical overlap with the query, so "auto" ESCALATES to the (failing)
    # embeddings path; "embed" always attempts it.
    store.add("an unrelated note about kittens and yarn")
    agent = Agent(
        provider=_BoomEmbedProvider(scenario="plain"),
        system_prompt="sys", tool_names=[],
        memory=store, recall_mode=mode, memory_top_k=2, memory_enabled=True,
    )
    n_before = len(store.records)
    final = agent.run("describe the photosynthesis pipeline")
    # The scripted answer still came back (the embed failure was swallowed).
    assert "mock provider" in final.lower()
    # The completed turn was still recorded despite the embed failure.
    assert len(store.records) == n_before + 1


# --------------------------------------------------------------------------- #
# truncation: a finish_reason=="length" turn is NOT recorded
# --------------------------------------------------------------------------- #

class _TruncatingProvider(MockProvider):
    def stream_chat(self, messages, tools):
        yield {"type": "text", "text": "a partial answer that got cut"}
        yield {"type": "done", "finish_reason": "length", "output_tokens": 6}


def test_truncated_turn_is_not_recorded(tmp_workspace):
    store = MemoryStore()
    agent = Agent(
        provider=_TruncatingProvider(),
        system_prompt="sys", tool_names=[],
        memory=store, recall_mode="bm25", memory_top_k=2, memory_enabled=True,
    )
    n_before = len(store.records)
    final = agent.run("give me the full design")
    # A truncated answer is a partial fact -> must NOT be stored as complete.
    assert len(store.records) == n_before
    # The returned text still carries the truncation marker (existing behaviour).
    assert "[output truncated at token limit]" in final
