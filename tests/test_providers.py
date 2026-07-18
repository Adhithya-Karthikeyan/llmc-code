"""Provider event-normalization tests. No network, no openai import."""

from __future__ import annotations

import importlib
import math
import sys

import pytest

from llmcode.agent import compute_tok_stats, format_footer
from llmcode.providers import (
    MockProvider,
    _clean_reasoning_text,
    _fence_is_sole_content,
    count_tool_blocks,
    effort_extra_body,
    parse_tool_block,
)

def _fence(body: str) -> str:
    return "```json\n" + body + "\n```"


def test_effort_extra_body():
    assert effort_extra_body("") == {}
    assert effort_extra_body(None) == {}
    assert effort_extra_body("low") == {"reasoning_effort": "low"}
    assert effort_extra_body("HIGH") == {"reasoning_effort": "high"}
    off = effort_extra_body("off")
    assert off["reasoning_effort"] == "minimal"
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert effort_extra_body("bogus") == {}


def test_mock_event_shapes():
    events = list(MockProvider(scenario="hello").stream_chat([{"role": "user", "content": "hi"}], None))

    # Exactly one done, always last.
    dones = [e for e in events if e["type"] == "done"]
    assert len(dones) == 1
    assert events[-1]["type"] == "done"
    assert events[-1]["finish_reason"] in ("stop", "tool_calls", "length", "error")

    for e in events:
        assert "type" in e
        if e["type"] == "tool_call":
            assert isinstance(e["id"], str) and e["id"]
            assert isinstance(e["name"], str) and e["name"]
            assert isinstance(e["arguments"], dict)  # already parsed
        elif e["type"] == "text":
            assert isinstance(e["text"], str)


def test_mock_first_turn_is_tool_call():
    events = list(MockProvider(scenario="hello").stream_chat([], None))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert tool_calls
    assert tool_calls[0]["name"] == "write_file"
    assert tool_calls[0]["arguments"]["path"] == "hello.py"


def test_parse_tool_block_extracts_call():
    text = (
        "Sure, let me read it.\n\n"
        "```json\n"
        '{"tool": "read_file", "input": {"path": "main.py"}}\n'
        "```\n"
    )
    parsed = parse_tool_block(text)
    assert parsed is not None
    assert parsed["name"] == "read_file"
    assert parsed["arguments"] == {"path": "main.py"}


def test_parse_tool_block_ignores_non_tool_json():
    text = "```json\n{\"foo\": 1}\n```"
    assert parse_tool_block(text) is None


def test_parse_tool_block_none_on_plain_text():
    assert parse_tool_block("just a normal answer, no fences") is None


def test_parse_tool_block_flat_args():
    # FLAT shape (weak local models): no input/arguments wrapper, so the
    # remaining top-level keys ARE the arguments. Must not drop "path" (PROV-4).
    parsed = parse_tool_block(_fence('{"tool": "read_file", "path": "a.py"}'))
    assert parsed == {"name": "read_file", "arguments": {"path": "a.py"}}


def test_parse_tool_block_wrapper_shapes_preserved():
    # PRIMARY wrapper paths must NOT regress with the flat fallback in place.
    a = parse_tool_block(_fence('{"tool": "read_file", "arguments": {"path": "m.py"}}'))
    assert a == {"name": "read_file", "arguments": {"path": "m.py"}}
    b = parse_tool_block(_fence('{"name": "glob", "input": {"pattern": "*.py"}}'))
    assert b == {"name": "glob", "arguments": {"pattern": "*.py"}}
    # Explicit empty wrapper stays empty (not back-filled from siblings).
    c = parse_tool_block(_fence('{"tool": "noop", "input": {}}'))
    assert c == {"name": "noop", "arguments": {}}


def test_fence_is_sole_content_lone_fence():
    text = "\n\n" + _fence('{"tool": "glob", "input": {"pattern": "*"}}') + "\n\n"
    assert _fence_is_sole_content(text) is True


def test_fence_is_sole_content_false_with_prose():
    text = "Here is how it works:\n" + _fence('{"tool": "glob", "input": {}}')
    assert _fence_is_sole_content(text) is False
    text2 = _fence('{"tool": "glob"}') + "\nand that's the call."
    assert _fence_is_sole_content(text2) is False


def test_count_tool_blocks():
    one = _fence('{"tool": "glob", "input": {"pattern": "a"}}')
    two = one + "\n" + _fence('{"tool": "grep", "input": {"pattern": "b"}}')
    assert count_tool_blocks(one) == 1
    assert count_tool_blocks(two) == 2
    # A non-tool example fence (no tool/name key) is not counted.
    assert count_tool_blocks(_fence('{"foo": 1}')) == 0
    assert count_tool_blocks("no fences here") == 0


class _FakeFn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _FakeToolCallDelta:
    """A streamed tool_call delta fragment (keyed by .index)."""

    def __init__(self, index=0, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _FakeFn(name, arguments)


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, content=None, finish_reason=None, tool_calls=None,
                 reasoning_content=None):
        self.delta = _FakeDelta(content, tool_calls, reasoning_content)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, completion_tokens):
        self.completion_tokens = completion_tokens


class _FakeChunk:
    def __init__(self, content=None, finish_reason=None, tool_calls=None,
                 usage=None, reasoning_content=None, choices=None):
        if choices is None:
            choices = [_FakeChoice(content, finish_reason, tool_calls, reasoning_content)]
        self.choices = choices
        self.usage = usage


class _FakeStream:
    """Minimal OpenAI-style stream: yields chunks for a single turn.

    ``close()`` records that the provider's finally released it (finding #3).
    """

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


def _local_with_stream(chunks, **provider_kwargs):
    from llmcode.providers import LocalProvider

    lp = LocalProvider(model="m", base_url="http://x/v1", api_key="k", **provider_kwargs)
    stream = _FakeStream(chunks)
    lp._last_create_kwargs = None  # test hook: records create() kwargs
    lp._last_stream = stream

    class _FakeCompletions:
        def create(_self, **kwargs):
            lp._last_create_kwargs = kwargs
            return stream

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    lp._client = _FakeClient()
    return lp


def test_local_prose_with_example_fence_not_executed():
    """A prose answer that merely CONTAINS a json example must NOT fire a call."""
    prose = "Use the read_file tool like:\n" + _fence(
        '{"tool": "read_file", "input": {"path": "x"}}'
    ) + "\nThat is the format."
    chunks = [_FakeChunk(content=prose), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([{"role": "user", "content": "how?"}], None))
    assert not [e for e in events if e["type"] == "tool_call"]
    assert events[-1]["finish_reason"] == "stop"


def test_local_two_tool_fences_not_executed():
    """Two adjacent tool-shaped fences are ambiguous -> not auto-executed."""
    body = (
        _fence('{"tool": "glob", "input": {"pattern": "a"}}')
        + "\n"
        + _fence('{"tool": "glob", "input": {"pattern": "b"}}')
    )
    chunks = [_FakeChunk(content=body), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    assert not [e for e in events if e["type"] == "tool_call"]


def test_local_sole_fence_is_executed():
    """A single lone tool-shaped fence IS executed via the text fallback."""
    body = _fence('{"tool": "glob", "input": {"pattern": "*.py"}}')
    chunks = [_FakeChunk(content=body), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == "glob"
    assert calls[0].get("_from_text_fence") is True


# --------------------------------------------------------------------------- #
# REASONING FALLBACK: an otherwise-empty turn surfaces (cleaned) reasoning
# --------------------------------------------------------------------------- #

def test_clean_reasoning_text_strips_wrapper_tags():
    assert _clean_reasoning_text("<think>\nThe answer is 42.\n</think>") == "The answer is 42."
    assert _clean_reasoning_text("<thinking>x</thinking>") == "x"
    assert _clean_reasoning_text("  plain  ") == "plain"
    assert _clean_reasoning_text("<think></think>") == ""
    assert _clean_reasoning_text("") == ""


def test_local_reasoning_only_turn_surfaces_as_answer():
    # The model routed its WHOLE answer into reasoning_content and emitted no
    # visible content -> the provider surfaces the cleaned reasoning as the answer
    # (so the agent doesn't see an empty turn), with the internal marker set.
    chunks = [
        _FakeChunk(reasoning_content="<think>"),
        _FakeChunk(reasoning_content="The answer is 42."),
        _FakeChunk(reasoning_content="</think>"),
        _FakeChunk(finish_reason="stop"),
    ]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([{"role": "user", "content": "q"}], None))
    texts = [e for e in events if e["type"] == "text"]
    assert len(texts) == 1
    assert texts[0]["text"] == "The answer is 42."   # <think> tags stripped + trimmed
    assert texts[0].get("_from_reasoning") is True
    assert events[-1]["type"] == "done"


def test_local_empty_turn_no_reasoning_emits_no_text():
    # GENUINELY empty: no content AND no reasoning -> no text event at all, so the
    # agent's "[no answer produced …]" sentinel still fires downstream.
    chunks = [_FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    assert not [e for e in events if e["type"] == "text"]
    assert events[-1]["finish_reason"] == "stop"


def test_local_visible_content_discards_reasoning():
    # NORMAL turn: visible content present -> reasoning stays discarded, no
    # fallback, byte-unchanged from before this feature.
    chunks = [
        _FakeChunk(content="Hello.", reasoning_content="<think>secret</think>"),
        _FakeChunk(finish_reason="stop"),
    ]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    texts = [e for e in events if e["type"] == "text"]
    assert [t["text"] for t in texts] == ["Hello."]   # only visible content
    assert not any(t.get("_from_reasoning") for t in texts)


def test_mock_done_includes_output_tokens_for_text_turn():
    # The 'plain' scenario is a single text turn -> done carries the word count.
    events = list(MockProvider(scenario="plain").stream_chat([], None))
    done = events[-1]
    assert done["type"] == "done"
    assert "output_tokens" in done
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert done["output_tokens"] == len(text.split())
    assert done["output_tokens"] > 0


def test_mock_done_output_tokens_zero_for_tool_turn():
    # 'hello' step 0 is a tool-call turn (no visible text) -> 0 tokens.
    events = list(MockProvider(scenario="hello").stream_chat([], None))
    done = events[-1]
    assert done["finish_reason"] == "tool_calls"
    assert done["output_tokens"] == 0


def test_compute_tok_stats_exact_rate():
    # Known token count + elapsed -> exact rate, ignoring text approximation.
    tokens, rate = compute_tok_stats("ignored text", 4, 2.0)
    assert tokens == 4
    assert rate == 2.0


def test_compute_tok_stats_approximates_when_tokens_none():
    # None tokens -> approximate from text: max(chars/4, words).
    text = "one two three four"  # 4 words, 18 chars -> chars//4 = 4
    tokens, rate = compute_tok_stats(text, None, 1.0)
    assert tokens == max(len(text) // 4, len(text.split()))
    assert tokens == 4
    assert rate == 4.0


def test_compute_tok_stats_divide_by_zero_floor():
    # elapsed 0 must not raise; rate is finite via the 1e-6 floor.
    tokens, rate = compute_tok_stats("x", 10, 0.0)
    assert tokens == 10
    assert rate == 10 / 1e-6
    assert rate > 0


def test_compute_tok_stats_empty_text_zero_tokens():
    tokens, rate = compute_tok_stats("", None, 1.0)
    assert tokens == 0
    assert rate == 0.0


def test_format_footer_exact_string():
    # Footer shows ONLY the generation speed.
    assert format_footer(2.0) == "2.0 tok/s"
    assert format_footer(38.4) == "38.4 tok/s"
    assert format_footer(0.0) == "0.0 tok/s"


# ----- native streamed tool-call assembly (finding #8) --------------------

def test_local_assembles_split_tool_call():
    """id in chunk 1, name in chunk 2, arguments fragmented across chunks 3-4 ->
    one tool_call event with parsed dict arguments."""
    chunks = [
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, id="call_x")]),
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, name="read_file")]),
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, arguments='{"a":')]),
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, arguments='1}')]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["id"] == "call_x"
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"] == {"a": 1}
    assert events[-1]["finish_reason"] == "tool_calls"


def test_local_multi_index_tool_calls_emitted_sorted():
    """Two tool calls at different .index values -> both emitted, sorted by index."""
    chunks = [
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=1, id="b", name="grep", arguments="{}")]),
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, id="a", name="glob", arguments="{}")]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    lp = _local_with_stream(chunks)
    calls = [e for e in lp.stream_chat([], None) if e["type"] == "tool_call"]
    assert [c["name"] for c in calls] == ["glob", "grep"]  # index 0 then 1


def test_local_malformed_tool_args_sets_parse_error():
    chunks = [
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, id="x", name="glob", arguments=":::")]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    lp = _local_with_stream(chunks)
    calls = [e for e in lp.stream_chat([], None) if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["arguments"] == {}
    assert calls[0].get("_parse_error")


# ----- Feature 1: temperature forwarded into the request ------------------

def test_local_forwards_temperature_into_request():
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks, temperature=0.15)
    list(lp.stream_chat([{"role": "user", "content": "q"}], None))
    assert lp._last_create_kwargs["temperature"] == 0.15


def test_local_default_temperature_is_low():
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)  # no temperature kwarg -> the 0.2 default
    list(lp.stream_chat([], None))
    assert lp._last_create_kwargs["temperature"] == 0.2


# ----- Feature 2: tool_choice forwarded only when set ---------------------

def test_local_forwards_tool_choice_required_when_set():
    tools = [{"type": "function", "function": {"name": "glob", "parameters": {}}}]
    chunks = [
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, id="x", name="glob", arguments="{}")]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    lp = _local_with_stream(chunks)
    list(lp.stream_chat([], tools, tool_choice="required"))
    assert lp._last_create_kwargs["tool_choice"] == "required"


def test_local_omits_tool_choice_by_default():
    tools = [{"type": "function", "function": {"name": "glob", "parameters": {}}}]
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    list(lp.stream_chat([], tools))  # no tool_choice -> key absent (server default)
    assert "tool_choice" not in lp._last_create_kwargs


def test_local_tool_choice_ignored_without_tools():
    # tool_choice is meaningless without tools and must not be sent.
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    list(lp.stream_chat([], None, tool_choice="required"))
    assert "tool_choice" not in lp._last_create_kwargs


def test_local_usage_chunk_sets_output_tokens():
    """A trailing chunk with EMPTY .choices but populated .usage must set
    done.output_tokens (the empty-choices skip happens AFTER usage capture)."""
    chunks = [
        _FakeChunk(content="hello"),
        _FakeChunk(finish_reason="stop"),
        _FakeChunk(choices=[], usage=_FakeUsage(completion_tokens=42)),
    ]
    lp = _local_with_stream(chunks)
    done = list(lp.stream_chat([], None))[-1]
    assert done["type"] == "done"
    assert done["output_tokens"] == 42


def test_local_gen_elapsed_present_from_first_token():
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    done = list(lp.stream_chat([], None))[-1]
    assert done["gen_elapsed"] is not None
    assert done["gen_elapsed"] >= 0.0


def test_local_closes_stream_on_completion():
    """finding #3: the provider's finally must close the HTTP stream."""
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    list(lp.stream_chat([], None))
    assert lp._last_stream.closed is True


def test_local_closes_stream_on_early_break():
    """A consumer that stops early (then GC/close) must still release the stream:
    closing the generator runs its finally."""
    chunks = [_FakeChunk(content="a"), _FakeChunk(content="b"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    gen = lp.stream_chat([], None)
    next(gen)  # consume one event only
    gen.close()  # simulate break/Ctrl+C finalization
    assert lp._last_stream.closed is True


# ----- error redaction (finding #17) --------------------------------------

def test_local_get_client_error_redacts_detail(monkeypatch):
    from llmcode.providers import LocalProvider

    lp = LocalProvider(model="m", base_url="http://x/v1", api_key="k")

    def _boom():
        raise ValueError("secret token sk-xxx")

    monkeypatch.setattr(lp, "_get_client", _boom)
    events = list(lp.stream_chat([], None))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text == "[provider error: ValueError]"
    assert "secret" not in text and "sk-xxx" not in text
    assert events[-1]["finish_reason"] == "error"


def test_local_mid_stream_error_redacts():
    class _RaisingStream:
        def __iter__(self):
            raise RuntimeError("http://internal/leak")

        def close(self):
            pass

    from llmcode.providers import LocalProvider

    lp = LocalProvider(model="m", base_url="http://x/v1", api_key="k")
    rs = _RaisingStream()

    class _FakeCompletions:
        def create(self, **kwargs):
            return rs

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    lp._client = _FakeClient()
    events = list(lp.stream_chat([], None))
    text = "".join(e["text"] for e in events if e["type"] == "text")
    assert text == "[stream error: RuntimeError]"
    assert "internal" not in text and "leak" not in text
    assert events[-1]["finish_reason"] == "error"


# ----- loopback proxy-tunneling defense in DEFAULT mode (finding #3) -------

def _capture_get_client_kwargs(monkeypatch, base_url, private=False):
    """Run _get_client with a fake openai (real httpx) and return its kwargs."""
    import types

    from llmcode.providers import LocalProvider

    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    lp = LocalProvider(model="m", base_url=base_url, api_key="k", private=private)
    lp._get_client()
    return captured


def test_default_loopback_client_disables_proxy(monkeypatch):
    # PROV-3: a loopback target in DEFAULT (network-on) mode must build an
    # explicit httpx client with trust_env=False so HTTP(S)_PROXY/ALL_PROXY
    # cannot tunnel loopback model traffic off-box (IP-pinning alone does not
    # stop proxying).
    captured = _capture_get_client_kwargs(monkeypatch, "http://127.0.0.1:1234/v1")
    http_client = captured.get("http_client")
    assert http_client is not None
    assert http_client.trust_env is False


def test_default_external_client_honors_proxy(monkeypatch):
    # The genuinely external (--allow-network) path is untouched: no explicit
    # httpx client is injected, so the SDK's proxy-honoring default is used.
    captured = _capture_get_client_kwargs(monkeypatch, "http://example.com:1234/v1")
    assert "http_client" not in captured


# ----- cache_prompt hint (finding #12) ------------------------------------

def test_local_cache_prompt_adds_extra_body():
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks, cache_prompt=True)
    list(lp.stream_chat([], None))
    eb = lp._last_create_kwargs.get("extra_body") or {}
    assert eb.get("cache_prompt") is True


def test_local_no_cache_prompt_by_default():
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)  # cache_prompt defaults False
    list(lp.stream_chat([], None))
    eb = lp._last_create_kwargs.get("extra_body") or {}
    assert "cache_prompt" not in eb


# ----- per-request generation cap (max_output_tokens) ---------------------

def test_local_max_output_tokens_injects_max_tokens():
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks, max_output_tokens=4096)
    list(lp.stream_chat([], None))
    assert lp._last_create_kwargs.get("max_tokens") == 4096


def test_local_no_max_tokens_by_default():
    # The default (None) must send NO max_tokens key at all (backward-compatible).
    chunks = [_FakeChunk(content="hi"), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)  # max_output_tokens defaults None
    list(lp.stream_chat([], None))
    assert "max_tokens" not in lp._last_create_kwargs


def test_local_truncation_normalized_to_length_when_cap_hit():
    # Cap active, server reports finish_reason="stop" but completion_tokens >= cap
    # -> the provider forces "length" so the agent's truncation marker fires.
    chunks = [
        _FakeChunk(content="partial"),
        _FakeChunk(finish_reason="stop"),
        _FakeChunk(choices=[], usage=_FakeUsage(completion_tokens=128)),
    ]
    lp = _local_with_stream(chunks, max_output_tokens=128)
    done = list(lp.stream_chat([], None))[-1]
    assert done["finish_reason"] == "length"


def test_local_no_normalization_below_cap():
    # Below the cap, a real "stop" must NOT be promoted to "length".
    chunks = [
        _FakeChunk(content="done early"),
        _FakeChunk(finish_reason="stop"),
        _FakeChunk(choices=[], usage=_FakeUsage(completion_tokens=10)),
    ]
    lp = _local_with_stream(chunks, max_output_tokens=128)
    done = list(lp.stream_chat([], None))[-1]
    assert done["finish_reason"] == "stop"


def test_local_no_normalization_without_cap():
    # With no cap set, finish_reason is left exactly as the server reported.
    chunks = [
        _FakeChunk(content="hi"),
        _FakeChunk(finish_reason="stop"),
        _FakeChunk(choices=[], usage=_FakeUsage(completion_tokens=9999)),
    ]
    lp = _local_with_stream(chunks)  # no cap
    done = list(lp.stream_chat([], None))[-1]
    assert done["finish_reason"] == "stop"


# ----- embeddings: MockProvider (offline) + LocalProvider (no network) ----

def test_mock_embeddings_deterministic_and_shape():
    p = MockProvider()
    v1 = p.embeddings(["the mcp server config"])
    v2 = p.embeddings(["the mcp server config"])
    assert v1 == v2  # same text -> identical vector
    assert len(v1) == 1 and len(v1[0]) == 64  # one 64-dim vector per input
    norm = math.sqrt(sum(x * x for x in v1[0]))
    assert abs(norm - 1.0) < 1e-9  # L2-normalized


def test_mock_embeddings_batch_shape():
    vs = MockProvider().embeddings(["a b", "c d e", "f"])
    assert len(vs) == 3
    assert all(len(v) == 64 for v in vs)


def test_mock_embeddings_shared_tokens_score_higher():
    from llmcode.memory import cosine

    p = MockProvider()
    a = p.embeddings(["mcp toggle command"])[0]
    b = p.embeddings(["mcp toggle servers"])[0]  # shares 2 tokens with a
    c = p.embeddings(["zzz qqq nothing"])[0]  # disjoint tokens
    assert cosine(a, b) > cosine(a, c)


def test_local_embeddings_requires_embed_model():
    from llmcode.providers import LocalProvider

    lp = LocalProvider(model="m", base_url="http://x/v1", api_key="k")  # embed_model None
    with pytest.raises(ValueError):
        lp.embeddings(["hello"])  # raises BEFORE any client/network access


def test_local_embeddings_returns_vectors_via_client():
    from llmcode.providers import LocalProvider

    class _Emb:
        def __init__(self, e):
            self.embedding = e

    class _Resp:
        data = [_Emb([0.1, 0.2]), _Emb([0.3, 0.4])]

    class _Embeddings:
        def create(self, model, input):
            assert model == "emb-model"
            assert input == ["a", "b"]
            return _Resp()

    class _Client:
        embeddings = _Embeddings()

    lp = LocalProvider(model="m", base_url="http://x/v1", api_key="k", embed_model="emb-model")
    lp._client = _Client()  # bypass _get_client (no network)
    assert lp.embeddings(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_local_embeddings_redacts_error_to_type_name():
    from llmcode.providers import LocalProvider

    class _Embeddings:
        def create(self, model, input):
            raise ValueError("secret sk-xxx http://internal/leak")

    class _Client:
        embeddings = _Embeddings()

    lp = LocalProvider(model="m", base_url="http://x/v1", api_key="k", embed_model="e")
    lp._client = _Client()
    with pytest.raises(RuntimeError) as ei:
        lp.embeddings(["x"])
    msg = str(ei.value)
    assert msg == "ValueError"  # only the type name, no SDK metadata
    assert "secret" not in msg and "internal" not in msg and "leak" not in msg


def test_providers_import_without_openai():
    # Ensure llmcode.providers is importable even if 'openai' is absent.
    saved = sys.modules.pop("openai", None)
    # importlib.reload re-runs the module body IN PLACE on the same module
    # object, replacing class objects with NEW identities. Every other module
    # that did `from llmcode.providers import MockProvider/LocalProvider` (and
    # every test that captured those names at import/collection time) still
    # holds the ORIGINAL class objects, so an isinstance() across them breaks
    # for the rest of the suite. Snapshot the originals and restore them in
    # finally so the reload this test needs does not leak new identities.
    import llmcode.providers as _prov
    _orig_classes = {
        k: getattr(_prov, k)
        for k in ("MockProvider", "LocalProvider", "Provider")
    }
    try:
        sys.modules["openai"] = None  # force ImportError on `import openai`
        importlib.reload(importlib.import_module("llmcode.providers"))
        mod = importlib.import_module("llmcode.providers")
        # MockProvider must work with no openai.
        events = list(mod.MockProvider().stream_chat([], None))
        assert events[-1]["type"] == "done"
        # LocalProvider can be constructed (openai only imported on first call).
        lp = mod.LocalProvider(model="m", base_url="http://x/v1", api_key="k")
        assert lp.name == "local"
    finally:
        sys.modules.pop("openai", None)
        if saved is not None:
            sys.modules["openai"] = saved
        # Reload with openai restored so the module returns to its normal state,
        # then put the ORIGINAL class objects back so identities stay consistent
        # with every other module's captured `from llmcode.providers import ...`.
        importlib.reload(importlib.import_module("llmcode.providers"))
        for k, v in _orig_classes.items():
            setattr(_prov, k, v)


# --------------------------------------------------------------------------- #
# build_provider threads embed_model into the LocalProvider (no network)
# --------------------------------------------------------------------------- #

def test_build_provider_threads_embed_model():
    from llmcode.repl import build_provider

    # No network: LocalProvider only imports/builds the openai client on first
    # call, so construction is fully offline.
    prov = build_provider(
        name="local", model="m", base_url="http://127.0.0.1:1234/v1",
        embed_model="E",
    )
    assert prov.name == "local"
    assert prov.embed_model == "E"


def test_build_provider_defaults_embed_model_none():
    from llmcode.repl import build_provider

    prov = build_provider(name="local", model="m", base_url="http://127.0.0.1:1234/v1")
    assert prov.embed_model is None


def test_config_embed_model_flows_to_provider():
    # Simulate the orchestrator/startup build path: the config's embed_model must
    # reach the constructed provider (the bug was build_provider dropping it).
    from llmcode.config import Config
    from llmcode.repl import build_provider

    config = Config(embed_model="nomic-test-embed")
    prov = build_provider(
        config.provider, config.model, config.base_url, config.effort,
        config.private, config.cache_prompt, config.max_output_tokens,
        embed_model=config.embed_model,
    )
    assert prov.embed_model == "nomic-test-embed"


# ----- Regression: sigil short-circuit in extract_text_tool_calls ----------

def test_extract_text_tool_calls_no_sigil_returns_none_fast():
    """A long pure-chat answer with no tool-call sigil must return None without
    running any parser (the common-path short-circuit, ~3 full-text regex passes
    avoided)."""
    from llmcode.providers import extract_text_tool_calls

    # A long answer that contains none of the sigils: no DeepSeek begin token,
    # no <function=, no <|python_tag|>, no [TOOL_CALLS], no triple backticks.
    text = "Here is a long plain answer. " * 5000
    assert extract_text_tool_calls(text) is None


def test_extract_text_tool_calls_short_circuit_still_detects_real_call():
    """Sanity: when a sigil IS present the short-circuit must NOT fire."""
    from llmcode.providers import extract_text_tool_calls

    text = "<function=read_file>{\"path\": \"main.py\"}</function>"
    calls = extract_text_tool_calls(text)
    assert calls and calls[0]["name"] == "read_file"


# ----- Regression: native streamed tool-call args list+join (kill O(n^2)) ---

def test_local_assembles_large_streamed_tool_call_args():
    """A large write_file payload streamed as MANY small argument deltas must
    assemble into the exact joined string and parse correctly (list+join, not
    O(n^2) string +=)."""
    import json as _json

    # Build a large payload fragmented across many deltas (the O(n^2) case).
    payload = {"path": "big.py", "content": "x = 1\n" * 4000}
    full = _json.dumps(payload)
    # Slice the JSON string into many small fragments (no respect for boundaries).
    frags = [full[i:i + 17] for i in range(0, len(full), 17)]
    chunks = [
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, id="call_x")]),
        _FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, name="write_file")]),
    ]
    for frag in frags:
        chunks.append(_FakeChunk(tool_calls=[_FakeToolCallDelta(index=0, arguments=frag)]))
    chunks.append(_FakeChunk(finish_reason="tool_calls"))

    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["path"] == "big.py"
    assert calls[0]["arguments"]["content"] == "x = 1\n" * 4000
    assert events[-1]["finish_reason"] == "tool_calls"


# ----- base_url normalization (bare host -> /v1) --------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        # Bare host (no path): /v1 is appended so the SDK hits the API root.
        ("http://127.0.0.1:1234", "http://127.0.0.1:1234/v1"),
        # Bare host with a trailing slash: same, without a doubled slash.
        ("http://127.0.0.1:1234/", "http://127.0.0.1:1234/v1"),
        # Already has the OpenAI path: unchanged.
        ("http://localhost:1234/v1", "http://localhost:1234/v1"),
        # Already has a path (even trailing-slash): unchanged, not re-appended.
        ("http://localhost:1234/v1/", "http://localhost:1234/v1/"),
        # An intentional custom endpoint (any non-empty path): respected as-is.
        ("http://host:8000/api", "http://host:8000/api"),
        # Garbage / non-URL string: returned unchanged, never raises.
        ("not a url", "not a url"),
    ],
)
def test_normalize_openai_base_url(raw, expected):
    from llmcode.providers import _normalize_openai_base_url

    assert _normalize_openai_base_url(raw) == expected


def test_local_provider_normalizes_bare_host_base_url():
    """A bare-host base_url is normalized to /v1 at construction so chat +
    embeddings + _get_client all target the OpenAI API root, not the server ROOT."""
    from llmcode.providers import LocalProvider

    lp = LocalProvider(model="m", base_url="http://127.0.0.1:1234", api_key="x")
    assert lp.base_url.endswith("/v1")
    assert lp.base_url == "http://127.0.0.1:1234/v1"


# ----- list_local_models: normalization + loopback proxy defense ----------

def _capture_list_models_kwargs(monkeypatch, base_url, private=False):
    """Run list_local_models with a fake openai (real httpx) and return the
    OpenAI(**kwargs) it was constructed with. No network: models.list() is stubbed."""
    import types

    from llmcode.providers import list_local_models

    captured: dict = {}

    class _FakeModels:
        def list(self):
            return types.SimpleNamespace(data=[])

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = _FakeModels()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    list_local_models(base_url, "k", private=private)
    return captured


def test_list_local_models_normalizes_bare_host(monkeypatch):
    # A bare-host loopback base_url is normalized to /v1 before the client is
    # built, so models.list() hits /v1/models (not the ROOT -> 0 models).
    captured = _capture_list_models_kwargs(monkeypatch, "http://127.0.0.1:1234")
    assert "/v1" in str(captured.get("base_url"))


def test_list_local_models_loopback_disables_proxy(monkeypatch):
    # A loopback metadata call in DEFAULT (non-private) mode must build an
    # explicit httpx client with trust_env=False so HTTP(S)_PROXY/ALL_PROXY
    # cannot tunnel the model-list call off-box (mirrors _get_client).
    captured = _capture_list_models_kwargs(monkeypatch, "http://127.0.0.1:1234/v1")
    http_client = captured.get("http_client")
    assert http_client is not None
    assert http_client.trust_env is False


def test_list_local_models_external_honors_proxy(monkeypatch):
    # A genuinely external (non-loopback, non-private) base_url stays on the
    # default client so proxy env vars are honored, as designed.
    captured = _capture_list_models_kwargs(monkeypatch, "http://example.com:1234/v1")
    assert "http_client" not in captured
