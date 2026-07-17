"""In-template (text) tool-call extraction tests for local models.

Covers ``extract_text_tool_calls`` for every supported chat-template format
(DeepSeek / Qwen+Hermes / Mistral / Llama), the false-positive guards, and a
stream-level check that a deepseek-style text turn yields tool_call events
instead of a (hallucinated) text answer. No network, no openai import.
"""

from __future__ import annotations

import json
import time

from llmcli.providers import extract_text_tool_calls

# DeepSeek control-token delimiters (NON-ASCII; built from code points so the
# literals in this test exactly match the ones in providers.py):
#   U+FF5C  ｜  FULLWIDTH VERTICAL LINE
#   U+2581  ▁  LOWER ONE EIGHTH BLOCK
_BAR = "｜"
_USC = "▁"
_DS_CALLS_BEGIN = f"<{_BAR}tool{_USC}calls{_USC}begin{_BAR}>"
_DS_CALLS_END = f"<{_BAR}tool{_USC}calls{_USC}end{_BAR}>"
_DS_CALL_BEGIN = f"<{_BAR}tool{_USC}call{_USC}begin{_BAR}>"
_DS_CALL_END = f"<{_BAR}tool{_USC}call{_USC}end{_BAR}>"
_DS_SEP = f"<{_BAR}tool{_USC}sep{_BAR}>"
_DS_OUT_BEGIN = f"<{_BAR}tool{_USC}outputs{_USC}begin{_BAR}>"
_DS_OUT_END = f"<{_BAR}tool{_USC}outputs{_USC}end{_BAR}>"


def _ds_call(name: str, args_json: str) -> str:
    return f"{_DS_CALL_BEGIN}function{_DS_SEP}{name}\n```json\n{args_json}\n```{_DS_CALL_END}"


# --------------------------------------------------------------------------- #
# DeepSeek-V2 / deepseek-coder-v2
# --------------------------------------------------------------------------- #

def test_deepseek_single_call():
    text = _DS_CALLS_BEGIN + _ds_call("read_file", '{"path": "main.py"}') + _DS_CALLS_END
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "main.py"}}]


def test_deepseek_single_call_missing_outer_end():
    # The outer <｜tool▁calls▁end｜> may be ABSENT for a single call.
    text = _DS_CALLS_BEGIN + _ds_call("glob", '{"pattern": "*.py"}')
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "glob", "arguments": {"pattern": "*.py"}}]


def test_deepseek_bare_args_no_fence():
    # Tolerant: NAME up to newline, ARGS the first JSON object even without a
    # ```json fence.
    text = _DS_CALL_BEGIN + f"function{_DS_SEP}run_bash\n" + '{"command": "ls"}' + _DS_CALL_END
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "run_bash", "arguments": {"command": "ls"}}]


def test_deepseek_multiple_calls():
    text = (
        _DS_CALLS_BEGIN
        + _ds_call("write_file", '{"path": "a.txt", "content": "x"}')
        + _ds_call("run_bash", '{"command": "cat a.txt"}')
        + _DS_CALLS_END
    )
    calls = extract_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["write_file", "run_bash"]
    assert calls[0]["arguments"] == {"path": "a.txt", "content": "x"}
    assert calls[1]["arguments"] == {"command": "cat a.txt"}


def test_deepseek_malformed_args_sets_parse_error():
    text = _DS_CALL_BEGIN + f"function{_DS_SEP}run_bash\n```json\n" + "{not valid}" + "\n```" + _DS_CALL_END
    calls = extract_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "run_bash"
    assert calls[0]["arguments"] == {}
    assert calls[0].get("_parse_error")


def test_deepseek_in_the_wild_ignores_hallucinated_tail():
    """EXACT in-the-wild shape: two real calls, then a HALLUCINATED tool-output
    block + a prose answer the server let the model continue with. Only the two
    real calls must be extracted; the fake tail is discarded."""
    text = (
        _DS_CALLS_BEGIN
        + _ds_call("write_file", '{"path": "answer.txt", "content": "5"}')
        + _ds_call("run_bash", '{"command": "cat answer.txt"}')
        + _DS_CALLS_END
        + _DS_OUT_BEGIN + '{"output":"5"}' + _DS_OUT_END
        + "\n\nI created answer.txt and it contains 5. Done."
    )
    calls = extract_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["write_file", "run_bash"]
    assert calls[0]["arguments"] == {"path": "answer.txt", "content": "5"}
    assert calls[1]["arguments"] == {"command": "cat answer.txt"}
    # No phantom third call from the hallucinated <｜tool▁outputs▁begin｜> block.
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# Qwen2.5 / Qwen3 + Nous Hermes-2-Pro / Hermes-3 (identical <tool_call> markup)
# --------------------------------------------------------------------------- #

def test_qwen_single_call():
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "m.py"}}\n</tool_call>'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "m.py"}}]


def test_qwen_nested_args_object():
    # Balanced slicing must not stop at the first inner '}'.
    text = '<tool_call>{"name": "cfg", "arguments": {"a": {"b": 1}, "c": 2}}</tool_call>'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "cfg", "arguments": {"a": {"b": 1}, "c": 2}}]


def test_qwen_multiple_calls():
    text = (
        '<tool_call>{"name": "glob", "arguments": {"pattern": "*.py"}}</tool_call>\n'
        '<tool_call>{"name": "grep", "arguments": {"pattern": "def"}}</tool_call>'
    )
    calls = extract_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["glob", "grep"]
    assert calls[1]["arguments"] == {"pattern": "def"}


def test_qwen_args_as_json_string_coerced():
    # arguments delivered as a JSON STRING -> coerced to a dict.
    text = '<tool_call>{"name": "glob", "arguments": "{\\"pattern\\": \\"*.py\\"}"}</tool_call>'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "glob", "arguments": {"pattern": "*.py"}}]


def test_qwen_malformed_args_string_sets_parse_error():
    text = '<tool_call>{"name": "glob", "arguments": "{bad json"}</tool_call>'
    calls = extract_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "glob"
    assert calls[0]["arguments"] == {}
    assert calls[0].get("_parse_error")


def test_qwen_strips_leading_think_block():
    text = (
        "<think>I should read the file first.</think>\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "x"}}</tool_call>'
    )
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "x"}}]


# --------------------------------------------------------------------------- #
# Mistral / Mixtral / Nemo
# --------------------------------------------------------------------------- #

def test_mistral_single_call():
    text = '[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "m.py"}}]'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "m.py"}}]


def test_mistral_multiple_calls_and_ignores_id():
    text = (
        '[TOOL_CALLS][{"id": "abc", "name": "glob", "arguments": {"pattern": "*"}}, '
        '{"name": "grep", "arguments": {"pattern": "x"}}]'
    )
    calls = extract_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["glob", "grep"]
    # id is ignored; only name/arguments survive.
    assert "id" not in calls[0]
    assert calls[0]["arguments"] == {"pattern": "*"}


def test_mistral_args_as_string_coerced():
    text = '[TOOL_CALLS] [{"name": "glob", "arguments": "{\\"pattern\\": \\"*.py\\"}"}]'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "glob", "arguments": {"pattern": "*.py"}}]


def test_mistral_malformed_element_args_sets_parse_error():
    text = '[TOOL_CALLS] [{"name": "run_bash", "arguments": "{oops"}]'
    calls = extract_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "run_bash"
    assert calls[0]["arguments"] == {}
    assert calls[0].get("_parse_error")


# --------------------------------------------------------------------------- #
# Llama 3.1 / 3.2 / 3.3
# --------------------------------------------------------------------------- #

def test_llama_function_tag_single():
    text = '<function=get_weather>{"location": "SF"}</function>'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "get_weather", "arguments": {"location": "SF"}}]


def test_llama_function_tag_multiple():
    text = (
        '<function=glob>{"pattern": "*.py"}</function>'
        '<function=grep>{"pattern": "def"}</function>'
    )
    calls = extract_text_tool_calls(text)
    assert [c["name"] for c in calls] == ["glob", "grep"]
    assert calls[1]["arguments"] == {"pattern": "def"}


def test_llama_function_tag_malformed_args():
    text = "<function=run_bash>{not json}</function>"
    calls = extract_text_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "run_bash"
    assert calls[0]["arguments"] == {}
    assert calls[0].get("_parse_error")


def test_llama_python_tag_parameters_key():
    text = '<|python_tag|>{"name": "read_file", "parameters": {"path": "m.py"}}'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "m.py"}}]


def test_llama_python_tag_arguments_key():
    text = '<|python_tag|>{"name": "glob", "arguments": {"pattern": "*"}}'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "glob", "arguments": {"pattern": "*"}}]


def test_llama_pythonic_bracket_form_is_skipped():
    # The ambiguous [func(a='x')] form is intentionally NOT recognized.
    assert extract_text_tool_calls("[read_file(path='m.py')]") is None


# --------------------------------------------------------------------------- #
# FALSE-POSITIVE GUARDS
# --------------------------------------------------------------------------- #

def test_guard_tool_call_inside_code_fence_is_ignored():
    # (a) A fenced example explaining the <tool_call> format must NOT execute.
    text = (
        "Qwen tool calls look like this:\n"
        "```\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "x"}}</tool_call>\n'
        "```\n"
        "Hope that helps!"
    )
    assert extract_text_tool_calls(text) is None


def test_guard_function_tag_inside_code_fence_is_ignored():
    text = (
        "Llama uses:\n```\n<function=foo>{\"a\": 1}</function>\n```\nThat's the shape."
    )
    assert extract_text_tool_calls(text) is None


def test_guard_plain_prose_returns_none():
    # (b) A normal answer with no markup is not a tool call.
    assert extract_text_tool_calls("The answer is 42. No tools were needed.") is None
    assert extract_text_tool_calls("") is None


def test_guard_generic_json_fence_returns_none_routes_to_existing_path():
    # (c) The generic ```json {"tool":..} fence is NOT a high-confidence format;
    # extract_text_tool_calls returns None so the existing strict generic-fence
    # path (parse_tool_block + sole-content gating) handles it instead.
    text = '```json\n{"tool": "read_file", "input": {"path": "x"}}\n```'
    assert extract_text_tool_calls(text) is None


# --------------------------------------------------------------------------- #
# ADVERSARIAL false-positive repros (independent security review). Each MUST
# return None (NOT be executed as a tool call).
# --------------------------------------------------------------------------- #

def test_guard_unterminated_code_fence_is_ignored():
    # HIGH-1: an UNCLOSED ``` (model forgot the closer, or truncated at the token
    # cap) still hides everything after it — markup inside must NOT execute.
    text = (
        "```\n"
        '<tool_call>{"name":"run_bash","arguments":{"command":"rm -rf ~"}}'
    )
    assert extract_text_tool_calls(text) is None


def test_guard_unterminated_fence_does_not_swallow_a_preceding_real_call():
    # Sanity (HIGH-1 direction): a real call BEFORE the unterminated fence is
    # still extracted; only the fenced remainder is stripped.
    text = (
        '<tool_call>{"name": "glob", "arguments": {"pattern": "*.py"}}</tool_call>\n'
        "```\n<tool_call>{\"name\":\"run_bash\",\"arguments\":{\"command\":\"rm -rf ~\"}}"
    )
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "glob", "arguments": {"pattern": "*.py"}}]


def test_guard_bare_function_tag_no_body_is_ignored():
    # HIGH-2: a bare <function=NAME> with no JSON body must NOT synthesize an
    # empty-args call.
    assert extract_text_tool_calls("use the <function=submit> directive") is None


def test_guard_function_tag_distant_brace_is_ignored():
    # HIGH-2: with no </function> and only a distant, unrelated {...}, the body is
    # not immediately after '>', so no call is synthesized.
    text = '<function=foo> ... later unrelated prose ... {"x":1}'
    assert extract_text_tool_calls(text) is None


def test_guard_tool_call_inside_single_backtick_inline_code_is_ignored():
    # HIGH-3: single-backtick inline code is documentation too.
    text = '`<tool_call>{"name":"run_bash","arguments":{"command":"rm -rf ~"}}</tool_call>`'
    assert extract_text_tool_calls(text) is None


def test_guard_deepseek_example_inside_code_fence_is_ignored():
    # MED-4: a fenced EXAMPLE of the DeepSeek control-token format must NOT
    # execute, even though DeepSeek is parsed before the string-level stripping.
    text = (
        "```\n"
        + _DS_CALLS_BEGIN
        + _ds_call("run_bash", '{"command":"rm -rf ~"}')
        + _DS_CALLS_END
        + "\n```"
    )
    assert extract_text_tool_calls(text) is None


def test_guard_invalid_tool_name_is_not_emitted():
    # LOW-6: a non-identifier synthesized name is garbage; refuse to emit it.
    assert extract_text_tool_calls('<function=rm -rf>{"x":1}</function>') is None
    text = '<tool_call>{"name": "../../etc/passwd", "arguments": {}}</tool_call>'
    assert extract_text_tool_calls(text) is None


# --------------------------------------------------------------------------- #
# ARG-PRESERVATION: backticks / fences INSIDE a REAL call's argument STRING
# VALUE must survive VERBATIM. The old example-guard worked by .sub("")-DELETING
# fenced/backtick spans from the text BEFORE the JSON was sliced, which silently
# corrupted args (HIGH-A): the wrong command would execute, or written content
# would be mangled. The fix slices the JSON body from the ORIGINAL text, so these
# spans are preserved.
# --------------------------------------------------------------------------- #

def test_qwen_preserves_backticks_in_command_arg():
    # Repro: deleting `date` from the command would run a DIFFERENT command.
    text = (
        '<tool_call>{"name":"run_bash","arguments":'
        '{"command":"echo `date`"}}</tool_call>'
    )
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "run_bash", "arguments": {"command": "echo `date`"}}]
    assert calls[0]["arguments"]["command"] == "echo `date`"  # backticks intact


def test_qwen_preserves_inline_backticks_in_content_arg():
    text = (
        '<tool_call>{"name":"write_file","arguments":'
        '{"path":"x","content":"use `ls` here"}}</tool_call>'
    )
    calls = extract_text_tool_calls(text)
    assert calls[0]["arguments"]["content"] == "use `ls` here"


def test_qwen_preserves_triple_fence_block_in_content_arg():
    # A ```python ... ``` block inside the content value must NOT be stripped.
    content = "see ```python\nx=1\n``` end"
    text = (
        "<tool_call>"
        + json.dumps({"name": "write_file", "arguments": {"path": "x", "content": content}})
        + "</tool_call>"
    )
    calls = extract_text_tool_calls(text)
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["content"] == content  # fence preserved verbatim


def test_llama_function_tag_preserves_triple_fence_block_in_content_arg():
    content = "see ```python\nx=1\n``` end"
    text = (
        "<function=write_file>"
        + json.dumps({"path": "x", "content": content})
        + "</function>"
    )
    calls = extract_text_tool_calls(text)
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["content"] == content  # fence preserved verbatim


def test_mistral_preserves_backticks_in_arg():
    text = '[TOOL_CALLS] [{"name":"run_bash","arguments":{"command":"echo `pwd`"}}]'
    calls = extract_text_tool_calls(text)
    assert calls[0]["arguments"]["command"] == "echo `pwd`"


def test_llama_function_tag_preserves_backticks_in_arg():
    text = '<function=run_bash>{"command":"echo `hi`"}</function>'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "run_bash", "arguments": {"command": "echo `hi`"}}]


# --------------------------------------------------------------------------- #
# SPLIT-SIGIL: a control sigil broken ACROSS a fence / inline-code span must NOT
# re-form into a live token. The old code .sub("")-stripped the span, which
# CONCATENATED the remainder back into a firing sigil (MED-B). Scanning the
# ORIGINAL (never-mutated) text means the sigil never exists contiguously, so no
# call is synthesized. Each MUST return None.
# --------------------------------------------------------------------------- #

def test_split_qwen_sigil_across_triple_fence_is_not_a_call():
    text = (
        "<tool_" + "```x```" + "call>"
        '{"name":"run_bash","arguments":{"command":"rm -rf ~"}}</tool_call>'
    )
    assert extract_text_tool_calls(text) is None


def test_split_llama_function_sigil_across_inline_code_is_not_a_call():
    text = "<func" + "`y`" + 'tion=run_bash>{"command":"x"}'
    assert extract_text_tool_calls(text) is None


def test_split_mistral_sigil_across_inline_code_is_not_a_call():
    text = "[TOOL" + "`z`" + '_CALLS] [{"name":"glob","arguments":{"pattern":"*"}}]'
    assert extract_text_tool_calls(text) is None


# --------------------------------------------------------------------------- #
# Stream-level: a deepseek-style TEXT turn yields tool_call events, not an answer
# --------------------------------------------------------------------------- #

class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = None


class _FakeChoice:
    def __init__(self, content=None, finish_reason=None):
        self.delta = _FakeDelta(content)
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content=None, finish_reason=None):
        self.choices = [_FakeChoice(content, finish_reason)]
        self.usage = None


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


def _local_with_stream(chunks):
    from llmcli.providers import LocalProvider

    lp = LocalProvider(model="m", base_url="http://x/v1", api_key="k")
    stream = _FakeStream(chunks)

    class _FakeCompletions:
        def create(self, **kwargs):
            return stream

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    lp._client = _FakeClient()
    return lp


def test_stream_deepseek_text_turn_yields_tool_calls_not_answer():
    """A server that left DeepSeek markup as plain text (no native tool_calls)
    must produce tool_call events + finish_reason='tool_calls', NOT a text
    answer carrying the model's hallucinated tool output."""
    body = (
        _DS_CALLS_BEGIN
        + _ds_call("write_file", '{"path": "answer.txt", "content": "5"}')
        + _ds_call("run_bash", '{"command": "cat answer.txt"}')
        + _DS_CALLS_END
        + _DS_OUT_BEGIN + '{"output":"5"}' + _DS_OUT_END
        + "\n\nDone, the file contains 5."
    )
    chunks = [_FakeChunk(content=body), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([{"role": "user", "content": "make it"}], None))

    calls = [e for e in events if e["type"] == "tool_call"]
    assert [c["name"] for c in calls] == ["write_file", "run_bash"]
    assert all(c.get("_from_text_fence") is True for c in calls)
    assert calls[0]["id"] == "call_0" and calls[1]["id"] == "call_1"
    assert calls[0]["arguments"] == {"path": "answer.txt", "content": "5"}
    # Terminal event is a tool_calls done, not a 'stop' text answer.
    assert events[-1]["type"] == "done"
    assert events[-1]["finish_reason"] == "tool_calls"


def test_stream_qwen_text_turn_yields_tool_call():
    body = '<tool_call>{"name": "glob", "arguments": {"pattern": "*.py"}}</tool_call>'
    chunks = [_FakeChunk(content=body), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == "glob"
    assert calls[0]["arguments"] == {"pattern": "*.py"}
    assert calls[0]["_from_text_fence"] is True
    assert events[-1]["finish_reason"] == "tool_calls"


def test_stream_malformed_args_propagate_parse_error_event():
    body = _DS_CALL_BEGIN + f"function{_DS_SEP}run_bash\n```json\n{{bad}}\n```" + _DS_CALL_END
    chunks = [_FakeChunk(content=body), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    calls = [e for e in lp.stream_chat([], None) if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["arguments"] == {}
    assert calls[0].get("_parse_error")


def test_stream_prose_answer_unaffected():
    # A plain prose turn (no markup) still streams text + a 'stop' done.
    chunks = [_FakeChunk(content="The answer is 42."), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    assert not [e for e in events if e["type"] == "tool_call"]
    assert events[-1]["finish_reason"] == "stop"


# --------------------------------------------------------------------------- #
# PERFORMANCE: degenerate token-repetition is a real local-model failure mode.
# ``extract_text_tool_calls`` runs SYNCHRONOUSLY after the stream, so a quadratic
# guard-span scan froze the REPL for ~6-10s on a single pathological reply. The
# guard lookup is now bisect-based (O(n log n) overall), so even 100KB+ of pure
# backticks must finish near-instantly. A generous 1.0s bound still fails hard on
# the old O(n^2) code (measured ~9.4s for ```*40000, ~6.5s for `*100000).
# --------------------------------------------------------------------------- #

def test_perf_degenerate_triple_backtick_run_is_not_quadratic():
    text = "```" * 40000  # 120KB of triple-backtick fences -> ~9.4s before the fix
    t0 = time.perf_counter()
    result = extract_text_tool_calls(text)
    elapsed = time.perf_counter() - t0
    assert result is None
    assert elapsed < 1.0, f"degenerate ``` parse took {elapsed:.3f}s (regression)"


def test_perf_degenerate_single_backtick_run_is_not_quadratic():
    text = "`" * 100000  # 100KB of single backticks -> ~6.5s before the fix
    t0 = time.perf_counter()
    result = extract_text_tool_calls(text)
    elapsed = time.perf_counter() - t0
    assert result is None
    assert elapsed < 1.0, f"degenerate ` parse took {elapsed:.3f}s (regression)"


def test_normal_mixed_content_answer_parses_correctly_and_fast():
    """A realistic markdown answer (many real ```fences``` + inline `code`) plus a
    fenced EXAMPLE call and ONE real trailing call: the example stays guarded, the
    real call is extracted with byte-exact args, and the whole parse stays fast."""
    block = (
        "Here is some prose with `inline_code` and a `second` snippet.\n"
        "```python\n"
        "def f(x):\n"
        "    return x + 1  # a real fenced code block\n"
        "```\n"
        "More text with `path/to/file.py` referenced inline.\n"
        "```json\n"
        '{"sample": "documentation only"}\n'
        "```\n"
    )
    fenced_example = (
        "Example of the format you could emit:\n"
        "```\n"
        '<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>\n'
        "```\n"
    )
    real_call = '<tool_call>{"name":"read_file","arguments":{"path":"main.py","note":"see `x`"}}</tool_call>'
    text = block * 60 + fenced_example + real_call  # ~30KB+ of mixed markdown

    t0 = time.perf_counter()
    calls = extract_text_tool_calls(text)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"normal mixed-content parse took {elapsed:.3f}s"
    # The fenced example is ignored; only the real trailing call is returned, with
    # its backtick-bearing arg value preserved byte-exact.
    assert calls == [
        {"name": "read_file", "arguments": {"path": "main.py", "note": "see `x`"}}
    ]
