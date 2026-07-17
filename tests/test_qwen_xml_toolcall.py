"""Qwen3 / Hermes XML tool-call format: <function=NAME><parameter=K>V</parameter>.

Ground-truth root cause (saved session llm_handler-c9a79f01ae39): qwen/qwen3.6
emits write_file as XML-parameter markup, NOT JSON. None of the other parsers
handled it, so the call was shown as prose and the file was never written
("write_file requires a string 'path'"). These tests pin the parser + the
empty-native-call recovery.
"""

from __future__ import annotations

from pathlib import Path

from llmcli.agent import Agent
from llmcli.providers import extract_text_tool_calls


def test_qwen_xml_write_file_exact_session_payload():
    # The literal text qwen emitted in the failing session (message [16]).
    text = (
        "Found the problem. Let me write it now.\n\n"
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>\nllm_handler_pkg/token_tracker.py\n</parameter>\n"
        "<parameter=content>\n"
        '"""Token budget tracking."""\nimport math\n\nclass TokenBudget:\n    pass\n'
        "</parameter>\n</function>\n</tool_call>"
    )
    calls = extract_text_tool_calls(text)
    assert calls is not None and len(calls) == 1
    c = calls[0]
    assert c["name"] == "write_file"
    assert c.get("_parse_error") is None
    assert c["arguments"]["path"] == "llm_handler_pkg/token_tracker.py"
    assert c["arguments"]["content"].startswith('"""Token budget tracking."""')
    assert "class TokenBudget" in c["arguments"]["content"]


def test_qwen_xml_no_closers_multiple_calls_split_correctly():
    # The screenshot bug: NO </parameter>/</function> closers, only </tool_call>;
    # several calls concatenated. Each must get its OWN distinct args (the first
    # call must not swallow the rest, and </tool_call> must not leak into a path).
    text = (
        "Let me read the files.\n\n"
        "<tool_call> <function=read_file> <parameter=limit> 200  "
        "<parameter=offset> 299  <parameter=path> a/llm_handler.py </tool_call> "
        "<tool_call> <function=read_file> <parameter=limit> 100  "
        "<parameter=offset> 100  <parameter=path> a/main.py </tool_call>"
    )
    calls = extract_text_tool_calls(text)
    assert calls is not None and len(calls) == 2
    assert calls[0]["arguments"] == {"limit": 200, "offset": 299, "path": "a/llm_handler.py"}
    assert calls[1]["arguments"] == {"limit": 100, "offset": 100, "path": "a/main.py"}
    # no </tool_call> leaked into any value
    assert all("tool_call" not in str(v) for c in calls for v in c["arguments"].values())


def test_qwen_xml_without_tool_call_wrapper():
    # Some templates omit the <tool_call> wrapper and emit <function=…> directly.
    text = (
        "<function=read_file><parameter=path>main.py</parameter></function>"
    )
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "main.py"}}]


def test_qwen_xml_edit_file_multi_params():
    text = (
        "<function=edit_file>"
        "<parameter=path>a.py</parameter>"
        "<parameter=old>x = 1</parameter>"
        "<parameter=new>x = 2</parameter>"
        "</function>"
    )
    calls = extract_text_tool_calls(text)
    assert calls == [
        {"name": "edit_file", "arguments": {"path": "a.py", "old": "x = 1", "new": "x = 2"}}
    ]


def test_qwen_xml_scalar_coercion_keeps_multiline_verbatim():
    text = (
        "<function=run_bash>"
        "<parameter=command>echo hi</parameter>"
        "<parameter=timeout>30</parameter>"
        "</function>"
    )
    calls = extract_text_tool_calls(text)
    args = calls[0]["arguments"]
    assert args["command"] == "echo hi"
    assert args["timeout"] == 30  # bare integer coerced


def test_qwen_xml_truncated_final_param_still_yields_path():
    # The content was cut off mid-stream (gentle cap): no closing </parameter> /
    # </function>. We must still recover path + the partial content.
    text = (
        "<function=write_file>\n"
        "<parameter=path>out.py</parameter>\n"
        "<parameter=content>\ndef f():\n    return 1\n    # cut off here"
    )
    calls = extract_text_tool_calls(text)
    assert calls is not None and calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["path"] == "out.py"
    assert "def f()" in calls[0]["arguments"]["content"]


def test_json_body_function_still_works():
    # Llama 3.x JSON body must NOT regress with the new param branch.
    text = '<function=read_file>{"path": "x.py"}</function>'
    calls = extract_text_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "x.py"}}]


def test_bare_function_tag_is_not_a_call():
    # No JSON, no parameters -> not a tool call (prose mentioning a function).
    assert extract_text_tool_calls("use <function=foo> somehow") is None


def test_content_containing_json_does_not_misfire_a_different_tool():
    # CRITICAL regression: write_file content that embeds a {"name":...} object
    # must NOT be sliced by _parse_qwen and fired as that tool. The real
    # write_file (with the JSON as its content) is what must execute.
    text = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>pkg.json</parameter>\n"
        '<parameter=content>{"name": "run_bash", "arguments": {"command": "echo pwned"}}</parameter>\n'
        "</function>\n</tool_call>"
    )
    calls = extract_text_tool_calls(text)
    assert calls is not None and len(calls) == 1
    assert calls[0]["name"] == "write_file"          # NOT run_bash
    assert calls[0]["arguments"]["path"] == "pkg.json"
    assert '"command": "echo pwned"' in calls[0]["arguments"]["content"]


def test_json_shape_call_with_function_tokens_in_content():
    # A legitimate JSON-format <tool_call> whose argument STRING contains the
    # literal "<function=" / "<parameter=" must still parse as the real call —
    # the _parse_qwen bail is gated on non-JSON bodies, so string-aware slicing
    # keeps those tokens inside the value (no content-to-execution misfire).
    text = (
        '<tool_call>{"name": "write_file", "arguments": '
        '{"path": "doc.md", "content": "see <function=x> and <parameter=y> tags"}}'
        "</tool_call>"
    )
    calls = extract_text_tool_calls(text)
    assert calls is not None and len(calls) == 1
    assert calls[0]["name"] == "write_file"          # NOT x
    assert calls[0]["arguments"]["path"] == "doc.md"
    assert calls[0]["arguments"]["content"] == "see <function=x> and <parameter=y> tags"


def test_single_line_scalar_content_stays_a_string():
    # MEDIUM regression: a one-line content of "42"/"true" must stay a STRING so
    # write_file accepts it (only true scalar keys like timeout are coerced).
    for val in ("42", "true", "null", "-7"):
        text = (
            f"<function=write_file><parameter=path>v.txt</parameter>"
            f"<parameter=content>{val}</parameter></function>"
        )
        args = extract_text_tool_calls(text)[0]["arguments"]
        assert args["content"] == val, val
        assert isinstance(args["content"], str)


# --- stream-level: through the REAL LocalProvider (where parsing happens) ----- #

class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, content=None, finish_reason=None, tool_calls=None):
        self.delta = _FakeDelta(content, tool_calls)
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content=None, finish_reason=None, tool_calls=None):
        self.choices = [_FakeChoice(content, finish_reason, tool_calls)]
        self.usage = None


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


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


def test_stream_qwen_xml_text_turn_yields_write_file_call():
    # The real provider must convert qwen's XML-param TEXT into a tool_call event.
    body = (
        "<tool_call>\n<function=write_file>\n"
        "<parameter=path>token_tracker.py</parameter>\n"
        '<parameter=content>\nVALUE = 42\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    chunks = [_FakeChunk(content=body), _FakeChunk(finish_reason="stop")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["path"] == "token_tracker.py"
    assert calls[0]["arguments"]["content"].strip() == "VALUE = 42"
    assert calls[0]["_from_text_fence"] is True
    assert events[-1]["finish_reason"] == "tool_calls"


class _FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeNativeTC:
    def __init__(self, index, name, arguments, id="call_0"):
        self.index = index
        self.id = id
        self.function = _FakeFn(name, arguments)


def test_stream_empty_native_call_recovers_args_from_text():
    # LM Studio made a NATIVE write_file call but dropped the XML body -> empty
    # args ({}). The same markup survives in the text stream; the provider must
    # recover the real path/content so the write isn't lost.
    markup = (
        "<function=write_file>"
        "<parameter=path>recovered.py</parameter>"
        "<parameter=content>print(1)</parameter>"
        "</function>"
    )
    native = _FakeNativeTC(index=0, name="write_file", arguments="")  # empty args
    chunks = [
        _FakeChunk(content=markup),
        _FakeChunk(tool_calls=[native]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"] == {"path": "recovered.py", "content": "print(1)"}
    assert "_parse_error" not in calls[0]


def test_stream_empty_native_call_without_text_markup_stays_empty():
    # No text markup to recover from -> args stay empty (the agent then feeds back
    # the actionable error). The recovery must not invent arguments.
    native = _FakeNativeTC(index=0, name="write_file", arguments="")
    chunks = [_FakeChunk(tool_calls=[native]), _FakeChunk(finish_reason="tool_calls")]
    lp = _local_with_stream(chunks)
    events = list(lp.stream_chat([], None))
    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls[0]["arguments"] == {}


def test_two_empty_native_calls_recover_distinct_args():
    # MEDIUM regression: two arg-less native write_file calls must backfill from
    # their OWN text markup, not both from the first one.
    markup = (
        "<function=write_file><parameter=path>a.py</parameter>"
        "<parameter=content>AAA</parameter></function>"
        "<function=write_file><parameter=path>b.py</parameter>"
        "<parameter=content>BBB</parameter></function>"
    )
    tc0 = _FakeNativeTC(index=0, name="write_file", arguments="", id="c0")
    tc1 = _FakeNativeTC(index=1, name="write_file", arguments="", id="c1")
    chunks = [
        _FakeChunk(content=markup),
        _FakeChunk(tool_calls=[tc0, tc1]),
        _FakeChunk(finish_reason="tool_calls"),
    ]
    lp = _local_with_stream(chunks)
    calls = [e for e in lp.stream_chat([], None) if e["type"] == "tool_call"]
    assert len(calls) == 2
    assert calls[0]["arguments"] == {"path": "a.py", "content": "AAA"}
    assert calls[1]["arguments"] == {"path": "b.py", "content": "BBB"}


def test_qwen_xml_text_call_writes_file_end_to_end(tmp_workspace):
    # Full path: a provider that yields the parsed tool_call (as the real provider
    # would) -> the agent executes it -> the file lands on disk.
    class _Prov:
        model = "m"

        def __init__(self, scripts):
            self.scripts, self.n = scripts, 0

        def stream_chat(self, messages, tools):
            s = self.scripts[min(self.n, len(self.scripts) - 1)]
            self.n += 1
            yield from s

    body = '"""tracker"""\nVALUE = 42\n'
    p = _Prov([
        [{"type": "tool_call", "id": "w1", "name": "write_file",
          "arguments": {"path": "token_tracker.py", "content": body}},
         {"type": "done", "finish_reason": "tool_calls"}],
        [{"type": "text", "text": "Done."}, {"type": "done", "finish_reason": "stop"}],
    ])
    agent = Agent(p, "sys", ["write_file"], console=None, auto_confirm=True,
                  max_iterations=10)
    agent.run("write token_tracker.py")
    written = Path(tmp_workspace) / "token_tracker.py"
    assert written.exists() and written.read_text() == body


# --- follow-through guard: tool markup shown as text -> re-prompt, not printed -- #

class _ScriptProv:
    model = "m"

    def __init__(self, scripts):
        self.scripts, self.n = scripts, 0

    def stream_chat(self, messages, tools=None, tool_choice=None):
        s = self.scripts[min(self.n, len(self.scripts) - 1)]
        self.n += 1
        yield from s


def test_unexecuted_tool_markup_triggers_followup_not_raw_output():
    from llmcli.agent import looks_like_unexecuted_tool_call
    assert looks_like_unexecuted_tool_call("<tool_call> <function=read_file> ...")
    assert looks_like_unexecuted_tool_call("sure, <function=foo> bar")
    assert not looks_like_unexecuted_tool_call("the file has 3 functions")

    # A provider that emits tool markup AS TEXT (not native/parsed) on turn 1,
    # then a clean answer after the nudge.
    markup_as_text = [
        {"type": "text", "text": "Let me read it.\n<tool_call> <function=read_file>"
                                 " <parameter=path> a.py </tool_call>"},
        {"type": "done", "finish_reason": "stop"},
    ]
    clean = [{"type": "text", "text": "Here is the answer."},
             {"type": "done", "finish_reason": "stop"}]
    p = _ScriptProv([markup_as_text, clean])
    out = Agent(p, "sys", ["read_file"], console=None, auto_confirm=True,
                max_iterations=4).run("what does a.py do?")
    assert out == "Here is the answer."          # recovered, NOT the raw markup
    assert "<function=" not in out
    assert p.n == 2                               # original + exactly one re-prompt


def test_xml_content_with_inner_markup_is_preserved_not_truncated():
    # HIGH bug: a write_file content that itself contains <parameter=/<function=
    # must be preserved verbatim, not truncated at the inner token (which also
    # invented a bogus arg). Hits when editing files that contain tool-call markup
    # — including llmc's OWN parser source.
    body = 'def f(): return "<parameter=foo> and <function=bar>"'
    text = (
        "<function=write_file><parameter=path>x.py</parameter>"
        f"<parameter=content>{body}</parameter></function>"
    )
    calls = extract_text_tool_calls(text)
    assert calls is not None and len(calls) == 1     # NOT split into bogus calls
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["path"] == "x.py"
    assert calls[0]["arguments"]["content"] == body  # full content preserved
    assert "foo" not in calls[0]["arguments"]        # no invented param


def test_followthrough_guard_ignores_fenced_examples():
    from llmcli.agent import looks_like_unexecuted_tool_call as f
    # markup shown INSIDE a code fence / inline code is an EXAMPLE -> no nudge
    assert f("Use ```\n<function=read_file>\n``` like this") is False
    assert f("call it via `<tool_call>` markup") is False
    assert f("<think><function=x></think> ok") is False   # leading think stripped
    # real, un-fenced markup still triggers
    assert f("Let me read it.\n<tool_call> <function=read_file> <parameter=path> a.py") is True
