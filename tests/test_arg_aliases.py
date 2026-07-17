"""Tool-argument alias normalization.

Local models post-trained on Claude Code / OpenAI tool schemas emit the WRONG
parameter KEY — `file_path` instead of `path`, `old_string`/`new_string` instead
of `old`/`new`, `cmd` instead of `command`. The value is right; only the key name
disagrees with llmc's schema, so a strict `args.get("path")` saw None and the
call failed forever ("write_file requires a string 'path'"). normalize_tool_args
maps the aliases onto the canonical keys so the call just works.
"""

from __future__ import annotations

from pathlib import Path

from llmcli.agent import Agent
from llmcli.tools import normalize_tool_args


# --- unit: the normalizer ---------------------------------------------------- #

def test_write_file_file_path_alias():
    out = normalize_tool_args("write_file", {"file_path": "a.py", "content": "x"})
    assert out["path"] == "a.py"
    assert out["content"] == "x"


def test_read_file_aliases():
    for key in ("file_path", "filepath", "filename", "file", "target_file"):
        out = normalize_tool_args("read_file", {key: "a.py"})
        assert out["path"] == "a.py", key


def test_edit_file_claude_style_keys():
    out = normalize_tool_args(
        "edit_file",
        {"file_path": "a.py", "old_string": "foo", "new_string": "bar"},
    )
    assert out["path"] == "a.py"
    assert out["old"] == "foo"
    assert out["new"] == "bar"


def test_run_bash_command_aliases():
    for key in ("cmd", "bash", "shell", "script"):
        out = normalize_tool_args("run_bash", {key: "ls -la"})
        assert out["command"] == "ls -la", key


def test_grep_glob_pattern_aliases():
    assert normalize_tool_args("grep", {"query": "TODO"})["pattern"] == "TODO"
    assert normalize_tool_args("glob", {"glob": "**/*.py"})["pattern"] == "**/*.py"


def test_canonical_key_is_never_clobbered():
    # When the real key is present, an alias must NOT override it.
    out = normalize_tool_args(
        "write_file", {"path": "real.py", "file_path": "wrong.py", "content": "c"}
    )
    assert out["path"] == "real.py"


def test_empty_content_is_preserved_over_absent_alias():
    # A genuine empty-file write keeps content="" (no alias to fill from).
    out = normalize_tool_args("write_file", {"path": "a.py", "content": ""})
    assert out["content"] == ""


def test_explicit_empty_content_wins_over_competing_alias():
    # An intentional empty-file write keeps content="" even when a content-alias
    # key co-occurs — the model's explicit canonical value is authoritative.
    out = normalize_tool_args(
        "write_file", {"path": "a.py", "content": "", "text": "X"}
    )
    assert out["content"] == ""


def test_null_canonical_is_backfilled_from_alias():
    # An explicit null (not just absent) canonical key is still backfilled.
    out = normalize_tool_args(
        "write_file", {"path": "a.py", "content": None, "file_text": "hi"}
    )
    assert out["content"] == "hi"


def test_write_file_does_not_borrow_edit_new_string():
    # new_string is edit_file's `new`, NOT write_file content — must not become a
    # destructive whole-file overwrite of just the fragment.
    out = normalize_tool_args("write_file", {"path": "a.py", "new_string": "frag"})
    assert "content" not in out or not out.get("content")


def test_unknown_tool_and_non_dict_pass_through():
    assert normalize_tool_args("mcp__server__do", {"file_path": "z"}) == {"file_path": "z"}
    assert normalize_tool_args("spawn_agent", {"role": "coder"}) == {"role": "coder"}
    assert normalize_tool_args("write_file", "not a dict") == "not a dict"


def test_normalizer_does_not_mutate_input_when_no_change():
    src = {"path": "a.py", "content": "x"}
    out = normalize_tool_args("write_file", src)
    assert out is src  # unchanged → same object, no needless copy


# --- integration: the agent loop actually writes the file -------------------- #

class _Prov:
    model = "m"

    def __init__(self, scripts):
        self.scripts = scripts
        self.n = 0

    def stream_chat(self, messages, tools):
        script = self.scripts[min(self.n, len(self.scripts) - 1)]
        self.n += 1
        yield from script


def _agent(prov, tools, **kw):
    kw.setdefault("max_iterations", 10)
    return Agent(prov, "sys", list(tools), console=None, auto_confirm=True, **kw)


def test_write_file_with_file_path_alias_actually_writes(tmp_workspace):
    # The screenshot bug end-to-end: the model emits write_file with `file_path`
    # (not `path`). Before the fix this looped forever on "requires a string
    # 'path'"; now the file lands on disk.
    body = "def f():\n    return 1\n"
    script_write = [
        {"type": "tool_call", "id": "w1", "name": "write_file",
         "arguments": {"file_path": "token_tracker.py", "content": body}},
        {"type": "done", "finish_reason": "tool_calls"},
    ]
    script_answer = [
        {"type": "text", "text": "Wrote the file."},
        {"type": "done", "finish_reason": "stop"},
    ]
    p = _Prov([script_write, script_answer])
    out = _agent(p, ["write_file"]).run("write token_tracker.py")
    assert out == "Wrote the file."
    written = Path(tmp_workspace) / "token_tracker.py"
    assert written.exists()
    assert written.read_text() == body


def test_edit_file_with_claude_keys_actually_edits(tmp_workspace):
    target = Path(tmp_workspace) / "m.py"
    target.write_text("x = 1\n")
    script_edit = [
        {"type": "tool_call", "id": "e1", "name": "edit_file",
         "arguments": {"file_path": "m.py", "old_string": "x = 1", "new_string": "x = 2"}},
        {"type": "done", "finish_reason": "tool_calls"},
    ]
    script_answer = [
        {"type": "text", "text": "Edited."},
        {"type": "done", "finish_reason": "stop"},
    ]
    p = _Prov([script_edit, script_answer])
    out = _agent(p, ["edit_file"]).run("bump x to 2")
    assert out == "Edited."
    assert target.read_text() == "x = 2\n"
