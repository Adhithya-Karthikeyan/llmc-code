"""Unit tests for the Claude-Code-style terminal renderer helpers.

These cover the PURE helpers (display-name mapping, per-tool result summaries,
the call label, and the error summary) plus the live two-line ⏺/⎿ tree the
agent loop renders, all driven by the offline MockProvider / direct calls.
"""

from __future__ import annotations

import pytest

from llmcode.agent import (
    Agent,
    display_name,
    error_summary,
    result_summary,
    tool_call_label,
)
from llmcode.providers import MockProvider
from llmcode.tools import FULL


# ----- DisplayName mapping -------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("read_file", "Read"),
    ("write_file", "Write"),
    ("edit_file", "Edit"),
    ("run_bash", "Bash"),
    ("glob", "Glob"),
    ("grep", "Grep"),
    ("web_fetch", "Fetch"),
    ("spawn_agent", "Task"),
    ("mcp__kyp-mem__kyp_search", "kyp-mem:kyp_search"),
    ("mcp__srv__do__thing", "srv:do__thing"),  # only first __ splits server
    ("some_unknown_tool", "some_unknown_tool"),  # unmapped -> raw
])
def test_display_name_mapping(raw, expected):
    assert display_name(raw) == expected


def test_tool_call_label_includes_concise_args():
    assert tool_call_label("read_file", {"path": "README.md"}) == "Read(README.md)"
    assert tool_call_label("run_bash", {"command": "pytest -q"}) == "Bash(pytest -q)"
    assert tool_call_label("glob", {"pattern": "**/*.py"}) == "Glob(**/*.py)"


def test_tool_call_label_truncates_long_bash():
    long = "echo " + "x" * 80
    label = tool_call_label("run_bash", {"command": long})
    assert label.startswith("Bash(echo ")
    assert label.endswith("...)")


def test_tool_call_label_no_args_no_parens():
    # An empty summary renders as a bare display name, no "()".
    assert tool_call_label("glob", {}) == "Glob"


# ----- result summaries ----------------------------------------------------

def test_result_summary_read_file():
    res = {"ok": True, "result": "line1\nline2\nline3\n"}
    assert result_summary("read_file", res) == "Read 3 lines"


def test_result_summary_write_file_bytes():
    res = {"ok": True, "result": {"path": "f.py", "bytes_written": 42}}
    assert result_summary("write_file", res) == "Wrote 42 bytes"


def test_result_summary_write_file_created_fallback():
    res = {"ok": True, "result": {"path": "f.py"}}
    assert result_summary("write_file", res) == "Created f.py"


def test_result_summary_edit_file():
    res = {"ok": True, "result": {"path": "f.py", "replacements": 1}}
    assert result_summary("edit_file", res) == "Updated f.py"


def test_result_summary_run_bash_stdout_first_line():
    res = {"ok": True, "result": {"stdout": "hello\nworld\nmore\n", "exit_code": 0}}
    assert result_summary("run_bash", res) == "hello (+2 lines)"


def test_result_summary_run_bash_single_line_no_extra():
    res = {"ok": True, "result": {"stdout": "only\n", "exit_code": 0}}
    assert result_summary("run_bash", res) == "only"


def test_result_summary_run_bash_no_stdout_shows_exit():
    res = {"ok": True, "result": {"stdout": "", "exit_code": 0}}
    assert result_summary("run_bash", res) == "exit 0"


def test_result_summary_run_bash_truncates_long_line():
    res = {"ok": True, "result": {"stdout": "x" * 120, "exit_code": 0}}
    out = result_summary("run_bash", res)
    assert len(out) == 60  # first 60 chars, single line, no "(+...)"


def test_result_summary_glob():
    res = {"ok": True, "result": {"matches": ["a", "b", "c"], "truncated": False}}
    assert result_summary("glob", res) == "3 files"


def test_result_summary_grep():
    res = {"ok": True, "result": {"matches": [{"file": "x"}], "truncated": False}}
    assert result_summary("grep", res) == "1 match"  # singular pluralization
    many = {"ok": True, "result": {"matches": [{"file": "x"}, {"file": "y"}]}}
    assert result_summary("grep", many) == "2 matches"


def test_result_summary_glob_pluralization():
    one = {"ok": True, "result": {"matches": ["a"], "truncated": False}}
    assert result_summary("glob", one) == "1 file"
    none = {"ok": True, "result": {"matches": [], "truncated": False}}
    assert result_summary("glob", none) == "0 files"


def test_result_summary_web_fetch():
    res = {"ok": True, "result": {"text": "hello world", "url": "http://x"}}
    assert result_summary("web_fetch", res) == "Fetched 11 chars"


def test_result_summary_spawn_agent_preview():
    # spawn_agent's payload is the sub-agent's summary string.
    res = {"ok": True, "result": "did the thing\nand more"}
    assert result_summary("spawn_agent", res) == "did the thing"


def test_result_summary_mcp_preview():
    res = {"ok": True, "result": "some mcp output text here"}
    assert result_summary("mcp__srv__tool", res) == "some mcp output text here"


def test_result_summary_tolerates_garbage_payload():
    # A weak model / odd tool can return a non-dict result; must not raise.
    assert result_summary("glob", {"ok": True, "result": None}) == "0 files"
    assert result_summary("read_file", {"ok": True}) == "Read 0 lines"
    assert result_summary("anything", "not a dict") == ""


# ----- error / declined summaries ------------------------------------------

def test_error_summary_first_line_only():
    res = {"ok": False, "error": "boom: bad thing\nsecond line ignored"}
    assert error_summary(res) == "boom: bad thing"


def test_error_summary_truncates():
    res = {"ok": False, "error": "e" * 200}
    assert len(error_summary(res)) == 70


def test_error_summary_default():
    assert error_summary({"ok": False}) == "failed"
    assert error_summary("not a dict") == "failed"


def test_error_summary_nonzero_exit_with_stderr():
    # run_bash marks ok=False with NO "error" key; the reason lives in stderr.
    res = {"ok": False, "result": {
        "exit_code": 1,
        "stdout": "",
        "stderr": "cat: /var/log/access.log: No such file or directory\nignored",
    }}
    assert error_summary(res) == (
        "exit 1: cat: /var/log/access.log: No such file or directory"
    )


def test_error_summary_nonzero_exit_no_stderr():
    # A command that fails silently (e.g. `exit 3`) still shows its exit code.
    res = {"ok": False, "result": {"exit_code": 3, "stdout": "", "stderr": ""}}
    assert error_summary(res) == "exit 3"


def test_error_summary_nonzero_exit_truncates():
    res = {"ok": False, "result": {"exit_code": 1, "stderr": "e" * 200}}
    assert len(error_summary(res)) == 70


def test_error_summary_prefers_explicit_error_over_exit():
    # When both an explicit error and an exit code exist (e.g. timeout), the
    # human-written error message wins over the raw exit code.
    res = {"ok": False, "error": "Command timed out after 15s.",
           "result": {"exit_code": -9, "stderr": "partial"}}
    assert error_summary(res) == "Command timed out after 15s."


# ----- live two-line tree rendering ----------------------------------------

def _run_capture(capsys, provider, **kw):
    from rich.console import Console

    agent = Agent(
        provider=provider, system_prompt="s", tool_names=FULL,
        auto_confirm=True, console=Console(markup=False), **kw,
    )
    agent.run("go")
    return capsys.readouterr().out, agent


def _run_capture_with_details(capsys, provider, **kw):
    """Run agent and also call render_details, returning (run_out, details_out, agent)."""
    from rich.console import Console

    agent = Agent(
        provider=provider, system_prompt="s", tool_names=FULL,
        auto_confirm=True, console=Console(markup=False), **kw,
    )
    agent.run("go")
    run_out = capsys.readouterr().out
    agent.render_details(agent.console)
    details_out = capsys.readouterr().out
    return run_out, details_out, agent


def test_live_tree_read_and_bash(tmp_workspace, capsys):
    # 'hello' scenario: write_file then run_bash, then a final text answer.
    run_out, details_out, agent = _run_capture_with_details(capsys, MockProvider(scenario="hello"))
    # run() emits the ◆ counts one-liner (✓2, no failures) + the dim ctrl-o hint.
    assert "◆" in run_out
    assert "2 tools" in run_out
    assert "✓2" in run_out
    assert "ctrl-o" in run_out
    # A modest all-green batch (≤5) AUTO-EXPANDS the tree inline (never hidden).
    assert "⏺ Write(hello.py)" in run_out
    assert "⎿" in run_out
    # render_details() reveals the full two-line tree per tool.
    assert "⏺ Write(hello.py)" in details_out
    assert "⎿  Wrote" in details_out
    assert "⏺ Bash(python3 hello.py)" in details_out
    assert "⎿  hi" in details_out  # run_bash stdout first line
    # Final Markdown answer + dim tok/s footer still present in run() output.
    assert "tok/s" in run_out
    # Full detail still buffered for Ctrl+O reveal.
    assert [r["name"] for r in agent.last_turn_details] == ["write_file", "run_bash"]


def test_live_tree_failure_renders_error(tmp_workspace, capsys):
    class _FailRead(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "read_file",
                       "arguments": {"path": "does_not_exist.txt"}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "ok."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    run_out, details_out, _ = _run_capture_with_details(capsys, _FailRead())
    # run() shows collapsed line with "1 tool (1 failed)" since the read failed.
    assert "⏺" in run_out
    assert "tool" in run_out
    # render_details() shows the ⏺/⎿ tree with the ✗ failure connector.
    assert "⏺ Read(does_not_exist.txt)" in details_out
    assert "✗" in details_out  # dim-red failure connector (unified ✗ glyph)


def test_live_tree_declined(tmp_workspace, capsys):
    from rich.console import Console

    agent = Agent(
        provider=MockProvider(scenario="hello"), system_prompt="s",
        tool_names=FULL, auto_confirm=False, confirm_fn=lambda t, a: False,
        console=Console(markup=False),
    )
    agent.run("go")
    capsys.readouterr()  # discard run() output
    agent.render_details(agent.console)
    out = capsys.readouterr().out
    assert "✗" in out  # declined calls show dim-red ✗ connector
    assert "declined" in out.lower() or "User declined" in out


def test_live_tree_unknown_tool(tmp_workspace, capsys):
    class _Unknown(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "does_not_exist",
                       "arguments": {}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "ok."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    run_out, details_out, _ = _run_capture_with_details(capsys, _Unknown())
    assert "⏺" in run_out  # collapsed summary in run()
    assert "✗" in details_out  # failure connector in render_details()
    assert "unknown" in details_out.lower() or "does_not_exist" in details_out


def test_live_tree_bad_args(tmp_workspace, capsys):
    class _BadArgs(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "tool_call", "id": "x", "name": "glob",
                   "arguments": {}, "_parse_error": "Expecting value"}
            yield {"type": "done", "finish_reason": "tool_calls"}

    from rich.console import Console
    agent = Agent(
        provider=_BadArgs(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, console=Console(markup=False), max_iterations=1,
    )
    agent.run("go")
    capsys.readouterr()  # discard run() output
    agent.render_details(agent.console)
    out = capsys.readouterr().out
    assert "✗" in out  # parse errors show dim-red ✗ connector
    assert "parse" in out.lower() or "invalid" in out.lower() or "JSON" in out


def test_weak_model_stray_token_does_not_break_render(tmp_workspace, capsys):
    """A model emitting a stray [TOOL_RESULT] as plain text must render cleanly
    (markup=False treats brackets literally; no crash, no markup error)."""
    class _Stray(MockProvider):
        def stream_chat(self, messages, tools):
            yield {"type": "text", "text": "[TOOL_RESULT] here is the answer: arr[i]"}
            yield {"type": "done", "finish_reason": "stop", "output_tokens": 3}

    out, _ = _run_capture(capsys, _Stray())
    assert "TOOL_RESULT" in out  # rendered literally, not swallowed


def test_non_tty_render_has_no_ansi(tmp_workspace):
    """A non-terminal console (force_terminal=False) emits NO ANSI escapes."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    console = Console(markup=False, file=buf, force_terminal=False)
    agent = Agent(
        provider=MockProvider(scenario="hello"), system_prompt="s",
        tool_names=FULL, auto_confirm=True, console=console,
    )
    agent.run("go")
    # render_details() writes the ⏺/⎿ tree to the same console/buf.
    agent.render_details(console)
    text = buf.getvalue()
    assert "\x1b[" not in text  # no ANSI residue when piped
    assert "⏺ Write(hello.py)" in text
    assert "⎿" in text


def test_every_theme_resolves_and_pipes_ansi_free(tmp_workspace):
    """EVERY theme key — legacy, new curated, and descriptive alias — resolves
    via palette_for/_make_console/_code_theme_for without error, and rendering a
    markdown answer through a NON-tty themed console emits zero ANSI escapes
    (the byte-clean piped guarantee holds for every theme, new ones included)."""
    import io
    from rich.console import Console
    from llmcode.config import THEMES
    from llmcode.repl import (
        _SPECS, _code_theme_for, _make_console, _resolve_theme, palette_for,
        to_rich_theme,
    )

    md = "**bold** and `inline code` and a [link](x)\n\n# Heading\n"
    for theme in THEMES:
        # Resolves without raising and lands on a real spec.
        assert _resolve_theme(theme) in _SPECS
        palette_for(theme)          # no error
        _make_console(theme)        # no error
        code_theme = _code_theme_for(theme)
        # Render the SAME rich Theme _make_console applies, but to a non-tty file
        # so the piped path is exercised: it must stay ANSI-free.
        buf = io.StringIO()
        console = Console(
            theme=to_rich_theme(_SPECS[_resolve_theme(theme)]),
            markup=False, highlight=False, file=buf, force_terminal=False,
        )
        agent = Agent(
            provider=MockProvider(scenario="hello"), system_prompt="s",
            tool_names=FULL, auto_confirm=True, console=console,
            code_theme=code_theme,
        )
        agent._print_markdown(md)
        agent.run("go")
        agent.render_details(console)
        assert "\x1b[" not in buf.getvalue(), f"{theme} leaked ANSI when piped"


# ----- themed answer box / link colour (rendering-defect fixes) -----

def test_answer_box_follows_palette_box_style(tmp_workspace):
    """The answer Panel uses the theme's box_style (ember → HEAVY, neon → DOUBLE,
    frost → SIMPLE), so the box matches the banner instead of a hardcoded ROUNDED."""
    import io
    from rich.console import Console
    from llmcode.repl import _SPECS, _resolve_theme, to_palette

    cases = {
        "ember": ("━", ("╭", "═")),   # HEAVY: heavy horizontal, no ROUNDED/DOUBLE
        "neon":  ("═", ("╭", "━")),   # DOUBLE: double horizontal
        # HORIZONTALS: light rule top+bottom, no side bars / no corners.
        "frost": ("─", ("╭", "━", "═", "│", "┃", "║")),
    }
    for theme, (must_have, must_not) in cases.items():
        spec = _SPECS[_resolve_theme(theme)]
        buf = io.StringIO()
        console = Console(markup=False, highlight=False, file=buf,
                          force_terminal=True, width=40, color_system="truecolor")
        agent = Agent(
            provider=MockProvider(scenario="hello"), system_prompt="s",
            tool_names=FULL, auto_confirm=True, console=console,
            accent=spec.accent, palette=to_palette(spec),
        )
        agent._print_markdown("hello world")
        out = buf.getvalue()
        assert must_have in out, f"{theme}: expected {must_have!r} box glyph"
        for glyph in must_not:
            assert glyph not in out, f"{theme}: unexpected {glyph!r} box glyph"


def test_answer_box_palette_less_agent_falls_back_to_rounded(tmp_workspace):
    """A palette-less agent (sub-agent/tests) keeps the historic ROUNDED box."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf,
                      force_terminal=True, width=40, color_system="truecolor")
    agent = Agent(
        provider=MockProvider(scenario="hello"), system_prompt="s",
        tool_names=FULL, auto_confirm=True, console=console,
        accent="#ffffff", palette=None,
    )
    agent._print_markdown("hello world")
    assert "╭" in buf.getvalue()  # ROUNDED corner


def test_frost_spec_box_style_is_visible_horizontals():
    """Frost/Nord uses HORIZONTALS (visible top+bottom rules) — MINIMAL, and on a
    Panel SIMPLE too, render as blank spaces, so the box would look borderless."""
    from rich import box
    from llmcode.repl import _SPECS, _resolve_theme, _box_for

    assert _SPECS[_resolve_theme("frost")].box_style == "HORIZONTALS"
    assert _box_for("HORIZONTALS") is box.HORIZONTALS


def test_link_url_uses_themed_link_colour():
    """markdown.link_url (rich colours the VISIBLE anchor text with it) is the
    themed link colour + underline, not faint — so link text reads as a link."""
    from llmcode.repl import _SPECS, _resolve_theme, to_rich_theme

    for theme in ("neon", "ember", "frost", "clean"):
        spec = _SPECS[_resolve_theme(theme)]
        style = to_rich_theme(spec).styles["markdown.link_url"]
        assert style.color.name == spec.link
        assert style.underline is True


# ----- newline-in-arg sanitization (the "one collapsed line" contract) -----

def test_tool_call_label_collapses_newline_in_arg():
    """A model emitting a newline inside an arg must NOT split the head line."""
    label = tool_call_label("glob", {"pattern": "*.py\nINJECTED SECOND LINE"})
    assert "\n" not in label
    assert label == "Glob(*.py INJECTED SECOND LINE)"


def test_result_summary_collapses_newline_in_path():
    """Created/Updated path summaries must stay a single line."""
    created = result_summary("write_file", {"ok": True, "result": {"path": "a\nb"}})
    assert "\n" not in created
    updated = result_summary("edit_file", {"ok": True, "result": {"path": "a\nb"}})
    assert "\n" not in updated


def test_live_tree_newline_arg_stays_single_line(tmp_workspace, capsys):
    """End-to-end: a newline-bearing glob pattern renders ONE head line."""
    class _NLGlob(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "glob",
                       "arguments": {"pattern": "*.py\nINJECTED"}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "ok."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    run_out, details_out, _ = _run_capture_with_details(capsys, _NLGlob())
    # The injected token must appear in render_details() output.
    assert "INJECTED" in details_out
    # The injected token must be on the SAME physical line as the head glyph,
    # i.e. there is no line whose entire content is the injected fragment.
    for ln in details_out.splitlines():
        assert ln.strip() != "INJECTED)"


# ----- successful run_bash with "Error:"-prefixed stdout is NOT red --------

def test_successful_bash_error_prefixed_stdout_not_red(tmp_workspace):
    """A 0-exit run_bash whose stdout begins with 'Error:' must render dim, not
    dim-red — the result color comes from the real ok flag, not the text."""
    import io
    from rich.console import Console

    class _ErrStdout(MockProvider):
        def stream_chat(self, messages, tools):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "run_bash",
                       "arguments": {"command": "echo Error: ok"}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "done."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf, force_terminal=True)
    agent = Agent(
        provider=_ErrStdout(), system_prompt="s", tool_names=FULL,
        auto_confirm=True, console=console,
    )
    agent.run("go")
    # render_details() writes the ⏺/⎿ tree to the same console/buf.
    agent.render_details(console)
    text = buf.getvalue()
    assert "Error: ok" in text  # the benign stdout is shown
    # Locate the result (⎿) line carrying the stdout and assert it is NOT red.
    # rich emits red as SGR "31" (dim red -> "2;31"/"31"); dim grey is "2".
    result_lines = [ln for ln in text.splitlines() if "⎿" in ln and "Error: ok" in ln]
    assert result_lines, "expected a ⎿ result line with the stdout"
    for ln in result_lines:
        assert "31m" not in ln  # no red SGR on a successful call


# ----- highlight=False keeps the dim footer uniform (no bold-cyan number) ---

def test_footer_not_bold_cyan_with_highlight_off(tmp_workspace):
    """The tok/s footer rate must render as uniform dim, never bold-cyan.

    Rich's ReprHighlighter (highlight=True) would recolor the number bold-cyan
    ('1;2;36m'); the REPL builds its console with highlight=False so the footer
    matches the Claude Code dim-grey aesthetic."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf, force_terminal=True)
    agent = Agent(
        provider=MockProvider(scenario="hello"), system_prompt="s",
        tool_names=FULL, auto_confirm=True, console=console,
    )
    agent.run("go")
    text = buf.getvalue()
    footer_lines = [ln for ln in text.splitlines() if "tok/s" in ln]
    assert footer_lines, "expected a tok/s footer line"
    for ln in footer_lines:
        assert "1;36" not in ln  # no bold-cyan
        assert "1;2;36" not in ln  # no bold-dim-cyan


def test_make_console_disables_highlight():
    """repl._make_console must build a non-highlighting console."""
    from llmcode.repl import _make_console

    console = _make_console()
    assert console._highlight is False  # ReprHighlighter disabled
    assert console._markup is False  # markup still off (literal brackets safe)


# ----- theme: ANSI (Dark mode) downsamples color to the 16 standard colors ----

def test_make_console_ansi_uses_standard_color_system(monkeypatch):
    """theme='ansi' pins the 16-color STANDARD system for a TTY; auto is not."""
    from rich.color import ColorSystem
    import llmcode.repl as r
    from llmcode.repl import _make_console

    # The ansi theme pins color_system="standard" ONLY for a real terminal, so
    # force the tty check on (under pytest stdout is captured / not a tty).
    monkeypatch.setattr(r, "_stdout_is_tty", lambda: True)

    ansi = _make_console("ansi")
    # color_system is the string name of the enum; the internal _color_system is
    # the enum itself. Assert via BOTH so either spelling is covered.
    assert ansi.color_system == "standard"
    assert ansi._color_system == ColorSystem.STANDARD
    # markup/highlight are still off (the app's always-on invariants).
    assert ansi._markup is False
    assert ansi._highlight is False

    auto = _make_console("auto")
    assert auto.color_system != "standard"
    # no-arg _make_console() uses the rendering fallback theme (auto), which —
    # like every non-ansi theme — leaves color_system auto-detected, not pinned.
    assert _make_console().color_system != "standard"


def test_make_console_ansi_non_tty_stays_ansi_free(tmp_workspace):
    """Piped/non-tty ANSI theme must NOT pin 'standard' (would leak ANSI into a
    file). With stdout not a tty, the ansi console falls back to auto-detect and
    emits clean, escape-free text — same as the auto theme when piped."""
    import io
    from rich.console import Console
    import llmcode.repl as r

    # Build the console exactly as _make_console would for a non-tty ansi run:
    # _stdout_is_tty() is False under pytest, so it returns the auto-detect
    # console; render through a non-terminal file and assert zero ANSI.
    assert r._stdout_is_tty() is False  # pytest captures stdout
    console = r._make_console("ansi")
    buf = io.StringIO()
    console.file = buf
    console._force_terminal = False
    agent = Agent(
        provider=MockProvider(scenario="hello"), system_prompt="s",
        tool_names=FULL, auto_confirm=True, console=console,
        code_theme="ansi_dark",
    )
    agent.run("go")
    assert "\x1b[" not in buf.getvalue()  # clean when piped


def test_code_theme_for_helper():
    from llmcode.repl import _code_theme_for

    assert _code_theme_for("ansi") == "ansi_dark"
    # auto is now the Frost (Nord) look, so its code fence uses the "nord" style.
    assert _code_theme_for("auto") == "nord"


def test_agent_default_code_theme_is_monokai():
    from llmcode.agent import Agent

    a = Agent(provider=MockProvider(), system_prompt="s", tool_names=[])
    assert a.code_theme == "monokai"


def test_agent_uses_ansi_dark_when_built_for_ansi_theme():
    """The orchestrator built under the ansi theme passes code_theme=ansi_dark
    to its Agent (and threads it to spawned sub-agents)."""
    from llmcode.config import Config
    from llmcode.repl import _build_orchestrator, _make_console

    cfg = Config(provider="mock", theme="ansi")
    agent = _build_orchestrator(
        MockProvider(), cfg, _make_console("ansi"), auto_confirm=True
    )
    assert agent.code_theme == "ansi_dark"

    cfg_auto = Config(provider="mock", theme="auto")
    agent_auto = _build_orchestrator(
        MockProvider(), cfg_auto, _make_console("auto"), auto_confirm=True
    )
    # auto == Frost (Nord): its fenced code uses the "nord" pygments style.
    assert agent_auto.code_theme == "nord"


def test_ansi_console_emits_no_truecolor_escapes(tmp_workspace):
    """A terminal ANSI-theme console must emit only basic ANSI SGR
    (\\x1b[32m / \\x1b[2m / \\x1b[90m) — never a truecolor \\x1b[38;2;R;G;Bm.

    Build the SAME console _make_console builds for a tty ansi run (the
    color_system='standard' branch) directly, since under pytest stdout is not a
    tty so _make_console would fall back to auto-detect."""
    import io
    import re
    from rich.console import Console

    buf = io.StringIO()
    # Mirror repl._make_console's ansi+tty branch exactly.
    console = Console(
        color_system="standard", markup=False, highlight=False,
        file=buf, force_terminal=True, width=100,
    )
    agent = Agent(
        provider=MockProvider(scenario="hello"), system_prompt="s",
        tool_names=FULL, auto_confirm=True, console=console,
        code_theme="ansi_dark",
    )
    # A markdown answer with a fenced code block exercises ansi_dark highlighting.
    agent._print_markdown(
        "Here is the answer.\n\n```python\ndef f(x):\n    return x + 1\n```\n"
    )
    agent.run("go")  # collapsed summary line + final answer + footer
    # render_details() writes the full ⏺/⎿ tree (green ⏺, dim ⎿) to the same buf.
    agent.render_details(console)
    text = buf.getvalue()
    assert "\x1b[" in text  # colored (it IS a terminal)
    # The crux of "ANSI colors only": no truecolor fg/bg escapes anywhere.
    assert not re.search(r"\x1b\[(?:38|48);2;", text)
    # And the basic ANSI codes the app relies on ARE present: green ⏺ + dim ⎿.
    assert "\x1b[32m" in text  # green glyph
    assert "\x1b[2m" in text   # dim result/footer


# ----- theme: ORANGE (orange-on-black; inline code is orange TEXT, no box) -----

def test_make_console_orange_inline_code_has_no_background_box():
    """THE key assertion: orange theme's markdown.code has NO bgcolor, so inline
    code renders as orange TEXT with no grey/black background box (rich's
    default markdown.code box is removed). orange now carries the warm Ember
    (Gruvbox) palette, so its inline-code orange is #fe8019."""
    from llmcode.repl import _make_console

    style = _make_console("orange").get_style("markdown.code")
    assert style.bgcolor is None  # no box
    assert style.bold is True
    # foreground is the Ember orange truecolor (254, 128, 25 == #fe8019).
    assert style.color is not None
    assert style.color.get_truecolor() == (254, 128, 25)


def test_make_console_every_theme_has_no_inline_code_box():
    """Step 2: EVERY theme (auto and ansi included) now gets a markdown rich
    Theme, so inline code renders as themed TEXT with NO default grey/black
    background box — the old auto/ansi default code box is gone."""
    from llmcode.config import THEMES
    from llmcode.repl import _make_console

    for theme in THEMES:
        style = _make_console(theme).get_style("markdown.code")
        assert style.bgcolor is None, f"{theme} inline code should have no box"


def test_orange_in_themes_and_load_config_accepts_it(tmp_path):
    import json
    from llmcode.config import DEFAULT_THEME, THEMES, load_config

    assert "orange" in THEMES
    good = tmp_path / "ok.json"
    good.write_text(json.dumps({"theme": "orange"}), encoding="utf-8")
    assert load_config(good).theme == "orange"
    # an unknown theme is rejected -> safe default kept (not the poisoned value).
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"theme": "neon-pink"}), encoding="utf-8")
    assert load_config(bad).theme == DEFAULT_THEME


def test_code_theme_for_orange_is_a_valid_pygments_style():
    from pygments.styles import get_style_by_name
    from llmcode.repl import _code_theme_for

    name = _code_theme_for("orange")
    get_style_by_name(name)  # must not raise
    assert name in ("native", "monokai")


# ----- theme: AMBER (the polished default — banner, gutter, gold bold) --------

def test_clean_is_default_theme():
    from llmcode.config import DEFAULT_THEME, THEMES, Config

    assert DEFAULT_THEME == "clean"
    assert "clean" in THEMES
    assert "amber" in THEMES  # amber stays available, just no longer the default
    assert Config().theme == "clean"  # fresh installs


def test_clean_theme_registered_and_selectable(tmp_path):
    """The new minimal-dark "clean" theme is registered, selectable via config,
    builds a console, has a palette, and maps to a valid pygments code theme."""
    import json
    from pygments.styles import get_style_by_name
    from llmcode.config import THEMES, load_config
    from llmcode.repl import _make_console, _code_theme_for, palette_for

    assert "clean" in THEMES
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"theme": "clean"}), encoding="utf-8")
    assert load_config(p).theme == "clean"
    # Builds a console without raising and has the Midnight (Tokyo Night) blue
    # accent palette (the default clean look is now cool-blue, not grey).
    _make_console("clean")
    pal = palette_for("clean")
    assert pal.accent == "#7aa2f7"
    # Code theme is a real, importable pygments style.
    get_style_by_name(_code_theme_for("clean"))


def test_make_console_clean_inline_code_has_no_background_box():
    """Clean (Midnight) inline code is cyan TEXT with NO bgcolor box."""
    from llmcode.repl import _make_console

    style = _make_console("clean").get_style("markdown.code")
    assert style.bgcolor is None  # no box
    assert style.color is not None
    assert style.color.get_truecolor() == (125, 207, 255)  # #7dcfff


def test_make_console_amber_inline_code_has_no_background_box():
    """Amber (Ember/Gruvbox) inline code is orange TEXT with NO bgcolor box."""
    from llmcode.repl import _make_console

    style = _make_console("amber").get_style("markdown.code")
    assert style.bgcolor is None  # no box
    assert style.bold is True
    assert style.color is not None
    assert style.color.get_truecolor() == (254, 128, 25)  # #fe8019


def test_amber_strong_is_gold():
    """**bold** words render in GOLD (Ember/Gruvbox #fabd2f) so key words pop."""
    from llmcode.repl import _make_console

    style = _make_console("amber").get_style("markdown.strong")
    assert style.bold is True
    assert style.color.get_truecolor() == (250, 189, 47)  # #fabd2f


def test_code_theme_for_amber_is_valid_pygments():
    from pygments.styles import get_style_by_name
    from llmcode.repl import _code_theme_for

    name = _code_theme_for("amber")
    get_style_by_name(name)  # must not raise
    assert name in ("native", "monokai")


def test_palette_for_known_and_unknown():
    from llmcode.repl import palette_for

    amber = palette_for("amber")
    assert amber.accent == "#fe8019"  # Ember (Gruvbox) orange
    assert amber.gutter == "▌"
    assert amber.prompt == "❯"
    # ansi stays inside the 16 basic colours and uses a thin gutter.
    ansi = palette_for("ansi")
    assert ansi.accent == "yellow"
    assert ansi.gutter == "│"
    # auto is the cool cyan Frost (Nord) accent.
    assert palette_for("auto").accent == "#88c0d0"
    # clean is the Midnight (Tokyo Night) blue default accent.
    assert palette_for("clean").accent == "#7aa2f7"
    # an unknown theme falls back to the clean palette (matches DEFAULT_THEME).
    assert palette_for("nope").accent == "#7aa2f7"
    # descriptive aliases resolve to their canonical spec's palette.
    assert palette_for("midnight").accent == "#7aa2f7"
    assert palette_for("frost").accent == "#88c0d0"
    assert palette_for("ember").accent == "#fe8019"
    # new curated themes have their own accents.
    assert palette_for("neon").accent == "#bd93f9"
    assert palette_for("blossom").accent == "#cba6f7"


# ----- the thin accent "Answer" box ----------------------------------------

def _tty_console(buf, width=60):
    """A console that IS a terminal (so the box is enabled) but emits no ANSI
    (color_system=None) — lets us assert the literal box border chars cleanly."""
    from rich.console import Console

    return Console(markup=False, highlight=False, file=buf, force_terminal=True,
                   color_system=None, width=width)


def test_agent_with_accent_renders_answer_box_on_tty(tmp_workspace):
    """A top-level agent with an accent ON A TERMINAL wraps its answer in a thin,
    title-less rounded box (reference design), with breathing-room blank lines
    around it."""
    import io

    buf = io.StringIO()
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=[],
        console=_tty_console(buf), accent="#ff9e3d",
    )
    agent._print_markdown("Hello **world** with `code`.")
    out = buf.getvalue()
    assert "Answer" not in out          # title-less box (no "Answer" caption)
    assert "╭" in out and "╰" in out   # rounded box top + bottom corners
    assert "│" in out                  # vertical box borders
    assert "world" in out and "code" in out
    # Breathing room: a blank line before the box and after it.
    lines = out.splitlines()
    assert lines and lines[0] == ""    # blank line above the box
    assert lines[-1] == ""             # blank line below the box


def test_accent_answer_is_plain_when_piped(tmp_workspace):
    """finding #1: a NON-terminal (piped/scripted) run must NOT get the box
    decoration — clean plain Markdown — AND no ANSI."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf,
                      force_terminal=False, width=60)
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=[],
        console=console, accent="#ff9e3d",
    )
    agent._print_markdown("# Title\n\nBody **bold** and `code`.")
    out = buf.getvalue()
    assert "╭" not in out and "╰" not in out  # no box decoration when piped
    assert "▌" not in out  # nor the old gutter bar
    assert "\x1b[" not in out  # and no ANSI
    assert "Title" in out and "Body" in out


def test_agent_without_accent_renders_plain_markdown(tmp_workspace):
    """No accent => plain Markdown, no box (back-compat for sub-agents/tests)."""
    import io

    buf = io.StringIO()
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=[],
        console=_tty_console(buf),  # a terminal, yet still no box without accent
    )
    agent._print_markdown("Hello world.")
    out = buf.getvalue()
    assert "╭" not in out  # no box border
    assert "Hello world." in out


def test_subagent_accent_does_not_box(tmp_workspace):
    """A nested sub-agent (non-empty line_prefix) keeps plain Markdown EVEN on a
    terminal, so the box never collides with the '↳' marker."""
    import io

    buf = io.StringIO()
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=[],
        console=_tty_console(buf), accent="#ff9e3d", line_prefix="  ↳ ",
    )
    agent._print_markdown("Sub result.")
    out = buf.getvalue()
    assert "╭" not in out  # no box
    assert "Sub result." in out


def test_accent_footer_rate_not_dimmed(tmp_workspace):
    """finding #2: the accent footer rate reads in full accent, NOT dim-accent
    (no leading '2;' dim SGR layered onto the number)."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf,
                      force_terminal=True, color_system="truecolor", width=60)
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=[],
        console=console, accent="#ff9e3d",
    )
    agent._print_footer("x" * 200, 200, 4.0)
    line = next(ln for ln in buf.getvalue().splitlines() if "tok/s" in ln)
    assert "38;2;255;158;61" in line       # the rate IS amber
    assert "2;38;2;255;158;61" not in line  # but NOT dimmed-amber


def test_footer_shows_model_time_speed(tmp_workspace):
    """The footer renders the reference 'Model: ... · Time: ...s · Speed: ...
    tok/s' line, with the model name read from the provider. The segment
    separator is the middle dot '·' so the one-shot footer and the pinned status
    bar read as the same object."""
    import io

    buf = io.StringIO()
    agent = Agent(
        provider=MockProvider(), system_prompt="s", tool_names=[],
        console=_tty_console(buf),  # terminal, no ANSI -> assert literal text
    )
    agent.provider.model = "qwen2.5-coder"
    agent._print_footer("x" * 200, 200, 1.49)
    line = next(ln for ln in buf.getvalue().splitlines() if "tok/s" in ln)
    assert "Model: qwen2.5-coder" in line
    assert "Time: 1.49s" in line
    assert "Speed:" in line and "tok/s" in line
    assert " · " in line  # middle-dot separator (matches the pinned status bar)
    assert " | " not in line  # the old pipe separator is gone


def test_left_heading_not_centered(tmp_workspace):
    """build_answer_markdown left-aligns headings (rich centers h1 by default)."""
    import io
    from rich.console import Console
    from llmcode.agent import build_answer_markdown

    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf, force_terminal=False, width=60)
    console.print(build_answer_markdown("# Short Title", "monokai"))
    line = next(ln for ln in buf.getvalue().splitlines() if "Short Title" in ln)
    # left-aligned => no big block of leading spaces (centering would indent it).
    assert line.lstrip() == line  # starts at column 0


# ----- Step 3: semantic tokens flow into the tool tree / footer / activity ----

def test_tool_tree_uses_palette_semantic_tokens_on_tty(tmp_workspace):
    """With a threaded palette on a TERMINAL, the tool-tree head is the theme
    ACCENT on success and the theme ERROR on failure, the ✓ is the SUCCESS token,
    and the ⎿ connector is MUTED — no more hardcoded green/red. Uses a truecolor
    console so the theme hex SGR are directly assertable."""
    import io
    from rich.console import Console
    from llmcode.repl import palette_for

    pal = palette_for("neon")  # accent #bd93f9, success #50fa7b, error #ff5555
    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf,
                      force_terminal=True, color_system="truecolor", width=100)
    agent = Agent(provider=MockProvider(), system_prompt="s", tool_names=[],
                  console=console, accent=pal.accent, palette=pal)
    # SUCCESS row: accent head glyph + success ✓.
    agent._render_tool_tree("read_file", {"path": "x.py"}, "3 lines", False)
    out = buf.getvalue()
    assert "38;2;189;147;249" in out  # accent head glyph (#bd93f9)
    assert "38;2;80;250;123" in out   # success ✓ (#50fa7b)
    # FAILURE row: head glyph + summary both read in the error token.
    buf2 = io.StringIO()
    console.file = buf2
    agent._render_tool_tree("run_bash", {"command": "false"}, "✗ boom", True)
    out2 = buf2.getvalue()
    assert "38;2;255;85;85" in out2   # error head + summary (#ff5555)


def test_tool_tree_and_footer_ansi_free_when_piped_even_with_palette(tmp_workspace):
    """The byte-clean guarantee: with a full palette threaded in, a NON-terminal
    (piped/scripted) console still emits ZERO ANSI from the tool tree, the footer,
    and the collapsed activity line — the semantic colour lives only behind the
    is_terminal gate."""
    import io
    from rich.console import Console
    from llmcode.repl import palette_for

    pal = palette_for("neon")
    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf,
                      force_terminal=False, width=100)
    agent = Agent(provider=MockProvider(), system_prompt="s", tool_names=[],
                  console=console, accent=pal.accent, palette=pal)
    agent._render_tool_tree("read_file", {"path": "x.py"}, "3 lines", False)
    agent._render_tool_tree("run_bash", {"command": "false"}, "✗ boom", True)
    agent._print_footer("hello there", 12, 0.5)
    agent.last_turn_details = [
        {"name": "read_file", "args": {"path": "x.py"}, "result": {}, "ok": True,
         "elapsed": 0.1},
        {"name": "run_bash", "args": {"command": "false"}, "result": {}, "ok": False,
         "elapsed": 0.1},
    ]
    agent._print_activity_summary()
    assert "\x1b[" not in buf.getvalue()  # clean when piped, palette notwithstanding


def test_activity_summary_glyph_uses_error_token_on_failure(tmp_workspace):
    """The collapsed activity ⏺ reads in the theme ERROR token when any call
    failed (accent otherwise), matching the expanded tool-tree head."""
    import io
    from rich.console import Console
    from llmcode.repl import palette_for

    pal = palette_for("neon")
    buf = io.StringIO()
    console = Console(markup=False, highlight=False, file=buf,
                      force_terminal=True, color_system="truecolor", width=100)
    agent = Agent(provider=MockProvider(), system_prompt="s", tool_names=[],
                  console=console, accent=pal.accent, palette=pal)
    agent.last_turn_details = [
        {"name": "run_bash", "args": {"command": "false"}, "result": {}, "ok": False,
         "elapsed": 0.1},
    ]
    agent._print_activity_summary()
    assert "38;2;255;85;85" in buf.getvalue()  # error #ff5555 glyph


# ----- build item 5: outcome counts + auto-expand ---------------------------

def _summary_agent(buf, *, tty=False, prefix="", details=None):
    """Agent wired to render its activity summary into ``buf``."""
    from rich.console import Console
    from llmcode.repl import palette_for

    pal = palette_for("neon")
    console = Console(
        markup=False, highlight=False, file=buf, width=100,
        force_terminal=tty, color_system="truecolor" if tty else None,
    )
    agent = Agent(provider=MockProvider(), system_prompt="s", tool_names=[],
                  console=console, accent=pal.accent, palette=pal,
                  line_prefix=prefix)
    agent.last_turn_details = details or []
    return agent


def _detail(name, ok, **result):
    return {"name": name, "args": {"path": "a.py"} if name == "read_file"
            else {"command": "false"}, "result": result, "ok": ok, "elapsed": 0.1}


def test_activity_summary_shows_pass_fail_counts(tmp_workspace):
    """The collapsed one-liner shows ◆ N tools · ✓x ✗y + the first failure reason
    + a dim ctrl-o hint (plain text on a non-tty)."""
    import io

    details = [
        _detail("read_file", True, result="x\ny"),
        _detail("run_bash", False, result={"exit_code": 1}),
    ]
    buf = io.StringIO()
    agent = _summary_agent(buf, tty=False, details=details)
    agent._print_activity_summary()
    text = buf.getvalue()
    assert "◆ 2 tools" in text
    assert "✓1" in text and "✗1" in text
    assert "run_bash exit 1" in text   # first failure reason
    assert "ctrl-o details" in text


def test_activity_summary_counts_carry_success_and_error_styles(tmp_workspace):
    """On a real terminal the ✓x count reads in the success token and ✗y + reason
    in the error token."""
    import io

    details = [
        _detail("read_file", True, result="x"),
        _detail("run_bash", False, result={"exit_code": 1}),
    ]
    buf = io.StringIO()
    agent = _summary_agent(buf, tty=True, details=details)
    agent._print_activity_summary()
    out = buf.getvalue()
    assert "38;2;80;250;123" in out    # neon success #50fa7b on ✓1
    assert "38;2;255;85;85" in out     # neon error #ff5555 on ✗1 + reason


def test_activity_summary_auto_expands_on_failure(tmp_workspace):
    """A failure forces the full ⏺/⎿ tree inline even for a LARGE batch (>5) — the
    failed tool's detail is never hidden behind Ctrl+O."""
    import io

    details = [_detail("read_file", True, result="x") for _ in range(7)]
    details.append(_detail("run_bash", False, result={"exit_code": 1}))
    buf = io.StringIO()
    agent = _summary_agent(buf, tty=False, details=details)
    agent._print_activity_summary()
    text = buf.getvalue()
    assert "✓7" in text and "✗1" in text
    assert "⏺" in text and "⎿" in text          # tree auto-expanded
    assert "⏺ Bash(false)" in text              # the failed tool is visible


def test_activity_summary_expands_small_all_green(tmp_workspace):
    """A modest all-green batch (≤5) still auto-expands the tree inline."""
    import io

    details = [_detail("read_file", True, result="x\ny") for _ in range(3)]
    buf = io.StringIO()
    agent = _summary_agent(buf, tty=False, details=details)
    agent._print_activity_summary()
    text = buf.getvalue()
    assert "✓3" in text
    assert "⏺" in text and "⎿" in text


def test_activity_summary_collapses_large_all_green(tmp_workspace):
    """A large all-green batch (>5, no failures) collapses to JUST the one-liner —
    the tree stays behind Ctrl+O."""
    import io

    details = [_detail("read_file", True, result="x") for _ in range(6)]
    buf = io.StringIO()
    agent = _summary_agent(buf, tty=False, details=details)
    agent._print_activity_summary()
    text = buf.getvalue()
    assert "◆ 6 tools" in text and "✓6" in text
    assert "ctrl-o details" in text
    assert "⏺" not in text and "⎿" not in text  # NOT expanded


def test_activity_summary_failed_batch_piped_stays_ansi_free(tmp_workspace):
    """Piped (non-tty) tool output — one-liner AND auto-expanded tree — is ANSI-free,
    palette notwithstanding."""
    import io

    details = [
        _detail("read_file", True, result="x"),
        _detail("run_bash", False, result={"exit_code": 1}),
    ]
    buf = io.StringIO()
    agent = _summary_agent(buf, tty=False, details=details)
    agent._print_activity_summary()
    text = buf.getvalue()
    assert "\x1b[" not in text          # no ANSI escapes when piped
    assert "✓1" in text and "✗1" in text
    assert "⏺" in text                  # tree still rendered as plain text


def test_build_orchestrator_threads_accent_and_gutter():
    """The orchestrator (and its sub-agents) get the theme's accent + gutter."""
    from llmcode.config import Config
    from llmcode.repl import _build_orchestrator, _make_console

    cfg = Config(provider="mock", theme="amber")
    agent = _build_orchestrator(
        MockProvider(), cfg, _make_console("amber"), auto_confirm=True
    )
    assert agent.accent == "#fe8019"  # Ember (Gruvbox) orange
    assert agent.gutter_char == "▌"

    cfg_ansi = Config(provider="mock", theme="ansi")
    agent_ansi = _build_orchestrator(
        MockProvider(), cfg_ansi, _make_console("ansi"), auto_confirm=True
    )
    assert agent_ansi.accent == "yellow"
    assert agent_ansi.gutter_char == "│"


def test_build_orchestrator_threads_full_palette():
    """Step 3: the orchestrator also receives the FULL palette (semantic tokens +
    spinner colours), not just the accent — so its tool tree / footer / spinner
    can reach success/error/muted/spinner."""
    from llmcode.config import Config
    from llmcode.repl import _build_orchestrator, _make_console, palette_for

    cfg = Config(provider="mock", theme="neon")
    agent = _build_orchestrator(
        MockProvider(), cfg, _make_console("neon"), auto_confirm=True
    )
    assert agent.palette is not None
    assert agent.palette.accent == palette_for("neon").accent  # #bd93f9
    assert agent.palette.error == "#ff5555"
    assert agent.palette.success == "#50fa7b"
    assert agent.palette.spinner == "#bd93f9"


def test_orange_render_emits_orange_fg_and_no_bg_box():
    """Render markdown with inline code through an orange-themed recording
    console and assert the output carries the Ember orange foreground SGR
    (38;2;254;128;25) and NO background-color SGR (48;2;) around the code."""
    from rich.console import Console
    from rich.markdown import Markdown
    from llmcode.repl import _orange_theme, _code_theme_for

    console = Console(
        theme=_orange_theme(), force_terminal=True, color_system="truecolor",
        markup=False, highlight=False,
    )
    with console.capture() as cap:
        console.print(Markdown(
            "Fix missing `import requests` in `trader.py`",
            code_theme=_code_theme_for("orange"),
        ))
    out = cap.get()
    assert "38;2;254;128;25" in out  # Ember orange inline-code foreground
    assert "48;2;" not in out        # no background-color box anywhere
