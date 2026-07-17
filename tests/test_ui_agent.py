"""Agent-side support for the combined terminal UI (pinned status bar + tree).

Covers the three agent changes that back the REPL's new bottom status bar:

  1. TOOL-TREE ✓ + ASCII fallback: a successful tool head gets a dim-green "✓"
     WHEN rendered to a real terminal (is_terminal); the ⏺/⎿ glyphs fall back to
     ASCII ("* " / "  L  ") when the console encoding can't represent them.
  2. last_turn_stats: after a rendered turn the agent exposes the SAME footer
     stats (model / elapsed / toks_per_sec / output_tokens); None beforehand.
  3. suppress_footer: when True the "Model | Time | Speed" line is NOT printed
     but last_turn_stats is still set. Default False keeps the footer as before.

All offline via MockProvider. The default (suppress_footer=False, non-terminal)
path is asserted byte-identical to the historic behaviour so the wider suite
stays green.
"""

from __future__ import annotations

import io

from rich.console import Console

from llmcli.agent import Agent
from llmcli.providers import MockProvider
from llmcli.tools import FULL


class _AsciiIO(io.StringIO):
    """A StringIO whose reported encoding is ASCII (mirrors a LANG=C terminal).

    rich's ``Console.encoding`` reads ``file.encoding``; this lets a test drive
    the encoding-can't-represent-glyph fallback deterministically."""

    encoding = "ascii"


def _tty_console(buf: io.StringIO | None = None) -> Console:
    """A real-terminal console writing to ``buf`` (utf-8), like the REPL's tty."""
    return Console(
        markup=False, highlight=False, file=buf or io.StringIO(),
        force_terminal=True, width=100,
    )


def _nontty_console(buf: io.StringIO | None = None) -> Console:
    """A piped/non-terminal console (force_terminal=False) — the test/pipe path."""
    return Console(markup=False, file=buf or io.StringIO(), force_terminal=False)


def _agent(console: Console, provider: MockProvider | None = None, **kw) -> Agent:
    return Agent(
        provider=provider or MockProvider(scenario="hello"),
        system_prompt="s", tool_names=FULL, auto_confirm=True,
        console=console, **kw,
    )


# ----- change 2: last_turn_stats -------------------------------------------

def test_last_turn_stats_none_before_any_turn(tmp_workspace):
    """A fresh agent exposes last_turn_stats == None (nothing rendered yet)."""
    agent = _agent(_nontty_console())
    assert agent.last_turn_stats is None


def test_last_turn_stats_populated_after_turn(tmp_workspace):
    """After a rendered turn, last_turn_stats has model/elapsed/toks_per_sec/
    output_tokens with sane types + values."""
    provider = MockProvider(scenario="hello")
    provider.model = "mock-model"
    agent = _agent(_nontty_console(), provider=provider)
    agent.run("go")

    stats = agent.last_turn_stats
    assert isinstance(stats, dict)
    assert set(stats) == {"model", "elapsed", "toks_per_sec", "output_tokens"}
    assert stats["model"] == "mock-model"
    assert isinstance(stats["elapsed"], float) and stats["elapsed"] >= 0.0
    assert isinstance(stats["toks_per_sec"], float) and stats["toks_per_sec"] >= 0.0
    assert isinstance(stats["output_tokens"], int) and stats["output_tokens"] > 0


def test_last_turn_stats_match_printed_footer(tmp_workspace):
    """The stored stats come from the SAME values the footer prints — the
    stored rate rounds to the tok/s value shown on the footer line."""
    buf = io.StringIO()
    provider = MockProvider(scenario="hello")
    provider.model = "mock-model"
    agent = _agent(_tty_console(buf), provider=provider)
    agent.run("go")

    footer_lines = [ln for ln in buf.getvalue().splitlines() if "tok/s" in ln]
    assert footer_lines, "expected a printed tok/s footer line"
    printed_rate = f"{agent.last_turn_stats['toks_per_sec']:.1f} tok/s"
    assert any(printed_rate in ln for ln in footer_lines)
    # Model rendered on the footer matches the stored model.
    assert any("mock-model" in ln for ln in footer_lines)


# ----- change 3: suppress_footer -------------------------------------------

def test_footer_prints_by_default(tmp_workspace):
    """DEFAULT (suppress_footer=False): the Model|Time|Speed line still prints."""
    buf = io.StringIO()
    provider = MockProvider(scenario="hello")
    provider.model = "mock-model"
    agent = _agent(_tty_console(buf), provider=provider)
    assert agent.suppress_footer is False  # default
    agent.run("go")

    text = buf.getvalue()
    assert "tok/s" in text
    assert "Model: mock-model" in text
    assert "Time:" in text and "Speed:" in text


def test_suppress_footer_skips_line_but_sets_stats(tmp_workspace):
    """suppress_footer=True: the footer line is NOT printed, yet last_turn_stats
    IS populated (the REPL's pinned bar shows the numbers instead)."""
    buf = io.StringIO()
    provider = MockProvider(scenario="hello")
    provider.model = "mock-model"
    agent = _agent(_tty_console(buf), provider=provider, suppress_footer=True)
    agent.run("go")

    text = buf.getvalue()
    # The footer LINE is gone …
    assert "tok/s" not in text
    assert "Speed:" not in text
    # … but the stats are still available for the status bar.
    assert agent.last_turn_stats is not None
    assert agent.last_turn_stats["model"] == "mock-model"
    assert agent.last_turn_stats["toks_per_sec"] >= 0.0
    assert agent.last_turn_stats["output_tokens"] > 0


def test_suppress_footer_leaves_answer_intact(tmp_workspace):
    """Suppressing the footer must not suppress the answer itself."""
    buf = io.StringIO()
    agent = _agent(_tty_console(buf), suppress_footer=True)
    agent.run("go")
    # The 'hello' scenario ends with a visible answer; only the footer is gone.
    assert buf.getvalue().strip() != ""
    assert "tok/s" not in buf.getvalue()


# ----- change 1: ✓ on success + ASCII fallback ------------------------------

def test_success_check_on_tool_head_when_tty(tmp_workspace):
    """On a real terminal, a SUCCESSFUL tool head line carries a "✓"."""
    buf = io.StringIO()
    agent = _agent(_tty_console(buf))
    agent.run("go")
    agent.render_details(agent.console)

    lines = buf.getvalue().splitlines()
    # Both 'hello' tools succeed -> each head line gets the check.
    assert any("Write(hello.py)" in ln and "✓" in ln for ln in lines)
    assert any("Bash(" in ln and "✓" in ln for ln in lines)


def test_no_check_on_non_tty(tmp_workspace):
    """Piped/non-terminal output stays byte-identical: NO "✓" is added, and the
    ⏺/⎿ glyphs are preserved (utf-8 console)."""
    buf = io.StringIO()
    agent = _agent(_nontty_console(buf))
    agent.run("go")
    agent.render_details(agent.console)

    text = buf.getvalue()
    assert "✓" not in text          # check-mark is TTY-only
    assert "⏺ Write(hello.py)" in text  # historic head line, unchanged
    assert "⎿" in text                  # historic connector, unchanged


def test_no_check_on_failed_call(tmp_workspace):
    """A FAILED call keeps the ✗ connector and gets NO ✓ on its head line."""
    class _FailRead(MockProvider):
        def stream_chat(self, messages, tools, tool_choice=None):
            step = self._step_from_history(messages)
            if step == 0:
                yield {"type": "tool_call", "id": "x", "name": "read_file",
                       "arguments": {"path": "does_not_exist.txt"}}
                yield {"type": "done", "finish_reason": "tool_calls"}
            else:
                yield {"type": "text", "text": "ok."}
                yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}

    buf = io.StringIO()
    agent = _agent(_tty_console(buf), provider=_FailRead())
    agent.run("go")
    agent.render_details(agent.console)

    lines = buf.getvalue().splitlines()
    head_lines = [ln for ln in lines if "Read(does_not_exist.txt)" in ln]
    assert head_lines, "expected the failed Read head line"
    assert all("✓" not in ln for ln in head_lines)  # no check on failure
    assert "✗" in buf.getvalue()  # failure still shown on the connector


def test_ascii_fallback_when_encoding_cannot_encode(tmp_workspace):
    """When the console encoding can't represent the glyphs (e.g. LANG=C ascii),
    the tree falls back to ASCII markers with no mojibake and no ✓."""
    buf = _AsciiIO()
    console = Console(markup=False, highlight=False, file=buf, force_terminal=True)
    agent = _agent(console)
    # Glyph selection reflects the ascii encoding.
    assert agent._tree_glyphs() == ("* ", "  L  ")

    agent.run("go")
    agent.render_details(console)
    text = buf.getvalue()
    assert "⏺" not in text and "⎿" not in text  # no unicode glyphs
    assert "* " in text                          # ascii head marker present
    assert "✓" not in text                       # check-mark also gated off


def test_tree_glyphs_helper_encoding_gate(tmp_workspace):
    """_tree_glyphs returns the unicode glyphs on a utf-8 console and ASCII on an
    ascii console; _console_can_encode agrees."""
    utf8 = _agent(_tty_console())
    assert utf8._tree_glyphs() == ("⏺ ", "  ⎿  ")
    assert utf8._console_can_encode("⏺⎿✓") is True

    ascii_console = Console(markup=False, file=_AsciiIO(), force_terminal=True)
    ascii_agent = _agent(ascii_console)
    assert ascii_agent._tree_glyphs() == ("* ", "  L  ")
    assert ascii_agent._console_can_encode("⏺") is False
