"""Spinner (braille working-indicator) tests — offline, deterministic, no sleeping.

The line builder is pure, so we assert on its output directly. The threaded
parts are exercised only through their TTY-guard (non-TTY => no thread, no
output) and the idempotent self-erasing stop().
"""

from __future__ import annotations

import io

import re

from llmcode.spinner import (
    _ASCII_FRAMES,
    _BRAILLE_FRAMES,
    _PULSE_PERIOD,
    AntSpinner,
    DuckSpinner,
    Spinner,
    _frame,
    _parse_hex,
    _sgr,
)


class _FakeConsole:
    """Minimal rich-Console stand-in: an is_terminal flag + a writable file."""

    def __init__(self, is_terminal: bool):
        self.is_terminal = is_terminal
        self.file = io.StringIO()


# ----- pure frame function -------------------------------------------------

def test_frame_cycles_the_braille_glyphs():
    # One glyph advances per tick and the cycle wraps at the end of the set.
    for i in range(len(_BRAILLE_FRAMES) * 3):
        line = _frame(i)
        assert line[0] == _BRAILLE_FRAMES[i % len(_BRAILLE_FRAMES)]
    # A full cycle visits every braille glyph exactly, in order.
    heads = [_frame(i)[0] for i in range(len(_BRAILLE_FRAMES))]
    assert "".join(heads) == _BRAILLE_FRAMES
    # Pure: same input -> same output, no hidden state.
    assert _frame(7) == _frame(7)
    # It wraps: tick 0 and tick len(frames) share the same glyph.
    assert _frame(0)[0] == _frame(len(_BRAILLE_FRAMES))[0]


def test_frame_renders_label_seconds_and_interrupt_hint():
    line = _frame(0, elapsed=3.0)
    assert "working" in line
    assert "3s" in line
    assert "ctrl-c to stop" in line
    # The real interrupt is ctrl-c, NOT esc.
    assert "esc" not in line.lower()


def test_frame_seconds_are_whole_and_increase_with_elapsed():
    # Fractions truncate to whole seconds.
    assert "0s" in _frame(0, elapsed=0.0)
    assert "0s" in _frame(0, elapsed=0.9)
    assert "3s" in _frame(0, elapsed=3.4)
    # Seconds grow monotonically as elapsed grows (1 -> 2 digits, no crash).
    for secs in (0, 1, 5, 12, 99, 120):
        assert f"{secs}s" in _frame(0, elapsed=float(secs))
    # Negative/odd clocks never render a negative count.
    assert "0s" in _frame(0, elapsed=-5.0)


def test_ascii_fallback_yields_ascii_glyphs_and_same_text():
    line = _frame(1, elapsed=3.0, ascii_mode=True)
    # The glyph comes from the classic |/-\ cycle, not braille.
    assert line[0] in _ASCII_FRAMES
    assert line[0] not in _BRAILLE_FRAMES
    # The informational text is preserved and fully ASCII-encodable.
    assert "working" in line
    assert "3s" in line
    assert "ctrl-c to stop" in line
    line.encode("ascii")  # must not raise: the whole line is plain ASCII


def test_ascii_frames_cycle_over_the_classic_glyphs():
    heads = [_frame(i, ascii_mode=True)[0] for i in range(len(_ASCII_FRAMES))]
    assert "".join(heads) == _ASCII_FRAMES


def test_ant_and_duck_aliases_point_at_spinner():
    # Back-compat aliases stay wired to the canonical Spinner.
    assert AntSpinner is Spinner
    assert DuckSpinner is Spinner


# ----- ascii-mode auto-detection ------------------------------------------

def test_ascii_env_switch_forces_ascii_mode(monkeypatch):
    monkeypatch.setenv("LLMCODE_ASCII_SPINNER", "1")
    sp = Spinner(_FakeConsole(is_terminal=True), enabled=True)
    assert sp.ascii_mode is True


def test_ascii_mode_when_stream_cannot_encode_braille():
    class _AsciiConsole(_FakeConsole):
        def __init__(self):
            super().__init__(is_terminal=True)
            # An ascii-only stream can't encode the braille glyph -> fallback.
            self.file = io.TextIOWrapper(io.BytesIO(), encoding="ascii")

    sp = Spinner(_AsciiConsole(), enabled=True)
    assert sp.ascii_mode is True


def test_unicode_stream_stays_on_braille():
    # A StringIO (no encoding attribute) is treated as unicode-capable.
    sp = Spinner(_FakeConsole(is_terminal=True), enabled=True)
    assert sp.ascii_mode is False


# ----- non-TTY guard: no thread, no output --------------------------------

def test_disabled_when_not_a_terminal_spawns_no_thread_and_writes_nothing():
    console = _FakeConsole(is_terminal=False)
    sp = Spinner(console)
    assert sp.enabled is False
    sp.start()
    assert sp._thread is None  # no thread spawned
    sp.stop()
    sp.stop()  # idempotent
    assert console.file.getvalue() == ""  # NOTHING written in non-TTY mode


def test_env_off_switch_disables_even_on_tty(monkeypatch):
    monkeypatch.setenv("LLMCODE_NO_SPINNER", "1")
    sp = Spinner(_FakeConsole(is_terminal=True))
    assert sp.enabled is False


def test_none_console_is_disabled():
    sp = Spinner(None)
    assert sp.enabled is False
    sp.start()
    sp.stop()  # must not raise


# ----- enabled stop() writes the clear sequence ---------------------------

def test_enabled_stop_writes_clear_sequence_and_is_idempotent():
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(console, enabled=True)
    # Do not start the thread (deterministic): call stop() directly. It must
    # write a self-erasing clear sequence (CR + blanks + erase-to-EOL + CR) and
    # restore the terminal cursor (show-cursor) it hid on start().
    sp.stop()
    out = console.file.getvalue()
    assert out.startswith("\r")
    assert " " in out  # the blanking spaces
    assert "\x1b[K" in out  # erase-to-end-of-line (no stale tail left behind)
    assert out.endswith("\x1b[?25h")  # cursor restored as the final act
    # Idempotent: a second stop appends another harmless clear, never raises.
    sp.stop()


def test_context_manager_on_disabled_is_clean():
    console = _FakeConsole(is_terminal=False)
    with DuckSpinner(console) as sp:
        assert sp.enabled is False
    assert console.file.getvalue() == ""


# ----- threaded path: _run loop, live timer, start/stop join --------------

def test_enabled_threaded_run_writes_frames_and_clears_on_stop(monkeypatch):
    """Start the real thread against a fake TTY, let a couple of frames write,
    then stop. Assert at least one '\\r...\\x1b[K' frame was drawn, the output
    ends with the blank clear + show-cursor, and the thread is gone after stop."""
    import time as _time

    import llmcode.spinner as sp_mod

    # Near-zero frame sleep so frames advance fast and the test stays quick.
    monkeypatch.setattr(sp_mod, "_FPS_SLEEP", 0.001)
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(console, enabled=True)
    sp.start()
    assert sp._thread is not None and sp._thread.is_alive()
    # Let a few frames tick.
    deadline = _time.time() + 1.0
    while _time.time() < deadline and console.file.getvalue().count("\x1b[K") < 1:
        _time.sleep(0.005)
    sp.stop()
    out = console.file.getvalue()
    assert "\r" in out
    assert "\x1b[K" in out  # at least one self-erasing frame was drawn
    assert "working" in out  # the status label rendered
    assert "ctrl-c to stop" in out  # the interrupt hint rendered
    assert out.endswith("\x1b[?25h")  # final act: cursor restored
    assert sp._thread is None  # joined + cleared on stop


def test_live_timer_counts_up_with_injected_clock(monkeypatch):
    """The rendered seconds track a monkeypatched monotonic clock: start at t0,
    advance the clock, and a later frame shows a larger whole-second count."""
    import llmcode.spinner as sp_mod

    fake = {"t": 100.0}
    monkeypatch.setattr(sp_mod, "_now", lambda: fake["t"])
    monkeypatch.setattr(sp_mod, "_FPS_SLEEP", 0.001)

    console = _FakeConsole(is_terminal=True)
    sp = Spinner(console, enabled=True)
    sp.start()  # anchors _start_ts at t=100.0
    # Wait for the first (0s) frame.
    import time as _time

    deadline = _time.time() + 1.0
    while _time.time() < deadline and "0s" not in console.file.getvalue():
        _time.sleep(0.005)
    assert "0s" in console.file.getvalue()
    # Jump the clock forward 7s and wait for a frame that reflects it.
    fake["t"] = 107.0
    deadline = _time.time() + 1.0
    while _time.time() < deadline and "7s" not in console.file.getvalue():
        _time.sleep(0.005)
    sp.stop()
    assert "7s" in console.file.getvalue()


# ----- theme colour: _sgr helper + gated glyph tinting --------------------

def test_sgr_only_emits_truecolor_for_hex():
    """_sgr returns a 24-bit SGR (prefix, reset) for a #rrggbb hex, and an empty
    ('', '') for anything else — None, '', an ANSI name, or a malformed string —
    so a non-hex spinner colour (e.g. the ansi theme) draws no SGR at all."""
    pre, reset = _sgr("#7aa2f7")
    assert pre == "\x1b[38;2;122;162;247m"
    assert reset == "\x1b[0m"
    # Everything non-hex opts out cleanly (never raises).
    for bad in (None, "", "yellow", "ansiyellow", "#12", "#gggggg", "7aa2f7"):
        assert _sgr(bad) == ("", "")


def test_disabled_spinner_with_color_writes_no_sgr():
    """A DISABLED spinner (non-tty / enabled=False) must emit ZERO SGR even when a
    colour is set — the byte-clean piped-output guarantee. start()/stop() are
    no-ops, so nothing (least of all a truecolor escape) is written."""
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(console, enabled=False, color="#7aa2f7", timer_color="#565f89")
    assert sp.enabled is False
    sp.start()
    sp.stop()
    out = console.file.getvalue()
    assert out == ""            # nothing written when disabled
    assert "\x1b" not in out    # in particular, no SGR / ANSI at all


def test_non_tty_colored_spinner_stays_ansi_free():
    """The same guarantee via the real TTY gate: a non-terminal console disables
    the spinner, so a colour set on it can never reach the (piped) stream."""
    console = _FakeConsole(is_terminal=False)
    sp = Spinner(console, color="#bd93f9", timer_color="#6272a4")
    assert sp.enabled is False
    with sp:
        pass
    assert console.file.getvalue() == ""


def test_enabled_colored_spinner_wraps_glyph_in_truecolor_sgr(monkeypatch):
    """An ENABLED (forced-tty) spinner with a #hex colour wraps ONLY the braille
    glyph in a 38;2; truecolor SGR (and dims the timer), while a plain enabled
    spinner emits none."""
    import time as _time
    import llmcode.spinner as sp_mod

    monkeypatch.setattr(sp_mod, "_FPS_SLEEP", 0.001)
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(console, enabled=True, color="#7aa2f7", timer_color="#565f89")
    sp.start()
    deadline = _time.time() + 1.0
    while _time.time() < deadline and "\x1b[K" not in console.file.getvalue():
        _time.sleep(0.005)
    sp.stop()
    out = console.file.getvalue()
    # The glyph carries the accent truecolor SGR; the timer carries its own.
    assert "\x1b[38;2;122;162;247m" in out   # glyph tint (#7aa2f7)
    assert "\x1b[38;2;86;95;137m" in out     # timer tint (#565f89)
    # The label/hint text still renders (line is not swallowed by the colour).
    assert "working" in out and "ctrl-c to stop" in out


def test_enabled_uncolored_spinner_emits_no_truecolor_sgr(monkeypatch):
    """A plain enabled spinner (no colour) draws exactly as before: frames, but
    NO truecolor SGR — only the historic \\r / erase-EOL / cursor control."""
    import time as _time
    import llmcode.spinner as sp_mod

    monkeypatch.setattr(sp_mod, "_FPS_SLEEP", 0.001)
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(console, enabled=True)  # no color
    sp.start()
    deadline = _time.time() + 1.0
    while _time.time() < deadline and "\x1b[K" not in console.file.getvalue():
        _time.sleep(0.005)
    sp.stop()
    out = console.file.getvalue()
    assert "\x1b[K" in out          # frames drew
    assert "38;2;" not in out       # but no truecolor tint anywhere


def test_ansi_theme_color_produces_no_sgr_even_when_enabled(monkeypatch):
    """An ANSI-name colour (the ansi theme leaves spinner empty; a name never
    matches) yields no truecolor SGR — the spinner stays plain."""
    import time as _time
    import llmcode.spinner as sp_mod

    monkeypatch.setattr(sp_mod, "_FPS_SLEEP", 0.001)
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(console, enabled=True, color="", timer_color="")
    sp.start()
    deadline = _time.time() + 1.0
    while _time.time() < deadline and "\x1b[K" not in console.file.getvalue():
        _time.sleep(0.005)
    sp.stop()
    assert "38;2;" not in console.file.getvalue()


def test_start_is_idempotent_while_alive(monkeypatch):
    """A second start() while the thread is alive must NOT spawn a second one."""
    import llmcode.spinner as sp_mod

    monkeypatch.setattr(sp_mod, "_FPS_SLEEP", 0.01)
    sp = Spinner(_FakeConsole(is_terminal=True), enabled=True)
    sp.start()
    try:
        first = sp._thread
        sp.start()  # idempotent: same thread, no second spawn
        assert sp._thread is first
    finally:
        sp.stop()
    assert sp._thread is None


# ----- reactor pulse: breathing ◆ core + verb + live tok/s ----------------

def test_parse_hex_valid_and_invalid():
    """_parse_hex returns (r,g,b) for #rrggbb and None for anything else — the
    endpoints the pulse interpolates between (a bad endpoint => no truecolor)."""
    assert _parse_hex("#7aa2f7") == (122, 162, 247)
    for bad in (None, "", "yellow", "#12", "#gggggg", "7aa2f7"):
        assert _parse_hex(bad) is None


def test_disabled_reactor_spinner_with_color_verb_and_rate_writes_nothing():
    """A DISABLED reactor spinner (non-tty) with color + pulse_to + verb + a live
    rate set must emit ZERO bytes — the byte-clean piped-output guarantee. No SGR,
    no ◆, nothing: start()/stop() are no-ops when disabled."""
    console = _FakeConsole(is_terminal=False)
    sp = Spinner(
        console, enabled=False, color="#7aa2f7", timer_color="#565f89",
        pulse_to="#bb9af7", verb="forging",
    )
    sp.set_rate(226.0)
    sp.set_verb("reading providers.py")
    assert sp.enabled is False
    sp.start()
    sp.stop()
    out = console.file.getvalue()
    assert out == ""            # nothing written when disabled
    assert "\x1b" not in out    # in particular, no SGR / ANSI at all
    assert "◆" not in out and "◇" not in out


def test_reactor_frame_emits_truecolor_core_and_pulses_across_frames():
    """An ENABLED reactor spinner (both color + pulse_to hex, unicode stream) draws
    the ◆/◇ core in a 38;2; truecolor SGR, shows the verb + tok/s, and the pulse
    changes the core colour across the breathing cycle."""
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(
        console, enabled=True, color="#101010", timer_color="#565f89",
        pulse_to="#f0f0f0", verb="forging",
    )
    sp.set_rate(226.0)
    frames = [sp._frame_reactor(i, 4.0) for i in range(_PULSE_PERIOD)]
    joined = "".join(frames)
    assert "38;2;" in joined                         # truecolor SGR for the core
    assert ("◆" in joined) or ("◇" in joined)         # the reactor core glyph
    assert "forging" in joined                        # the activity verb
    assert "226" in joined and "tok/s" in joined      # live decode rate
    # The pulse BREATHES: the core colour is not constant across the cycle.
    core_colors = set(re.findall(r"38;2;\d+;\d+;\d+m[◆◇]", joined))
    assert len(core_colors) > 1
    # Valley (i=0) shows the dim ◇ shape at the dim colour; peak (mid-cycle) the ◆.
    assert "◇" in frames[0]
    assert "◆" in frames[_PULSE_PERIOD // 2]


def test_reactor_set_rate_and_verb_reflected_and_hint_demoted():
    """set_rate / set_verb are reflected live; before any rate is set the tok/s
    segment is OMITTED (never a bogus 0); the interrupt hint is demoted to a bare
    'ctrl-c' (not the long 'ctrl-c to stop')."""
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(
        console, enabled=True, color="#101010", timer_color="#565f89",
        pulse_to="#f0f0f0",
    )
    # No rate yet -> no tok/s segment (don't show 0).
    line0 = sp._frame_reactor(0, 1.0)
    assert "tok/s" not in line0
    # Publish a rate + verb from the "caller" thread; both show up.
    sp.set_rate(240.0)
    sp.set_verb("reading providers.py")
    line1 = sp._frame_reactor(3, 2.0)
    assert "240" in line1 and "tok/s" in line1
    assert "reading providers.py" in line1
    # Interrupt hint demoted: 'ctrl-c' present, but NOT the old 'ctrl-c to stop'.
    assert "ctrl-c" in line1
    assert "ctrl-c to stop" not in line1
    # A non-positive / None rate hides the segment again.
    sp.set_rate(0)
    assert "tok/s" not in sp._frame_reactor(4, 3.0)
    sp.set_rate(None)
    assert "tok/s" not in sp._frame_reactor(5, 3.0)


def test_reactor_ascii_fallback_no_truecolor_no_mojibake():
    """When the stream can't encode the diamonds, the reactor degrades to a plain
    <>/<*> pulse: NO truecolor SGR, NO unicode glyphs (no mojibake), fully
    ASCII-encodable — while still showing the verb + tok/s."""
    class _AsciiConsole(_FakeConsole):
        def __init__(self):
            super().__init__(is_terminal=True)
            self.file = io.TextIOWrapper(io.BytesIO(), encoding="ascii")

    sp = Spinner(
        _AsciiConsole(), enabled=True, color="#101010", timer_color="#565f89",
        pulse_to="#f0f0f0", verb="forging",
    )
    assert sp._core_ascii is True
    sp.set_rate(226.0)
    line = sp._frame_reactor(6, 4.0)
    assert "38;2;" not in line                 # no truecolor in the ascii fallback
    assert "\x1b" not in line                   # no ANSI/SGR at all
    assert "◆" not in line and "◇" not in line and "·" not in line  # no mojibake
    line.encode("ascii")                        # fully ASCII-encodable -> no crash
    assert "forging" in line and "226" in line and "tok/s" in line
    assert ("<*>" in line) or ("<>" in line)    # the ascii core pulse


def test_reactor_threaded_run_writes_truecolor_verb_and_clears(monkeypatch):
    """The real daemon thread on a reactor spinner writes the pulsing core (38;2;),
    the verb, and the demoted hint, then clears + restores the cursor on stop —
    proving the _run reactor branch, not just the pure frame builder."""
    import time as _time
    import llmcode.spinner as sp_mod

    monkeypatch.setattr(sp_mod, "_FPS_SLEEP", 0.001)
    console = _FakeConsole(is_terminal=True)
    sp = Spinner(
        console, enabled=True, color="#101010", timer_color="#565f89",
        pulse_to="#f0f0f0", verb="forging",
    )
    sp.set_rate(226.0)
    sp.start()
    deadline = _time.time() + 1.0
    while _time.time() < deadline and console.file.getvalue().count("\x1b[K") < 2:
        _time.sleep(0.005)
    sp.stop()
    out = console.file.getvalue()
    assert "38;2;" in out                    # truecolor core drew
    assert "forging" in out                   # the verb rendered
    assert "226" in out and "tok/s" in out    # live rate rendered
    assert "ctrl-c" in out                    # demoted interrupt hint
    assert out.endswith("\x1b[?25h")          # final act: cursor restored
    assert sp._thread is None                  # joined + cleared on stop
