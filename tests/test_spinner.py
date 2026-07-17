"""Spinner (braille working-indicator) tests — offline, deterministic, no sleeping.

The line builder is pure, so we assert on its output directly. The threaded
parts are exercised only through their TTY-guard (non-TTY => no thread, no
output) and the idempotent self-erasing stop().
"""

from __future__ import annotations

import io

from llmcli.spinner import (
    _ASCII_FRAMES,
    _BRAILLE_FRAMES,
    AntSpinner,
    DuckSpinner,
    Spinner,
    _frame,
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
    monkeypatch.setenv("LLMCLI_ASCII_SPINNER", "1")
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
    monkeypatch.setenv("LLMCLI_NO_SPINNER", "1")
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

    import llmcli.spinner as sp_mod

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
    import llmcli.spinner as sp_mod

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


def test_start_is_idempotent_while_alive(monkeypatch):
    """A second start() while the thread is alive must NOT spawn a second one."""
    import llmcli.spinner as sp_mod

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
