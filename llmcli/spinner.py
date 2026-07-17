"""A tiny "working" indicator with a smooth braille spinner (stdlib only).

While llmcli is busy — waiting on the (often long) model think/generate, or
running a batch of tools — a small braille dots spinner animates in place next
to a short status line that shows a LIVE elapsed timer and the interrupt hint::

    ⠙ working · 3s · ctrl-c to stop

The glyph advances one braille frame per tick; the seconds count up in whole
seconds since :meth:`Spinner.start`. "ctrl-c to stop" is the real interrupt (a
SIGINT handler sets the cancel flag).

Hard guarantees (the stream must stay byte-for-byte clean for non-TTY use):
  - TTY-ONLY: when ``console.is_terminal`` is False (piped one-shot, pytest
    capture), the spinner is DISABLED. ``start()`` spawns no thread and writes
    nothing; ``stop()`` writes nothing. Existing behavior/tests are unchanged.
  - SELF-ERASING: each tick redraws in place with a carriage-return ("\r")
    followed by a fixed-width pad and an erase-to-end-of-line ("\x1b[K"). As the
    seconds tick from 1 to 2+ digits the line grows, so the pad/erase clears any
    leftover chars and no stale tail (ghosting) can ever remain.
  - NO CURSOR JANK: ``start()`` hides the hardware cursor ("\x1b[?25l") so it
    can't bob at the end of the line; ``stop()`` (and the worker on exit)
    restores it ("\x1b[?25h"). All ANSI writes are TTY-gated and swallow errors.
  - UNICODE-SAFE: if the output stream can't encode the braille glyph (or the
    LLMCLI_ASCII_SPINNER off-switch is set) the spinner degrades to a plain
    ASCII spinner ("|/-\\") with the same "working · Ns · ctrl-c to stop" text.
    Detection never raises.
  - NEVER INTERLEAVE: the caller stops the spinner BEFORE any other console
    write (the buffered answer, narration, the collapsed tool line, the tok/s
    footer, or a y/N confirm prompt). The spinner only runs while nothing else
    writes.

The animation line builder (:func:`_frame`) is a PURE function so it can be
unit-tested without threads or sleeping.
"""
# NOTE: historically this animated an ASCII "ant" (and a "duck" before that).
# The public surface keeps ``Spinner`` (and back-compat ``AntSpinner`` /
# ``DuckSpinner`` aliases) so nothing that imported the old names breaks.

from __future__ import annotations

import os
import sys
import threading
import time

# Off-switch: set LLMCLI_NO_SPINNER=1 (or any non-empty value) to force-disable
# the spinner even on a real TTY. Default is ON for terminals.
_ENV_DISABLE = "LLMCLI_NO_SPINNER"

# Force the ASCII fallback spinner even on a unicode-capable terminal. Also
# triggered automatically when the stream can't encode the braille glyph.
_ENV_ASCII = "LLMCLI_ASCII_SPINNER"

# Standard braille "dots" spinner cycle: one glyph per tick, wraps at the end.
_BRAILLE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# ASCII fallback spinner glyphs (classic |/-\ cycle).
_ASCII_FRAMES = "|/-\\"

# Short status label + the real interrupt hint (SIGINT sets the cancel flag).
_LABEL = "working"
_HINT = "ctrl-c to stop"
# Field separator: a middle dot on unicode terminals, plain ASCII "-" otherwise.
_SEP = "·"
_SEP_ASCII = "-"

_FPS_SLEEP = 0.1  # ~10 fps

# Indirection so tests can monkeypatch the clock deterministically.
_now = time.monotonic

# ANSI control sequences (TTY-only). Erase-to-end-of-line keeps a shorter/longer
# frame from smearing a stale tail; hide/show cursor stops the hardware cursor
# from bobbing at the end of the line.
_ERASE_EOL = "\x1b[K"
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"


def _frame(i: int, elapsed: float = 0.0, ascii_mode: bool = False) -> str:
    """Return the raw status line for tick ``i`` (PURE — no I/O, no state).

    ``i`` selects the spinner glyph (advancing one frame per tick, wrapping at
    the end of the cycle). ``elapsed`` is seconds since ``start()``; it is shown
    truncated to whole seconds. ``ascii_mode`` swaps the braille glyphs + middle
    dot for a plain ASCII spinner + separator so the line is fully encodable on
    a non-unicode terminal. The result looks like ``⠙ working · 3s · ctrl-c to
    stop`` (or ``/ working - 3s - ctrl-c to stop`` in ASCII mode).
    """
    frames = _ASCII_FRAMES if ascii_mode else _BRAILLE_FRAMES
    glyph = frames[i % len(frames)]
    sep = _SEP_ASCII if ascii_mode else _SEP
    secs = max(0, int(elapsed))
    return f"{glyph} {_LABEL} {sep} {secs}s {sep} {_HINT}"


def _stream_supports_unicode(stream) -> bool:
    """True if ``stream`` can encode the braille glyph. Never raises.

    A stream without an ``encoding`` (e.g. ``io.StringIO`` used in tests) is
    treated as unicode-capable. An encoding that can't represent the braille
    glyph (e.g. ``ascii``/``cp1252``) — or any error probing it — falls back to
    ASCII so writes can't blow up mid-animation.
    """
    enc = getattr(stream, "encoding", None)
    if not enc:
        return True
    try:
        _BRAILLE_FRAMES[0].encode(enc)
        return True
    except Exception:  # noqa: BLE001 - detection must never crash
        return False


class Spinner:
    """A daemon-thread braille spinner bound to a rich ``console``.

    Use as a context manager or via explicit ``start()`` / ``stop()``. ``stop()``
    is idempotent and always erases the spinner's line when enabled.
    """

    def __init__(self, console, enabled: bool | None = None):
        self.console = console
        if enabled is None:
            # Default ON only for a real TTY, and only if the env off-switch is
            # not set. A None/typeless console (no is_terminal) => disabled.
            is_tty = bool(getattr(console, "is_terminal", False)) if console is not None else False
            enabled = is_tty and not os.environ.get(_ENV_DISABLE)
        self.enabled = bool(enabled)
        # Pick the glyph set once: honor the explicit ASCII off-switch, else fall
        # back to ASCII only when the stream can't encode the braille glyph.
        self.ascii_mode = bool(os.environ.get(_ENV_ASCII)) or not _stream_supports_unicode(self._stream())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ts = 0.0
        # Size the pad/erase for the widest line we expect to draw. The seconds
        # field grows over a long run, so pad generously; erase-to-EOL is the
        # real backstop that guarantees no ghosting when the line changes width.
        self._max_line = len(_frame(0, 0.0, self.ascii_mode)) + 8

    # ----- stream helpers ------------------------------------------------
    def _stream(self):
        """The underlying writable stream (console.file), else sys.stdout."""
        f = getattr(self.console, "file", None)
        return f if f is not None else sys.stdout

    def _write(self, s: str) -> None:
        try:
            stream = self._stream()
            stream.write(s)
            stream.flush()
        except Exception:  # noqa: BLE001 - drawing must never crash the loop
            pass

    # ----- animation -----------------------------------------------------
    def _run(self) -> None:
        i = 0
        while not self._stop_event.is_set():
            elapsed = _now() - self._start_ts
            # Pad to a fixed width so a shorter line can never leave a stale tail
            # behind; the trailing erase-to-EOL is a belt-and-braces cleanup that
            # also clears the extra column when the seconds field gains a digit.
            self._write(
                "\r" + _frame(i, elapsed, self.ascii_mode).ljust(self._max_line) + _ERASE_EOL
            )
            i += 1
            # Event.wait doubles as the sleep AND a prompt stop signal.
            if self._stop_event.wait(_FPS_SLEEP):
                break
        # The worker clears its OWN line as its last act, so no stray frame can
        # ever print after stop()'s teardown (closes the join-timeout ordering
        # gap) and the cursor is restored exactly once on the same thread.
        self._write("\r" + (" " * self._max_line) + _ERASE_EOL + "\r" + _SHOW_CURSOR)

    def start(self) -> None:
        """Spawn the animator thread (no-op if disabled or already running)."""
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        # Anchor the elapsed timer to now (monotonic so it can't run backwards).
        self._start_ts = _now()
        # Hide the hardware cursor so it can't bob at the end of the line.
        self._write(_HIDE_CURSOR)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the animator and fully erase its line. Idempotent.

        The worker thread clears its own line and restores the cursor as its
        final act, so no stray frame can print after teardown. We still emit an
        idempotent erase + cursor-restore here to cover the case where the
        thread never started or its join timed out on a clogged stream.
        """
        if not self.enabled:
            return
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
            self._thread = None
        # Idempotent backstop: erase the line and restore the cursor. Safe to
        # repeat even after the worker already did the same on exit.
        self._write(
            "\r" + (" " * self._max_line) + _ERASE_EOL + "\r" + _SHOW_CURSOR
        )

    # ----- context manager ----------------------------------------------
    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# Back-compat aliases: older imports referenced the (former) ant/duck indicator.
AntSpinner = Spinner
DuckSpinner = Spinner
