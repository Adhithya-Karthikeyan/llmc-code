"""A tiny "working" indicator with a smooth braille spinner (stdlib only).

While llmcode is busy — waiting on the (often long) model think/generate, or
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
    LLMCODE_ASCII_SPINNER off-switch is set) the spinner degrades to a plain
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

import math
import os
import sys
import threading
import time

# Off-switch: set LLMCODE_NO_SPINNER=1 (or any non-empty value) to force-disable
# the spinner even on a real TTY. Default is ON for terminals.
_ENV_DISABLE = "LLMCODE_NO_SPINNER"

# Force the ASCII fallback spinner even on a unicode-capable terminal. Also
# triggered automatically when the stream can't encode the braille glyph.
_ENV_ASCII = "LLMCODE_ASCII_SPINNER"

# Standard braille "dots" spinner cycle: one glyph per tick, wraps at the end.
_BRAILLE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# ASCII fallback spinner glyphs (classic |/-\ cycle).
_ASCII_FRAMES = "|/-\\"

# --- Reactor pulse ("The Local Reactor") ---------------------------------
# The signature ◆ core BREATHES: its shape + truecolor brightness ramp between a
# dim valley (the spinner colour) and a bright peak (``pulse_to``) on a ~1s sine
# keyed to the frame counter. ◇ is the dim/valley shape, ◆ the peak — the shape
# swap gives the beat a stronger read. On a stream that can't encode the diamonds
# we fall back to an ASCII ``<>``/``<*>`` pulse with NO truecolor at all (no
# mojibake, no SGR). This is the "your GPU under load" microinteraction.
_CORE_GLYPH = "◆"        # peak / bright shape
_CORE_GLYPH_DIM = "◇"    # valley / dim shape (stronger beat)
_ASCII_CORE = "<*>"      # ascii peak pulse
_ASCII_CORE_DIM = "<>"   # ascii valley pulse
_RATE_UNIT = "tok/s"     # live decode-rate unit (your hardware, ticking up)
# Frames per breath (~1s at the default ~10 fps). Keyed to the frame counter so
# the pulse is deterministic regardless of the wall-clock frame sleep (tests
# monkeypatch _FPS_SLEEP, which must not change the breathing curve's shape).
_PULSE_PERIOD = 12

# Short status label + the real interrupt hint (SIGINT sets the cancel flag).
_LABEL = "working"
_HINT = "ctrl-c to stop"
# Demoted interrupt hint for the reactor pulse: a dim, right-anchored "ctrl-c".
_HINT_SHORT = "ctrl-c"
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


def _sgr(hex_or_none: str | None) -> tuple[str, str]:
    """Truecolor SGR ``(prefix, reset)`` for a ``#rrggbb`` hex, else ``("", "")``.

    Returns the ANSI 24-bit foreground escape + a full reset so a caller can wrap
    a substring (the glyph / the timer) in colour. Any NON-hex value — ``None``,
    ``""``, an ANSI colour NAME ("yellow"), or a malformed string — yields
    ``("", "")`` so the frame stays uncolored and the spinner draws exactly as
    before (this is how the ansi theme, whose ``spinner`` is empty, opts out of
    truecolor). Never raises: a bad hex just falls back to no colour.
    """
    if (
        not hex_or_none
        or not isinstance(hex_or_none, str)
        or not hex_or_none.startswith("#")
        or len(hex_or_none) < 7
    ):
        return "", ""
    try:
        r = int(hex_or_none[1:3], 16)
        g = int(hex_or_none[3:5], 16)
        b = int(hex_or_none[5:7], 16)
    except ValueError:  # non-hex chars after the "#"
        return "", ""
    return f"\x1b[38;2;{r};{g};{b}m", "\x1b[0m"


def _parse_hex(hex_or_none: str | None) -> tuple[int, int, int] | None:
    """Parse ``#rrggbb`` -> ``(r, g, b)`` ints, else ``None``. Never raises.

    Mirrors the validation in :func:`_sgr` so the reactor pulse can interpolate
    between two theme hexes (the dim valley and the bright peak). A non-hex value
    (``None``, ``""``, an ANSI colour name, or a malformed string) yields ``None``
    so the pulse cleanly opts out of truecolor.
    """
    if (
        not hex_or_none
        or not isinstance(hex_or_none, str)
        or not hex_or_none.startswith("#")
        or len(hex_or_none) < 7
    ):
        return None
    try:
        return (
            int(hex_or_none[1:3], 16),
            int(hex_or_none[3:5], 16),
            int(hex_or_none[5:7], 16),
        )
    except ValueError:  # non-hex chars after the "#"
        return None


def _stream_can_encode(stream, chars: str) -> bool:
    """True if ``stream`` can encode ``chars``. Never raises.

    A stream without an ``encoding`` (e.g. ``io.StringIO`` used in tests) is
    treated as capable. An encoding that can't represent ``chars`` (e.g.
    ``ascii``/``cp1252``) — or any error probing it — returns False so writes
    can't blow up mid-animation and the caller can degrade to ASCII.
    """
    enc = getattr(stream, "encoding", None)
    if not enc:
        return True
    try:
        chars.encode(enc)
        return True
    except Exception:  # noqa: BLE001 - detection must never crash
        return False


def _stream_supports_unicode(stream) -> bool:
    """True if ``stream`` can encode the braille glyph. Never raises.

    A stream without an ``encoding`` (e.g. ``io.StringIO`` used in tests) is
    treated as unicode-capable. An encoding that can't represent the braille
    glyph (e.g. ``ascii``/``cp1252``) — or any error probing it — falls back to
    ASCII so writes can't blow up mid-animation.
    """
    return _stream_can_encode(stream, _BRAILLE_FRAMES[0])


class Spinner:
    """A daemon-thread braille spinner bound to a rich ``console``.

    Use as a context manager or via explicit ``start()`` / ``stop()``. ``stop()``
    is idempotent and always erases the spinner's line when enabled.
    """

    def __init__(
        self,
        console,
        enabled: bool | None = None,
        *,
        color: str | None = None,
        timer_color: str | None = None,
        glyph: str = _CORE_GLYPH,
        pulse_to: str | None = None,
        verb: str = _LABEL,
    ):
        self.console = console
        # Optional per-theme colours (truecolor ``#hex``). ``color`` tints ONLY the
        # braille glyph; ``timer_color`` dims ONLY the "Ns" seconds field. Both are
        # emitted as raw SGR (this stream bypasses rich) and ONLY on an enabled
        # (TTY-gated) spinner, so piped/non-tty output stays byte-for-byte clean.
        # A non-hex/empty value (e.g. the ansi theme) yields no SGR — see ``_sgr``.
        self.color = color
        self.timer_color = timer_color
        # REACTOR PULSE (opt-in): when ``pulse_to`` is a real ``#hex`` AND ``color``
        # is too, the spinner engages the breathing ◆ core — its shape + colour ramp
        # between ``color`` (dim valley) and ``pulse_to`` (bright peak) — plus an
        # activity verb and live tok/s. Constructions WITHOUT ``pulse_to`` keep the
        # exact classic braille/coloured frame (no behaviour change).
        self.glyph = glyph or _CORE_GLYPH
        self.pulse_to = pulse_to
        # Thread-safe cross-thread state (the daemon thread READS these; the caller
        # WRITES them via set_verb/set_rate). Each is a single immutable value
        # (str / float / None), so a bare attribute read/write is atomic under the
        # GIL — the daemon can never observe a torn/mid-mutation structure.
        self._verb = str(verb) if verb else _LABEL
        self._rate: float | None = None
        if enabled is None:
            # Default ON only for a real TTY, and only if the env off-switch is
            # not set. A None/typeless console (no is_terminal) => disabled.
            is_tty = bool(getattr(console, "is_terminal", False)) if console is not None else False
            enabled = is_tty and not os.environ.get(_ENV_DISABLE)
        self.enabled = bool(enabled)
        # Pick the glyph set once: honor the explicit ASCII off-switch, else fall
        # back to ASCII only when the stream can't encode the braille glyph.
        self.ascii_mode = bool(os.environ.get(_ENV_ASCII)) or not _stream_supports_unicode(self._stream())
        # The reactor core (◆/◇) has its own encode gate: a stream that can't take
        # the braille glyph almost never takes the diamonds either, but probe them
        # explicitly so the ASCII <>/<*> fallback is guaranteed mojibake-free.
        self._core_ascii = self.ascii_mode or not _stream_can_encode(
            self._stream(), _CORE_GLYPH + _CORE_GLYPH_DIM
        )
        # Reactor engages only when we can pulse: BOTH a dim valley (``color``) and
        # a bright peak (``pulse_to``) must be real ``#hex`` (else there's nothing
        # to interpolate). Computed once; the SGR it emits can only ever reach an
        # enabled (TTY-gated) stream because the animator thread never starts when
        # disabled — so piped/non-tty output stays byte-for-byte clean.
        self._reactor = bool(_sgr(color)[0]) and bool(_sgr(pulse_to)[0])
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
    def _frame_colored(self, i: int, elapsed: float) -> str:
        """``_frame`` with ONLY the glyph tinted (``self.color``) and ONLY the
        ``Ns`` timer dimmed (``self.timer_color``) via truecolor SGR.

        The rest of the line (label, separators, interrupt hint) stays the
        terminal default — the whole line is never colored (keeps it calm). Each
        coloured span resets immediately, so the frame ends in the default colour
        BEFORE ``_run`` appends the erase-to-EOL; no colour can bleed past the
        line. Used only when the spinner is enabled AND ``color`` is a real hex.
        """
        frames = _ASCII_FRAMES if self.ascii_mode else _BRAILLE_FRAMES
        glyph = frames[i % len(frames)]
        sep = _SEP_ASCII if self.ascii_mode else _SEP
        secs = max(0, int(elapsed))
        g_pre, g_reset = _sgr(self.color)
        t_pre, t_reset = _sgr(self.timer_color)
        glyph_part = f"{g_pre}{glyph}{g_reset}"
        timer_part = f"{t_pre}{secs}s{t_reset}"
        return f"{glyph_part} {_LABEL} {sep} {timer_part} {sep} {_HINT}"

    # ----- reactor: thread-safe live state -------------------------------
    def set_verb(self, verb: str) -> None:
        """Thread-safe: update the activity verb shown in the reactor frame
        (``forging`` / ``reading providers.py`` / ``running <cmd>`` / ``thinking``).

        The caller (agent.run's main thread) writes it; the daemon thread reads a
        single immutable string, so no lock is needed — the write is atomic under
        the GIL and can never be observed half-applied. A falsy value is ignored so
        the verb never regresses to blank mid-run.
        """
        try:
            if verb:
                self._verb = str(verb)
        except Exception:  # noqa: BLE001 - a setter must never raise into the caller
            pass

    def set_rate(self, tok_per_s: float | None) -> None:
        """Thread-safe: publish the live decode rate (tok/s) for the reactor frame.

        THREAD-SAFETY (design risk flag): this is a single float attribute updated
        from the caller's streaming snapshot (tokens_so_far / elapsed). The daemon
        thread reads ONLY this float — never a mid-mutation structure — so it can
        never see torn state. ``None`` (or a non-positive/uncoercible value) hides
        the ``tok/s`` segment entirely (we never flash a bogus ``0``).
        """
        try:
            self._rate = float(tok_per_s) if tok_per_s is not None else None
        except (TypeError, ValueError):
            self._rate = None

    # ----- reactor: pulse maths ------------------------------------------
    def _pulse_t(self, i: int) -> float:
        """Breathing curve in ``[0, 1]`` for frame ``i`` — valley(0) -> peak(1) ->
        valley(0) over ``_PULSE_PERIOD`` frames. Keyed to the frame counter (not the
        wall clock) so the pulse is deterministic and test-stable."""
        period = _PULSE_PERIOD if _PULSE_PERIOD > 1 else 2
        phase = (i % period) / float(period)
        return (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0

    def _pulse_rgb(self, t: float) -> tuple[int, int, int] | None:
        """Interpolate the glyph RGB between ``color`` (valley) and ``pulse_to``
        (peak) at ``t`` in ``[0, 1]``. ``None`` if either endpoint isn't a hex."""
        c0 = _parse_hex(self.color)
        c1 = _parse_hex(self.pulse_to)
        if c0 is None or c1 is None:
            return None
        return tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1))  # type: ignore[return-value]

    def _frame_reactor(self, i: int, elapsed: float) -> str:
        """The reactor frame: a breathing ◆ core + activity verb + live tok/s + a
        dim, right-anchored ``ctrl-c``. Reads the thread-safe verb/rate snapshots.

        Unicode: the core shape (◆ peak / ◇ valley) and its truecolor ramp pulse on
        ``_pulse_t``; the ``Ns`` timer and the ``tok/s`` unit stay dim
        (``timer_color``); the rate NUMBER carries the accent (``color``) so it
        reads. ASCII fallback (stream can't encode the diamonds): a plain
        ``<>``/``<*>`` pulse with the ``-`` separator and ZERO SGR — no truecolor,
        no mojibake. Never raises (the ``_run`` caller also guards)."""
        try:
            t = self._pulse_t(i)
        except Exception:  # noqa: BLE001 - the daemon must never die on a frame build
            t = 0.0
        ascii_core = self._core_ascii
        sep = _SEP_ASCII if ascii_core else _SEP
        # ASCII fallback emits NO truecolor at all (byte-clean + mojibake-free);
        # unicode mode dims the timer/unit and accents the rate number.
        if ascii_core:
            t_pre = t_reset = ""
            n_pre = n_reset = ""
        else:
            t_pre, t_reset = _sgr(self.timer_color)
            n_pre, n_reset = _sgr(self.color)
        verb = self._verb or _LABEL          # atomic snapshot read
        secs = max(0, int(elapsed))
        rate = self._rate                     # atomic snapshot read
        # --- breathing core (plain shape for width; styled shape for the wire) ---
        if ascii_core:
            core_plain = _ASCII_CORE if t >= 0.5 else _ASCII_CORE_DIM
            core_styled = core_plain
        else:
            glyph = self.glyph if t >= 0.5 else _CORE_GLYPH_DIM
            rgb = self._pulse_rgb(t)
            core_plain = glyph
            core_styled = (
                f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{glyph}\x1b[0m" if rgb else glyph
            )
        # --- core + verb + timer ---
        plain = f"{core_plain} {verb} {sep} {secs}s"
        styled = f"{core_styled} {verb} {sep} {t_pre}{secs}s{t_reset}"
        # --- live tok/s (omit entirely when unset / non-positive) ---
        if rate is not None and rate > 0:
            num = str(int(round(rate)))
            plain += f" {sep} {num} {_RATE_UNIT}"
            styled += f" {sep} {n_pre}{num}{n_reset} {t_pre}{_RATE_UNIT}{t_reset}"
        # --- dim interrupt hint, right-anchored when the console width is known ---
        try:
            width = int(getattr(self.console, "width", 0) or 0)
        except Exception:  # noqa: BLE001 - width probing must never raise
            width = 0
        need = len(plain) + len(_HINT_SHORT)
        gap = width - need if (width and width > need + 2) else 2
        gap_str = " " * max(2, gap)
        hint_part = _HINT_SHORT if ascii_core else f"{t_pre}{_HINT_SHORT}{t_reset}"
        return styled + gap_str + hint_part

    def _run(self) -> None:
        i = 0
        # Reactor pulse when both a valley + peak hex are set; else the classic
        # coloured frame when a real ``#hex`` glyph colour is set; else plain. The
        # thread only runs on an enabled (TTY-gated) spinner, so any SGR here can
        # never reach a piped/non-tty stream.
        reactor = self._reactor
        colored = bool(_sgr(self.color)[0])
        while not self._stop_event.is_set():
            elapsed = _now() - self._start_ts
            # Pad to a fixed width so a shorter line can never leave a stale tail
            # behind; the trailing erase-to-EOL is a belt-and-braces cleanup that
            # also clears the extra column when the seconds field gains a digit.
            # Padding is sized on the PLAIN (visible) width so the SGR bytes in the
            # coloured frame don't skew it; the erase-to-EOL is the real backstop.
            # The whole frame build is guarded so a bad verb/rate read can never
            # kill the daemon (which would leave the cursor hidden).
            try:
                if reactor:
                    # The reactor frame self-anchors ``ctrl-c``; the trailing
                    # erase-to-EOL clears any longer previous tail, so no manual pad.
                    self._write("\r" + self._frame_reactor(i, elapsed) + _ERASE_EOL)
                elif colored:
                    plain_len = len(_frame(i, elapsed, self.ascii_mode))
                    pad = " " * max(0, self._max_line - plain_len)
                    self._write("\r" + self._frame_colored(i, elapsed) + pad + _ERASE_EOL)
                else:
                    self._write(
                        "\r" + _frame(i, elapsed, self.ascii_mode).ljust(self._max_line) + _ERASE_EOL
                    )
            except Exception:  # noqa: BLE001 - a frame-build error must never kill the daemon
                pass
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
