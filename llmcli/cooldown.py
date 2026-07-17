"""Thermal pacing for long local-GPU runs.

A process-global "pacer" that inserts a short cool-down break every N minutes of
work so the local GPU does not overheat during long agent loops. Unlike gentle
mode (which paces the GAP between turns), this fires EVEN MID-WORK: the caller
invokes :func:`maybe_pause` at a safe checkpoint (top of each agent iteration or
before each model generation) and, when an interval has elapsed, the call blocks
for ``duration_seconds`` and then returns.

Design mirrors ``repl.gentle_wait``: the decision math is pure and every
side-effect (clock read, sleeping) is dependency-injected, so tests can simulate
hours of calls with a fake clock and a recording fake sleep WITHOUT ever calling
the real :func:`time.sleep`. State is module-global and guarded by a simple lock
so it is safe to call very frequently from any thread; the not-due path is cheap
(one clock read + one comparison).
"""

from __future__ import annotations

import threading
import time

# --- Module-level configuration (defaults) ---------------------------------
# Take a break of ``duration_seconds`` after every ``interval_seconds`` of work.
interval_seconds: float = 600.0  # 10 minutes between breaks
duration_seconds: float = 60.0  # 60 second cool-down break
enabled: bool = True

# --- Internal state (guarded by _LOCK) -------------------------------------
_LOCK = threading.Lock()
# Monotonic timestamp of the last break's END (baseline for the next interval).
# ``None`` means "not yet initialized" so the FIRST call seeds it without pausing.
_last_break: float | None = None


def _clock(now: float | None) -> float:
    """Return ``now`` if given, else a monotonic clock read."""
    return time.monotonic() if now is None else float(now)


def configure(
    *,
    enabled: bool | None = None,
    interval_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Update whichever settings are provided; silently ignore bad values.

    ``interval_seconds`` must be > 0 and ``duration_seconds`` must be >= 0; a
    value that fails validation (including non-numeric) is dropped so a bad
    config never disables pacing or wedges it into a busy-sleep loop.
    """
    with _LOCK:
        _apply_config(enabled, interval_seconds, duration_seconds)


def _apply_config(
    new_enabled: bool | None,
    new_interval: float | None,
    new_duration: float | None,
) -> None:
    """Validate and assign config globals (call while holding ``_LOCK``)."""
    global enabled, interval_seconds, duration_seconds
    if new_enabled is not None:
        enabled = bool(new_enabled)
    if new_interval is not None:
        try:
            value = float(new_interval)
        except (TypeError, ValueError):
            value = None
        if value is not None and value > 0.0:
            interval_seconds = value
    if new_duration is not None:
        try:
            value = float(new_duration)
        except (TypeError, ValueError):
            value = None
        if value is not None and value >= 0.0:
            duration_seconds = value


def reset(now: float | None = None) -> None:
    """Reset the "last break" baseline to ``now`` (or the current clock).

    Call at the START of a work session so the first break happens one full
    interval in, not immediately.
    """
    global _last_break
    with _LOCK:
        _last_break = _clock(now)


def maybe_pause(
    *,
    now: float | None = None,
    sleep=time.sleep,
    notify=None,
) -> float:
    """Pause for a cool-down break if an interval has elapsed; else do nothing.

    Returns the seconds actually slept (0.0 when no break fired). Behaviour:

    * disabled or ``duration_seconds`` <= 0 -> return 0.0, never sleep;
    * FIRST call (no baseline yet) -> seed the baseline to ``now`` and return
      0.0 (no immediate pause);
    * ``now - last_break >= interval_seconds`` -> optionally ``notify(msg)``,
      then ``sleep(duration_seconds)``, advance the baseline past the break, and
      return the seconds slept;
    * otherwise -> return 0.0.

    ``sleep`` and ``now`` are injectable so tests never block on the real clock.
    """
    # Snapshot config/state under the lock, but perform notify()/sleep() OUTSIDE
    # the lock so a real break does not hold the mutex for ``duration_seconds``.
    with _LOCK:
        if not enabled or duration_seconds <= 0.0:
            return 0.0
        current = _clock(now)
        global _last_break
        if _last_break is None:
            _last_break = current
            return 0.0
        if current - _last_break < interval_seconds:
            return 0.0
        pause_for = duration_seconds

    msg = (
        f"thermal break: pausing {pause_for:g}s to cool the GPU "
        f"(every {interval_seconds / 60.0:g} min)"
    )
    if notify is not None:
        notify(msg)
    sleep(pause_for)

    # Advance the baseline to the END of the break so the next break is one full
    # interval later. Prefer a fresh clock read when the caller uses a real
    # clock; when ``now`` was injected, derive the post-sleep time from it so a
    # fake clock stays deterministic.
    post = _clock(now) if now is None else float(now) + pause_for
    with _LOCK:
        _last_break = post
    return pause_for


def status() -> dict:
    """Snapshot for a ``/cooldown status`` command.

    ``seconds_until_next_break`` is the remaining time until the next break is
    due (0.0 when already due or not yet initialized); it reflects a real
    monotonic clock read.
    """
    with _LOCK:
        cur_enabled = enabled
        cur_interval = interval_seconds
        cur_duration = duration_seconds
        baseline = _last_break
    if baseline is None:
        remaining = cur_interval
    else:
        elapsed = time.monotonic() - baseline
        remaining = cur_interval - elapsed
    if remaining < 0.0:
        remaining = 0.0
    return {
        "enabled": cur_enabled,
        "interval_seconds": cur_interval,
        "duration_seconds": cur_duration,
        "seconds_until_next_break": remaining,
    }
