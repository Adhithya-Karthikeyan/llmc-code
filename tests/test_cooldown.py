"""Tests for llmcli.cooldown thermal pacing.

All tests use a fake clock (explicit ``now=`` values) and a recording fake
sleep so the real :func:`time.sleep` is NEVER invoked. Each test resets the
module config to defaults so process-global state does not leak between tests.
"""

import pytest

from llmcli import cooldown


class FakeSleep:
    """Records every sleep duration without actually sleeping."""

    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)

    @property
    def total(self):
        return sum(self.calls)


@pytest.fixture(autouse=True)
def _reset_module():
    """Restore module defaults and clear the baseline around every test."""
    cooldown.configure(enabled=True, interval_seconds=600.0, duration_seconds=60.0)
    cooldown._last_break = None
    yield
    cooldown.configure(enabled=True, interval_seconds=600.0, duration_seconds=60.0)
    cooldown._last_break = None


def test_first_call_never_pauses():
    sleep = FakeSleep()
    slept = cooldown.maybe_pause(now=0.0, sleep=sleep)
    assert slept == 0.0
    assert sleep.calls == []


def test_crossing_interval_triggers_one_pause_and_resets_baseline():
    sleep = FakeSleep()
    # Seed baseline at t=0.
    assert cooldown.maybe_pause(now=0.0, sleep=sleep) == 0.0
    # Just before the interval -> no pause.
    assert cooldown.maybe_pause(now=599.0, sleep=sleep) == 0.0
    assert sleep.calls == []
    # At/after the interval -> exactly one pause of duration_seconds.
    assert cooldown.maybe_pause(now=600.0, sleep=sleep) == 60.0
    assert sleep.calls == [60.0]
    # Baseline moved to end of break (600 + 60 = 660); next break due at 1260.
    assert cooldown.maybe_pause(now=1200.0, sleep=sleep) == 0.0
    assert cooldown.maybe_pause(now=1260.0, sleep=sleep) == 60.0
    assert sleep.calls == [60.0, 60.0]


def test_twenty_five_minutes_fires_exactly_two_breaks():
    sleep = FakeSleep()
    notes = []
    cooldown.reset(now=0.0)
    breaks = 0
    # Frequent calls every 30 (fake) seconds across 25 minutes (1500s).
    t = 0.0
    while t <= 1500.0:
        slept = cooldown.maybe_pause(now=t, sleep=sleep, notify=notes.append)
        if slept > 0.0:
            breaks += 1
        t += 30.0
    assert breaks == 2
    assert sleep.calls == [60.0, 60.0]
    # notify called once per break with a descriptive message.
    assert len(notes) == 2
    for msg in notes:
        assert "thermal break" in msg
        assert "60s" in msg


def test_disabled_never_pauses():
    sleep = FakeSleep()
    cooldown.configure(enabled=False)
    cooldown.reset(now=0.0)
    for t in (0.0, 600.0, 1200.0, 2400.0):
        assert cooldown.maybe_pause(now=t, sleep=sleep) == 0.0
    assert sleep.calls == []


def test_zero_duration_never_pauses():
    sleep = FakeSleep()
    cooldown.configure(duration_seconds=0.0)
    cooldown.reset(now=0.0)
    for t in (0.0, 600.0, 1200.0, 2400.0):
        assert cooldown.maybe_pause(now=t, sleep=sleep) == 0.0
    assert sleep.calls == []


def test_notify_optional_when_break_fires():
    sleep = FakeSleep()
    cooldown.reset(now=0.0)
    # No notify callback -> should not raise, still pauses.
    assert cooldown.maybe_pause(now=700.0, sleep=sleep) == 60.0
    assert sleep.calls == [60.0]


def test_configure_validation_ignores_bad_values():
    # Bad interval (<= 0) and bad duration (< 0) are ignored.
    cooldown.configure(interval_seconds=-5.0, duration_seconds=-1.0)
    assert cooldown.interval_seconds == 600.0
    assert cooldown.duration_seconds == 60.0
    # Non-numeric values are ignored too.
    cooldown.configure(interval_seconds="nope", duration_seconds=object())
    assert cooldown.interval_seconds == 600.0
    assert cooldown.duration_seconds == 60.0
    # Zero interval is invalid; zero duration is valid.
    cooldown.configure(interval_seconds=0.0)
    assert cooldown.interval_seconds == 600.0
    cooldown.configure(duration_seconds=0.0)
    assert cooldown.duration_seconds == 0.0
    # Valid values apply.
    cooldown.configure(interval_seconds=300.0, duration_seconds=30.0)
    assert cooldown.interval_seconds == 300.0
    assert cooldown.duration_seconds == 30.0


def test_configure_custom_interval_and_duration_behaviour():
    sleep = FakeSleep()
    cooldown.configure(interval_seconds=300.0, duration_seconds=30.0)
    cooldown.reset(now=0.0)
    assert cooldown.maybe_pause(now=299.0, sleep=sleep) == 0.0
    assert cooldown.maybe_pause(now=300.0, sleep=sleep) == 30.0
    assert sleep.calls == [30.0]


def test_status_math():
    cooldown.configure(enabled=True, interval_seconds=600.0, duration_seconds=60.0)
    cooldown._last_break = None
    st = cooldown.status()
    assert st["enabled"] is True
    assert st["interval_seconds"] == 600.0
    assert st["duration_seconds"] == 60.0
    # Uninitialized baseline -> full interval remaining.
    assert st["seconds_until_next_break"] == 600.0


def test_status_remaining_never_negative(monkeypatch):
    cooldown.reset(now=0.0)
    # Force baseline far in the past relative to a fixed monotonic reading.
    cooldown._last_break = 0.0
    monkeypatch.setattr(cooldown.time, "monotonic", lambda: 10_000.0)
    st = cooldown.status()
    assert st["seconds_until_next_break"] == 0.0


def test_status_reflects_disabled_flag():
    cooldown.configure(enabled=False)
    assert cooldown.status()["enabled"] is False


def test_reset_makes_first_break_one_interval_later():
    sleep = FakeSleep()
    cooldown.reset(now=1000.0)
    # Baseline is 1000; a break should not fire before 1600.
    assert cooldown.maybe_pause(now=1599.0, sleep=sleep) == 0.0
    assert cooldown.maybe_pause(now=1600.0, sleep=sleep) == 60.0
    assert sleep.calls == [60.0]
