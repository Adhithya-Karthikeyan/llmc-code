"""Per-project session persistence tests (llmcli/session.py + REPL wiring).

All offline/deterministic: a temp HOME (so we never touch the real
~/.llm-cli/sessions) and a MockProvider. Covers the pure session helpers
(stable/sanitized ids, save/load round-trip, corrupt/missing -> None, the
<=1-message skip, meta shape, derive_title, relative_time buckets) and the REPL
surface (/resume, /forget, the -c startup load, and a one-shot with no session).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import llmcli.repl as r
import llmcli.session as s
from llmcli.config import Config
from llmcli.providers import MockProvider


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    """Point Path.home() at a temp dir so sessions land under tmp, not real HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(s.Path, "home", classmethod(lambda cls: home))
    return home


# --------------------------------------------------------------------------- #
# session_id: stable per cwd, differs across cwds, sanitizes the basename
# --------------------------------------------------------------------------- #

def test_session_id_stable_for_same_cwd(tmp_path):
    a = s.session_id(str(tmp_path))
    b = s.session_id(str(tmp_path))
    assert a == b


def test_session_id_differs_across_cwds(tmp_path):
    d1 = tmp_path / "alpha"
    d2 = tmp_path / "beta"
    d1.mkdir()
    d2.mkdir()
    assert s.session_id(str(d1)) != s.session_id(str(d2))


def test_session_id_differs_for_same_basename_different_paths(tmp_path):
    # Same basename "proj" but different parents -> different ids (hash of abspath).
    p1 = tmp_path / "one" / "proj"
    p2 = tmp_path / "two" / "proj"
    p1.mkdir(parents=True)
    p2.mkdir(parents=True)
    id1, id2 = s.session_id(str(p1)), s.session_id(str(p2))
    assert id1 != id2
    assert id1.startswith("proj-") and id2.startswith("proj-")


def test_session_id_sanitizes_basename(tmp_path):
    weird = tmp_path / "my project!@#"
    weird.mkdir()
    sid = s.session_id(str(weird))
    # Everything before the 12-hex suffix must be filesystem-safe.
    slug = sid.rsplit("-", 1)[0]
    assert all(c.isalnum() or c in "._-" for c in slug)
    assert " " not in sid and "!" not in sid and "@" not in sid


def test_session_id_shape_is_basename_dash_12hex(tmp_path):
    sid = s.session_id(str(tmp_path))
    suffix = sid.rsplit("-", 1)[1]
    assert len(suffix) == 12
    assert all(c in "0123456789abcdef" for c in suffix)


# --------------------------------------------------------------------------- #
# save / load round-trip + best-effort behavior
# --------------------------------------------------------------------------- #

def _msgs():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi"},
    ]


def test_save_load_round_trip(tmp_path):
    cwd = str(tmp_path)
    s.save_session(cwd, _msgs(), model="qwen", title="hello world")
    data = s.load_session(cwd)
    assert data is not None
    assert data["messages"] == _msgs()
    assert data["model"] == "qwen"
    assert data["title"] == "hello world"
    assert data["cwd"] == str(tmp_path)
    # updated_at is a parseable ISO-8601 timestamp.
    assert datetime.fromisoformat(data["updated_at"]).tzinfo is not None


def test_save_creates_sessions_dir(tmp_path, _tmp_home):
    assert not s.sessions_dir().exists()
    s.save_session(str(tmp_path), _msgs(), model="m", title="t")
    assert s.sessions_dir().exists()
    assert s.session_path(str(tmp_path)).exists()


def test_load_missing_returns_none(tmp_path):
    assert s.load_session(str(tmp_path)) is None


def test_load_corrupt_returns_none(tmp_path):
    s.save_session(str(tmp_path), _msgs(), model="m", title="t")
    s.session_path(str(tmp_path)).write_text("{not valid json", encoding="utf-8")
    assert s.load_session(str(tmp_path)) is None


def test_load_invalid_shape_returns_none(tmp_path):
    # Valid JSON, but not the expected object-with-messages-list shape.
    s.sessions_dir().mkdir(parents=True, exist_ok=True)
    s.session_path(str(tmp_path)).write_text('["a", "b"]', encoding="utf-8")
    assert s.load_session(str(tmp_path)) is None


def test_load_filters_non_dict_messages(tmp_path):
    # A hand-edited file with stray scalars in the list must not crash any
    # consumer that does msg.get(...) — load drops the non-dict elements.
    s.sessions_dir().mkdir(parents=True, exist_ok=True)
    s.session_path(str(tmp_path)).write_text(
        '{"messages": [42, {"role": "user", "content": "ok"}, "junk"]}',
        encoding="utf-8",
    )
    data = s.load_session(str(tmp_path))
    assert data is not None
    assert data["messages"] == [{"role": "user", "content": "ok"}]


def test_save_skips_when_one_or_fewer_messages(tmp_path):
    # Just the system prompt = nothing to remember -> no file written.
    s.save_session(str(tmp_path), [{"role": "system", "content": "sys"}], "m", "t")
    assert not s.session_path(str(tmp_path)).exists()
    assert s.load_session(str(tmp_path)) is None
    # Empty list too.
    s.save_session(str(tmp_path), [], "m", "t")
    assert not s.session_path(str(tmp_path)).exists()


def test_save_never_raises_on_oserror(tmp_path, monkeypatch):
    # Force the atomic write to fail; save_session must swallow it.
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(s.tempfile, "mkstemp", _boom)
    # Should NOT raise.
    s.save_session(str(tmp_path), _msgs(), "m", "t")


# --------------------------------------------------------------------------- #
# session_meta: shape, no big messages list, None when absent
# --------------------------------------------------------------------------- #

def test_session_meta_shape_without_messages(tmp_path):
    s.save_session(str(tmp_path), _msgs(), model="qwen", title="hello world")
    meta = s.session_meta(str(tmp_path))
    assert meta is not None
    assert set(meta) == {"updated_at", "title", "message_count", "model"}
    assert "messages" not in meta
    assert meta["message_count"] == 3
    assert meta["model"] == "qwen"
    assert meta["title"] == "hello world"


def test_session_meta_none_when_no_session(tmp_path):
    assert s.session_meta(str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# clear_session
# --------------------------------------------------------------------------- #

def test_clear_session_deletes_file(tmp_path):
    s.save_session(str(tmp_path), _msgs(), "m", "t")
    assert s.session_path(str(tmp_path)).exists()
    s.clear_session(str(tmp_path))
    assert not s.session_path(str(tmp_path)).exists()


def test_clear_session_missing_is_noop(tmp_path):
    # No file present -> must not raise.
    s.clear_session(str(tmp_path))


# --------------------------------------------------------------------------- #
# derive_title
# --------------------------------------------------------------------------- #

def test_derive_title_first_user_message():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first thing I asked"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    assert s.derive_title(msgs) == "first thing I asked"


def test_derive_title_single_lines_and_truncates():
    long = "line one\nline two " + "x" * 100
    title = s.derive_title([{"role": "user", "content": long}])
    assert "\n" not in title
    # ~60 chars + an ellipsis when truncated.
    assert len(title) <= 61
    assert title.endswith("…")


def test_derive_title_fallback_when_no_user_message():
    assert s.derive_title([{"role": "system", "content": "sys"}]) == "(no title)"
    assert s.derive_title([]) == "(no title)"


# --------------------------------------------------------------------------- #
# relative_time buckets + tolerance
# --------------------------------------------------------------------------- #

def _iso_ago(**kw):
    return (datetime.now(timezone.utc) - timedelta(**kw)).isoformat()


def test_relative_time_just_now():
    assert s.relative_time(_iso_ago(seconds=5)) == "just now"


def test_relative_time_minutes():
    assert s.relative_time(_iso_ago(minutes=5)) == "5m ago"


def test_relative_time_hours():
    assert s.relative_time(_iso_ago(hours=3)) == "3h ago"


def test_relative_time_days():
    assert s.relative_time(_iso_ago(days=2)) == "2d ago"


def test_relative_time_tolerates_bad_input():
    assert s.relative_time("not a date") == "just now"
    assert s.relative_time("") == "just now"
    assert s.relative_time(None) == "just now"


def test_relative_time_handles_naive_timestamp():
    # A naive ISO string (no tz) is treated as UTC, not a crash.
    naive = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None)
    assert s.relative_time(naive.isoformat()) == "2h ago"


# --------------------------------------------------------------------------- #
# REPL wiring: /resume, /forget, -c startup load, one-shot without a session
# --------------------------------------------------------------------------- #

class _FakeMCP:
    def __init__(self, *a, **k):
        pass

    def registry(self):
        return {}

    def start_all(self, **k):
        pass

    def shutdown_all(self):
        pass

    def status(self):
        return []


@pytest.fixture
def repl(monkeypatch, tmp_path):
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    monkeypatch.chdir(tmp_path)
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_resume_loads_saved_history_into_agent(repl, tmp_path, capsys):
    saved = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "remembered question"},
        {"role": "assistant", "content": "remembered answer"},
    ]
    s.save_session(str(tmp_path), saved, model="m", title="remembered question")

    assert repl._dispatch_slash("/resume") is True
    assert repl.agent.messages == saved
    out = capsys.readouterr().out
    assert "resumed 3 messages" in out


def test_resume_no_session_prints_message(repl, capsys):
    assert repl._dispatch_slash("/resume") is True
    assert "no saved session for this directory." in capsys.readouterr().out


def test_resume_corrupt_head_does_not_crash(repl, tmp_path, capsys):
    # A file whose messages head is a non-dict scalar must not crash /resume.
    # After filtering, two real messages survive (>=2) so a resume still happens;
    # the agent's own system prompt heads the rebuilt history.
    s.sessions_dir().mkdir(parents=True, exist_ok=True)
    s.session_path(str(tmp_path)).write_text(
        '{"updated_at": "", "messages": ["junk", '
        '{"role": "user", "content": "real"}, '
        '{"role": "assistant", "content": "reply"}]}',
        encoding="utf-8",
    )
    assert repl._dispatch_slash("/resume") is True
    # The scalar was filtered; the agent's own system prompt heads the history.
    assert repl.agent.messages[0]["role"] == "system"
    assert {"role": "user", "content": "real"} in repl.agent.messages
    assert "resumed" in capsys.readouterr().out


def test_resume_empty_history_reports_no_session(repl, tmp_path, capsys):
    # A loadable file with an empty messages list (after filtering) is treated
    # like "none" — no phantom "resumed 1 messages".
    s.sessions_dir().mkdir(parents=True, exist_ok=True)
    s.session_path(str(tmp_path)).write_text('{"messages": []}', encoding="utf-8")
    assert repl._dispatch_slash("/resume") is True
    out = capsys.readouterr().out
    assert "no saved session for this directory." in out
    assert "resumed" not in out


def test_forget_deletes_session(repl, tmp_path, capsys):
    s.save_session(str(tmp_path), _msgs(), model="m", title="t")
    assert s.session_path(str(tmp_path)).exists()
    assert repl._dispatch_slash("/forget") is True
    assert not s.session_path(str(tmp_path)).exists()
    assert "forgot this project's saved session and memory." in capsys.readouterr().out


def test_forget_clears_conversation_memory(repl, tmp_path):
    """/forget must delete the persisted memory store AND reset the live one so a
    forgotten project starts with no recalled records."""
    import llmcli.memory as memory_mod

    # Seed + persist a conversation-memory record for this project.
    repl.agent.memory.add("a remembered project fact")
    repl.agent.memory.save(memory_mod.store_path(str(tmp_path)))
    assert memory_mod.store_path(str(tmp_path)).exists()
    assert repl.agent.memory.records

    assert repl._dispatch_slash("/forget") is True

    # The persisted store file is gone and the live store is empty.
    assert not memory_mod.store_path(str(tmp_path)).exists()
    assert repl.agent.memory.records == []
    assert repl.agent.memory.vectors == {}


def test_save_session_helper_persists_current_history(repl, tmp_path):
    repl.agent.messages += [
        {"role": "user", "content": "save me"},
        {"role": "assistant", "content": "ok"},
    ]
    repl._save_session()
    data = s.load_session(str(tmp_path))
    assert data is not None
    assert data["title"] == "save me"
    assert data["message_count"] if False else len(data["messages"]) == 3


def test_continue_flag_loads_session_at_startup(monkeypatch, tmp_path, capsys):
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    monkeypatch.chdir(tmp_path)

    saved = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    s.save_session(str(tmp_path), saved, model="m", title="earlier")

    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True, resume=True)

    # Drive run() with a fake PromptSession that EOFs immediately (no turns).
    class _FakeSession:
        def prompt(self, text):
            raise EOFError

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda *a, **k: _FakeSession())
    repl.run()

    assert repl.agent.messages == saved
    out = capsys.readouterr().out
    assert "continuing previous session" in out
    # The dim startup hint must NOT also appear when we already resumed.
    assert "/resume to continue" not in out


def test_startup_hint_shown_when_session_exists_and_not_resuming(monkeypatch, tmp_path, capsys):
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    monkeypatch.chdir(tmp_path)
    s.save_session(str(tmp_path), _msgs(), model="m", title="hello world")

    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    class _FakeSession:
        def prompt(self, text):
            raise EOFError

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda *a, **k: _FakeSession())
    repl.run()
    out = capsys.readouterr().out
    assert "↩ last session" in out
    assert "/resume to continue" in out


def test_fresh_launch_no_session_prints_no_hint(monkeypatch, tmp_path, capsys):
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    monkeypatch.chdir(tmp_path)  # no saved session here

    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)

    class _FakeSession:
        def prompt(self, text):
            raise EOFError

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession", lambda *a, **k: _FakeSession())
    repl.run()
    out = capsys.readouterr().out
    assert "last session" not in out
    assert "↩" not in out
    # The startup banner replaces the old flat "llm-cli ready." line; it still
    # shows the app name + a "ready" marker.
    assert "llm-cli" in out and "ready" in out


def test_one_shot_run_once_without_session(monkeypatch, tmp_path):
    """run_once with resume=True but NO saved session must still work (no crash)
    and produce a result; and the completed turn is saved back."""
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    monkeypatch.setattr(r, "load_mcp_config", lambda *a, **k: {})
    monkeypatch.chdir(tmp_path)

    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    out = r.run_once(MockProvider(), cfg, "hello", auto_confirm=True, resume=True)
    assert isinstance(out, str)
    # The one-shot persisted its conversation so a follow-up -c can build on it.
    assert s.session_path(str(tmp_path)).exists()
