"""Foundation-wave REPL feature tests: diff preview, /undo, /mode, /init,
/doctor, custom macros, /commit message generation, and JSON one-shot output.

All tests are offline: MockProvider (or a tiny stub) + a tmp cwd, with HOME
redirected to a tmp dir so session/checkpoint state stays isolated. The Repl is
constructed exactly like tests/test_repl_slash.py (patched load_config /
save_config / MCPManager) so no disk/network/subprocess is touched unless a test
opts in.
"""

from __future__ import annotations

import json
import signal
import threading

import pytest

import llmcode.gitint as gitint
import llmcode.repl as r
from llmcode.config import Config
from llmcode.providers import MockProvider
from llmcode.tools import get_tool


class _FakeMCP:
    def __init__(self, *a, **k):
        self.configs = {}
        self._running = False

    def registry(self):
        return {}

    def is_running(self):
        return self._running

    def start_all(self, **k):
        self._running = True

    def shutdown_all(self):
        self._running = False

    def status(self):
        return []


class _StubProvider:
    """Minimal provider whose stream_chat yields a fixed text answer."""

    def __init__(self, text: str):
        self._text = text
        self.model = "stub"
        self.base_url = None

    def stream_chat(self, messages, tools, tool_choice=None):
        yield {"type": "text", "text": self._text}
        yield {"type": "done", "finish_reason": "stop", "output_tokens": 1}


class _FakeSession:
    """Records the prompt string and answers 'n' so confirm returns False.

    Mirrors prompt_toolkit's PromptSession enough for the confirm path: it holds
    a `placeholder` (the main-input ghost) and its `prompt()` accepts a per-call
    `placeholder` override + records both, so the suppression fix is testable.
    """

    def __init__(self, placeholder="Ask anything · / commands · @ files"):
        self.prompts: list[str] = []
        self.placeholders: list = []
        self.placeholder = placeholder

    def prompt(self, text, placeholder=None, **kwargs):
        self.prompts.append(text)
        self.placeholders.append(placeholder)
        return "n"


@pytest.fixture
def repl(monkeypatch, tmp_path):
    # Isolate cwd + HOME so session/checkpoint/rules I/O stays in tmp.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    disk = Config(provider="mock", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", model="m")
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


# --------------------------------------------------------------------------- #
# Diff preview
# --------------------------------------------------------------------------- #

def test_diff_preview_for_edit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "sample.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    cfg = Config(diff_preview=True)
    confirm = r.make_ptk_confirm(_FakeSession(), cfg)
    tool = get_tool("edit_file")
    assert tool is not None
    decided = confirm(tool, {"path": str(f), "old": "x = 1", "new": "x = 42"})
    assert decided is False  # fake session answered 'n'
    out = capsys.readouterr().out
    # The unified diff shows the removed old line and the added new line.
    assert "-x = 1" in out
    assert "+x = 42" in out


def test_diff_preview_off_shows_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n", encoding="utf-8")
    cfg = Config(diff_preview=False)
    confirm = r.make_ptk_confirm(_FakeSession(), cfg)
    tool = get_tool("edit_file")
    confirm(tool, {"path": str(f), "old": "x = 1", "new": "x = 2"})
    out = capsys.readouterr().out
    assert "+x = 2" not in out  # preview suppressed when diff_preview is off


def test_diff_preview_new_file_note(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "brand_new.txt"
    cfg = Config(diff_preview=True)
    confirm = r.make_ptk_confirm(_FakeSession(), cfg)
    tool = get_tool("write_file")
    confirm(tool, {"path": str(target), "content": "line1\nline2\n"})
    out = capsys.readouterr().out
    assert "new file" in out


# --------------------------------------------------------------------------- #
# Confirmation prompt: ghost-placeholder suppression + restore
# --------------------------------------------------------------------------- #

def test_confirm_suppresses_ghost_placeholder():
    # The y/N confirm must pass an EMPTY placeholder so the main-input ghost
    # ("Ask anything · …") does not render glued onto the confirmation line.
    ghost = "Ask anything · / commands · @ files"
    sess = _FakeSession(placeholder=ghost)
    confirm = r.make_ptk_confirm(sess, Config())
    tool = get_tool("write_file")
    confirm(tool, {"path": "hello.py", "content": "print('x')\n"})
    # Exactly one prompt was issued and it overrode the placeholder to empty
    # (empty string, NOT None — None is a no-op in prompt_toolkit).
    assert sess.placeholders == [""]
    assert sess.prompts and sess.prompts[0].startswith("\nRun write_file")


def test_confirm_restores_session_placeholder():
    # prompt() mutates session.placeholder; the confirm must restore it so the
    # main input keeps its ghost after the first tool confirmation.
    ghost = "Ask anything · / commands · @ files"
    sess = _FakeSession(placeholder=ghost)
    confirm = r.make_ptk_confirm(sess, Config())
    tool = get_tool("write_file")
    confirm(tool, {"path": "hello.py", "content": "print('x')\n"})
    assert sess.placeholder == ghost


def test_confirm_semantics_preserved():
    # y/yes -> True, anything else -> False; EOF/KeyboardInterrupt -> False.
    class _Yes(_FakeSession):
        def prompt(self, text, placeholder=None, **kwargs):
            self.placeholders.append(placeholder)
            return "y"

    class _Interrupt(_FakeSession):
        def prompt(self, text, placeholder=None, **kwargs):
            raise KeyboardInterrupt

    tool = get_tool("write_file")
    args = {"path": "hello.py", "content": "print('x')\n"}
    assert r.make_ptk_confirm(_Yes(), Config())(tool, args) is True
    assert r.make_ptk_confirm(_FakeSession(), Config())(tool, args) is False
    # Interrupt path still restores the placeholder (finally runs) and denies.
    sess = _Interrupt(placeholder="ghost")
    assert r.make_ptk_confirm(sess, Config())(tool, args) is False
    assert sess.placeholder == "ghost"


# --------------------------------------------------------------------------- #
# /undo
# --------------------------------------------------------------------------- #

def test_undo_calls_checkpoint_undo(repl, monkeypatch, capsys):
    calls = {}

    def fake_undo(root, *, session=None):
        calls["root"] = root
        calls["session"] = session
        return {
            "undone": True, "id": "ck-1", "label": "",
            "restored": ["a.py"], "deleted": [], "errors": [],
            "message": "Undid checkpoint ck-1: restored 1.",
        }

    monkeypatch.setattr(r.checkpoint, "undo", fake_undo)
    assert repl._dispatch_slash("/undo") is True
    assert "root" in calls
    # /undo is scoped to THIS launch's checkpoint session token.
    assert calls["session"] == repl._ckpt_session
    out = capsys.readouterr().out
    assert "Undid checkpoint" in out
    assert "restored a.py" in out


# --------------------------------------------------------------------------- #
# /mode
# --------------------------------------------------------------------------- #

def test_mode_sets_and_persists_valid(repl, monkeypatch, capsys):
    saved = {}
    monkeypatch.setattr(r, "save_config", lambda cfg, *a, **k: saved.setdefault("cfg", cfg))
    assert repl._dispatch_slash("/mode plan") is True
    assert repl.config.permission_mode == "plan"
    assert "cfg" in saved  # persisted via _persist_config -> save_config
    assert "mode -> plan" in capsys.readouterr().out


def test_mode_rejects_invalid(repl, capsys):
    before = repl.config.permission_mode
    assert repl._dispatch_slash("/mode bogus") is True
    assert repl.config.permission_mode == before
    out = capsys.readouterr().out
    assert "unknown mode" in out


def test_mode_no_arg_shows_current(repl, capsys):
    assert repl._dispatch_slash("/mode") is True
    out = capsys.readouterr().out
    assert "permission mode:" in out
    assert "auto-edit" in out  # lists the available modes


# --------------------------------------------------------------------------- #
# /init
# --------------------------------------------------------------------------- #

def test_init_writes_agents_md_when_absent(repl, tmp_path, capsys):
    assert repl._dispatch_slash("/init") is True
    agents = tmp_path / "AGENTS.md"
    assert agents.is_file()
    assert "# Project rules" in agents.read_text(encoding="utf-8")
    assert "created AGENTS.md" in capsys.readouterr().out


def test_init_reports_existing(repl, tmp_path, capsys):
    (tmp_path / "AGENTS.md").write_text("# existing\n", encoding="utf-8")
    assert repl._dispatch_slash("/init") is True
    out = capsys.readouterr().out
    assert "already exists" in out


# --------------------------------------------------------------------------- #
# /doctor
# --------------------------------------------------------------------------- #

def test_doctor_runs_without_raising(repl, capsys):
    assert repl._dispatch_slash("/doctor") is True
    out = capsys.readouterr().out
    assert "doctor" in out
    # Several checks are reported (marks are ✓ or ✗).
    assert ("✓" in out) or ("✗" in out)


def test_doctor_prints_server_tuning_advice(repl, monkeypatch, capsys):
    # A detected local window -> the ranked server-tuning section is printed.
    monkeypatch.setattr(r, "detect_context_length", lambda *a, **k: 262144)
    repl._cmd_doctor("")
    out = capsys.readouterr().out
    assert "Server tuning" in out
    assert "flash attention" in out
    assert "cache-type" in out
    assert "Speculative" in out
    # 262144 > 4 * default context_budget (12000) -> the reload-smaller-ctx tip fires.
    assert "smaller" in out and "262144" in out


def test_doctor_tuning_unavailable_without_endpoint(repl, monkeypatch, capsys):
    # No detectable window (lookup returns None) -> graceful skip, no exception.
    monkeypatch.setattr(r, "detect_context_length", lambda *a, **k: None)
    repl._cmd_doctor("")
    out = capsys.readouterr().out
    assert "server tuning advice unavailable" in out
    assert "flash attention" not in out


# --------------------------------------------------------------------------- #
# Custom macros
# --------------------------------------------------------------------------- #

def test_macro_discovered_and_expands_arguments(repl, tmp_path, monkeypatch):
    cmd_dir = tmp_path / ".llmcode" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "greet.md").write_text("Say hello to $ARGUMENTS now.", encoding="utf-8")
    submitted = {}
    monkeypatch.setattr(repl, "_submit", lambda text: submitted.setdefault("text", text))
    assert repl._dispatch_slash("/greet world") is True
    assert submitted.get("text") == "Say hello to world now."


def test_commands_lists_macros(repl, tmp_path, capsys):
    cmd_dir = tmp_path / ".llmcode" / "commands"
    cmd_dir.mkdir(parents=True)
    (cmd_dir / "deploy.md").write_text("deploy $ARGUMENTS", encoding="utf-8")
    assert repl._dispatch_slash("/commands") is True
    out = capsys.readouterr().out
    assert "/deploy" in out


# --------------------------------------------------------------------------- #
# /commit
# --------------------------------------------------------------------------- #

def test_commit_generates_message_and_commits(repl, monkeypatch, capsys):
    captured = {}
    monkeypatch.setattr(gitint, "is_repo", lambda root: True)
    monkeypatch.setattr(gitint, "diff", lambda root, path=None: "diff --git a/x b/x\n+hi\n")

    def fake_commit(root, msg):
        captured["msg"] = msg
        return {"ok": True, "commit_hash": "abcdef123456"}

    monkeypatch.setattr(gitint, "commit_all", fake_commit)
    # Provider yields the generated one-line message.
    repl.provider = _StubProvider("feat: add greeting output")
    assert repl._dispatch_slash("/commit") is True
    assert captured.get("msg") == "feat: add greeting output"
    out = capsys.readouterr().out
    assert "committed" in out


def test_commit_uses_explicit_message(repl, monkeypatch, capsys):
    captured = {}

    def fake_commit(root, msg):
        captured["msg"] = msg
        return {"ok": True, "commit_hash": "deadbeef"}

    monkeypatch.setattr(gitint, "is_repo", lambda root: True)
    monkeypatch.setattr(gitint, "commit_all", fake_commit)
    assert repl._dispatch_slash("/commit fix: typo") is True
    assert captured.get("msg") == "fix: typo"


def test_commit_not_a_repo(repl, monkeypatch, capsys):
    monkeypatch.setattr(gitint, "is_repo", lambda root: False)
    assert repl._dispatch_slash("/commit") is True
    assert "not a git repository" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# JSON one-shot output
# --------------------------------------------------------------------------- #

def test_json_output_emits_parseable_object(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(
        provider="mock", model="testmodel", output_format="json",
        mcp_enabled=False, memory_enabled=False,
    )
    provider = MockProvider(scenario="plain")
    result = r.run_once(provider, cfg, "hi there", auto_confirm=True)
    out = capsys.readouterr().out.strip()
    # Exactly one parseable JSON object on stdout (decorative UI suppressed).
    obj = json.loads(out)
    assert obj["ok"] is True
    assert obj["model"] == "testmodel"
    assert obj["answer"] == result
    assert isinstance(obj["answer"], str) and obj["answer"]


def test_text_output_is_not_json(monkeypatch, tmp_path, capsys):
    # Regression guard: default text mode must NOT emit a JSON object line.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", model="testmodel", mcp_enabled=False, memory_enabled=False)
    r.run_once(MockProvider(scenario="plain"), cfg, "hi there", auto_confirm=True)
    out = capsys.readouterr().out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip())


# --------------------------------------------------------------------------- #
# /help lists the new commands
# --------------------------------------------------------------------------- #

def test_help_lists_new_commands(repl, capsys):
    assert repl._dispatch_slash("/help") is True
    out = capsys.readouterr().out
    for cmd in ("/undo", "/diff", "/commit", "/init", "/copy", "/mode",
                "/branch", "/fork", "/doctor", "/commands"):
        assert cmd in out, f"{cmd} missing from /help"


def test_help_notes_shell_not_sandboxed(repl, capsys):
    # Security note: !<cmd> runs in the user's own shell, NOT sandboxed.
    assert repl._dispatch_slash("/help") is True
    out = capsys.readouterr().out
    assert "!<cmd>" in out
    assert "NOT sandboxed" in out


# --------------------------------------------------------------------------- #
# /undo is session-scoped
# --------------------------------------------------------------------------- #

def test_undo_is_session_scoped(repl, tmp_path, monkeypatch):
    # A checkpoint written under a DIFFERENT session token must NOT be undone by
    # this REPL's /undo (which uses its own per-launch token).
    monkeypatch.chdir(tmp_path)
    cwd = str(tmp_path)
    f = tmp_path / "note.txt"
    f.write_text("original\n", encoding="utf-8")
    r.checkpoint.snapshot(
        [str(f)], root=cwd, label="write_file", session="other-session"
    )
    f.write_text("changed\n", encoding="utf-8")

    # This session (fresh token) sees nothing to undo -> file untouched.
    res = r.checkpoint.undo(cwd, session=repl._ckpt_session)
    assert res.get("undone") is False
    assert f.read_text(encoding="utf-8") == "changed\n"

    # The other session CAN undo its own checkpoint.
    res2 = r.checkpoint.undo(cwd, session="other-session")
    assert res2.get("undone") is True
    assert f.read_text(encoding="utf-8") == "original\n"


def test_undo_fresh_session_reports_nothing(repl, tmp_path, monkeypatch, capsys):
    # A foreign-session checkpoint exists, but /undo (this launch's token) reports
    # "nothing to undo" through the REPL path.
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "x.py"
    f.write_text("a = 1\n", encoding="utf-8")
    r.checkpoint.snapshot(
        [str(f)], root=str(tmp_path), label="write_file", session="stale-session"
    )
    assert repl._dispatch_slash("/undo") is True
    assert "Nothing to undo" in capsys.readouterr().out
    # The stale checkpoint's file was never touched.
    assert f.read_text(encoding="utf-8") == "a = 1\n"


# --------------------------------------------------------------------------- #
# JSON one-shot robustness (exactly ONE object on success AND on error)
# --------------------------------------------------------------------------- #

def test_json_output_emits_one_object_on_error(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())

    class _BoomAgent:
        memory = None
        messages: list = []

        def run(self, prompt, images=None):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(r, "_build_orchestrator", lambda *a, **k: _BoomAgent())
    cfg = Config(
        provider="mock", model="m", output_format="json",
        mcp_enabled=False, memory_enabled=False,
    )
    result = r.run_once(MockProvider(), cfg, "hi", auto_confirm=True)
    out = capsys.readouterr().out
    # Exactly ONE parseable object, even though the run raised.
    obj = json.loads(out.strip())
    assert obj["ok"] is False
    assert "RuntimeError" in obj["error"]
    assert "provider exploded" in obj["error"]
    assert result == ""


# --------------------------------------------------------------------------- #
# Two-stage Ctrl-C (SIGINT): 1st = cooperative cancel, 2nd = re-raise
# --------------------------------------------------------------------------- #

class _SignalingAgent:
    """Fake agent whose run() raises SIGINT twice to exercise the two-stage handler."""

    def __init__(self, cancel_event):
        self.cancel_event = cancel_event
        self.messages: list = [{"role": "system", "content": "x"}]

    def run(self, text, images=None):
        # 1st SIGINT: the installed handler should just SET the cooperative event.
        signal.raise_signal(signal.SIGINT)
        for _ in range(100_000):
            if self.cancel_event.is_set():
                break
        assert self.cancel_event.is_set(), "1st SIGINT did not set cancel_event"
        # 2nd SIGINT in the SAME turn: the handler must re-raise KeyboardInterrupt,
        # which unwinds through this loop.
        signal.raise_signal(signal.SIGINT)
        for _ in range(50_000_000):
            pass
        return "should not reach"


def test_two_stage_sigint_reraises(repl, monkeypatch, capsys):
    # Force the TTY/main-thread gate so _submit installs the SIGINT handler.
    monkeypatch.setattr(r, "_stdout_is_tty", lambda: True)
    fake = _SignalingAgent(repl._cancel_event)
    repl.agent = fake
    repl._submit("do work")
    out = capsys.readouterr().out
    assert "[interrupted]" in out  # 2nd SIGINT force-broke the wedged run
    assert repl._cancel_event.is_set()  # 1st SIGINT set the cooperative flag


# --------------------------------------------------------------------------- #
# _build_orchestrator propagates permission_mode + cancel_event to sub-agents
# --------------------------------------------------------------------------- #

def test_build_orchestrator_propagates_to_spawn_tool(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    captured: dict = {}
    real_make = r.make_spawn_agent_tool

    def spy(*a, **k):
        captured.update(k)
        return real_make(*a, **k)

    monkeypatch.setattr(r, "make_spawn_agent_tool", spy)
    cancel = threading.Event()
    cfg = Config(
        provider="mock", model="m", permission_mode="plan",
        mcp_enabled=False, memory_enabled=False,
    )
    console = r._make_console(cfg.theme)
    r._build_orchestrator(
        MockProvider(), cfg, console, auto_confirm=True,
        cancel_event=cancel, checkpoint_session="tok123",
    )
    assert captured.get("permission_mode") == "plan"
    assert captured.get("cancel_event") is cancel
