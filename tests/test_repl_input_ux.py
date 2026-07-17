"""Final input-layer UX tests (completer, ! passthrough, @-mention expansion,
cooperative SIGINT, context gauge, live macro routing).

All offline: MockProvider / recording stubs, patched load_config / save_config /
MCPManager, HOME + cwd redirected to a tmp dir so no real disk/network/subprocess
state is touched. The Repl is constructed exactly like tests/test_repl_slash.py.
"""

from __future__ import annotations

import signal

import pytest
from prompt_toolkit.document import Document

import llmcli.repl as r
from llmcli.config import Config
from llmcli.providers import MockProvider


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


def _make_repl(monkeypatch, tmp_path):
    """Construct an offline Repl with HOME + cwd isolated to ``tmp_path``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)
    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", private=True,
                 base_url="http://127.0.0.1:1234/v1", model="m")
    repl = r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)
    monkeypatch.setattr(repl, "_save_session", lambda *a, **k: None)
    return repl


# ---------------------------------------------------------------- completer

def _texts(completer, line, cursor=None):
    doc = Document(line, len(line) if cursor is None else cursor)
    return [c.text for c in completer.get_completions(doc, None)]


def test_completer_offers_slash_commands(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    completer = repl._make_input_completer()
    texts = _texts(completer, "/mod")
    assert "/model" in texts and "/models" in texts
    # Unrelated commands are not offered for the "/mod" prefix.
    assert "/help" not in texts


def test_completer_includes_macros(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    cmds = tmp_path / ".llmcli" / "commands"
    cmds.mkdir(parents=True)
    (cmds / "deploy.md").write_text("do a deploy $ARGUMENTS", encoding="utf-8")
    completer = repl._make_input_completer()
    assert "/deploy" in _texts(completer, "/dep")


def test_completer_fuzzy_files(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    # Seed the cache directly so the test does not depend on a repo walk.
    repl._completion_file_cache = [
        "src/app.py", "src/apparatus.py", "tests/test_x.py", "README.md",
    ]
    completer = repl._make_input_completer()
    texts = _texts(completer, "look at @src/ap")
    assert "src/app.py" in texts
    assert "README.md" not in texts  # fuzzy-filtered out


def test_completer_noop_and_never_raises(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    completer = repl._make_input_completer()
    # Plain text (no slash, no @) -> no completions.
    assert _texts(completer, "hello world") == []
    # A slash line with a space (past the command token) -> no command completion.
    assert _texts(completer, "/model gpt") == []


def test_completion_files_cached(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    calls = {"n": 0}

    def _fake_project_files(root, **k):
        calls["n"] += 1
        return ["a.py"]

    monkeypatch.setattr(r.mentions, "project_files", _fake_project_files)
    assert repl._completion_files() == ["a.py"]
    assert repl._completion_files() == ["a.py"]
    assert calls["n"] == 1  # walked once, then cached
    repl._refresh_completion_files()
    repl._completion_files()
    assert calls["n"] == 2


# --------------------------------------------------------- ! shell passthrough

def test_shell_passthrough_runs_shell_not_model(monkeypatch, tmp_path, capsys):
    repl = _make_repl(monkeypatch, tmp_path)
    called = {"submit": False}
    monkeypatch.setattr(repl, "_submit", lambda *a, **k: called.__setitem__("submit", True))
    repl._submit_or_stage("!echo hi")
    assert called["submit"] is False  # model was NOT invoked
    assert "hi" in capsys.readouterr().out


def test_bare_bang_is_not_passthrough(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(repl, "_submit", lambda line, **k: seen.__setitem__("line", line))
    repl._submit_or_stage("!")
    assert seen.get("line") == "!"  # a bare "!" falls through unchanged


# ------------------------------------------------------- @-mention expansion

class _RecAgent:
    def __init__(self):
        self.received = None
        self.messages = [{"role": "system", "content": "s"}]

    def run(self, line, images=None):
        self.received = line

    def render_details(self, console):
        pass


def test_mention_expansion_prepends_context(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    (tmp_path / "note.txt").write_text("SECRET_MARKER_XYZ", encoding="utf-8")
    agent = _RecAgent()
    repl.agent = agent
    repl._submit("please explain @note.txt")
    assert agent.received is not None
    # The model sees the attached file content prepended, then the original line.
    assert "# Attached context" in agent.received
    assert "SECRET_MARKER_XYZ" in agent.received
    assert agent.received.endswith("please explain @note.txt")


def test_no_mention_is_byte_identical(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    agent = _RecAgent()
    repl.agent = agent
    repl._submit("just a normal question")
    assert agent.received == "just a normal question"


# --------------------------------------------------- cooperative SIGINT

def test_sigint_handler_sets_cancel_and_restores(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    monkeypatch.setattr(r, "_stdout_is_tty", lambda: True)  # force handler install
    original = signal.getsignal(signal.SIGINT)
    captured = {}

    class _CtrlCAgent:
        messages = [{"role": "system", "content": "s"}]

        def run(self, line, images=None):
            # During the turn a handler is installed; invoking it must SET the
            # cooperative cancel event (and must NOT raise).
            assert not repl._cancel_event.is_set()
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            captured["set"] = repl._cancel_event.is_set()

        def render_details(self, console):
            pass

    repl.agent = _CtrlCAgent()
    repl._submit("hi")
    assert captured["set"] is True
    # The previous SIGINT handler is restored after the turn.
    assert signal.getsignal(signal.SIGINT) is original


def test_no_sigint_handler_when_not_tty(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    monkeypatch.setattr(r, "_stdout_is_tty", lambda: False)
    original = signal.getsignal(signal.SIGINT)

    class _A:
        messages = [{"role": "system", "content": "s"}]

        def run(self, line, images=None):
            assert signal.getsignal(signal.SIGINT) is original  # not swapped

        def render_details(self, console):
            pass

    repl.agent = _A()
    repl._submit("hi")
    assert signal.getsignal(signal.SIGINT) is original


# ----------------------------------------------------------- context gauge

def test_context_gauge_computed_correctly(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    repl.agent.messages = [
        {"role": "system", "content": "x" * 400},
        {"role": "user", "content": "y" * 1200},
    ]
    used = r.Agent._estimate_tokens(repl.agent.messages)
    ceiling = r._effective_soft_limit(repl.provider, repl.config)
    pct = int(round(100 * used / ceiling))
    expected = f"context: {pct}% (~{r._fmt_tok(used)} / {r._fmt_tok(ceiling)} tok)"
    assert repl._context_gauge_line() == expected


def test_fmt_tok():
    assert r._fmt_tok(4400) == "4.4k"
    assert r._fmt_tok(12000) == "12k"
    assert r._fmt_tok(999) == "999"


# --------------------------------------------------- pinned bottom status bar

def _bar_text(repl):
    """Plain text of the cached status bar (strips FormattedText styling)."""
    from prompt_toolkit.formatted_text import fragment_list_to_text
    cache = repl._status_cache
    if isinstance(cache, str):
        return cache
    return fragment_list_to_text(cache)


def test_status_bar_resting_form(monkeypatch, tmp_path):
    """Before any turn (last_turn_stats None) the bar shows model + ctx% and NO
    tok/s/time; the tmp cwd is not a repo so the git segment is omitted."""
    repl = _make_repl(monkeypatch, tmp_path)
    repl.agent.messages = [
        {"role": "system", "content": "x" * 400},
        {"role": "user", "content": "y" * 1200},
    ]
    repl._refresh_status_bar()
    text = _bar_text(repl)
    assert "m" in text  # the short model name
    assert "ctx" in text and "%" in text
    assert "tok/s" not in text  # no turn yet -> no speed/time
    assert " · " in text  # segments joined with " · "


def test_status_bar_strips_org_prefix(monkeypatch, tmp_path):
    """The model segment drops any leading "org/" prefix (qwen/x -> x)."""
    repl = _make_repl(monkeypatch, tmp_path)
    repl.provider.model = "qwen/qwen3.6-35b-a3b"
    repl._refresh_status_bar()
    text = _bar_text(repl)
    assert "qwen3.6-35b-a3b" in text
    assert "qwen/" not in text


def test_status_bar_includes_stats_after_turn(monkeypatch, tmp_path):
    """Once last_turn_stats is populated the bar appends tok/s + time."""
    repl = _make_repl(monkeypatch, tmp_path)
    repl.agent.last_turn_stats = {
        "model": "m", "elapsed": 0.42, "toks_per_sec": 223.4, "output_tokens": 50,
    }
    repl._refresh_status_bar()
    text = _bar_text(repl)
    assert "223 tok/s" in text
    assert "0.42s" in text


def test_status_bar_git_segment(monkeypatch, tmp_path):
    """When cwd is a repo the bar shows branch + '*' when dirty (git helpers are
    stubbed so no subprocess runs)."""
    import llmcli.gitint as gitint
    repl = _make_repl(monkeypatch, tmp_path)
    monkeypatch.setattr(gitint, "is_repo", lambda root: True)
    monkeypatch.setattr(gitint, "current_branch", lambda root: "feature")
    monkeypatch.setattr(gitint, "is_dirty", lambda root: True)
    repl._refresh_status_bar()
    assert "feature*" in _bar_text(repl)


def test_status_bar_callable_reads_cache_without_git(monkeypatch, tmp_path):
    """_status_bar (the bottom_toolbar callable, hit per keystroke) must ONLY read
    the cache — never touch git or recompute — so it stays cheap."""
    import llmcli.gitint as gitint
    repl = _make_repl(monkeypatch, tmp_path)
    sentinel = "SENTINEL_CACHE"
    repl._status_cache = sentinel

    def _boom(*a, **k):
        raise AssertionError("git must not be called from the bottom_toolbar")

    monkeypatch.setattr(gitint, "is_repo", _boom)
    monkeypatch.setattr(gitint, "current_branch", _boom)
    monkeypatch.setattr(gitint, "is_dirty", _boom)
    assert repl._status_bar() is sentinel


def test_status_bar_never_raises_on_bad_agent(monkeypatch, tmp_path):
    """Every segment is guarded: a stub agent without messages/last_turn_stats
    must not make _refresh_status_bar raise."""
    repl = _make_repl(monkeypatch, tmp_path)

    class _Bare:
        pass

    repl.agent = _Bare()
    repl._refresh_status_bar()  # must not raise
    # model still resolves from the provider even with a bare agent.
    assert "tok/s" not in _bar_text(repl)


def test_interactive_agent_suppresses_footer(monkeypatch, tmp_path):
    """The REPL orchestrator (_new_agent) suppresses its own footer line — the
    pinned bar owns those stats — while the one-shot default keeps it False."""
    repl = _make_repl(monkeypatch, tmp_path)
    assert repl.agent.suppress_footer is True
    assert repl._new_agent().suppress_footer is True
    # One-shot / -p path builds via _build_orchestrator without the flag -> False.
    console = r._make_console(repl.config.theme)
    one_shot = r._build_orchestrator(
        repl.provider, repl.config, console, auto_confirm=True
    )
    assert one_shot.suppress_footer is False


def test_banner_shows_short_model_no_provider_prefix(monkeypatch, tmp_path, capsys):
    """Banner line 1 is tidied to '◆ <short-model>   ● ready': no '<provider> ·'
    prefix and any 'org/' prefix on the model is stripped."""
    repl = _make_repl(monkeypatch, tmp_path)
    repl.config.model = "qwen/qwen3.6-35b-a3b"
    repl.config.provider = "mock"
    repl._print_banner()
    out = capsys.readouterr().out
    assert "qwen3.6-35b-a3b" in out
    assert "qwen/" not in out          # org prefix stripped
    assert "mock ·" not in out         # provider prefix dropped
    assert "ready" in out


# ---------------------------------------------------- live macro routing

class _FakeSession:
    def __init__(self, lines):
        self._lines = iter(lines)

    def prompt(self, text):
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError


def test_macro_slash_routes_to_dispatch(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    cmds = tmp_path / ".llmcli" / "commands"
    cmds.mkdir(parents=True)
    (cmds / "greet.md").write_text("say hello $ARGUMENTS", encoding="utf-8")

    dispatched = []
    real_dispatch = repl._dispatch_slash

    def _spy(line):
        dispatched.append(line)
        return real_dispatch(line)

    monkeypatch.setattr(repl, "_dispatch_slash", _spy)
    # The macro expands and is submitted as a model turn; capture that text.
    submitted = []
    monkeypatch.setattr(repl, "_submit", lambda line, **k: submitted.append(line))

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession",
                        lambda *a, **k: _FakeSession(["/greet world"]))
    repl.run()

    assert dispatched == ["/greet world"]  # routed to dispatch, not the model path
    assert submitted == ["say hello world"]  # macro expanded + $ARGUMENTS filled


def test_non_command_slash_not_hijacked(monkeypatch, tmp_path):
    repl = _make_repl(monkeypatch, tmp_path)
    dispatched = []
    monkeypatch.setattr(repl, "_dispatch_slash",
                        lambda line: dispatched.append(line) or True)
    staged = []
    monkeypatch.setattr(repl, "_submit_or_stage", lambda line: staged.append(line))

    import prompt_toolkit
    monkeypatch.setattr(prompt_toolkit, "PromptSession",
                        lambda *a, **k: _FakeSession(["/build the app"]))
    repl.run()

    assert dispatched == []  # not one of ours + not a macro -> not hijacked
    assert staged == ["/build the app"]  # sent to the model path unchanged
