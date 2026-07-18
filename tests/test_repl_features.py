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


def test_diff_preview_themed_render_is_ansi_free_when_piped(tmp_path, monkeypatch, capsys):
    """Step 3: the diff preview now renders on the THEMED console with semantic
    +/-/context tokens, but a piped/non-tty run (capsys) must stay byte-clean —
    the text is present, with ZERO ANSI escapes, for a truecolor theme."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n", encoding="utf-8")
    cfg = Config(diff_preview=True, theme="neon")  # a truecolor theme
    confirm = r.make_ptk_confirm(_FakeSession(), cfg)
    tool = get_tool("edit_file")
    confirm(tool, {"path": str(f), "old": "x = 1", "new": "x = 42"})
    out = capsys.readouterr().out
    assert "-x = 1" in out and "+x = 42" in out  # the diff content is shown
    assert "\x1b[" not in out  # …with no ANSI (byte-clean piped output)


def test_confirm_header_and_diff_ansi_free_when_piped(tmp_path, monkeypatch, capsys):
    """Step 6: the confirm now prints a themed "⏺ <tool> <path>" header ABOVE the
    diff so the change belongs to the action. On a piped/non-tty run (capsys) the
    whole preview must stay byte-clean: header + path + diff present, ZERO ANSI
    escapes, even for a truecolor theme."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "sample.py"
    f.write_text("x = 1\n", encoding="utf-8")
    cfg = Config(diff_preview=True, theme="neon")  # a truecolor theme
    confirm = r.make_ptk_confirm(_FakeSession(), cfg)
    tool = get_tool("edit_file")
    confirm(tool, {"path": str(f), "old": "x = 1", "new": "x = 42"})
    out = capsys.readouterr().out
    assert "⏺ edit_file" in out          # themed action header above the diff
    assert "sample.py" in out             # header carries the path (basename)
    assert "-x = 1" in out and "+x = 42" in out
    # The header must sit ABOVE the diff, not below it.
    assert out.index("⏺ edit_file") < out.index("-x = 1")
    assert "\x1b[" not in out             # byte-clean piped output


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
    # The prompt is now a themed FormattedText (warning ⚠ glyph + accent action
    # label + default-fg "[y/N]"); flatten it to plain text to assert structure.
    from prompt_toolkit.formatted_text import fragment_list_to_text
    assert sess.prompts
    plain = fragment_list_to_text(sess.prompts[0])
    assert plain.startswith("\n")
    assert "Run write_file" in plain and "[y/N]" in plain


def test_confirm_restores_session_placeholder():
    # prompt() mutates session.placeholder; the confirm must restore it so the
    # main input keeps its ghost after the first tool confirmation.
    ghost = "Ask anything · / commands · @ files"
    sess = _FakeSession(placeholder=ghost)
    confirm = r.make_ptk_confirm(sess, Config())
    tool = get_tool("write_file")
    confirm(tool, {"path": "hello.py", "content": "print('x')\n"})
    assert sess.placeholder == ghost


def test_ansi_theme_warning_is_16_colour_not_truecolor_yellow():
    # The ansi ThemeSpec's `warning` must be the explicit 16-colour ptk name
    # "ansiyellow", not the bare "yellow" rich style — prompt_toolkit resolves
    # bare "yellow" to truecolor #ffff00 when color_depth is forced to 24-bit,
    # which would leak truecolor into the otherwise strictly-16-colour ansi
    # theme's confirm ⚠ glyph.
    from llmcode.repl import _SPECS, _resolve_theme

    assert _SPECS[_resolve_theme("ansi")].warning == "ansiyellow"


def test_ansi_theme_confirm_prompt_uses_ansiyellow_warning_style():
    # The confirm y/N prompt colours its leading "⚠" glyph with pal.warning;
    # for the ansi theme that must be the ptk 16-colour name, never a hex
    # truecolor spelling.
    from prompt_toolkit.formatted_text import fragment_list_to_text

    sess = _FakeSession()
    confirm = r.make_ptk_confirm(sess, Config(theme="ansi"))
    tool = get_tool("write_file")
    confirm(tool, {"path": "hello.py", "content": "print('x')\n"})
    assert sess.prompts
    fragments = sess.prompts[0]
    warn_fragment = next(f for f in fragments if "⚠" in f[1] or "!" in f[1])
    assert warn_fragment[0] == "ansiyellow"
    assert "#" not in warn_fragment[0]  # never a truecolor hex spelling
    assert fragment_list_to_text(fragments)  # sanity: still renders text


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


def test_help_grouped_sections_and_network_subpage(repl, capsys):
    """Step 6: /help renders grouped sections; the long NETWORK/SSRF prose moves
    behind `/help network`. Default /help stays scannable + ANSI-free when piped."""
    assert repl._dispatch_slash("/help") is True
    out = capsys.readouterr().out
    # Section titles present, default view no longer carries the SSRF prose.
    for title in ("Core", "Model", "Context", "Git & files", "Tuning", "Themes"):
        assert title in out, f"section {title} missing from /help"
    assert "SSRF" not in out
    assert "loopback-pinned" not in out
    assert "Run /help network" in out       # pointer to the moved section
    assert "\x1b[" not in out               # byte-clean piped output

    # `/help network` surfaces the moved security prose.
    assert repl._dispatch_slash("/help network") is True
    net = capsys.readouterr().out
    assert "SSRF" in net and "loopback-pinned" in net
    assert "\x1b[" not in net


def test_banner_degrades_glyphs_under_ascii_console(repl):
    """Step 6: the banner ASCII-degrades its ◆/● glyphs on a console whose output
    encoding can't represent them (LANG=C), instead of emitting mojibake — the
    banner is the FIRST thing users see and previously had no such guard."""
    import io
    from rich.console import Console

    class _AsciiFile(io.StringIO):
        encoding = "ascii"

    buf = _AsciiFile()
    repl.console = Console(file=buf, force_terminal=False, width=80,
                           markup=False, highlight=False)
    # Precondition: this console genuinely cannot encode the banner glyphs.
    assert r._enc_can(repl.console, "◆") is False
    repl.config.model = "m"
    repl._print_banner()
    out = buf.getvalue()
    assert "◆" not in out and "●" not in out   # un-encodable glyphs degraded away
    assert "*" in out                          # ASCII diamond fallback used
    assert "ready" in out


# --------------------------------------------------------------------------- #
# Startup block-letter wordmark (big "llmc-code") + its gating
# --------------------------------------------------------------------------- #

# A row of the embedded ANSI-Shadow art — a substring no other banner text emits,
# so its presence/absence cleanly proves whether the big wordmark rendered. The
# gradient hero colours each cell individually, so the raw ANSI stream splits the
# row with escapes; strip ANSI first to check the underlying glyphs.
_WORDMARK_MARK = "╚══════╝╚══════╝"

_ANSI_RE = __import__("re").compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def test_banner_wordmark_on_wide_truecolor_tty(repl):
    """A wide, UTF-8-capable real terminal opens with the "Local Reactor" hero:
    the diagonal-gradient ``llmc-code`` wordmark + value-prop tagline + the
    reactor ribbon (``<model> · ◆ core ready · honest lock badge``)."""
    import io
    from rich.console import Console

    buf = io.StringIO()   # UTF-8 StringIO -> _enc_can("█…") is True
    repl.console = Console(file=buf, force_terminal=True, width=100,
                           color_system="truecolor")
    assert getattr(repl.console, "is_terminal", False) is True
    assert repl.console.size.width >= r._WORDMARK_WIDTH
    repl.config.theme = "neon"
    repl.config.private = False            # network on -> honest badge is "local"
    repl.config.model = "qwen/qwen3.6-35b-a3b"
    repl._print_banner()
    out = buf.getvalue()
    flat = _strip_ansi(out)
    # The wordmark art renders (its glyphs survive once ANSI is stripped).
    assert _WORDMARK_MARK in flat
    assert "██████" in flat
    # The gradient is a per-cell RGB lerp: MANY distinct truecolor codes appear
    # (a flat wordmark would repeat a single 38;2;r;g;b run).
    codes = set(__import__("re").findall(r"38;2;\d+;\d+;\d+", out))
    assert len(codes) > 10                 # a true gradient, not a flat fill
    # Tagline + reactor ribbon (contiguous single-style runs -> survive raw too).
    assert "a coding agent that runs on your machine" in out
    assert "◆ core ready" in out
    assert "⬡ local" in out                # network on -> "local", NOT "offline"
    assert "no egress" not in out          # honesty: no offline claim when networked
    assert "qwen3.6-35b-a3b" in out
    assert "\x1b[" in out                  # a real tty is styled (ANSI present)


def test_banner_hero_badge_honest_offline_when_private(repl):
    """HONESTY: only when private mode is on does the ribbon claim ``⬡ offline ·
    no egress``; the model is always local but we never claim offline on network."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    repl.console = Console(file=buf, force_terminal=True, width=100,
                           color_system="truecolor")
    repl.config.private = True
    repl.config.model = "m"
    repl._print_banner()
    out = buf.getvalue()
    assert "⬡ offline · no egress" in out
    assert "◆ core ready" in out


def test_banner_returning_run_skips_wordmark(repl):
    """On a subsequent launch (the ~/.llmcode/seen marker is present) the wide-tty
    startup compresses to a two-line ribbon — no big wordmark, no verbose tail."""
    import io
    import os
    from rich.console import Console

    # Simulate a prior first-run by writing the marker (HOME is the tmp fixture).
    marker = os.path.expanduser("~/.llmcode/seen")
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("1")

    buf = io.StringIO()
    repl.console = Console(file=buf, force_terminal=True, width=100,
                           color_system="truecolor")
    repl.config.private = True
    repl.config.model = "qwen/qwen3.6-35b-a3b"
    repl._print_banner()
    out = buf.getvalue()
    flat = _strip_ansi(out)
    assert _WORDMARK_MARK not in flat          # big wordmark skipped
    assert "██████" not in flat
    assert "◆ llmc-code" in out                # compressed ribbon shown
    assert "qwen3.6-35b-a3b" in out
    assert "⬡ offline" in out


def test_banner_first_run_writes_marker(repl):
    """The first-run hero writes the ~/.llmcode/seen marker so the NEXT launch
    takes the compressed path."""
    import io
    import os
    from rich.console import Console

    marker = os.path.expanduser("~/.llmcode/seen")
    assert not os.path.exists(marker)          # fresh HOME -> first run
    buf = io.StringIO()
    repl.console = Console(file=buf, force_terminal=True, width=100,
                           color_system="truecolor")
    repl.config.model = "m"
    repl._print_banner()
    assert os.path.exists(marker)              # marker now recorded


def test_banner_narrow_terminal_falls_back_to_compact(repl):
    """A terminal narrower than the wordmark falls back to the compact framed
    banner — no big art (which would wrap and break) — while still showing the
    model + ready + the framed 'llmc-code' title."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    # Width below _WORDMARK_WIDTH (74): the wordmark gate must reject this.
    repl.console = Console(file=buf, force_terminal=True, width=40,
                           color_system="truecolor")
    assert repl.console.size.width < r._WORDMARK_WIDTH
    repl.config.model = "m"
    repl._print_banner()
    out = buf.getvalue()
    assert _WORDMARK_MARK not in out       # NO big wordmark on a narrow terminal
    assert "██████" not in out             # no block art at all
    assert "llmc-code" in out              # compact framed banner (its title) shown
    assert "ready" in out


def test_banner_non_terminal_no_wordmark_and_ansi_free(repl):
    """A non-terminal/piped console prints NO big wordmark and stays byte-clean
    (ANSI-free) — the piped-output guarantee scripts/tests rely on."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    # force_terminal=False + a generous width: even though it is wide enough, a
    # non-tty console must NOT get the wordmark (is_terminal gate) and must emit
    # no ANSI at all.
    repl.console = Console(file=buf, force_terminal=False, width=120)
    assert getattr(repl.console, "is_terminal", False) is False
    repl.config.model = "m"
    repl._print_banner()
    out = buf.getvalue()
    assert _WORDMARK_MARK not in out       # no big art off a real terminal
    assert "██████" not in out
    assert "\x1b[" not in out              # byte-clean, ANSI-free (piped guarantee)
    assert "ready" in out                  # compact banner content still present
    # The reactor hero's new glyphs/gradient never leak into the piped path — the
    # ⬡ lock badge lives only on the wide-tty hero/ribbon (the compact fallback is
    # unchanged and carries no ⬡).
    assert "⬡" not in out


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
