"""The orchestrator/sub-agent prompts inject the current project + cwd so the
model knows which project it is in — critical for kyp-mem, which does NOT
auto-detect the project and otherwise gets the wrong/recent one."""

from __future__ import annotations

from llmcli.prompts import orchestrator_prompt, project_name, role_prompt


def test_tool_use_mechanics_directive_present():
    # PROV-1: tool-call mechanics lines must appear so small local models
    # don't emit tool calls as prose or invent tool output.
    p = orchestrator_prompt(has_memory_tool=False)
    assert "Emit a tool call (not prose) to invoke a tool." in p
    assert "Never invent or paste tool output you did not receive." in p


def test_project_name_is_cwd_basename():
    assert project_name("/Users/x/Projects/apps/llm-cli") == "llm-cli"
    assert project_name("/Users/x/Projects/apps/llm-cli/") == "llm-cli"
    assert project_name("/tmp/myproj") == "myproj"


def test_orchestrator_prompt_injects_cwd_and_project():
    p = orchestrator_prompt(has_memory_tool=False, workspace="/Users/x/apps/llm-cli")
    assert "/Users/x/apps/llm-cli" in p          # the working directory
    assert "Project name: llm-cli" in p          # the derived project name
    assert "# Environment" in p


def test_orchestrator_prompt_without_workspace_has_no_env_block():
    # Back-compat: no workspace => no Environment block (old behaviour).
    p = orchestrator_prompt(has_memory_tool=False)
    assert "# Environment" not in p


def test_memory_project_rule_only_with_memory_tool_and_workspace():
    # memory tool + workspace => the explicit kyp-mem PROJECT SCOPE rule, naming
    # the exact project and the kyp_* calls to pass it to.
    p = orchestrator_prompt(has_memory_tool=True, workspace="/a/b/llm-cli")
    assert "PROJECT SCOPE" in p
    assert 'kyp_session_create(project="llm-cli"' in p
    assert 'kyp_project_context(project="llm-cli"' in p
    assert 'write notes under the "llm-cli/..." path' in p
    # no memory tool => no kyp-mem scope rule even with a workspace.
    p2 = orchestrator_prompt(has_memory_tool=False, workspace="/a/b/llm-cli")
    assert "PROJECT SCOPE" not in p2


def test_memory_project_rule_sits_under_memory_header():
    # The PROJECT SCOPE rule must be in the SAME block as "# Memory notes"
    # (no blank line separating it into a dangling bullet).
    p = orchestrator_prompt(has_memory_tool=True, workspace="/a/b/llm-cli")
    g = p.index("GROUND FIRST")
    s = p.index("PROJECT SCOPE")
    assert g < s
    assert "\n\n" not in p[g:s]  # contiguous block, not a detached bullet


def test_role_prompt_injects_workspace_for_subagents():
    coder = role_prompt("coder", has_memory_tool=True, workspace="/a/b/llm-cli")
    assert "Project name: llm-cli" in coder
    assert "PROJECT SCOPE" in coder  # writing role + memory tool
    explorer = role_prompt("explorer", workspace="/a/b/llm-cli")
    assert "Project name: llm-cli" in explorer
    assert "PROJECT SCOPE" not in explorer  # read-only role: no memory write rule


def test_build_orchestrator_injects_live_cwd(tmp_path, monkeypatch):
    from llmcli.config import Config
    from llmcli.repl import _build_orchestrator, _make_console
    from llmcli.providers import MockProvider

    proj = tmp_path / "my-cool-project"
    proj.mkdir()
    monkeypatch.chdir(proj)
    cfg = Config(provider="mock", theme="amber")
    agent = _build_orchestrator(
        MockProvider(), cfg, _make_console("amber"), auto_confirm=True
    )
    system = agent.messages[0]["content"]
    assert "my-cool-project" in system
    assert str(proj) in system
