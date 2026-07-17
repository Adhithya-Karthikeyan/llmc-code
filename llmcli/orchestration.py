"""Multi-agent orchestration: the ``spawn_agent`` delegation tool.

The orchestrator can spawn fresh, role-scoped sub-agents. Each sub-agent has
its own message history, system prompt, and a restricted tool subset, runs its
own agent loop to a final answer, and returns a concise summary string to the
parent.

Roles:
  - explorer : read-only tools (read_file, glob, grep)
  - coder    : full tools (read/write/edit/bash/glob/grep)
  - reviewer : read-only tools
"""

from __future__ import annotations

import time
from typing import Callable

from .agent import Agent
from .prompts import role_prompt
from .providers import Provider
from .tools import FULL, READ_ONLY, Tool

ROLE_TOOLS: dict[str, list[str]] = {
    "explorer": READ_ONLY,
    "coder": FULL,
    "reviewer": READ_ONLY,
}

# Egress-capable tools removed from EVERY agent's tool set in --private lockdown
# mode (the model must never even see an external-egress surface). In the default
# network-on mode these tools are KEPT. Lives in one place so the filter is
# applied identically by every helper (finding #31).
_EGRESS_TOOLS = frozenset({"web_fetch"})


def _drop_egress_tools(names: list[str], private: bool) -> list[str]:
    """Return ``names`` with egress tools removed when ``private`` is True."""
    if not private:
        return list(names)
    return [n for n in names if n not in _EGRESS_TOOLS]


def _has_memory_tool(tool_names) -> bool:
    """True if any loaded tool is a memory MCP tool (mcp__*mem*)."""
    return any(
        isinstance(n, str) and n.startswith("mcp__") and "mem" in n.lower()
        for n in (tool_names or [])
    )


def _role_tools(role: str, private: bool) -> list[str]:
    """Tools granted to a sub-agent role, with web_fetch dropped when private.

    In private mode web_fetch is an external-egress surface, so the model must
    not even see it in any agent's tool set (orchestrator or sub-agent)."""
    return _drop_egress_tools(ROLE_TOOLS[role], private)


def make_spawn_agent_tool(
    provider: Provider,
    console=None,
    auto_confirm: bool = False,
    max_iterations: int = 12,
    private: bool = False,
    confirm_fn: Callable[[Tool, dict], bool] | None = None,
    code_theme: str = "monokai",
    accent: str | None = None,
    workspace: str | None = None,
    context_budget: int = 0,
    context_ceiling: int = 0,
    context_adaptive: bool = True,
    code_search_tool: Tool | None = None,
    gentle_mode: bool = False,
    gentle_spawn_gap_seconds: float = 5.0,
    is_terminal: bool = False,
    permission_mode: str = "default",
    cancel_event=None,
    auto_fix_tools: bool = False,
    auto_fix_max_attempts: int = 2,
    cooldown_enabled: bool = False,
) -> Tool:
    """Build the ``spawn_agent`` Tool bound to a provider/console/config.

    When invoked, it creates a fresh role-scoped :class:`Agent`, runs the given
    task through that agent's own loop, and returns the sub-agent's final
    summary string as the tool result.

    ``code_search_tool`` (when given) is the injected, provider-bound code_search
    Tool. It is added to EVERY sub-agent's tool set so a delegated explorer/coder/
    reviewer can semantically find code too. It is safe to share (it reads local
    files only and never spawns, so it can't recurse).
    """

    # Per-orchestrator spawn counter (mutable so it persists across spawns within
    # one orchestrator). The wait only fires BEFORE the 2nd+ spawn — never the
    # first — so a single delegation never pays the gap. The tool is rebuilt per
    # orchestrator, so no reset across orchestrators is needed.
    spawn_count = [0]

    def _spawn(args: dict) -> dict:
        role = args.get("role")
        task = args.get("task")
        if role not in ROLE_TOOLS:
            return {
                "ok": False,
                "error": f"Unknown role '{role}'. Choose: {', '.join(ROLE_TOOLS)}.",
            }
        if not isinstance(task, str) or not task.strip():
            return {"ok": False, "error": "spawn_agent requires a non-empty 'task' string."}

        granted_names = _role_tools(role, private)
        # When a code_search tool is injected, give it to the sub-agent too. Since
        # it is NOT in the global import-time REGISTRY (it is provider/workspace-
        # bound), the sub-agent needs an explicit registry that carries it — the
        # role's other tools all live in the global REGISTRY, so we merge them in.
        sub_registry = None
        if code_search_tool is not None:
            from .tools import REGISTRY
            sub_registry = {n: REGISTRY[n] for n in granted_names if n in REGISTRY}
            sub_registry[code_search_tool.name] = code_search_tool
            granted_names = granted_names + [code_search_tool.name]
        if console is not None:
            granted = ", ".join(granted_names)
            console.print(
                f"[orchestrator] spawning {role} sub-agent (tools: {granted})...",
                markup=False,
            )

        # KV-CACHE SLOT: the orchestrator pins llama.cpp slot 0 to keep its warm
        # prefix, but sub-agents reuse this SAME provider instance. A sub-agent's
        # large, distinct prompt on slot 0 would evict the orchestrator's slot-0 KV
        # prefix, forcing a full re-prefill when control returns. Give the sub-agent
        # a clone that does NOT pin slot 0 (id_slot=None) so its requests land
        # elsewhere and never collide. Guarded: providers without with_id_slot (e.g.
        # MockProvider in tests) fall back to sharing the provider unchanged.
        sub_provider = provider
        if hasattr(provider, "with_id_slot"):
            sub_provider = provider.with_id_slot(None)

        # Sub-agents are told the project/cwd too (via workspace) so a delegated
        # coder/explorer knows which project it is in for any project-scoped tool.
        sub = Agent(
            provider=sub_provider,
            system_prompt=role_prompt(role, has_memory_tool=False, workspace=workspace),
            tool_names=granted_names,
            console=console,
            auto_confirm=auto_confirm,
            max_iterations=max_iterations,
            # Forward the SAME prompt_toolkit-safe confirm_fn the orchestrator
            # uses (finding #1): without it the sub-agent fell back to builtin
            # input(), which conflicts with prompt_toolkit's live event loop and
            # broke/garbled confirmation for a spawned coder's gated tool calls.
            confirm_fn=confirm_fn,
            # Nesting marker so the sub-agent's tool lines + footer are visually
            # distinct from the orchestrator's column-0 lines.
            line_prefix="  ↳ ",
            # Keep a spawned sub-agent's Markdown code highlighting consistent
            # with the orchestrator's active theme (ansi_dark under Dark mode).
            code_theme=code_theme,
            # Accent the sub-agent's footer/glyph to match the theme. The "▌
            # Answer" gutter stays OFF for sub-agents (its non-empty line_prefix
            # disables it), so the bar never collides with the "↳" marker.
            accent=accent,
            # Don't print the sub-agent's full answer live: it is RETURNED to the
            # orchestrator, which renders it once. Without this the delegated
            # audit printed twice (sub-agent copy + orchestrator re-render). The
            # collapsed "↳ ⏺ N tools" activity line still shows.
            render_answer=False,
            # Context guard: prevent window overflow in long sub-agent runs.
            # Without these, _maybe_auto_compact is a no-op (default 0) and
            # a sub-agent running up to max_iterations can overflow the window.
            context_budget=context_budget,
            context_ceiling=context_ceiling,
            # Forward the user's adaptive-budget setting so a sub-agent honours
            # /context off too. Without this it always fell back to the Agent
            # default (True), ignoring a user who disabled adaptive budgeting.
            context_adaptive=context_adaptive,
            # Inject the code_search-carrying registry when present (None keeps the
            # old behaviour: the sub-agent falls back to the global REGISTRY).
            registry=sub_registry,
            # Inherit the orchestrator's permission mode so a sub-agent spawned
            # while read-only/plan is active ALSO blocks writes/edits/run_bash —
            # closing the "spawn a coder to escape read-only" escalation. Defaults
            # to "default" so existing callers/tests are unchanged.
            permission_mode=permission_mode,
            # Propagate the interrupt so a long delegated run stops promptly when
            # the user cancels, instead of ignoring it until the sub-agent returns.
            cancel_event=cancel_event,
            # A long delegated coder run should self-heal failed tool calls and
            # pace the GPU too. Defaults keep existing callers/tests unchanged.
            auto_fix_tools=auto_fix_tools,
            auto_fix_max_attempts=auto_fix_max_attempts,
            cooldown_enabled=cooldown_enabled,
        )
        try:
            # Gentle cool-down: when gentle is on AND this is a real terminal AND
            # the spawn gap is positive AND this is NOT the first spawn, wait the
            # gap before running the sub-agent. This spaces out multi-spawn
            # orchestrator bursts to lower AVERAGE GPU load/heat (it does NOT cap
            # GPU %). The first spawn never waits, so a single delegation is free.
            # Parallel spawns each independently check the counter — that's fine.
            if (
                gentle_mode
                and is_terminal
                and gentle_spawn_gap_seconds > 0
                and spawn_count[0] >= 1
            ):
                if console is not None:
                    console.print(
                        "(gentle: cooling down between sub-agents…)",
                        style="dim",
                    )
                time.sleep(gentle_spawn_gap_seconds)
            spawn_count[0] += 1
            summary = sub.run(task)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Sub-agent failed: {type(exc).__name__}: {exc}"}
        if console is not None:
            console.print(f"  ↳ {role} done", markup=False, style="dim")
        return {"ok": True, "result": summary}

    return Tool(
        name="spawn_agent",
        description=(
            "Delegate a self-contained task to a fresh role-scoped sub-agent. "
            "Roles: 'explorer' (read-only investigation), 'coder' (full tools, "
            "implements + validates a scoped change), 'reviewer' (read-only "
            "critique). The sub-agent runs its own loop and returns a summary. "
            "It cannot see your conversation - include all needed context in 'task'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["explorer", "coder", "reviewer"],
                    "description": "The sub-agent role to spawn.",
                },
                "task": {
                    "type": "string",
                    "description": "Crisp, self-contained task description for the sub-agent.",
                },
            },
            "required": ["role", "task"],
        },
        fn=_spawn,
        # Spawning a sub-agent fans work out to a role-scoped agent that may
        # itself run side-effecting tools (the 'coder' role gets full tools).
        # Gate the spawn so the user sees and approves the delegation, instead of
        # only seeing each individual tool prompt buried inside the sub-agent.
        requires_confirmation=True,
    )


def _orchestrator_base_tools(private: bool) -> list[str]:
    """Full orchestrator tool list, minus web_fetch when private (no egress)."""
    return _drop_egress_tools(list(FULL), private)


def orchestrator_registry(
    spawn_tool: Tool, mcp_tools: dict[str, Tool] | None = None, private: bool = False,
    extra_tools: list[Tool] | None = None,
) -> dict[str, Tool]:
    """Registry for an orchestrator run: full tools + spawn_agent + MCP tools.

    ``mcp_tools`` is a per-run (session) registry of MCP-backed Tools. It is
    MERGED here so the orchestrator can call them, WITHOUT mutating the global
    import-time ``tools.REGISTRY``. When ``private`` is True, web_fetch is
    excluded so the model can never reach an external URL.

    ``extra_tools`` are additional INJECTED, provider-bound Tools (e.g.
    code_search) that are not in the global REGISTRY; they are merged in so the
    orchestrator can call them.
    """
    from .tools import REGISTRY

    reg = {name: REGISTRY[name] for name in _orchestrator_base_tools(private) if name in REGISTRY}
    reg[spawn_tool.name] = spawn_tool
    if mcp_tools:
        reg.update(mcp_tools)
    for t in (extra_tools or []):
        reg[t.name] = t
    return reg


def orchestrator_tool_names(
    spawn_tool: Tool, mcp_tools: dict[str, Tool] | None = None, private: bool = False,
    extra_tools: list[Tool] | None = None,
) -> list[str]:
    names = _orchestrator_base_tools(private) + [spawn_tool.name]
    if mcp_tools:
        names.extend(mcp_tools)
    names.extend(t.name for t in (extra_tools or []))
    return names
