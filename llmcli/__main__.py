"""CLI entry point.

`python -m llmcli` and the `llmc` console script both call :func:`main`.

Flags:
  --provider {local,mock}   backend (default 'local')
  --model <name>            model for the active provider
  --base-url <url>          LM Studio base URL
  -p/--prompt "<text>"      run one prompt and exit (one-shot)
  -c/--continue             resume this project's saved session at startup
  --auto-pilot              auto-confirm dangerous tools (non-interactive/mock);
                            --yes/-y are back-compat aliases
  --max-iterations <n>      cap tool-use iterations per turn
  --mode <name>             permission mode {default,auto-edit,read-only,plan,
                            full-auto}; session-only unless --save
  --output-format {text,json}  output rendering; session-only unless --save
  --no-checkpoints          disable pre-edit file snapshots for this session
  --git-autocommit          auto-commit each change for this session
  --theme {amber,auto,ansi,orange} color theme; 'amber' (default) = warm polished
                            look; 'ansi' = Dark mode (ANSI colors only);
                            'orange' = orange-on-black (orange inline code, no box)
  --mcp {on,off}            Enable/disable MCP servers (off = no MCP tools,
                            smaller/faster prompts). Session-only unless --save
  --save                    persist the given flags as the new default

CLI flags are SESSION-ONLY overrides and do NOT change your saved defaults in
~/.llm-cli/config.json unless you pass --save. This prevents a throwaway run
like `llmc --provider mock -p ...` from silently clobbering your default
provider/model. (Inside the REPL, /provider and /model still persist, since
those are explicit deliberate changes.)
"""

from __future__ import annotations

import argparse
import sys

from .config import (
    EFFORT_LEVELS,
    OUTPUT_FORMATS,
    PERMISSION_MODES,
    PROVIDER_LOCAL,
    PROVIDERS,
    THEMES,
    is_loopback_url,
    load_config,
    save_config,
)
from .repl import Repl, build_provider, run_once
from .tools import set_private


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="llmc",
        description="A lightweight, personal agentic CLI for LOCAL LLMs (LM Studio).",
    )
    parser.add_argument(
        "--provider", choices=list(PROVIDERS), default=None,
        help="Backend provider (default: from config, else 'local').",
    )
    parser.add_argument("--model", default=None, help="Model for the active provider.")
    parser.add_argument("--base-url", default=None, help="LM Studio base URL.")
    parser.add_argument("-p", "--prompt", default=None, help="Run one prompt and exit.")
    # dest="continue_" because "continue" is a Python keyword (not a valid
    # identifier for args.continue). Resumes this project's saved session at
    # startup: the REPL loads it before the first turn; a one-shot (-p) prepends
    # the saved history. Session memory is LOCAL-ONLY (~/.llm-cli/sessions).
    parser.add_argument(
        "-c", "--continue", dest="continue_", action="store_true",
        help="Resume this project's saved session (from ~/.llm-cli/sessions) "
             "at startup. Without it, a fresh launch starts clean (use /resume "
             "in the REPL to load it later).",
    )
    # --auto-pilot is the canonical flag; --yes / -y are hidden back-compat
    # aliases into the SAME dest ("yes") so existing scripts + muscle memory
    # (and the args.yes reader below) keep working unchanged.
    parser.add_argument(
        "--auto-pilot", dest="yes", action="store_true",
        help="Auto-confirm dangerous tools (write/edit/bash). Aliases: --yes, -y.",
    )
    parser.add_argument(
        "-y", "--yes", dest="yes", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help=(
            "Cap provider turns per user message. Counts provider calls, so N=1 "
            "allows no tool-react cycle (tools may run but the model never sees "
            "their results). Use >=2 for any tool-using task."
        ),
    )
    parser.add_argument(
        "--effort", choices=list(EFFORT_LEVELS), default=None,
        help="Reasoning effort: off|low|medium|high (best-effort; see /effort).",
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=None,
        help=(
            "Per-request generation cap (max_tokens) for the local model. "
            "0 or -1 = unbounded (the default). On reasoning models the cap "
            "counts reasoning tokens, so a low cap may cut off the answer."
        ),
    )
    parser.add_argument(
        "--theme", choices=list(THEMES), default=None,
        help="Color theme: amber (default, warm polished look) | auto (truecolor) "
             "| ansi (Dark mode, ANSI colors only — uses your terminal's own "
             "16-color palette) | orange (orange-on-black; orange inline code, no "
             "box). Session-only unless --save.",
    )
    # NETWORK is the DEFAULT (private=False): web_fetch on (SSRF-safe), run_bash
    # un-sandboxed, non-loopback base_url + all MCP servers allowed. --private
    # opts INTO the offline lockdown for this session. --allow-network is kept as
    # an accepted NO-OP alias (network is already the default) for back-compat.
    # They are mutually exclusive; default (neither) leaves the persisted value
    # (which itself defaults to network-on). Flags are session-only unless --save.
    parser.add_argument(
        "--context", default=None, metavar="N|auto|fixed|off",
        help="Working-context budget in ~tokens (auto-trims after each turn so "
             "decode stays fast). N sets the budget; 'auto'/'fixed' toggle "
             "per-request flexing; 'off' disables it (trim only near the model's "
             "window). Session-only unless --save.",
    )
    parser.add_argument(
        "--mcp", choices=["on", "off"], default=None,
        help="Enable/disable MCP servers (from ~/.llm-cli/mcp.json) for this "
             "session. 'off' starts no servers and offers no MCP tools (smaller "
             "prompt, faster). Session-only unless --save. Toggle in-REPL with "
             "/mcp on|off.",
    )
    # Foundation-wave flags: they only carry values onto the session config
    # (behavior is wired in later waves). Session-only unless --save, mirroring
    # --mcp. default=None marks "not passed" so an un-flagged run is untouched.
    parser.add_argument(
        "--mode", choices=list(PERMISSION_MODES), default=None,
        help="Permission mode: default (confirm destructive tools) | auto-edit | "
             "read-only | plan | full-auto. Session-only unless --save.",
    )
    parser.add_argument(
        "--output-format", choices=list(OUTPUT_FORMATS), default=None,
        help="Output rendering: text (default) | json. Session-only unless --save.",
    )
    parser.add_argument(
        "--no-checkpoints", dest="checkpoints", action="store_false", default=None,
        help="Disable pre-write/edit file snapshots for this session (they are "
             "on by default, powering a later /undo). Session-only unless --save.",
    )
    parser.add_argument(
        "--git-autocommit", dest="git_autocommit", action="store_true", default=None,
        help="Auto-commit each file change for this session (off by default). "
             "Session-only unless --save.",
    )
    # MANDATORY self-healing + thermal cooldown (both ON by default). These flags
    # only OPT OUT / override values; default=None marks "not passed" so an
    # un-flagged run keeps the config defaults. Session-only unless --save.
    parser.add_argument(
        "--no-auto-fix", dest="auto_fix_tools", action="store_false", default=None,
        help="Disable MANDATORY tool self-healing for this session (a failed tool "
             "call is auto-corrected + retried before the model sees it; on by "
             "default). Session-only unless --save.",
    )
    parser.add_argument(
        "--no-cooldown", dest="cooldown_enabled", action="store_false", default=None,
        help="Disable the MANDATORY thermal cooldown for this session (a short "
             "break every N seconds of continuous work to let the GPU cool; on by "
             "default). Session-only unless --save.",
    )
    parser.add_argument(
        "--cooldown-interval", type=float, default=None, metavar="SECONDS",
        help="Seconds of continuous work between thermal breaks (default 600). "
             "Session-only unless --save.",
    )
    parser.add_argument(
        "--cooldown-duration", type=float, default=None, metavar="SECONDS",
        help="Seconds each thermal break pauses generation (default 60). "
             "Session-only unless --save.",
    )
    privacy = parser.add_mutually_exclusive_group()
    privacy.add_argument(
        "--private", dest="private", action="store_true", default=None,
        help="Opt into the offline lockdown for this session: no external egress "
             "— base_url must be loopback (IP-pinned); web_fetch off; run_bash "
             "network-sandboxed; only private_ok MCP servers start. Add --save to "
             "persist.",
    )
    privacy.add_argument(
        "--allow-network", dest="private", action="store_false", default=None,
        help="Accepted NO-OP alias for back-compat: network is already enabled by "
             "default. Explicitly keeps network on (web_fetch + non-loopback "
             "base_url + un-sandboxed run_bash). Add --save to persist.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Persist the given flags as the new default in ~/.llm-cli/config.json.",
    )
    # SEED flag (Feature: reproducibility): when set, the MockProvider uses the
    # seed to produce a deterministic scenario. For local models, the seed is
    # passed through to the provider (if supported) to enable reproducible outputs.
    parser.add_argument(
        "--seed", type=int, default=None, metavar="N",
        help="Deterministic seed for reproducible runs (MockProvider uses it). "
             "When set, combined with temperature=0, local models can produce "
             "reproducible outputs. Default: None (no seed).",
    )
    # DRY-RUN flag (Feature): when set, the CLI parses everything but never
    # actually runs the provider or any tools. Useful for debugging prompt
    # construction, tool selection, or configuration loading without side effects.
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Parse arguments and show configuration, but never call the provider "
             "or execute tools. Prints the effective config and returns 0. "
             "Useful for debugging without side effects.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    config = load_config()
    # CLI flags are SESSION-ONLY overrides applied on top of the saved config.
    # They are NOT written back unless --save is given, so a throwaway run like
    # `llmc --provider mock -p ...` can never clobber the user's saved default.
    if args.provider is not None:
        config.provider = args.provider
    if args.model is not None:
        config.model = args.model
    if args.base_url is not None:
        config.base_url = args.base_url
    if args.max_iterations is not None and args.max_iterations > 0:
        config.max_iterations = args.max_iterations
    if args.effort is not None:
        # "unset" is the command-surface sentinel for the server-default (send
        # nothing) state; the stored value is "" (finding #22).
        config.effort = "" if args.effort == "unset" else args.effort
    if args.max_output_tokens is not None:
        # Normalize the "unbounded" sentinels: <=0 (covers 0 and -1) => None so no
        # max_tokens is ever sent. A positive value caps generation this session.
        config.max_output_tokens = (
            args.max_output_tokens if args.max_output_tokens > 0 else None
        )
    if args.theme is not None:
        config.theme = args.theme
    if args.mcp is not None:
        config.mcp_enabled = args.mcp == "on"
    # Foundation-wave overrides (session-only unless --save; None = not passed).
    if args.mode is not None:
        config.permission_mode = args.mode
    if args.output_format is not None:
        config.output_format = args.output_format
    if args.checkpoints is not None:
        # store_false: present => False, absent => None (leaves config default).
        config.checkpoints_enabled = args.checkpoints
    if args.git_autocommit is not None:
        config.git_autocommit = args.git_autocommit
    # Self-healing + cooldown overrides (session-only unless --save; None = not
    # passed). store_false flags: present => False, absent => None (keeps default).
    if args.auto_fix_tools is not None:
        config.auto_fix_tools = args.auto_fix_tools
    if args.cooldown_enabled is not None:
        config.cooldown_enabled = args.cooldown_enabled
    if args.cooldown_interval is not None and args.cooldown_interval > 0:
        config.cooldown_interval_seconds = args.cooldown_interval
    if args.cooldown_duration is not None and args.cooldown_duration >= 0:
        config.cooldown_duration_seconds = args.cooldown_duration
    if args.context is not None:
        c = args.context.strip().lower()
        if c in ("off", "0", "none"):
            config.context_budget = 0
        elif c == "auto":
            config.context_adaptive = True
        elif c == "fixed":
            config.context_adaptive = False
        else:
            try:
                n = int(args.context)
                config.context_budget = n if n > 0 else 0
            except ValueError:
                print(
                    f"error: --context expects an integer, auto, fixed, or off "
                    f"(got {args.context!r})",
                    file=sys.stderr,
                )
                return 2
    # Resolve private mode: an explicit --private/--allow-network overrides the
    # persisted value; otherwise keep what load_config gave us (default OFF =
    # network enabled).
    if args.private is not None:
        config.private = args.private

    # SEED: session-only override. When set, MockProvider uses the seed for
    # deterministic scenarios. For local models, passed through to the provider
    # (if supported) to enable reproducible outputs with temperature=0.
    if args.seed is not None:
        config.seed = args.seed

    # DRY-RUN: parse everything but never actually call the provider or run tools.
    # Useful for debugging prompt construction, tool selection, or config loading.
    if args.dry_run:
        print(
            f"[dry-run] provider={config.provider} model={config.model} "
            f"base_url={config.base_url!r} private={config.private} "
            f"seed={config.seed} theme={config.theme} "
            f"mcp={config.mcp_enabled} gentle={config.gentle_mode} "
            f"rerank={config.rerank}",
            file=sys.stderr,
        )
        return 0

    # PRIVATE-mode enforcement (entry point #1, the --base-url flag): validate
    # the effective base_url BEFORE building anything so a non-loopback URL from
    # the flag is refused up front with a clear error, never silently used.
    if config.private and config.provider == PROVIDER_LOCAL and not is_loopback_url(config.base_url):
        print(
            f"error: private mode: refusing a non-loopback base_url "
            f"({config.base_url!r}). It must be loopback (127.0.0.0/8, ::1, or "
            "localhost) so project data never leaves the machine. Re-run with "
            "--allow-network to use an external server.",
            file=sys.stderr,
        )
        return 2

    # Make the tools layer (run_bash sandbox + web_fetch guard) agree with the
    # resolved mode for this process.
    set_private(config.private)

    if args.save:
        save_config(config)
        print(
            f"[saved defaults] provider={config.provider} model={config.model} "
            f"private={config.private}",
            file=sys.stderr,
        )

    try:
        provider = build_provider(
            config.provider, config.model, config.base_url, config.effort,
            config.private, config.cache_prompt, config.max_output_tokens,
            embed_model=config.embed_model, temperature=config.temperature,
            gentle_mode=config.gentle_mode, gentle_max_tokens=config.gentle_max_tokens,
            seed=config.seed, id_slot=config.id_slot,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Clear private-mode indicator in the one-shot startup banner (the REPL
    # prints its own via _status()).
    if args.prompt is not None:
        if config.private:
            print(
                "private mode: ON — offline lockdown, no external egress",
                file=sys.stderr,
            )
        else:
            print(
                "private mode: OFF — network enabled; web_fetch is SSRF-safe "
                "(blocks internal/metadata, validates redirects, http/https only) "
                "and confirmation-gated; run_bash is confirmation-gated and has "
                "FULL filesystem + network access (the y/N prompt is the boundary "
                "— review each command, avoid --yes on untrusted tasks). --private "
                "adds an offline no-network sandbox for run_bash + drops web_fetch.",
                file=sys.stderr,
            )

    # Auto-confirm ONLY when --yes is passed. One-shot runs are otherwise
    # interactive: a non-TTY one-shot without --yes will have every gated tool
    # (write/edit/bash) declined (input() raises EOFError -> decline), so pass
    # --yes for unattended runs.
    auto_confirm = bool(args.yes)

    if args.prompt is not None:
        final = run_once(
            provider, config, args.prompt, auto_confirm, resume=args.continue_
        )
        # Trailing-newline tidy-up for the human/text path only. In JSON mode
        # run_once already printed exactly one object; a stray blank line here
        # would break a consumer expecting a single JSON object on stdout.
        if config.output_format != "json" and final and not final.endswith("\n"):
            print()
        return 0

    Repl(
        config=config, provider=provider, auto_confirm=auto_confirm,
        resume=args.continue_,
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
