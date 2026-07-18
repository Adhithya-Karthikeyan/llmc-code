"""Configuration: LM Studio defaults, model selection, persisted settings.

LOCAL ONLY. There is no Anthropic/Claude provider. Models are served by
LM Studio's OpenAI-compatible server.

Settings precedence (highest first):
  1. Explicit CLI flags (handled in __main__).
  2. Persisted config file at ~/.llmcode/config.json.
  3. Built-in defaults below.

API keys are read from the environment (or a harmless default) and are NEVER
written to disk by this tool.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import tempfile
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

# LM Studio's OpenAI-compatible server defaults.
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "local-model"

# Providers available in this app. LOCAL = LM Studio, MOCK = offline scripted.
PROVIDER_LOCAL = "local"
PROVIDER_MOCK = "mock"
PROVIDERS = (PROVIDER_LOCAL, PROVIDER_MOCK)

# Color/markdown themes for the terminal renderer (Config.theme, /theme, --theme).
# "auto" keeps the current behavior: rich's default color_system (truecolor/256
# as the terminal allows) and monokai-highlighted code blocks. "ansi" is a "Dark
# mode (ANSI colors only)" — the console downsamples EVERY color to the 16 basic
# ANSI colors (which the terminal then renders per the user's own palette) and
# code blocks use the ANSI-only "ansi_dark" highlighter, so NO truecolor escape
# (\x1b[38;2;R;G;Bm) is ever emitted, only basic SGR (\x1b[32m, \x1b[2m, ...).
# "orange" is an orange-on-black markdown theme: inline code and accents render
# as orange TEXT with NO background box (overrides rich's markdown.* styles).
# "amber" is the polished warm theme: a framed startup banner, gold bold words,
# amber headers/bullets, orange inline code (no box), and a styled "❯" prompt.
# "clean" is the DEFAULT minimal DARK theme: near-monochrome grey scale with a
# single soft accent, LOW-KEY dim/grey borders, and white-ish bold for emphasis.
# The shared layout (banner + a thin rounded BOX around each answer with
# breathing room above/below + prompt + colour-coded status) is palette-driven,
# so EVERY theme gets it — only the accent/border colour differs.
THEME_CLEAN = "clean"
THEME_AUTO = "auto"
THEME_ANSI = "ansi"
THEME_ORANGE = "orange"
THEME_AMBER = "amber"
THEMES = (THEME_CLEAN, THEME_AUTO, THEME_ANSI, THEME_ORANGE, THEME_AMBER)
# The default theme for fresh installs. Existing configs keep their saved theme;
# load_config falls back to this when a persisted theme is missing/unknown.
DEFAULT_THEME = THEME_CLEAN

# Reasoning-effort levels for /effort and --effort. "" = unset (send nothing,
# server default). "off" best-effort disables thinking. low/medium/high map to
# the standard reasoning_effort param. NOTE: honored only by models/servers that
# support it; qwen3.6 on LM Studio currently ignores these.
#
# "unset" is a user-facing SENTINEL that maps back to "" (the server-default,
# send-nothing state). Without it, once a user picks /effort low there is no
# in-REPL way to return to the server default — "off" is NOT the same (it sends
# reasoning_effort=minimal + enable_thinking=False). The empty string is the
# stored value; "unset" is only the spelling accepted at the command surface.
EFFORT_LEVELS = ("unset", "off", "low", "medium", "high")
# The stored effort value for the "unset" sentinel (send nothing to the server).
EFFORT_UNSET = ""

# Conversation-memory recall modes (Config.recall_mode). "auto" = hybrid BM25 ∪
# embeddings with graceful BM25-only fallback; "embed" = same hybrid (embeddings
# attempted, BM25 fallback on failure); "bm25" = lexical BM25 only (no embed
# endpoint needed); "off" = retrieval disabled. An unknown persisted value keeps
# the safe "auto" default.
RECALL_MODES = ("auto", "embed", "bm25", "off")

# code_search recall modes (Config.code_search_recall). Same semantics as
# RECALL_MODES but govern the deliberately-invoked code_search TOOL: "bm25" =
# lexical-only (never calls embeddings); "auto"/"embed" re-enable semantic code
# recall; "off" disables. Default is "bm25" (see the field below).
CODE_SEARCH_RECALL_MODES = ("auto", "bm25", "embed", "off")

# Permission modes (Config.permission_mode, --mode). Foundation for a later
# feature wave; these only carry the value today (no behavior is wired yet).
# "default" = CURRENT behavior (confirm destructive write/edit/bash tools).
# "auto-edit" = auto-apply edits (later); "read-only" = block writes (later);
# "plan" = plan-only, no execution (later); "full-auto" = confirm nothing
# (later). An unknown persisted value keeps the safe "default".
PERMISSION_MODES = ("default", "auto-edit", "read-only", "plan", "full-auto")

# Output formats (Config.output_format, --output-format). "text" = the current
# human-readable rendering; "json" = machine-readable output (later wave). An
# unknown persisted value keeps the safe "text" default.
OUTPUT_FORMATS = ("text", "json")

CONFIG_DIR = Path.home() / ".llmcode"
CONFIG_PATH = CONFIG_DIR / "config.json"


def is_loopback_url(url: str) -> bool:
    """True iff ``url``'s host is the local machine (loopback) ONLY.

    PRIVATE-mode gate: the provider base_url must resolve to the local box so
    project data is never shipped off-machine. Accepts:
      - the literal ``localhost`` hostname (case-insensitive),
      - any IPv4 address in 127.0.0.0/8,
      - the IPv6 loopback ``::1``,
      - a hostname that resolves EXCLUSIVELY to loopback addresses.

    Rejects everything else, including the unspecified address 0.0.0.0 / ::
    (it is reachable from off-box and binds all interfaces), public hosts, and
    a host that resolves to ANY non-loopback address. A trailing-dot host
    (``localhost.``) is normalized away before the name check so it cannot
    sneak past as a distinct label.
    """
    if not isinstance(url, str) or not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    host = parsed.hostname
    if not host:
        return False
    host = host.strip()
    # Strip a single trailing dot (FQDN root) so 'localhost.' == 'localhost'.
    if host.endswith(".") and host != ".":
        host = host[:-1]
    if not host:
        return False
    lowered = host.lower()
    if lowered == "localhost":
        return True
    # REJECT non-canonical numeric host encodings (bare-integer '2130706433',
    # 0x-hex '0x7f000001', octal '017700000001') BEFORE any resolution
    # (finding #4). ``ipaddress.ip_address(str)`` does NOT parse these, but the
    # OS resolver / HTTP client DOES treat them as 127.0.0.1 — that
    # validator-vs-client divergence is the classic rebinding seam. We require a
    # canonical dotted-decimal IPv4 (contains '.') or a canonical IPv6 form
    # (contains ':'; urllib already stripped the brackets in .hostname). A bare
    # token with neither a '.' nor a ':' that is all-digits or 0x-hex is a
    # numeric host encoding and is refused outright; a normal hostname label
    # (e.g. 'mybox') falls through to DNS resolution below. (A legitimate
    # hostname is never all-digits, and 'localhost' was handled above.)
    if ":" not in host and "." not in host:
        h = lowered
        if h.startswith("0x") or h.isdigit():
            return False  # non-canonical numeric host form: refuse.
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr is not None:
        # is_loopback covers 127.0.0.0/8 and ::1. The unspecified address
        # (0.0.0.0 / ::) is NOT loopback and is explicitly rejected here.
        return bool(addr.is_loopback)
    # A hostname (not 'localhost', not a literal IP): resolve it and require
    # EVERY resolved address to be loopback. A single non-loopback answer
    # (DNS-rebinding / split-horizon trick) fails closed.
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    if not infos:
        return False
    for info in infos:
        ip = info[4][0]
        # Strip a possible IPv6 scope id (e.g. 'fe80::1%lo0').
        ip = ip.split("%", 1)[0]
        try:
            if not ipaddress.ip_address(ip).is_loopback:
                return False
        except ValueError:
            return False
    return True


def resolve_loopback_ip(url: str) -> str | None:
    """Resolve ``url``'s host to a single VALIDATED loopback IP, or ``None``.

    DNS-rebinding / TOCTOU defense for the provider path (finding #1): the
    provider must connect to the exact IP we validated, not re-resolve the
    hostname per request. This returns a literal loopback IP to pin to:

      - a literal loopback IP (127.0.0.0/8, ::1) -> that literal (no resolution);
      - the name 'localhost' or any other hostname -> resolved, with EVERY
        answer required to be loopback (mirrors :func:`is_loopback_url`), and the
        first loopback address returned as the pin target.

    Returns ``None`` when the host is missing, is a non-canonical numeric form,
    or resolves to (or contains) any non-loopback address. Callers should pin
    the connection to the returned literal IP so no later re-resolution can
    redirect traffic off-box.
    """
    if not is_loopback_url(url):
        return None
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return None
    host = parsed.hostname
    if not host:
        return None
    host = host.strip()
    if host.endswith(".") and host != ".":
        host = host[:-1]
    # A literal loopback IP: pin to it directly (no resolution can change it).
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr is not None:
        return host if addr.is_loopback else None
    # 'localhost' or a hostname: resolve and pin to the first loopback answer.
    # is_loopback_url already guaranteed EVERY answer is loopback, but re-resolve
    # here so the pinned IP is one we just validated (single, tight window).
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return None
    for info in infos:
        ip = info[4][0].split("%", 1)[0]
        try:
            if ipaddress.ip_address(ip).is_loopback:
                return ip
        except ValueError:
            continue
        # A non-loopback answer appeared between the two resolutions: fail closed.
        return None
    return None


@dataclass
class Config:
    """Runtime configuration for the CLI.

    Only non-secret fields are persisted to ``CONFIG_PATH``. The API key lives
    in the environment and is resolved lazily via :func:`get_api_key`.
    """

    provider: str = PROVIDER_LOCAL
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    # tool-use rounds per turn. 12 was too low for real codebase tasks; 50 was
    # still getting hit on legitimately long multi-step work (read/edit/test/fix).
    # 80 is a generous backstop for real multi-step work; the duplicate-batch loop
    # guard in Agent.run (breaks early if the model re-issues the SAME tool batch
    # repeatedly) is the normal stop, so a stuck model can no longer spin to the
    # cap. The cap is a backstop, not the normal stop condition. (Was 200 — an
    # excessive backstop that let a confused model burn many extra rounds.)
    max_iterations: int = 80
    effort: str = ""  # "" = unset; else one of EFFORT_LEVELS
    # Terminal color/markdown theme. DEFAULT "clean" (minimal dark, low-key grey)
    # for fresh installs; "amber" is the warm polished look; "auto" is rich's
    # truecolor default; "ansi" is the Dark mode (ANSI colors only) — see THEMES.
    theme: str = DEFAULT_THEME
    # PRIVATE / OFFLINE lockdown mode. DEFAULT OFF: network is enabled out of the
    # box (web_fetch on, run_bash un-sandboxed, non-loopback base_url + MCP all
    # start). The ALWAYS-ON safety guards (web_fetch SSRF + IP-pinning, run_bash
    # confirmation + workspace confinement, write/edit confinement) stay active
    # regardless of this flag. Setting it True opts INTO the full offline lockdown
    # (no egress) enforced in tools/providers/MCP. Use --private to enable.
    private: bool = False
    # Auto-compaction safety valve (findings #4/#26). When the rough chars/4
    # token estimate of the live history exceeds this soft budget, Agent.run()
    # compacts BEFORE the next provider call, so a multi-turn file-reading
    # session can't silently grow the prompt past the local model's context
    # window. ~24k tokens leaves headroom under a typical 32k window while still
    # letting a couple of large (60KB) tool results coexist. 0 disables it.
    context_soft_limit: int = 24_000
    # KV-cache reuse hint (finding #12). When True, the local provider adds
    # extra_body.cache_prompt=true so llama.cpp/LM Studio explicitly reuse the
    # prompt cache across turns. Backward-compatible: servers that don't know the
    # key ignore it. Default ON (the prompt prefix is append-only and stable).
    cache_prompt: bool = True
    # Opt-in per-request generation cap. When set, the local provider sends
    # max_tokens=N so LM Studio stops generating after N tokens this turn.
    # Backward-compatible: DEFAULT None = unbounded (the field is omitted, NOT
    # sent as -1 — LM Studio does not honor -1 as unbounded; omitting it is the
    # correct way to leave generation uncapped).
    max_output_tokens: int | None = None  # per-request generation cap (tokens); None = unbounded
    # Adaptive working-context budget (rough chars/4 tokens). After EVERY turn the
    # live history is trimmed (older turns -> a running summary) to keep it near
    # this budget, so context stays small and decode stays fast regardless of how
    # much the model has read. ADAPTIVE: the per-turn budget flexes UP for a big
    # request (audit/refactor/whole-project/long prompt) and stays tight for a
    # simple one. The model's real window is still an upper ceiling. 0 = OFF
    # (fall back to the near-window safety valve only). Tune with /context or
    # --context. Much tighter than the old "80% of the window" default — that is
    # what kept big sessions slow.
    context_budget: int = 12_000
    context_adaptive: bool = True  # flex the budget per request (vs a flat budget)
    # PERF read-budget guard threshold (bytes). How many bytes of CODE/output the
    # context-bloating tools (read_file/grep/repo_map/code_search/glob) may pull
    # into ONE user turn before the agent appends a one-time "stop reading" nudge.
    # ~32KB ≈ ~8K tokens, where extra context starts visibly slowing local decode.
    # Tune lower to nudge sooner on small models, higher to allow heavier reads.
    read_nudge_bytes: int = 32_000
    # MCP servers (from ~/.llmcode/mcp.json) on/off. DEFAULT True (back-compat:
    # configured servers start and their tools are offered). Turning it OFF stops
    # the server subprocesses AND removes their tools from every request — which
    # shrinks the per-turn prompt (each MCP tool's JSON schema is sent every
    # turn), so decode speed (tok/s) recovers. Toggle in-session with /mcp on|off
    # or at launch with --mcp on|off (session-only unless --save).
    mcp_enabled: bool = True
    # Conversation-memory retrieval (STAGE 1: offline core; agent wiring is later).
    # embed_model: the LM Studio /v1/embeddings model id used to vector-encode the
    # query + corpus when recall uses embeddings (768-dim nomic by default).
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    # recall_mode: how memory is searched — one of RECALL_MODES (see above).
    recall_mode: str = "auto"
    # memory_enabled: master switch for the retrieval feature. Default ON.
    memory_enabled: bool = True
    # memory_top_k: max records returned per recall (guard: a real int > 0).
    memory_top_k: int = 3
    # Sampling temperature sent on EVERY chat/tool turn. LOW by default (0.2):
    # code + tool-call turns need deterministic, well-formed output, and the
    # server default (~0.7-0.8) is the root cause of malformed tool calls and
    # hallucinated output on local models. Clamped to 0.0-2.0 by load_config; set
    # at runtime with /temp.
    temperature: float = 0.2
    # Constrained-decode retry (Feature 2): when the model emits a tool call the
    # system cannot parse, re-issue the SAME request once with
    # tool_choice="required" to force a clean NATIVE tool call before falling back
    # to the corrective-text behavior. Default ON; a server that rejects
    # tool_choice or a still-malformed retry falls back gracefully.
    constrained_retry: bool = True
    # Auto-verify after edit (Feature 3): a shell command run AUTOMATICALLY after a
    # turn that wrote/edited files but never ran a command itself; its output is
    # fed back so the model can fix failures. Empty = DISABLED (opt-in to avoid
    # surprise + speed cost). Set at runtime with /verify (e.g. "python -m pytest -q").
    verify_cmd: str = ""
    # Reviewer gate (Feature 4): after a turn makes a code change, spawn the
    # read-only reviewer sub-agent to critique it and feed findings back before the
    # final answer. Now OPT-IN / default OFF (was default-ON; it caused ~2x latency
    # on every code-writing turn by spawning a full reviewer sub-agent). Enable it
    # deliberately when you want the gate; top-level orchestrator only (sub-agents
    # lack spawn_agent so they can never trigger it / recurse).
    review_writes: bool = False
    # Max on-disk size (bytes) of a single image attached via /image. base64
    # images bloat the prompt and slow local decode, so larger files are
    # rejected with an actionable error. Mirrors images.MAX_IMAGE_BYTES.
    max_image_bytes: int = 5_000_000
    # GENTLE mode: reduce average GPU load/heat on local hardware. DEFAULT ON.
    # Two effects when on: (1) it LOWERS the effective per-request output-token
    # cap to gentle_max_tokens (shorter generation bursts) — never RAISING a
    # smaller existing cap, only capping an unset/larger one; (2) it paces user
    # turns with a minimum cool-down between the END of one generation and the
    # START of the next (gentle_gap_seconds). HONEST NOTE: this does NOT cap the
    # GPU utilization percentage — it only lowers AVERAGE load/heat by shortening
    # bursts and spacing turns. When off, behavior is exactly as before.
    gentle_mode: bool = True
    # Output-token cap applied WHEN gentle is on (guard: a real int > 0). 1024
    # frequently truncated tool-call JSON / long answers (finish_reason=length),
    # forcing an automatic uncapped retry = a full second generation, which
    # INCREASED total tokens/heat. 4096 leaves room for reasoning + content + a
    # file write so the retry rarely fires.
    gentle_max_tokens: int = 4096
    # Minimum seconds between the end of one generation and the start of the next
    # when gentle is on (guard: a real number >= 0). Only rapid back-to-back
    # turns wait; if the user spent longer than this typing, there is zero wait.
    gentle_gap_seconds: float = 2.0
    # Seconds to wait between SEQUENTIAL sub-agent spawns when gentle is on and the
    # spawn is in a real terminal (guard: a real number >= 0). Only fires BEFORE
    # the 2nd+ spawn (never before the first), spacing out multi-spawn orchestrator
    # bursts to lower AVERAGE load/heat. HONEST NOTE: this does NOT cap GPU % — it
    # only spaces sub-agent bursts; parallel spawns each independently check the
    # gap. Zero disables. Default 5.0s.
    gentle_spawn_gap_seconds: float = 5.0
    # SEED: deterministic replay control (Feature: reproducibility). When set, the
    # MockProvider uses the seed to produce a deterministic scenario. For local
    # models, the seed is passed through to the provider (if supported) to enable
    # reproducible outputs when combined with temperature=0. DEFAULT None (no seed).
    seed: int | None = None
    # ID_SLOT: pin every request to a fixed llama.cpp slot so the prefix KV cache
    # stays warm across turns (each turn re-prefills only the delta instead of the
    # full prompt, vs LM Studio's inconsistent -sps prefix-match heuristic). 0 = pin
    # to slot 0 (LM Studio typically runs 1 slot; safe default). None = do not pin.
    id_slot: int | None = 0
    # RERANK: gated LLM-judge reranker for weak-signal retrieval (the paraphrase
    # case where BM25/embeddings return a junk order). DEFAULT OFF — zero
    # latency/behavior change on the existing path. When on, retrieval asks the
    # chat model to re-order a candidate pool and narrow it to top_k.
    rerank: bool = False
    # rerank_candidates: the candidate-pool width the reranker sees (guard: a real
    # int >= 1). Larger = more thorough but costs a bigger rerank chat prompt.
    rerank_candidates: int = 20
    # code_search_recall: recall mode for the code_search TOOL (one of
    # CODE_SEARCH_RECALL_MODES). Default BM25-only avoids embedding-model GPU
    # swaps that evict the chat model on single-GPU/local servers; identifiers are
    # lexical so BM25 is strong for code. Set to "auto"/"embed" (via /codeembed)
    # to re-enable semantic code recall.
    code_search_recall: str = "bm25"
    # ----- Foundation wave: fields carried now, BEHAVIOR wired in later waves.
    # Each default PRESERVES today's behavior; only the value is persisted here.
    # permission_mode: one of PERMISSION_MODES. "default" = current behavior
    # (confirm destructive write/edit/bash tools). Set at launch with --mode.
    permission_mode: str = "default"
    # git_autocommit: later, auto-commit each file change. DEFAULT OFF so an
    # un-flagged run behaves exactly as today (no git side effects).
    git_autocommit: bool = False
    # checkpoints_enabled: later, snapshot files before write/edit for /undo.
    # DEFAULT ON (the snapshot is local + cheap); disable with --no-checkpoints.
    checkpoints_enabled: bool = True
    # hooks_enabled: later, run lifecycle hook scripts. DEFAULT OFF (opt-in, since
    # hook scripts run arbitrary local commands).
    hooks_enabled: bool = False
    # rules_file_enabled: later, auto-load a project rules file into context.
    # DEFAULT ON (harmless when no rules file exists).
    rules_file_enabled: bool = True
    # output_format: one of OUTPUT_FORMATS. "text" = the current human-readable
    # rendering; "json" (later) = machine-readable. Set at launch with
    # --output-format.
    output_format: str = "text"
    # diff_preview: later, show a diff before applying an edit. DEFAULT ON.
    diff_preview: bool = True
    # ----- MANDATORY self-healing + thermal cooldown ------------------------
    # auto_fix_tools: when a tool call FAILS, the harness itself consults the
    # deterministic remediator (remediation.py) for a SAFE corrected retry (a path
    # that landed outside the workspace, or a unique-basename file-not-found)
    # BEFORE handing the error back to the model. DEFAULT ON (bounded + non-
    # destructive); disable with --no-auto-fix.
    auto_fix_tools: bool = True
    # auto_fix_max_attempts: max corrected retries per failed call (guard: int in
    # 1..5). A bad/out-of-range value keeps the safe default.
    auto_fix_max_attempts: int = 2
    # cooldown_enabled: MANDATORY thermal pacing — take a short break every
    # cooldown_interval_seconds of continuous work (even mid-turn) so the local
    # GPU cools. DEFAULT ON; disable with --no-cooldown or /cooldown off.
    cooldown_enabled: bool = True
    # cooldown_interval_seconds: work between breaks (guard: > 0). 10 minutes.
    cooldown_interval_seconds: float = 600.0
    # cooldown_duration_seconds: how long each break pauses (guard: >= 0). 60s.
    cooldown_duration_seconds: float = 60.0

    def to_dict(self) -> dict:
        # All fields are plain serializable scalars, so asdict covers them and
        # stays correct automatically when a new field is added.
        return asdict(self)


def get_api_key() -> str:
    """Resolve the API key for the OpenAI-compatible client.

    LM Studio ignores the value, but the OpenAI SDK requires a non-empty key.
    """
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LMSTUDIO_API_KEY")
        or DEFAULT_API_KEY
    )


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load config, merging file values over defaults. Missing file => defaults."""
    cfg = Config()
    if not path.exists():
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Warn before silently reverting to defaults: the next save_config would
        # overwrite the corrupt file, so the user must know it was ignored.
        print(
            f"[warn] config at {path} is invalid JSON; using defaults "
            "(it will be overwritten on the next setting change).",
            file=sys.stderr,
        )
        return cfg
    except OSError:
        return cfg
    if not isinstance(data, dict):
        return cfg
    if isinstance(data.get("provider"), str) and data["provider"] in PROVIDERS:
        cfg.provider = data["provider"]
    if isinstance(data.get("model"), str) and data["model"]:
        cfg.model = data["model"]
    # Resolve private FIRST (it is the gate that validates base_url below).
    # Default OFF (network enabled); only an explicit `true` opts INTO lockdown.
    if isinstance(data.get("private"), bool):
        cfg.private = data["private"]
    if isinstance(data.get("base_url"), str) and data["base_url"]:
        candidate = data["base_url"]
        # In private mode a PERSISTED non-loopback base_url must be REFUSED, not
        # silently used: loading it would point provider traffic off-box. We
        # warn and fall back to the safe loopback default rather than honoring a
        # poisoned config that would ship prompts + file contents externally.
        if cfg.private and not is_loopback_url(candidate):
            print(
                f"[warn] private mode: refusing persisted non-loopback base_url "
                f"{candidate!r}; using {DEFAULT_BASE_URL}. "
                "Pass --allow-network (with --save) to persist an external URL.",
                file=sys.stderr,
            )
        else:
            cfg.base_url = candidate
    if (
        isinstance(data.get("max_iterations"), int)
        and not isinstance(data["max_iterations"], bool)  # bool is an int subclass
        and data["max_iterations"] > 0
    ):
        cfg.max_iterations = data["max_iterations"]
    if (
        isinstance(data.get("effort"), str)
        and data["effort"] in EFFORT_LEVELS
        # "unset" is a command-surface sentinel only; the stored value is "".
        and data["effort"] != "unset"
    ):
        cfg.effort = data["effort"]
    if (
        isinstance(data.get("context_soft_limit"), int)
        and not isinstance(data["context_soft_limit"], bool)
        and data["context_soft_limit"] >= 0
    ):
        cfg.context_soft_limit = data["context_soft_limit"]
    if isinstance(data.get("cache_prompt"), bool):
        cfg.cache_prompt = data["cache_prompt"]
    if isinstance(data.get("mcp_enabled"), bool):
        cfg.mcp_enabled = data["mcp_enabled"]
    # Conversation-memory fields (typed guards mirroring the existing ones).
    if isinstance(data.get("embed_model"), str) and data["embed_model"]:
        cfg.embed_model = data["embed_model"]
    # recall_mode: accept only a known mode; unknown/typo keeps the safe default.
    if isinstance(data.get("recall_mode"), str) and data["recall_mode"] in RECALL_MODES:
        cfg.recall_mode = data["recall_mode"]
    # code_search_recall: accept only a known mode; unknown/typo keeps the default.
    if (
        isinstance(data.get("code_search_recall"), str)
        and data["code_search_recall"] in CODE_SEARCH_RECALL_MODES
    ):
        cfg.code_search_recall = data["code_search_recall"]
    if isinstance(data.get("memory_enabled"), bool):
        cfg.memory_enabled = data["memory_enabled"]
    # memory_top_k: a real int > 0 only (bool is an int subclass — reject it).
    if (
        isinstance(data.get("memory_top_k"), int)
        and not isinstance(data["memory_top_k"], bool)
        and data["memory_top_k"] > 0
    ):
        cfg.memory_top_k = data["memory_top_k"]
    if (
        isinstance(data.get("context_budget"), int)
        and not isinstance(data["context_budget"], bool)
        and data["context_budget"] >= 0
    ):
        cfg.context_budget = data["context_budget"]
    if isinstance(data.get("context_adaptive"), bool):
        cfg.context_adaptive = data["context_adaptive"]
    if (
        isinstance(data.get("read_nudge_bytes"), int)
        and not isinstance(data["read_nudge_bytes"], bool)
        and data["read_nudge_bytes"] > 0
    ):
        cfg.read_nudge_bytes = data["read_nudge_bytes"]
    # Per-request generation cap: accept only a real int > 0 (bool is rejected:
    # isinstance(True, int) is True in Python). A persisted -1/0 (the "unbounded"
    # sentinels) leaves it None so no max_tokens is ever sent to the server.
    if (
        isinstance(data.get("max_output_tokens"), int)
        and not isinstance(data["max_output_tokens"], bool)
        and data["max_output_tokens"] > 0
    ):
        cfg.max_output_tokens = data["max_output_tokens"]
    # Theme: accept only a known value; anything else (typo, removed theme) keeps
    # the safe default so a poisoned/old config can't select an unknown theme.
    if isinstance(data.get("theme"), str) and data["theme"] in THEMES:
        cfg.theme = data["theme"]
    # temperature: a real number (int or float, NOT bool — bool is an int subclass)
    # within [0.0, 2.0]. Out-of-range or wrong-typed values keep the safe 0.2
    # default rather than letting a poisoned config raise the server to a hot,
    # malformed-output temperature. Mirrors the context_budget validation pattern.
    temp = data.get("temperature")
    if (
        isinstance(temp, (int, float))
        and not isinstance(temp, bool)
        and 0.0 <= float(temp) <= 2.0
    ):
        cfg.temperature = float(temp)
    if isinstance(data.get("constrained_retry"), bool):
        cfg.constrained_retry = data["constrained_retry"]
    # verify_cmd: any string is valid (empty string is the DISABLED state).
    if isinstance(data.get("verify_cmd"), str):
        cfg.verify_cmd = data["verify_cmd"]
    if isinstance(data.get("review_writes"), bool):
        cfg.review_writes = data["review_writes"]
    # max_image_bytes: accept only a real int > 0 (bool rejected: it is an int
    # subclass). A bad/missing value keeps the safe default cap.
    if (
        isinstance(data.get("max_image_bytes"), int)
        and not isinstance(data["max_image_bytes"], bool)
        and data["max_image_bytes"] > 0
    ):
        cfg.max_image_bytes = data["max_image_bytes"]
    # GENTLE mode (default ON). Only an explicit bool flips it; anything else
    # keeps the safe default.
    if isinstance(data.get("gentle_mode"), bool):
        cfg.gentle_mode = data["gentle_mode"]
    # gentle_max_tokens: a real int > 0 only (bool is an int subclass — reject
    # it). A bad/non-positive value keeps the safe default cap.
    if (
        isinstance(data.get("gentle_max_tokens"), int)
        and not isinstance(data["gentle_max_tokens"], bool)
        and data["gentle_max_tokens"] > 0
    ):
        cfg.gentle_max_tokens = data["gentle_max_tokens"]
    # gentle_gap_seconds: a real number (int or float, NOT bool) >= 0. A
    # negative or wrong-typed value keeps the safe default gap.
    gap = data.get("gentle_gap_seconds")
    if (
        isinstance(gap, (int, float))
        and not isinstance(gap, bool)
        and float(gap) >= 0.0
    ):
        cfg.gentle_gap_seconds = float(gap)
    # gentle_spawn_gap_seconds: a real number (int or float, NOT bool) >= 0. A
    # negative or wrong-typed value keeps the safe default spawn gap.
    sgap = data.get("gentle_spawn_gap_seconds")
    if (
        isinstance(sgap, (int, float))
        and not isinstance(sgap, bool)
        and float(sgap) >= 0.0
    ):
        cfg.gentle_spawn_gap_seconds = float(sgap)
    # SEED: accept only a real int >= 0 (bool rejected: it is an int subclass).
    # A persisted -1 means "no seed". Unknown/non-numeric types keep the None default.
    seed = data.get("seed")
    if (
        isinstance(seed, int)
        and not isinstance(seed, bool)
        and seed >= 0
    ):
        cfg.seed = seed
    # ID_SLOT: accept only a real int (bool rejected: it is an int subclass) with
    # value >= 0, OR None. A negative/wrong-typed value keeps the safe default 0.
    id_slot = data.get("id_slot")
    if id_slot is None or (
        isinstance(id_slot, int)
        and not isinstance(id_slot, bool)
        and id_slot >= 0
    ):
        cfg.id_slot = id_slot
    # RERANK: only an explicit bool flips it; anything else keeps the safe
    # default (off) so a typo/junk value never silently enables a chat call.
    if isinstance(data.get("rerank"), bool):
        cfg.rerank = data["rerank"]
    # rerank_candidates: a real int >= 1 only (bool is an int subclass — reject
    # it). A bad/non-positive value keeps the safe default pool width.
    rc = data.get("rerank_candidates")
    if (
        isinstance(rc, int)
        and not isinstance(rc, bool)
        and rc >= 1
    ):
        cfg.rerank_candidates = rc
    # ----- Foundation wave fields (typed guards mirroring the existing ones).
    # permission_mode: accept only a known mode; unknown/typo keeps "default".
    if (
        isinstance(data.get("permission_mode"), str)
        and data["permission_mode"] in PERMISSION_MODES
    ):
        cfg.permission_mode = data["permission_mode"]
    # git_autocommit (default OFF): only an explicit bool flips it.
    if isinstance(data.get("git_autocommit"), bool):
        cfg.git_autocommit = data["git_autocommit"]
    # checkpoints_enabled (default ON): only an explicit bool flips it.
    if isinstance(data.get("checkpoints_enabled"), bool):
        cfg.checkpoints_enabled = data["checkpoints_enabled"]
    # hooks_enabled (default OFF): only an explicit bool flips it.
    if isinstance(data.get("hooks_enabled"), bool):
        cfg.hooks_enabled = data["hooks_enabled"]
    # rules_file_enabled (default ON): only an explicit bool flips it.
    if isinstance(data.get("rules_file_enabled"), bool):
        cfg.rules_file_enabled = data["rules_file_enabled"]
    # output_format: accept only a known format; unknown/typo keeps "text".
    if (
        isinstance(data.get("output_format"), str)
        and data["output_format"] in OUTPUT_FORMATS
    ):
        cfg.output_format = data["output_format"]
    # diff_preview (default ON): only an explicit bool flips it.
    if isinstance(data.get("diff_preview"), bool):
        cfg.diff_preview = data["diff_preview"]
    # ----- self-healing + cooldown (typed guards mirroring the existing ones).
    # auto_fix_tools (default ON): only an explicit bool flips it.
    if isinstance(data.get("auto_fix_tools"), bool):
        cfg.auto_fix_tools = data["auto_fix_tools"]
    # auto_fix_max_attempts: a real int in 1..5 only (bool is an int subclass —
    # reject it). A bad/out-of-range value keeps the safe default.
    if (
        isinstance(data.get("auto_fix_max_attempts"), int)
        and not isinstance(data["auto_fix_max_attempts"], bool)
        and 1 <= data["auto_fix_max_attempts"] <= 5
    ):
        cfg.auto_fix_max_attempts = data["auto_fix_max_attempts"]
    # cooldown_enabled (default ON): only an explicit bool flips it.
    if isinstance(data.get("cooldown_enabled"), bool):
        cfg.cooldown_enabled = data["cooldown_enabled"]
    # cooldown_interval_seconds: a real number (int or float, NOT bool) > 0. A
    # non-positive or wrong-typed value keeps the safe default interval.
    ci = data.get("cooldown_interval_seconds")
    if (
        isinstance(ci, (int, float))
        and not isinstance(ci, bool)
        and float(ci) > 0.0
    ):
        cfg.cooldown_interval_seconds = float(ci)
    # cooldown_duration_seconds: a real number (int or float, NOT bool) >= 0. A
    # negative or wrong-typed value keeps the safe default duration.
    cd = data.get("cooldown_duration_seconds")
    if (
        isinstance(cd, (int, float))
        and not isinstance(cd, bool)
        and float(cd) >= 0.0
    ):
        cfg.cooldown_duration_seconds = float(cd)
    return cfg


def save_config(cfg: Config, path: Path = CONFIG_PATH) -> None:
    """Persist non-secret config to disk. Never writes any API key."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write (mirrors session.save_session): a crash / full disk mid-
        # write must not truncate config.json and silently lose ALL settings.
        # Write a temp file in the same dir, fsync, then os.replace().
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".config-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(cfg.to_dict(), indent=2))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        # Persistence is best-effort; failing to save must not crash the app.
        pass
