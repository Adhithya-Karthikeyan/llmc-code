"""Shared token-estimate helper.

This mirrors the rough ``chars // 4`` formula used by
``Agent._estimate_tokens`` (agent.py) so the two stay consistent without one
having to import the other (the agent's hot path is left untouched). Used by the
repo_map binary-search token-fit so the rendered map can be sized to a budget.

Keep this in sync with ``Agent._estimate_tokens`` if the formula ever changes.
"""

from __future__ import annotations

__all__ = ["estimate_text_tokens"]


def estimate_text_tokens(s: str) -> int:
    """Rough token estimate (chars / 4) over arbitrary text.

    Matches ``Agent._estimate_tokens`` (chars // 4) so a rendered repo_map's
    token cost is estimated the same way the agent accounts for messages.
    """
    if not s:
        return 0
    return len(s) // 4