"""Whole-project /audit hint heuristic tests (Feature 2).

``looks_like_whole_repo_request`` is a conservative pure-string detector used to
print a one-line /audit hint before a free-text turn. It must fire on explicit
whole-codebase phrasings and stay quiet on normal single-file questions.
"""

from __future__ import annotations

import pytest

from llmcli.repl import looks_like_whole_repo_request


@pytest.mark.parametrize(
    "text",
    [
        "review the whole codebase and summarize it",
        "can you explain this codebase to me?",
        "go through all the files and find dead code",
        "audit everything for security issues",
        "read the entire project and write docs",
        "I want a review of the entire codebase",
        "how does the whole system work end to end?",
        "explain the entire project structure",
        "look across the codebase for usages",
    ],
)
def test_whole_repo_phrasings_match(text):
    assert looks_like_whole_repo_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "what does foo() in bar.py do?",
        "fix the typo in README",
        "why is this test failing?",
        "add a timeout param to fetchData()",
        "explain this function",
        "rename the variable on line 12",
        "",
        "how does caching work in providers.py?",
    ],
)
def test_single_file_questions_do_not_match(text):
    assert looks_like_whole_repo_request(text) is False
