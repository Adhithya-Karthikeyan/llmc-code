"""Tests for the conservative final-answer preamble normalizer + tok guards."""

from __future__ import annotations

from llmcli.agent import compute_tok_stats, normalize_final_answer


def test_strips_leading_preamble_clause():
    # finding #2: only strip when the clause after the opener is content-free
    # filler ("do it") or the opener IS the whole first sentence. "I will do it."
    # is filler -> stripped.
    assert normalize_final_answer("I will do it. Done.") == "Done."
    # Opener as the entire first sentence: "Okay" then the real answer.
    assert normalize_final_answer("Okay, \n- one\n- two") == "- one\n- two"


def test_keeps_substantive_first_sentence_after_opener():
    # finding #2 regression: a substantive first clause must NOT be dropped.
    assert (
        normalize_final_answer("Sure, the file has 3 functions. They are foo, bar, baz.")
        == "Sure, the file has 3 functions. They are foo, bar, baz."
    )
    assert (
        normalize_final_answer("Okay, the answer is. 5 plus 5 equals 10.")
        == "Okay, the answer is. 5 plus 5 equals 10."
    )


def test_colon_is_not_a_strip_point():
    # A ':' introduces the actual content, so it is NOT a sentence break we cut
    # at (finding #9): the informative clause after the colon must be preserved.
    assert (
        normalize_final_answer("Let me explain: the answer is 42")
        == "Let me explain: the answer is 42"
    )


def test_here_is_openers_are_not_stripped():
    # "here is/here's/here are" routinely precede the answer's subject, so they
    # are no longer stripped at all (finding #9): the topic clause is real content.
    assert (
        normalize_final_answer("Here is how recursion works: a function calls itself.")
        == "Here is how recursion works: a function calls itself."
    )
    assert (
        normalize_final_answer("Here are the three steps. First do X.")
        == "Here are the three steps. First do X."
    )


def test_keeps_clean_answers_unchanged():
    assert normalize_final_answer("## Result\n- ok") == "## Result\n- ok"
    assert normalize_final_answer("yes, tests pass") == "yes, tests pass"
    assert normalize_final_answer("") == ""


def test_never_touches_code_fences_or_headers():
    code = "```python\nprint('here is')\n```"
    assert normalize_final_answer(code) == code
    header = "## Sure\n- x"
    assert normalize_final_answer(header) == header


def test_does_not_strip_when_no_sentence_break():
    # A single short line that merely STARTS with an opener word is left alone.
    assert normalize_final_answer("Sure thing") == "Sure thing"


def test_tok_stats_rejects_bool_output_tokens():
    # bool is an int subclass; True must NOT be counted as "1 tok".
    tokens, _ = compute_tok_stats("some words here", True, 1.0)
    assert tokens != 1  # falls back to text-based estimate, not the bool
    assert tokens > 0
