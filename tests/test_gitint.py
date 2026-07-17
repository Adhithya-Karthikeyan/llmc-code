"""Tests for llmcli.gitint — real tmp git repos + non-repo graceful degradation.

The git-backed tests build a genuine repository (init, local user config, an
initial commit) in a pytest ``tmp_path`` and assert the helpers observe real
state. They are skipped cleanly when git is unavailable. The non-repo tests
need no git and assert every helper degrades without raising.
"""

from __future__ import annotations

import subprocess

import pytest

from llmcli import gitint

# Every test that drives real git is gated on the binary being present.
requires_git = pytest.mark.skipif(
    not gitint.git_available(), reason="git executable not on PATH"
)


def _run(root, *args):
    """Run a raw git command in ``root`` for test setup (asserts success)."""
    subprocess.run(
        ["git", *args],
        cwd=str(root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A tmp git repo with local identity and one committed file."""
    _run(tmp_path, "init")
    # Local (not global) config so the test never touches the user's ~/.gitconfig
    # and works on a machine with no global identity set.
    _run(tmp_path, "config", "user.email", "test@example.com")
    _run(tmp_path, "config", "user.name", "Test User")
    # Pin a deterministic default branch name regardless of git's global default.
    _run(tmp_path, "checkout", "-b", "main")
    (tmp_path / "hello.txt").write_text("hello\n")
    _run(tmp_path, "add", "hello.txt")
    _run(tmp_path, "commit", "-m", "initial commit")
    return tmp_path


# --------------------------------------------------------------------------- #
# git_available
# --------------------------------------------------------------------------- #

def test_git_available_returns_bool():
    assert isinstance(gitint.git_available(), bool)


# --------------------------------------------------------------------------- #
# repo-backed behavior
# --------------------------------------------------------------------------- #

@requires_git
def test_is_repo_true(repo):
    assert gitint.is_repo(str(repo)) is True


@requires_git
def test_current_branch(repo):
    assert gitint.current_branch(str(repo)) == "main"


@requires_git
def test_clean_repo_is_not_dirty(repo):
    assert gitint.is_dirty(str(repo)) is False
    assert gitint.short_status(str(repo)) == ""


@requires_git
def test_dirty_after_edit(repo):
    (repo / "hello.txt").write_text("hello\nworld\n")
    assert gitint.is_dirty(str(repo)) is True
    status = gitint.short_status(str(repo))
    assert "hello.txt" in status


@requires_git
def test_dirty_with_untracked_file(repo):
    (repo / "new.txt").write_text("brand new\n")
    assert gitint.is_dirty(str(repo)) is True
    assert "new.txt" in gitint.short_status(str(repo))


@requires_git
def test_diff_working_tree(repo):
    (repo / "hello.txt").write_text("hello\nworld\n")
    text = gitint.diff(str(repo))
    assert "hello.txt" in text
    assert "+world" in text


@requires_git
def test_diff_path_filter(repo):
    (repo / "hello.txt").write_text("changed\n")
    (repo / "other.txt").write_text("other\n")
    _run(repo, "add", "other.txt")
    text = gitint.diff(str(repo), path="hello.txt")
    assert "hello.txt" in text
    # Path-filtered diff must not include the unrelated staged file.
    assert "other.txt" not in text


@requires_git
def test_diff_staged(repo):
    (repo / "hello.txt").write_text("staged change\n")
    _run(repo, "add", "hello.txt")
    # Working-tree diff is now empty (all staged); staged diff shows the change.
    assert gitint.diff(str(repo), staged=False) == ""
    staged_text = gitint.diff(str(repo), staged=True)
    assert "+staged change" in staged_text


@requires_git
def test_diff_clean_is_empty(repo):
    assert gitint.diff(str(repo)) == ""


@requires_git
def test_diff_byte_capped(repo):
    # Write a change far larger than the 20KB cap and confirm truncation.
    big = "".join(f"line {i}\n" for i in range(20_000))
    (repo / "hello.txt").write_text(big)
    text = gitint.diff(str(repo))
    assert text.endswith(gitint._TRUNC_MARKER)
    assert len(text.encode("utf-8")) <= gitint._DIFF_MAX_BYTES + len(gitint._TRUNC_MARKER)


@requires_git
def test_last_commit(repo):
    lc = gitint.last_commit(str(repo))
    assert lc is not None
    assert lc["subject"] == "initial commit"
    assert len(lc["hash"]) == 40  # full SHA-1


@requires_git
def test_commit_all_success(repo):
    (repo / "hello.txt").write_text("modified\n")
    (repo / "added.txt").write_text("added\n")
    result = gitint.commit_all(str(repo), "second commit")
    assert result["ok"] is True
    assert len(result["commit_hash"]) == 40
    # Tree is clean again and the new commit is HEAD.
    assert gitint.is_dirty(str(repo)) is False
    assert gitint.last_commit(str(repo))["subject"] == "second commit"
    # Both the modified and the untracked file were committed.
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=str(repo), check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout
    assert "added.txt" in tracked


@requires_git
def test_commit_all_nothing_to_commit(repo):
    result = gitint.commit_all(str(repo), "no changes")
    assert result["ok"] is False
    assert "error" in result
    assert "nothing" in result["error"].lower()


@requires_git
def test_commit_all_empty_message_refused(repo):
    (repo / "hello.txt").write_text("changed\n")
    result = gitint.commit_all(str(repo), "   ")
    assert result["ok"] is False
    assert "message" in result["error"].lower()
    # Nothing should have been committed.
    assert gitint.is_dirty(str(repo)) is True


# --------------------------------------------------------------------------- #
# non-repo graceful degradation (no git invocation needed to succeed)
# --------------------------------------------------------------------------- #

def test_non_repo_is_repo_false(tmp_path):
    assert gitint.is_repo(str(tmp_path)) is False


def test_non_repo_is_dirty_false(tmp_path):
    assert gitint.is_dirty(str(tmp_path)) is False


def test_non_repo_short_status_empty(tmp_path):
    assert gitint.short_status(str(tmp_path)) == ""


def test_non_repo_current_branch_none(tmp_path):
    assert gitint.current_branch(str(tmp_path)) is None


def test_non_repo_diff_empty(tmp_path):
    assert gitint.diff(str(tmp_path)) == ""
    assert gitint.diff(str(tmp_path), path="whatever.txt", staged=True) == ""


def test_non_repo_last_commit_none(tmp_path):
    assert gitint.last_commit(str(tmp_path)) is None


def test_non_repo_commit_all_fails_cleanly(tmp_path):
    result = gitint.commit_all(str(tmp_path), "msg")
    assert result["ok"] is False
    assert "error" in result
    assert "repositor" in result["error"].lower()
