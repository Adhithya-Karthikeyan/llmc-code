"""Tests for llmcli.remediation — the SAFE self-healing tool-retry layer.

All tests inject an explicit ``project_files`` list so no real repo walk is
needed, and use ``tmp_path`` for the workspace ``root``. Error strings mirror
the real messages produced in ``llmcli.tools``.
"""

from __future__ import annotations

from llmcli.remediation import remediate


def _confine_err(root: str, path: str, verb: str = "read") -> dict:
    """A workspace-confinement refusal matching llmcli.tools' real text."""
    return {
        "ok": False,
        "error": (
            f"Refusing to {verb} outside the workspace root ({root}): {path}"
            " — use a workspace-RELATIVE path instead (e.g. 'subdir/name.ext');"
        ),
    }


# --- Rule 1: workspace confinement -----------------------------------------


def test_confinement_unique_basename_match(tmp_path):
    root = str(tmp_path)
    pf = ["src/app.py", "docs/readme.md", "tests/test_app.py"]
    args = {"path": "/Users/other/app.py"}
    result = _confine_err(root, "/Users/other/app.py")

    out = remediate("read_file", args, result, root=root, project_files=pf)

    assert out is not None
    new_args, explanation = out
    assert new_args["path"] == "src/app.py"
    assert "src/app.py" in explanation
    # Original args are untouched (pure).
    assert args == {"path": "/Users/other/app.py"}
    # Only the path key changed; everything else identical.
    assert set(new_args) == set(args)


def test_confinement_preserves_other_args(tmp_path):
    root = str(tmp_path)
    pf = ["src/app.py"]
    args = {"path": "/abs/app.py", "content": "hello world", "overwrite": False}
    result = _confine_err(root, "/abs/app.py", verb="write")

    out = remediate("write_file", args, result, root=root, project_files=pf)

    assert out is not None
    new_args, _ = out
    assert new_args["path"] == "src/app.py"
    # Content and other flags are copied verbatim, never invented or changed.
    assert new_args["content"] == "hello world"
    assert new_args["overwrite"] is False
    assert args["path"] == "/abs/app.py"  # original unchanged


def test_confinement_ambiguous_basename_returns_none(tmp_path):
    root = str(tmp_path)
    pf = ["src/app.py", "lib/app.py"]  # two files named app.py -> ambiguous
    args = {"path": "/Users/other/app.py"}
    result = _confine_err(root, "/Users/other/app.py")

    assert remediate("read_file", args, result, root=root, project_files=pf) is None


def test_confinement_no_basename_match_returns_none(tmp_path):
    root = str(tmp_path)
    pf = ["src/app.py"]
    args = {"path": "/Users/other/nonexistent.py"}
    result = _confine_err(root, "/Users/other/nonexistent.py")

    assert remediate("read_file", args, result, root=root, project_files=pf) is None


def test_confinement_glob_uses_root_key(tmp_path):
    root = str(tmp_path)
    pf = ["pkg/module.py"]
    args = {"pattern": "*.py", "root": "/elsewhere/module.py"}
    result = {
        "ok": False,
        "error": f"Refusing to glob outside the workspace root ({root}): /elsewhere/module.py",
    }

    out = remediate("glob", args, result, root=root, project_files=pf)
    assert out is not None
    new_args, _ = out
    assert new_args["root"] == "pkg/module.py"
    assert new_args["pattern"] == "*.py"


def test_confinement_lazy_project_files(tmp_path):
    """When project_files is None, it is computed via mentions.project_files."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("print('hi')\n", encoding="utf-8")
    root = str(tmp_path)
    args = {"path": "/Users/other/app.py"}
    result = _confine_err(root, "/Users/other/app.py")

    out = remediate("read_file", args, result, root=root)  # no project_files
    assert out is not None
    new_args, _ = out
    assert new_args["path"] == "src/app.py"


# --- Rule 2: file not found -------------------------------------------------


def test_not_found_unique_basename_match(tmp_path):
    root = str(tmp_path)
    pf = ["src/utils.py", "docs/readme.md"]
    args = {"path": "utils.py"}
    result = {"ok": False, "error": "File not found: utils.py"}

    out = remediate("read_file", args, result, root=root, project_files=pf)
    assert out is not None
    new_args, explanation = out
    assert new_args["path"] == "src/utils.py"
    assert "utils.py" in explanation
    assert args["path"] == "utils.py"  # original unchanged


def test_not_found_edit_file(tmp_path):
    root = str(tmp_path)
    pf = ["lib/helpers.py"]
    args = {"path": "helpers.py", "old": "a", "new": "b"}
    result = {"ok": False, "error": "File not found: helpers.py"}

    out = remediate("edit_file", args, result, root=root, project_files=pf)
    assert out is not None
    new_args, _ = out
    assert new_args["path"] == "lib/helpers.py"
    # edit anchors are copied unchanged, never guessed.
    assert new_args["old"] == "a"
    assert new_args["new"] == "b"


def test_not_found_ambiguous_returns_none(tmp_path):
    root = str(tmp_path)
    pf = ["a/utils.py", "b/utils.py"]
    args = {"path": "utils.py"}
    result = {"ok": False, "error": "File not found: utils.py"}

    assert remediate("read_file", args, result, root=root, project_files=pf) is None


def test_not_found_no_match_returns_none(tmp_path):
    root = str(tmp_path)
    pf = ["src/other.py"]
    args = {"path": "utils.py"}
    result = {"ok": False, "error": "File not found: utils.py"}

    assert remediate("read_file", args, result, root=root, project_files=pf) is None


# --- Refusals we must NOT auto-fix -----------------------------------------


def test_write_already_exists_returns_none(tmp_path):
    root = str(tmp_path)
    pf = ["src/app.py"]
    args = {"path": "/abs/app.py", "content": "x"}
    result = {
        "ok": False,
        "error": "File already exists: src/app.py. Pass overwrite=true to replace it, "
        "or use edit_file for a targeted change.",
    }

    # Never auto-overwrite: no correction offered.
    assert remediate("write_file", args, result, root=root, project_files=pf) is None


def test_edit_text_not_found_returns_none(tmp_path):
    root = str(tmp_path)
    pf = ["src/app.py"]
    args = {"path": "src/app.py", "old": "foo", "new": "bar"}
    result = {
        "ok": False,
        "error": "'old' string not found in file (no changes made). "
        "The text may differ in indentation, whitespace, or line endings.",
    }

    # Can't guess the missing anchor text.
    assert remediate("edit_file", args, result, root=root, project_files=pf) is None


def test_run_bash_failure_returns_none(tmp_path):
    args = {"cmd": "false"}
    result = {"ok": False, "error": "Command exited with status 1"}
    assert remediate("run_bash", args, result, root=str(tmp_path), project_files=[]) is None


def test_unrelated_error_returns_none(tmp_path):
    root = str(tmp_path)
    pf = ["src/app.py"]
    args = {"path": "app.py"}
    result = {"ok": False, "error": "Path is not a file: app.py"}
    assert remediate("read_file", args, result, root=root, project_files=pf) is None


# --- Success and robustness -------------------------------------------------


def test_successful_result_returns_none(tmp_path):
    result = {"ok": True, "result": {"path": "src/app.py", "bytes_written": 3}}
    assert remediate("read_file", {"path": "x"}, result, root=str(tmp_path),
                     project_files=["src/x"]) is None


def test_never_raises_on_malformed_inputs(tmp_path):
    root = str(tmp_path)
    # args not a dict
    assert remediate("read_file", None, {"ok": False, "error": "outside the workspace root"},
                     root=root, project_files=[]) is None
    # result not a dict
    assert remediate("read_file", {"path": "x"}, None, root=root, project_files=[]) is None
    # result dict with no error key
    assert remediate("read_file", {"path": "x"}, {"ok": False}, root=root, project_files=[]) is None
    # non-string path arg under a confinement error
    assert remediate("read_file", {"path": 123},
                     {"ok": False, "error": "outside the workspace root"},
                     root=root, project_files=[]) is None
    # tool_name not a string
    assert remediate(None, {"path": "x"}, {"ok": False, "error": "File not found: x"},
                     root=root, project_files=["a/x"]) is None
    # error not a string
    assert remediate("read_file", {"path": "x"}, {"ok": False, "error": 42},
                     root=root, project_files=[]) is None
