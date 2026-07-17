"""Image-attachment tests (llmcli/images.py + agent/repl/session wiring).

All offline/deterministic: a MockProvider, temp files, and a temp HOME so the
session round-trip never touches the real ~/.llm-cli. Covers the pure encoder
(valid bytes -> data url + mime, oversized + missing + non-image rejection), the
``text_of`` content-shape helper, the agent's STRING-vs-LIST content decision
(text-only path must stay byte-identical for the prompt cache), token-estimation
and summarization ignoring base64 blobs, the /image REPL command, and a
save/load round-trip of a session carrying multimodal list content.
"""

from __future__ import annotations

import base64

import pytest

import llmcli.repl as r
import llmcli.session as s
from llmcli.agent import Agent
from llmcli.config import Config
from llmcli.images import (
    MAX_IMAGE_BYTES,
    encode_image,
    encode_image_bytes,
    extract_image_paths,
    is_image_path,
    normalize_dropped_path,
    text_of,
)
from llmcli.providers import MockProvider

# A real 1x1 transparent PNG (so the encoder sees genuine image bytes).
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYPhfDwAChw"
    "GA60e6kgAAAABJRU5ErkJggg=="
)


# --------------------------------------------------------------------------- #
# encode_image / encode_image_bytes
# --------------------------------------------------------------------------- #

def test_encode_image_bytes_builds_data_url_part():
    part = encode_image_bytes(b"hello", "image/png")
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"hello"


def test_encode_image_valid_png(tmp_path):
    p = tmp_path / "shot.png"
    p.write_bytes(_PNG_1X1)
    part = encode_image(str(p))
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _PNG_1X1


def test_encode_image_jpg_mime(tmp_path):
    p = tmp_path / "pic.jpg"
    p.write_bytes(_PNG_1X1)  # bytes don't matter; mime is from the extension
    part = encode_image(str(p))
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_encode_image_oversized_rejected(tmp_path):
    p = tmp_path / "big.png"
    p.write_bytes(b"x" * 2048)
    with pytest.raises(ValueError, match="too large"):
        encode_image(str(p), max_bytes=1024)


def test_encode_image_missing_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        encode_image(str(tmp_path / "nope.png"))


def test_encode_image_non_image_rejected(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="not a supported image"):
        encode_image(str(p))


def test_max_image_bytes_default_matches_config():
    assert Config().max_image_bytes == MAX_IMAGE_BYTES


# --------------------------------------------------------------------------- #
# text_of: string passthrough, list extraction, None
# --------------------------------------------------------------------------- #

def test_text_of_passes_strings_through():
    assert text_of("just text") == "just text"


def test_text_of_extracts_text_parts_ignoring_images():
    content = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]
    out = text_of(content)
    assert out == "what is this"
    assert "QUJD" not in out  # the base64 blob never leaks


def test_text_of_handles_none_and_other():
    assert text_of(None) == ""
    assert text_of(123) == ""


# --------------------------------------------------------------------------- #
# agent: LIST content with images, STRING content (byte-identical) without
# --------------------------------------------------------------------------- #

def _img_part():
    return encode_image_bytes(_PNG_1X1, "image/png")


def test_agent_text_only_keeps_string_content():
    agent = Agent(provider=MockProvider(scenario="plain"), system_prompt="sys", tool_names=[])
    agent.run("hello there")
    user_msg = agent.messages[1]
    # The text-only path is byte-identical to before: a plain string, not a list.
    assert user_msg == {"role": "user", "content": "hello there"}
    assert isinstance(user_msg["content"], str)


def test_agent_builds_list_content_with_images():
    agent = Agent(provider=MockProvider(scenario="plain"), system_prompt="sys", tool_names=[])
    part = _img_part()
    agent.run("describe this", images=[part])
    content = agent.messages[1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1] == part


def test_agent_empty_images_list_stays_string():
    # An empty list is falsy -> the fast string path must still be taken.
    agent = Agent(provider=MockProvider(scenario="plain"), system_prompt="sys", tool_names=[])
    agent.run("hi", images=[])
    assert agent.messages[1]["content"] == "hi"


# --------------------------------------------------------------------------- #
# token estimation + summarization ignore base64 image blobs
# --------------------------------------------------------------------------- #

def test_estimate_tokens_ignores_image_blob():
    huge = "A" * 100_000
    list_msg = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge}"}},
        ],
    }]
    str_msg = [{"role": "user", "content": "hi"}]
    # The giant base64 string must NOT inflate the estimate.
    assert Agent._estimate_tokens(list_msg) == Agent._estimate_tokens(str_msg)


def test_serialize_for_summary_ignores_image_blob():
    huge = "Z" * 50_000
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "look here"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{huge}"}},
        ],
    }]
    out = Agent._serialize_for_summary(msgs)
    assert "look here" in out
    assert huge not in out


# --------------------------------------------------------------------------- #
# /image REPL command
# --------------------------------------------------------------------------- #

@pytest.fixture
def repl(monkeypatch):
    disk = Config(private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    monkeypatch.setattr(r, "load_config", lambda *a, **k: disk)
    monkeypatch.setattr(r, "save_config", lambda *a, **k: None)

    class _FakeMCP:
        def __init__(self, *a, **k):
            self.configs = {}

        def registry(self):
            return {}

        def shutdown_all(self):
            pass

    monkeypatch.setattr(r, "MCPManager", lambda *a, **k: _FakeMCP())
    cfg = Config(provider="mock", private=True, base_url="http://127.0.0.1:1234/v1", model="m")
    return r.Repl(config=cfg, provider=MockProvider(), auto_confirm=True)


def test_image_stage(repl, tmp_path, monkeypatch):
    monkeypatch.setattr(repl, "_save_session", lambda: None)
    p = tmp_path / "a.png"
    p.write_bytes(_PNG_1X1)
    assert repl._dispatch_slash(f"/image {p}") is True
    assert len(repl._staged_images) == 1
    label, part = repl._staged_images[0]
    assert label.startswith("a.png")
    assert part["type"] == "image_url"


def test_image_stage_multiple_then_list_then_clear(repl, tmp_path, capsys):
    a = tmp_path / "a.png"; a.write_bytes(_PNG_1X1)
    b = tmp_path / "b.png"; b.write_bytes(_PNG_1X1)
    repl._dispatch_slash(f"/image {a}")
    repl._dispatch_slash(f"/image {b}")
    assert len(repl._staged_images) == 2
    capsys.readouterr()
    repl._dispatch_slash("/image")  # list
    out = capsys.readouterr().out
    assert "a.png" in out and "b.png" in out
    repl._dispatch_slash("/image clear")
    assert repl._staged_images == []


def test_image_stage_and_send_clears_buffer(repl, tmp_path, monkeypatch):
    sent = {}

    def _fake_run(line, images=None):
        sent["line"] = line
        sent["images"] = images
        return "ok"

    monkeypatch.setattr(repl.agent, "run", _fake_run)
    monkeypatch.setattr(repl, "_save_session", lambda: None)
    p = tmp_path / "a.png"
    p.write_bytes(_PNG_1X1)
    repl._dispatch_slash(f"/image {p} what is in this picture")
    assert sent["line"] == "what is in this picture"
    assert sent["images"] and sent["images"][0]["type"] == "image_url"
    # Staged images are consumed by the send.
    assert repl._staged_images == []


def test_image_bad_path_stages_nothing(repl, tmp_path, capsys):
    repl._dispatch_slash(f"/image {tmp_path / 'missing.png'}")
    out = capsys.readouterr().out
    assert "[error]" in out
    assert repl._staged_images == []


def test_submit_attaches_staged_images_then_clears(repl, tmp_path, monkeypatch):
    seen = {}

    def _fake_run(line, images=None):
        seen["images"] = images
        return "ok"

    monkeypatch.setattr(repl.agent, "run", _fake_run)
    monkeypatch.setattr(repl, "_save_session", lambda: None)
    p = tmp_path / "a.png"
    p.write_bytes(_PNG_1X1)
    repl._dispatch_slash(f"/image {p}")
    repl._submit("now describe it")
    assert seen["images"] and len(seen["images"]) == 1
    assert repl._staged_images == []


# --------------------------------------------------------------------------- #
# session round-trip with multimodal list content
# --------------------------------------------------------------------------- #

def test_session_roundtrip_list_content(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(s.Path, "home", classmethod(lambda cls: home))
    cwd = str(tmp_path / "proj")
    part = encode_image_bytes(_PNG_1X1, "image/png")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "see this"}, part]},
        {"role": "assistant", "content": "I see it"},
    ]
    s.save_session(cwd, messages, "m", s.derive_title(messages))
    loaded = s.load_session(cwd)
    assert loaded is not None
    user_msg = loaded["messages"][1]
    assert user_msg["content"][0] == {"type": "text", "text": "see this"}
    assert user_msg["content"][1] == part


# --------------------------------------------------------------------------- #
# normalize_dropped_path: quotes, escaped spaces, file://+%20, trailing, tilde
# --------------------------------------------------------------------------- #

def test_normalize_strips_trailing_space():
    assert normalize_dropped_path("/Users/a/foo.png ") == "/Users/a/foo.png"


def test_normalize_strips_single_and_double_quotes():
    assert normalize_dropped_path("'/Users/a/my photo.png'") == "/Users/a/my photo.png"
    assert normalize_dropped_path('"/Users/a/foo.png"') == "/Users/a/foo.png"


def test_normalize_unescapes_backslash_spaces():
    assert normalize_dropped_path("/Users/a/my\\ photo.png") == "/Users/a/my photo.png"


def test_normalize_file_url_with_percent_encoding():
    assert (
        normalize_dropped_path("file:///Users/a/my%20photo.png")
        == "/Users/a/my photo.png"
    )


def test_normalize_expands_tilde():
    import os
    assert normalize_dropped_path("~/foo.png") == os.path.expanduser("~/foo.png")


def test_normalize_empty_and_whitespace_return_none():
    assert normalize_dropped_path("") is None
    assert normalize_dropped_path("   ") is None
    assert normalize_dropped_path("''") is None


# --------------------------------------------------------------------------- #
# is_image_path: existing image True, missing False, existing non-image False
# --------------------------------------------------------------------------- #

def test_is_image_path_existing_image_true(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_PNG_1X1)
    assert is_image_path(str(p)) is True


def test_is_image_path_missing_false(tmp_path):
    assert is_image_path(str(tmp_path / "nope.png")) is False


def test_is_image_path_existing_non_image_false(tmp_path):
    p = tmp_path / "notes.pdf"
    p.write_bytes(b"%PDF-1.4")
    assert is_image_path(str(p)) is False


def test_is_image_path_none_false():
    assert is_image_path(None) is False


# --------------------------------------------------------------------------- #
# extract_image_paths
# --------------------------------------------------------------------------- #

def test_extract_bare_path_only(tmp_path):
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    paths, text = extract_image_paths(str(p))
    assert paths == [str(p)]
    assert text == ""


def test_extract_path_with_trailing_space(tmp_path):
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    paths, text = extract_image_paths(f"{p} ")
    assert paths == [str(p)]
    assert text == ""


def test_extract_quoted_path_with_spaces(tmp_path):
    p = tmp_path / "my photo.png"; p.write_bytes(_PNG_1X1)
    paths, text = extract_image_paths(f"'{p}'")
    assert paths == [str(p)]
    assert text == ""


def test_extract_escaped_space_path(tmp_path):
    p = tmp_path / "my photo.png"; p.write_bytes(_PNG_1X1)
    escaped = str(p).replace(" ", "\\ ")
    paths, text = extract_image_paths(escaped)
    assert paths == [str(p)]
    assert text == ""


def test_extract_file_url(tmp_path):
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    paths, text = extract_image_paths(f"file://{p}")
    assert paths == [str(p)]
    assert text == ""


def test_extract_multiple_images(tmp_path):
    a = tmp_path / "a.png"; a.write_bytes(_PNG_1X1)
    b = tmp_path / "b.png"; b.write_bytes(_PNG_1X1)
    paths, text = extract_image_paths(f"{a} {b}")
    assert paths == [str(a), str(b)]
    assert text == ""


def test_extract_path_then_text(tmp_path):
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    paths, text = extract_image_paths(f"{p} explain the chart")
    assert paths == [str(p)]
    assert text == "explain the chart"


def test_extract_text_then_path(tmp_path):
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    paths, text = extract_image_paths(f"what is in this {p}")
    assert paths == [str(p)]
    assert text == "what is in this"


def test_extract_pure_prose_with_fake_png_stays_text():
    paths, text = extract_image_paths("does foo.png look right to you")
    assert paths == []
    # Caller uses the ORIGINAL line on the empty-paths fast path; remaining is
    # only the rejoined tokens, which here are unchanged plain words.
    assert "foo.png" in text


def test_extract_unbalanced_quote_falls_back(tmp_path):
    # A lone quote makes shlex raise; the whitespace-split fallback still runs
    # and a real image token is still detected.
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    paths, _ = extract_image_paths(f"it's {p}")
    assert paths == [str(p)]


# --------------------------------------------------------------------------- #
# REPL drag-and-drop submit behaviour (_submit_or_stage)
# --------------------------------------------------------------------------- #

def test_drop_image_no_text_stages_and_does_not_run(repl, tmp_path, monkeypatch, capsys):
    called = {"run": False}
    monkeypatch.setattr(repl.agent, "run", lambda *a, **k: called.__setitem__("run", True))
    monkeypatch.setattr(repl, "_save_session", lambda: None)
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    repl._submit_or_stage(str(p))
    assert called["run"] is False
    assert len(repl._staged_images) == 1
    out = capsys.readouterr().out
    assert "attached" in out and "a.png" in out


def test_drop_image_with_text_sends_with_image(repl, tmp_path, monkeypatch):
    sent = {}
    def _fake_run(line, images=None):
        sent["line"] = line
        sent["images"] = images
    monkeypatch.setattr(repl.agent, "run", _fake_run)
    monkeypatch.setattr(repl, "_save_session", lambda: None)
    p = tmp_path / "a.png"; p.write_bytes(_PNG_1X1)
    repl._submit_or_stage(f"{p} what is this")
    assert sent["line"] == "what is this"
    assert sent["images"] and sent["images"][0]["type"] == "image_url"
    assert repl._staged_images == []  # consumed by the send


def test_drop_combines_with_image_staged_command(repl, tmp_path, monkeypatch):
    sent = {}
    def _fake_run(line, images=None):
        sent["images"] = images
    monkeypatch.setattr(repl.agent, "run", _fake_run)
    monkeypatch.setattr(repl, "_save_session", lambda: None)
    a = tmp_path / "a.png"; a.write_bytes(_PNG_1X1)
    b = tmp_path / "b.png"; b.write_bytes(_PNG_1X1)
    repl._dispatch_slash(f"/image {a}")          # stage one via /image
    repl._submit_or_stage(f"{b} compare these")  # drop a second + ask
    assert sent["images"] and len(sent["images"]) == 2


def test_text_only_fast_path_passes_original_line(repl, monkeypatch):
    seen = {}
    monkeypatch.setattr(repl, "_submit", lambda line: seen.__setitem__("line", line))
    repl._submit_or_stage("what is the capital of France")
    assert seen["line"] == "what is the capital of France"
