"""web_fetch: registration, SSRF guard, scheme/URL validation, HTML extraction.

All offline: SSRF checks use IP literals (no DNS) and a localhost URL that is
rejected before any network call.
"""

from __future__ import annotations

import pytest

import llmcli.tools as _tools_mod
from llmcli.tools import (
    _CGNAT_NET,
    REGISTRY,
    _WEB_MAX_REDIRECTS,
    _html_to_text,
    _ip_is_safe,
    _resolve_safe_ip,
    _web_fetch,
)


@pytest.fixture(autouse=True)
def _network_allowed(monkeypatch):
    """web_fetch is disabled in private mode; these tests exercise its network
    path, so run them with private mode OFF (restored automatically)."""
    monkeypatch.setattr(_tools_mod, "_PRIVATE", False)


def test_web_fetch_registered_and_unconfirmed():
    assert "web_fetch" in REGISTRY
    assert REGISTRY["web_fetch"].requires_confirmation is False


def test_resolve_safe_ip_blocks_private_and_loopback():
    # The fetch path uses _resolve_safe_ip directly (the _host_is_safe wrapper
    # was deleted, finding #14): a None ip means blocked.
    assert _resolve_safe_ip("127.0.0.1", 80)[0] is None
    assert _resolve_safe_ip("10.0.0.5", 80)[0] is None
    assert _resolve_safe_ip("192.168.1.1", 80)[0] is None
    assert _resolve_safe_ip("169.254.169.254", 80)[0] is None  # cloud metadata
    assert _resolve_safe_ip("", 80)[0] is None


def test_resolve_safe_ip_allows_public_ip():
    ip, why = _resolve_safe_ip("8.8.8.8", 80)
    assert ip == "8.8.8.8"
    assert why == ""


def test_web_fetch_rejects_non_http_scheme():
    assert _web_fetch({"url": "file:///etc/passwd"})["ok"] is False
    assert _web_fetch({"url": "ftp://example.com/x"})["ok"] is False


def test_web_fetch_blocks_localhost_url():
    r = _web_fetch({"url": "http://127.0.0.1:1234/v1/models"})
    assert r["ok"] is False
    assert "blocked" in r["error"]


def test_web_fetch_requires_url():
    assert _web_fetch({})["ok"] is False
    assert _web_fetch({"url": 123})["ok"] is False


def test_web_fetch_rejects_non_dict_args():
    # A bare string must return a clean error dict, not raise AttributeError.
    r = _web_fetch("http://example.com")
    assert r["ok"] is False
    assert "dict" in r["error"]


def test_ip_is_safe_blocks_cgnat_and_private():
    # RFC 6598 carrier-grade NAT is NOT flagged by ipaddress.is_private/reserved.
    assert _ip_is_safe("100.64.1.1")[0] is False
    assert _ip_is_safe("100.127.255.254")[0] is False
    assert _ip_is_safe("10.0.0.1")[0] is False
    assert _ip_is_safe("169.254.169.254")[0] is False
    assert _ip_is_safe("not-an-ip")[0] is False
    assert _ip_is_safe("8.8.8.8")[0] is True


def test_cgnat_net_membership():
    import ipaddress

    assert ipaddress.ip_address("100.100.0.1") in _CGNAT_NET
    assert ipaddress.ip_address("8.8.8.8") not in _CGNAT_NET


def test_resolve_safe_ip_blocks_loopback_literal():
    ip, why = _resolve_safe_ip("127.0.0.1", 80)
    assert ip is None
    assert "private/local" in why
    ip2, why2 = _resolve_safe_ip("", 80)
    assert ip2 is None


def test_html_to_text_returns_note_tuple():
    text, note = _html_to_text("<p>hi</p>")
    assert "hi" in text
    assert note == ""


def test_html_to_text_caps_oversize_body():
    # A body well past the extraction cap must be flagged, not OOM/hang.
    huge = "<html><body>" + ("<p>x</p>" * 200_000) + "</body></html>"
    text, note = _html_to_text(huge)
    assert isinstance(text, str)
    assert "capped" in note


def test_web_fetch_blocks_redirect_to_private(monkeypatch):
    """A 302 to a private host on the 2nd hop must be blocked (re-validated)."""
    import llmcli.tools as t

    # First hop: a public-looking host returns a redirect to a private host.
    def fake_resolve(host, port):
        if host == "evil.example":
            return "8.8.8.8", ""  # passes the guard
        if host == "internal":
            return None, "refusing to fetch a private/local address (127.0.0.1)."
        return None, "unknown"

    def fake_get(parsed, safe_ip):
        if parsed.hostname == "evil.example":
            return 302, {"location": "http://internal/secret"}, b""
        raise AssertionError("must not connect to the private host")

    monkeypatch.setattr(t, "_resolve_safe_ip", fake_resolve)
    monkeypatch.setattr(t, "_http_get_pinned", fake_get)
    r = t._web_fetch({"url": "http://evil.example/start"})
    assert r["ok"] is False
    assert "blocked" in r["error"]


def test_web_fetch_download_truncated(monkeypatch):
    import llmcli.tools as t

    monkeypatch.setattr(t, "_resolve_safe_ip", lambda h, p: ("8.8.8.8", ""))
    big = b"A" * (t._WEB_MAX_BYTES + 100)
    monkeypatch.setattr(
        t, "_http_get_pinned",
        lambda parsed, ip: (200, {"content-type": "text/plain"}, big),
    )
    r = t._web_fetch({"url": "http://example.com/big"})
    assert r["ok"] is True
    assert r["result"]["download_truncated"] is True


def test_web_fetch_html_detection_uppercase(monkeypatch):
    import llmcli.tools as t

    monkeypatch.setattr(t, "_resolve_safe_ip", lambda h, p: ("8.8.8.8", ""))
    body = b"<HTML><BODY><P>Hello</P><SCRIPT>x()</SCRIPT></BODY></HTML>"
    # Non-html content-type, but <HTML sniffing must still trigger extraction.
    monkeypatch.setattr(
        t, "_http_get_pinned",
        lambda parsed, ip: (200, {"content-type": "application/octet-stream"}, body),
    )
    r = t._web_fetch({"url": "http://example.com/x"})
    assert r["ok"] is True
    assert "Hello" in r["result"]["text"]
    assert "x()" not in r["result"]["text"]


def test_html_to_text_strips_scripts_and_styles():
    html = (
        "<html><head><style>.x{color:red}</style></head>"
        "<body><h1>Title</h1><script>evil()</script><p>Body text.</p></body></html>"
    )
    t, note = _html_to_text(html)
    assert "Title" in t and "Body text." in t
    assert "evil()" not in t and "color:red" not in t
    assert note == ""


# ----- redirect-cap branch (finding #6) -----------------------------------

def test_web_fetch_redirect_cap(monkeypatch):
    """A server that always 302s to a fresh public host must hit the hop cap and
    fail with 'too many redirects', proving the bound (not an unbounded loop)."""
    import llmcli.tools as t

    calls = {"n": 0}

    monkeypatch.setattr(t, "_resolve_safe_ip", lambda h, p: ("8.8.8.8", ""))

    def fake_get(parsed, safe_ip):
        calls["n"] += 1
        # Always redirect to a brand-new public host (relative-safe absolute URL).
        return 302, {"location": f"http://pub{calls['n']}.example/next"}, b""

    monkeypatch.setattr(t, "_http_get_pinned", fake_get)
    r = t._web_fetch({"url": "http://start.example/x"})
    assert r["ok"] is False
    assert "too many redirects" in r["error"]
    # The cap allows _WEB_MAX_REDIRECTS hops + the initial request = +1 calls.
    assert calls["n"] == _WEB_MAX_REDIRECTS + 1


def test_web_fetch_relative_redirect_resolved(monkeypatch):
    """A 302 with a RELATIVE Location must be urljoin'd against the current URL."""
    import llmcli.tools as t

    monkeypatch.setattr(t, "_resolve_safe_ip", lambda h, p: ("8.8.8.8", ""))
    seen_urls: list[str] = []

    def fake_get(parsed, safe_ip):
        seen_urls.append(parsed.geturl())
        if parsed.path == "/start":
            return 302, {"location": "/next"}, b""  # relative
        return 200, {"content-type": "text/plain"}, b"final body"

    monkeypatch.setattr(t, "_http_get_pinned", fake_get)
    r = t._web_fetch({"url": "http://host.example/start"})
    assert r["ok"] is True
    assert r["result"]["text"] == "final body"
    # The relative '/next' was resolved against the original host.
    assert "http://host.example/next" in seen_urls


def test_web_fetch_redirect_to_new_public_host_is_repinned(monkeypatch):
    """Each redirect hop must be re-resolved + re-pinned (SSRF re-check)."""
    import llmcli.tools as t

    resolved_hosts: list[str] = []

    def fake_resolve(host, port):
        resolved_hosts.append(host)
        return "8.8.8.8", ""

    def fake_get(parsed, safe_ip):
        if parsed.hostname == "first.example":
            return 302, {"location": "http://second.example/p"}, b""
        return 200, {"content-type": "text/plain"}, b"ok"

    monkeypatch.setattr(t, "_resolve_safe_ip", fake_resolve)
    monkeypatch.setattr(t, "_http_get_pinned", fake_get)
    r = t._web_fetch({"url": "http://first.example/start"})
    assert r["ok"] is True
    # Both hops were independently resolved (re-pinned per hop).
    assert resolved_hosts == ["first.example", "second.example"]
