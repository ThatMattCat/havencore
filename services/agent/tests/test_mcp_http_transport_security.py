"""Tests for mcp-tools' DNS-rebinding allowlists (_transport_security).

Origins are normally derived from the host allowlist (http:// + https://
per host), which can never match a browser-extension MCP client (e.g.
Island's connector sends ``Origin: chrome-extension://<id>``) — those are
admitted verbatim via MCP_HTTP_ALLOWED_ORIGINS.
"""
from __future__ import annotations

from mcp.server.transport_security import TransportSecurityMiddleware

from selene_agent.mcp_http_app import _transport_security
from selene_agent.utils import config

ISLAND_ORIGIN = "chrome-extension://pbgclfkmlfkllfbpdjpkndahdbjcgbmk"


def test_origins_derived_from_hosts(monkeypatch):
    monkeypatch.setattr(config, "HOST_IP_ADDRESS", "10.0.0.134")
    monkeypatch.setenv("MCP_HTTP_ALLOWED_HOSTS", "selene.renman.wtf")
    monkeypatch.delenv("MCP_HTTP_ALLOWED_ORIGINS", raising=False)

    settings = _transport_security()

    assert "selene.renman.wtf" in settings.allowed_hosts
    assert "https://selene.renman.wtf" in settings.allowed_origins
    assert "http://10.0.0.134:*" in settings.allowed_origins
    assert ISLAND_ORIGIN not in settings.allowed_origins


def test_extra_origins_added_verbatim(monkeypatch):
    monkeypatch.setenv(
        "MCP_HTTP_ALLOWED_ORIGINS",
        f" {ISLAND_ORIGIN} , chrome-extension://other ,,",
    )

    settings = _transport_security()

    assert ISLAND_ORIGIN in settings.allowed_origins
    assert "chrome-extension://other" in settings.allowed_origins
    # Extra origins must not leak into the Host allowlist.
    assert ISLAND_ORIGIN not in settings.allowed_hosts


def test_sdk_middleware_accepts_allowed_extension_origin(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_ALLOWED_ORIGINS", ISLAND_ORIGIN)

    middleware = TransportSecurityMiddleware(_transport_security())

    assert middleware._validate_origin(ISLAND_ORIGIN)
    assert not middleware._validate_origin("chrome-extension://unlisted")
