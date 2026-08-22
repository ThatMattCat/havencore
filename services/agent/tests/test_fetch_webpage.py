"""Tests for the native fetch_webpage tool (mcp_general_tools.fetch_tools).

The replacement for the retired upstream ``mcp-server-fetch`` server. HTTP is
mocked with ``httpx.MockTransport`` behind ``fetch_tools._make_client``; the
MCP-surface tests go through ``server.mcp.call_tool`` to pin the module
wiring and the plain-text (non-``isError``) error contract.
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from selene_agent.modules.mcp_general_tools import fetch_tools

_HTML_PAGE = """
<html>
  <head>
    <title>Example Domain</title>
    <style>body { color: red; }</style>
    <script>console.log("tracking");</script>
  </head>
  <body>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples.</p>
  </body>
</html>
"""


def _patched_client(handler):
    """Patch _make_client to serve canned responses through MockTransport."""
    return patch.object(
        fetch_tools,
        "_make_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    )


def _make_server():
    from selene_agent.modules.mcp_general_tools.mcp_server import GeneralToolsServer

    return GeneralToolsServer()


@pytest.mark.asyncio
async def test_html_happy_path_converts_to_markdown_via_mcp_surface():
    def handler(request):
        assert request.url == "https://example.com/"
        return httpx.Response(200, text=_HTML_PAGE,
                              headers={"content-type": "text/html; charset=UTF-8"})

    server = _make_server()
    with _patched_client(handler):
        result = await server.mcp.call_tool("fetch_webpage", {
            "url": "https://example.com/",
            "max_length": None,  # explicit null -> default 10000
            "start_index": None,  # explicit null -> default 0
        })

    assert not result.is_error
    text = result.content[0].text
    assert "# Example Domain" in text  # h1 -> ATX heading
    assert "illustrative examples" in text
    assert "console.log" not in text  # script stripped
    assert "color: red" not in text  # style stripped
    assert "<truncated" not in text  # short page, no paging note


@pytest.mark.asyncio
async def test_plain_text_passthrough_truncation_and_start_index_paging():
    body = "".join(f"line{i:04d}\n" for i in range(400))  # 3600 chars of text/plain

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/plain"})

    with _patched_client(handler):
        first = await fetch_tools.fetch_webpage("http://x.test/doc", max_length=1000)
    assert first.startswith("line0000")
    assert f"<truncated, {len(body) - 1000} more chars" in first
    assert "start_index=1000" in first

    with _patched_client(handler):
        second = await fetch_tools.fetch_webpage(
            "http://x.test/doc", max_length=1000, start_index=1000
        )
    assert second.startswith(body[1000:1010])
    assert "start_index=2000" in second

    with _patched_client(handler):
        past_end = await fetch_tools.fetch_webpage("http://x.test/doc", start_index=99999)
    assert "No more content available" in past_end
    assert str(len(body)) in past_end


@pytest.mark.asyncio
async def test_non_http_scheme_rejected_without_any_request():
    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("no request should be made for a bad scheme")

    server = _make_server()
    with _patched_client(handler):
        result = await server.mcp.call_tool("fetch_webpage", {
            "url": "ftp://example.com/file",
        })

    assert not result.is_error
    assert result.content[0].text == (
        "Error: only http:// and https:// URLs are supported (got: ftp://example.com/file)"
    )


@pytest.mark.asyncio
async def test_http_error_status_becomes_friendly_text():
    def handler(request):
        return httpx.Response(404, text="nope")

    server = _make_server()
    with _patched_client(handler):
        result = await server.mcp.call_tool("fetch_webpage", {
            "url": "https://example.com/missing",
        })

    assert not result.is_error
    assert result.content[0].text == (
        "Error fetching https://example.com/missing: HTTP 404 Not Found"
    )


@pytest.mark.asyncio
async def test_transport_error_becomes_friendly_text():
    def handler(request):
        raise httpx.ConnectError("[Errno -2] Name or service not known")

    with _patched_client(handler):
        result = await fetch_tools.fetch_webpage("https://no-such-host.invalid/")

    assert result.startswith("Error fetching https://no-such-host.invalid/:")
    assert "Name or service not known" in result


@pytest.mark.asyncio
async def test_binary_content_type_refused_with_informative_error():
    def handler(request):
        return httpx.Response(200, content=b"\x89PNG...",
                              headers={"content-type": "image/png"})

    with _patched_client(handler):
        result = await fetch_tools.fetch_webpage("https://example.com/logo.png")

    assert "non-text content" in result
    assert "image/png" in result


@pytest.mark.asyncio
async def test_out_of_range_max_length_rejected_by_schema():
    """Field(ge=1000, le=50000) enforces the advertised range at validation;
    direct call_tool() re-raises as ToolError (over the wire: is_error=True)."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()
    with pytest.raises(ToolError, match="max_length"):
        await server.mcp.call_tool("fetch_webpage", {
            "url": "https://example.com/",
            "max_length": 100,
        })
