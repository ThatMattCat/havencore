"""Tests for the GitHub MCP tools (selene_agent.modules.mcp_github_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline, the happy-path wire format with explicit nulls, and the
module's error contract (in-impl errors serialize as indented ``{"error":
...}`` JSON like any success payload; an unexpected exception becomes the
compact ``{"error": str(e)}`` form — both ordinary, non-``isError`` content).
git/ripgrep and the GitHub REST API are mocked at the ``subprocess.run`` /
``requests`` layer; ``_bootstrap_clone`` is stubbed so no test touches the
network or the clone volume.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_github_tools.tools.json"


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default. Real defaults (max_results=50, state='open'), ranges, enums,
    descriptions, types, and required lists must all still match the baseline.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if prop.get("default", "sentinel") is None:
            prop.pop("default")
    return schema


def _make_server():
    from selene_agent.modules.mcp_github_tools import github_mcp_server

    # No clone/fetch in tests — __init__ must not shell out to git.
    with patch.object(github_mcp_server, "_bootstrap_clone"):
        return github_mcp_server.GitHubMCPServer()


@pytest.mark.asyncio
async def test_mcp_surface_matches_schema_baseline():
    baseline = json.loads(_FIXTURE.read_text())
    server = _make_server()
    tools = {t.name: t for t in await server.mcp.list_tools()}

    assert sorted(tools) == [t["name"] for t in baseline]  # fixture is name-sorted
    for expected in baseline:
        tool = tools[expected["name"]]
        assert tool.description == expected["description"]
        assert tool.output_schema is None  # structured_output=False: plain text results
        got = _strip_generator_noise(tool.input_schema)
        want = _strip_generator_noise(expected["inputSchema"])
        assert got == want, f"schema drift for {expected['name']}"


@pytest.mark.asyncio
async def test_search_code_happy_path_tolerates_explicit_null_glob():
    """Explicit JSON null for the optional glob must behave as "not provided"
    (LLMs send them routinely), the ripgrep invocation keeps the .git
    lockdown glob, and the result stays one indented-JSON text block."""
    from selene_agent.modules.mcp_github_tools import github_mcp_server

    server = _make_server()
    root = str(Path(github_mcp_server.GITHUB_CLONE_PATH).resolve())
    proc = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=f"{root}/services/agent/foo.py:12:hit line\n", stderr="",
    )

    with patch.object(github_mcp_server.subprocess, "run", return_value=proc) as run:
        result = await server.mcp.call_tool("github_search_code", {
            "query": "hit",
            "glob": None,
        })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["query"] == "hit"
    assert payload["glob"] is None  # null glob -> no -g filter added
    assert payload["match_count"] == 1
    assert payload["output"] == "services/agent/foo.py:12:hit line"
    assert result.content[0].text == json.dumps(payload, indent=2)

    cmd = run.call_args.args[0]
    assert "!.git/**" in cmd  # credential-store lockdown survives conversion
    assert cmd.count("-g") == 1  # no glob arg when the filter is null
    assert cmd[-2:] == ["hit", github_mcp_server.GITHUB_CLONE_PATH]


@pytest.mark.asyncio
async def test_in_impl_error_stays_indented_json():
    """Errors the impl *returns* (here: path escaping the clone root) keep the
    success serialization — indented JSON, ordinary content."""
    server = _make_server()

    result = await server.mcp.call_tool("github_read_file", {
        "path": "../../etc/passwd",
    })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert "not readable" in payload["error"]
    assert result.content[0].text == json.dumps(payload, indent=2)


@pytest.mark.asyncio
async def test_impl_exception_becomes_compact_error_payload():
    """An unexpected impl exception must surface as the *compact*
    {"error": ...} JSON (no indent) — the pre-MCPServer contract kept its
    success/error asymmetry."""
    from selene_agent.modules.mcp_github_tools import github_mcp_server

    server = _make_server()

    with patch.object(github_mcp_server.subprocess, "run", side_effect=RuntimeError("kaboom")):
        result = await server.mcp.call_tool("github_search_code", {"query": "x"})

    assert not result.is_error
    assert result.content[0].text == json.dumps({"error": "kaboom"})


@pytest.mark.asyncio
async def test_create_issue_rate_limit_enforced_before_any_api_call():
    """The per-hour deque limiter survives conversion: at the cap, the tool
    refuses without touching the GitHub API."""
    from selene_agent.modules.mcp_github_tools import github_mcp_server

    server = _make_server()
    now = time.time()
    for _ in range(github_mcp_server.GITHUB_MAX_ISSUES_PER_HOUR):
        server._issue_create_times.append(now)

    with patch.object(github_mcp_server.requests, "post") as post:
        result = await server.mcp.call_tool("github_create_issue", {
            "title": "t", "body": "b", "labels": None,
        })

    post.assert_not_called()
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert "rate limit" in payload["error"]


@pytest.mark.asyncio
async def test_create_issue_stale_rate_limit_entries_expire():
    """Timestamps older than the hour window are evicted, so the call goes
    through (and appends the provenance footer + records the new timestamp)."""
    from selene_agent.modules.mcp_github_tools import github_mcp_server

    server = _make_server()
    stale = time.time() - 3700
    for _ in range(github_mcp_server.GITHUB_MAX_ISSUES_PER_HOUR):
        server._issue_create_times.append(stale)

    resp = MagicMock(status_code=201)
    resp.json.return_value = {"number": 7, "html_url": "http://x", "title": "t"}
    with patch.object(github_mcp_server.requests, "post", return_value=resp) as post:
        result = await server.mcp.call_tool("github_create_issue", {
            "title": "t", "body": "b",
        })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload == {"success": True, "number": 7, "url": "http://x", "title": "t"}
    assert "_Filed by Selene (HavenCore assistant)_" in post.call_args.kwargs["json"]["body"]
    assert len(server._issue_create_times) == 1  # stale evicted, this call recorded


@pytest.mark.asyncio
async def test_call_tool_rejects_out_of_range_max_results():
    """The Field(ge/le) range now rejects out-of-schema values at validation;
    direct call_tool() re-raises as ToolError (over the wire: is_error=True).
    The old dispatch silently clamped instead."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()

    with pytest.raises(ToolError, match="max_results"):
        await server.mcp.call_tool("github_search_code", {
            "query": "x",
            "max_results": 500,
        })
