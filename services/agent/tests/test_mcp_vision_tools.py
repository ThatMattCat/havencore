"""Tests for the mcp_vision_tools MCP server.

The HTTP layer (`_post_json`) is mocked — these tests assert that each tool
constructs the right payload, picks the right endpoint (chokepoint vs.
direct-to-vllm), and surfaces the right shape back to the LLM. The mcp 2.0
``MCPServer`` decorator surface is covered at the bottom: schema parity
against the fixtures baseline, the happy-path wire format with explicit
nulls, and the ``{"error": ...}`` indented-JSON error contract.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from selene_agent.modules.mcp_vision_tools.server import (
    ASK_URL_ENDPOINT,
    DEFAULT_DESCRIBE_PROMPT,
    DEFAULT_IDENTIFY_PROMPT,
    DEFAULT_OCR_PROMPT,
    VisionMCPServer,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_vision_tools.tools.json"


def _ask_url_response(text: str) -> dict:
    return {"response": text, "latency_ms": 100, "usage": {}}


def _vllm_chat_response(text: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {},
    }


@pytest.fixture
def server():
    return VisionMCPServer()


# --- describe_image -------------------------------------------------------


async def test_describe_image_uses_default_prompt_when_omitted(server):
    server._post_json = AsyncMock(return_value=_ask_url_response("a cat"))
    result = await server._describe_image({"image_url": "http://x/img.jpg"})
    assert result["description"] == "a cat"
    assert result["prompt"] == DEFAULT_DESCRIBE_PROMPT
    server._post_json.assert_awaited_once()
    url, payload = server._post_json.await_args.args
    assert url == ASK_URL_ENDPOINT
    assert payload["text"] == DEFAULT_DESCRIBE_PROMPT
    assert payload["image_url"] == "http://x/img.jpg"


async def test_describe_image_passes_through_custom_prompt(server):
    server._post_json = AsyncMock(return_value=_ask_url_response("a person"))
    await server._describe_image(
        {"image_url": "http://x/img.jpg", "prompt": "who is in this image?"}
    )
    payload = server._post_json.await_args.args[1]
    assert payload["text"] == "who is in this image?"


async def test_describe_image_requires_image_url(server):
    server._post_json = AsyncMock()
    result = await server._describe_image({})
    assert "error" in result
    server._post_json.assert_not_called()


# --- describe_camera_snapshot ---------------------------------------------


async def test_describe_camera_snapshot_picks_matching_url(server):
    fake_snap = AsyncMock(
        return_value={
            "success": True,
            "urls": [
                "http://10.0.0.1/snap/front_door_20260502.jpg",
                "http://10.0.0.1/snap/backyard_20260502.jpg",
                "http://10.0.0.1/snap/garage_20260502.jpg",
            ],
        }
    )
    snapshotter = MagicMock(get_camera_snapshots=fake_snap)
    server._snapshotter = snapshotter
    server._post_json = AsyncMock(return_value=_ask_url_response("backyard scene"))

    result = await server._describe_camera_snapshot({"camera_name": "backyard"})

    assert "error" not in result
    assert "backyard" in result["image_url"]
    assert result["description"] == "backyard scene"
    payload = server._post_json.await_args.args[1]
    assert payload["image_url"] == "http://10.0.0.1/snap/backyard_20260502.jpg"


async def test_describe_camera_snapshot_token_match_fallback(server):
    fake_snap = AsyncMock(
        return_value={
            "success": True,
            "urls": ["http://10.0.0.1/snap/front_door_cam_20260502.jpg"],
        }
    )
    server._snapshotter = MagicMock(get_camera_snapshots=fake_snap)
    server._post_json = AsyncMock(return_value=_ask_url_response("ok"))

    result = await server._describe_camera_snapshot({"camera_name": "front door"})
    assert "error" not in result
    assert "front_door" in result["image_url"]


async def test_describe_camera_snapshot_no_match_returns_error(server):
    fake_snap = AsyncMock(
        return_value={
            "success": True,
            "urls": ["http://10.0.0.1/snap/garage.jpg"],
        }
    )
    server._snapshotter = MagicMock(get_camera_snapshots=fake_snap)
    server._post_json = AsyncMock()

    result = await server._describe_camera_snapshot({"camera_name": "basement"})
    assert "error" in result
    assert result["available_urls"] == ["http://10.0.0.1/snap/garage.jpg"]
    server._post_json.assert_not_called()


async def test_describe_camera_snapshot_handles_capture_failure(server):
    fake_snap = AsyncMock(
        return_value={"success": False, "error": "timeout"}
    )
    server._snapshotter = MagicMock(get_camera_snapshots=fake_snap)
    server._post_json = AsyncMock()

    result = await server._describe_camera_snapshot({"camera_name": "backyard"})
    assert result["error"] == "timeout"
    server._post_json.assert_not_called()


async def test_describe_camera_snapshot_requires_camera_name(server):
    server._post_json = AsyncMock()
    result = await server._describe_camera_snapshot({})
    assert "error" in result


# --- compare_snapshots ----------------------------------------------------


async def test_compare_snapshots_sends_both_images_in_one_call(server):
    server._post_json = AsyncMock(return_value=_vllm_chat_response("nothing changed"))
    with patch(
        "selene_agent.modules.mcp_vision_tools.server.config"
    ) as mock_config:
        mock_config.VISION_API_BASE = "http://10.0.0.1:8001/v1"
        mock_config.VISION_SERVED_NAME = "gpt-4-vision"
        result = await server._compare_snapshots(
            {
                "image_url_a": "http://x/a.jpg",
                "image_url_b": "http://x/b.jpg",
                "focus": "the porch",
            }
        )

    assert result["comparison"] == "nothing changed"
    assert result["focus"] == "the porch"
    url, payload = server._post_json.await_args.args
    assert url.endswith("/chat/completions")
    assert "10.0.0.1:8001" in url  # direct vllm-vision, not the chokepoint
    content = payload["messages"][0]["content"]
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"] == "http://x/a.jpg"
    assert image_parts[1]["image_url"]["url"] == "http://x/b.jpg"
    text_part = next(p for p in content if p["type"] == "text")
    assert "the porch" in text_part["text"]


async def test_compare_snapshots_requires_both_urls(server):
    server._post_json = AsyncMock()
    result = await server._compare_snapshots({"image_url_a": "http://x/a.jpg"})
    assert "error" in result
    server._post_json.assert_not_called()


# --- identify_object ------------------------------------------------------


async def test_identify_object_appends_hint(server):
    server._post_json = AsyncMock(return_value=_ask_url_response("a fern"))
    result = await server._identify_object(
        {"image_url": "http://x/img.jpg", "hint": "plant"}
    )
    assert result["identification"] == "a fern"
    assert result["hint"] == "plant"
    payload = server._post_json.await_args.args[1]
    assert payload["text"].startswith(DEFAULT_IDENTIFY_PROMPT)
    assert "plant" in payload["text"]


async def test_identify_object_without_hint_uses_default_only(server):
    server._post_json = AsyncMock(return_value=_ask_url_response("unknown"))
    await server._identify_object({"image_url": "http://x/img.jpg"})
    payload = server._post_json.await_args.args[1]
    assert payload["text"] == DEFAULT_IDENTIFY_PROMPT


# --- read_text_in_image ---------------------------------------------------


async def test_read_text_uses_low_temperature(server):
    server._post_json = AsyncMock(return_value=_ask_url_response("LINE 1\nLINE 2"))
    result = await server._read_text_in_image({"image_url": "http://x/receipt.jpg"})
    assert result["text"] == "LINE 1\nLINE 2"
    payload = server._post_json.await_args.args[1]
    assert payload["text"] == DEFAULT_OCR_PROMPT
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] >= 512


# --- match_camera_url helper ---------------------------------------------


def test_match_camera_url_substring():
    urls = [
        "http://x/snap/front_door.jpg",
        "http://x/snap/backyard.jpg",
    ]
    assert (
        VisionMCPServer._match_camera_url("backyard", urls)
        == "http://x/snap/backyard.jpg"
    )


def test_match_camera_url_token_split():
    urls = ["http://x/snap/front_door_cam.jpg"]
    assert (
        VisionMCPServer._match_camera_url("front door", urls)
        == "http://x/snap/front_door_cam.jpg"
    )


def test_match_camera_url_returns_none_on_miss():
    urls = ["http://x/snap/garage.jpg"]
    assert VisionMCPServer._match_camera_url("attic", urls) is None


def test_match_camera_url_empty_urls():
    assert VisionMCPServer._match_camera_url("backyard", []) is None


# --- error path: ValueError surfaces as text on _ask_url failure ----------


async def test_ask_url_propagates_http_error(server):
    server._post_json = AsyncMock(side_effect=ValueError("vision API error (502)"))
    with pytest.raises(ValueError, match="vision API error"):
        await server._ask_url("hi", "http://x/img.jpg")


# --- MCP surface (mcp 2.0 MCPServer decorator API) ------------------------


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default. Descriptions, types, and required lists must all still match
    the baseline.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if prop.get("default", "sentinel") is None:
            prop.pop("default")
    return schema


async def test_mcp_surface_matches_schema_baseline(server):
    baseline = json.loads(_FIXTURE.read_text())
    tools = {t.name: t for t in await server.mcp.list_tools()}

    assert sorted(tools) == [t["name"] for t in baseline]  # fixture is name-sorted
    for expected in baseline:
        tool = tools[expected["name"]]
        assert tool.description == expected["description"]
        assert tool.output_schema is None  # structured_output=False: plain text results
        got = _strip_generator_noise(tool.input_schema)
        want = _strip_generator_noise(expected["inputSchema"])
        assert got == want, f"schema drift for {expected['name']}"


async def test_mcp_call_tool_returns_json_text_and_tolerates_explicit_nulls(server):
    """Explicit JSON null for the optional prompt must behave as "not
    provided" (LLMs send them routinely) — the impl falls back to the default
    describe prompt — and the result stays one indented-JSON text block, this
    module's pre-MCPServer wire format."""
    server._post_json = AsyncMock(return_value=_ask_url_response("a quiet porch"))

    result = await server.mcp.call_tool("describe_image", {
        "image_url": "http://x/img.jpg",
        "prompt": None,
    })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload == {
        "image_url": "http://x/img.jpg",
        "prompt": DEFAULT_DESCRIBE_PROMPT,
        "description": "a quiet porch",
    }
    assert result.content[0].text == json.dumps(payload, indent=2)


async def test_mcp_call_tool_network_error_keeps_friendly_message(server):
    """The aiohttp.ClientError branch of the old handler survives: an
    unreachable vision service maps to the same friendly text in ordinary
    (non-isError) content."""
    exc = aiohttp.ClientConnectionError("nope")
    server._post_json = AsyncMock(side_effect=exc)

    result = await server.mcp.call_tool("read_text_in_image", {
        "image_url": "http://x/receipt.jpg",
    })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload == {"error": f"vision service unreachable: {exc}"}
    assert result.content[0].text == json.dumps(payload, indent=2)


async def test_mcp_call_tool_value_error_becomes_error_payload(server):
    """ValueError (the impls' own failure signal, e.g. a non-2xx from the
    chokepoint) surfaces as {"error": str(e)} — never a protocol error."""
    server._post_json = AsyncMock(side_effect=ValueError("vision API error (502): bad gateway"))

    result = await server.mcp.call_tool("identify_object", {
        "image_url": "http://x/img.jpg",
        "hint": None,
    })

    assert not result.is_error
    assert json.loads(result.content[0].text) == {
        "error": "vision API error (502): bad gateway"
    }


async def test_mcp_call_tool_missing_required_param_raises_tool_error(server):
    """Missing image_url now fails pydantic validation; direct call_tool()
    re-raises as ToolError (over the wire: is_error=True)."""
    from mcp.server.mcpserver.exceptions import ToolError

    server._post_json = AsyncMock()
    with pytest.raises(ToolError, match="image_url"):
        await server.mcp.call_tool("describe_image", {})
    server._post_json.assert_not_called()
