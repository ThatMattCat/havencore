"""Tests for the orchestrator's companion-camera → vision chaining.

Covers ``identify_object_in_photo`` and ``read_text_from_image``: after the
companion app POSTs an upload, the orchestrator should chain to the vision
pipeline with the right prompt + parameters and return the vision response
to the LLM as the tool result.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from selene_agent import orchestrator as orch_module
from selene_agent.api import companion as companion_module
from selene_agent.api.companion import register_pending_upload
from selene_agent.orchestrator import (
    AgentOrchestrator,
    VISION_CHAINED_TOOLS,
    _build_identify_prompt,
    _build_ocr_prompt,
)


@pytest.fixture(autouse=True)
def _reset_companion_state():
    companion_module.reset_blob_store_for_testing()
    yield
    companion_module.reset_blob_store_for_testing()


def _make_orchestrator() -> AgentOrchestrator:
    orch = AgentOrchestrator(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="test-sid",
    )
    return orch


def _make_tool_call(name: str, tool_call_id: str, args: dict[str, Any]):
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


async def test_identify_object_in_photo_chains_to_vision(monkeypatch):
    captured_calls: list[dict[str, Any]] = []

    async def fake_call_vision(messages, *, max_tokens, temperature):
        captured_calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return ("A monstera deliciosa, a tropical houseplant.", 123, {})

    monkeypatch.setattr(
        "selene_agent.api.vision._call_vision", fake_call_vision
    )

    orch = _make_orchestrator()
    tool_call_id = "call_identify_001"
    tc = _make_tool_call(
        "identify_object_in_photo", tool_call_id, {"hint": "plant"}
    )

    async def _drive():
        # Simulate the companion app POSTing the upload after a tiny delay
        # so the orchestrator's wait_for actually awaits something.
        loop = asyncio.get_event_loop()

        async def _post_after_delay():
            await asyncio.sleep(0.05)
            fut = companion_module._pending_uploads.get(tool_call_id)
            assert fut is not None, "orchestrator did not register a future"
            fut.set_result({
                "image_url": "http://agent:6002/api/companion/blob/abc",
                "mime": "image/jpeg",
                "captured_at": 1234567890.0,
                "device_id": "matts-s24",
            })

        post_task = asyncio.create_task(_post_after_delay())
        result = await orch._execute_tool_call(tc)
        await post_task
        return result

    raw = await _drive()
    payload = json.loads(raw)
    assert payload["status"] == "captured_and_analyzed"
    assert payload["image_url"] == "http://agent:6002/api/companion/blob/abc"
    assert payload["identification"] == "A monstera deliciosa, a tropical houseplant."
    assert "captured_at" in payload

    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["max_tokens"] == 300
    assert call["temperature"] == 0.7
    user_content = call["messages"][0]["content"]
    text_part = next(p for p in user_content if p["type"] == "text")
    assert "Identify the primary subject" in text_part["text"]
    assert "Hint from the user about what this might be: plant" in text_part["text"]
    image_part = next(p for p in user_content if p["type"] == "image_url")
    assert image_part["image_url"]["url"] == "http://agent:6002/api/companion/blob/abc"


async def test_read_text_from_image_chains_with_ocr_prompt(monkeypatch):
    captured = {}

    async def fake_call_vision(messages, *, max_tokens, temperature):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        captured["temperature"] = temperature
        return ("MILK\n$3.49\nEGGS\n$5.99\nTOTAL: $9.48", 50, {})

    monkeypatch.setattr(
        "selene_agent.api.vision._call_vision", fake_call_vision
    )

    orch = _make_orchestrator()
    tool_call_id = "call_ocr_001"
    tc = _make_tool_call("read_text_from_image", tool_call_id, {})

    async def _post_after_delay():
        await asyncio.sleep(0.05)
        fut = companion_module._pending_uploads.get(tool_call_id)
        fut.set_result({
            "image_url": "http://agent:6002/api/companion/blob/xyz",
            "mime": "image/jpeg",
            "captured_at": 1234567890.0,
            "device_id": "matts-s24",
        })

    post_task = asyncio.create_task(_post_after_delay())
    result = await orch._execute_tool_call(tc)
    await post_task

    payload = json.loads(result)
    assert payload["status"] == "captured_and_analyzed"
    assert payload["text"].startswith("MILK")

    assert captured["max_tokens"] == 1024
    assert captured["temperature"] == 0.1
    text_part = next(
        p for p in captured["messages"][0]["content"] if p["type"] == "text"
    )
    assert "Transcribe all visible text" in text_part["text"]


async def test_take_photo_does_not_chain_to_vision(monkeypatch):
    called = {"vision_invoked": False}

    async def fake_call_vision(*a, **kw):
        called["vision_invoked"] = True
        return ("should not be called", 0, {})

    monkeypatch.setattr(
        "selene_agent.api.vision._call_vision", fake_call_vision
    )

    orch = _make_orchestrator()
    tool_call_id = "call_take_001"
    tc = _make_tool_call("take_photo", tool_call_id, {})

    async def _post_after_delay():
        await asyncio.sleep(0.05)
        fut = companion_module._pending_uploads.get(tool_call_id)
        fut.set_result({
            "image_url": "http://agent:6002/api/companion/blob/abc",
            "mime": "image/jpeg",
            "captured_at": 1234567890.0,
            "device_id": "matts-s24",
        })

    post_task = asyncio.create_task(_post_after_delay())
    result = await orch._execute_tool_call(tc)
    await post_task

    payload = json.loads(result)
    assert payload["status"] == "captured"
    assert "identification" not in payload
    assert "text" not in payload
    assert called["vision_invoked"] is False


async def test_vision_chain_error_returns_structured_error(monkeypatch):
    async def fake_call_vision(*a, **kw):
        raise RuntimeError("vllm-vision unreachable")

    monkeypatch.setattr(
        "selene_agent.api.vision._call_vision", fake_call_vision
    )

    orch = _make_orchestrator()
    tool_call_id = "call_identify_err_001"
    tc = _make_tool_call("identify_object_in_photo", tool_call_id, {})

    async def _post_after_delay():
        await asyncio.sleep(0.05)
        fut = companion_module._pending_uploads.get(tool_call_id)
        fut.set_result({
            "image_url": "http://agent:6002/api/companion/blob/abc",
            "mime": "image/jpeg",
            "captured_at": 1234567890.0,
            "device_id": "matts-s24",
        })

    post_task = asyncio.create_task(_post_after_delay())
    result = await orch._execute_tool_call(tc)
    await post_task

    payload = json.loads(result)
    assert payload["status"] == "vision_error"
    assert "vllm-vision unreachable" in payload["error"]
    assert payload["image_url"] == "http://agent:6002/api/companion/blob/abc"


def test_identify_prompt_includes_hint_when_present():
    p = _build_identify_prompt({"hint": "bird"})
    assert "Identify the primary subject" in p
    assert "Hint from the user about what this might be: bird" in p


def test_identify_prompt_skips_hint_when_blank():
    p = _build_identify_prompt({"hint": "  "})
    assert "Hint from the user" not in p


def test_ocr_prompt_is_static():
    assert _build_ocr_prompt({}) == _build_ocr_prompt({"hint": "ignored"})


def test_camera_tool_constants_aligned():
    """Catch the easy-to-forget mismatch where a tool gets added to one
    frozenset but not its siblings."""
    assert orch_module.PRE_EXECUTE_DEVICE_ACTION_TOOLS <= orch_module.DEVICE_ACTION_TOOLS
    assert orch_module.COMPANION_UPLOAD_TOOLS <= orch_module.PRE_EXECUTE_DEVICE_ACTION_TOOLS
    assert set(VISION_CHAINED_TOOLS.keys()) <= orch_module.COMPANION_UPLOAD_TOOLS
