"""Tests for the orchestrator's companion-camera → face-identify chaining.

Covers ``who_is_in_view``: after the companion app POSTs an upload, the
orchestrator should pull the captured JPEG out of the in-process BlobStore
and POST it as multipart to the face-recognition service's ``/api/identify``
endpoint, then surface the identity to the LLM.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from selene_agent import orchestrator as orch_module
from selene_agent.api import companion as companion_module
from selene_agent.api.companion import get_blob_store
from selene_agent.orchestrator import (
    AgentOrchestrator,
    FACE_IDENTIFY_TOOLS,
    VISION_CHAINED_TOOLS,
)


@pytest.fixture(autouse=True)
def _reset_companion_state():
    companion_module.reset_blob_store_for_testing()
    yield
    companion_module.reset_blob_store_for_testing()


def _make_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="test-sid",
    )


def _make_tool_call(name: str, tool_call_id: str, args: dict[str, Any]):
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _put_blob(token: str, data: bytes, tool_call_id: str) -> None:
    """Drop a blob into the live store under a known token.

    The real BlobStore mints its token internally; tests need a deterministic
    token so they can refer to it from the mocked upload payload, so we
    bypass put() and write directly to the internal dict.
    """
    from selene_agent.api.companion import _Blob
    import time as _time

    store = get_blob_store()
    store._blobs[token] = _Blob(
        data=data,
        mime="image/jpeg",
        expires_at=_time.time() + 600,
        tool_call_id=tool_call_id,
        device_id="matts-s24",
    )


class _FakePostCtx:
    """Async context manager standing in for aiohttp's ``session.post(...)``."""

    def __init__(self, status: int, body: dict[str, Any], capture: dict):
        self._status = status
        self._body = body
        self._capture = capture

    async def __aenter__(self):
        resp = MagicMock()
        resp.status = self._status

        async def _json(content_type=None):
            return self._body

        resp.json = _json
        return resp

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, status: int, body: dict[str, Any], capture: dict):
        self._status = status
        self._body = body
        self._capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, data=None, **kwargs):
        self._capture["url"] = url
        self._capture["data"] = data
        return _FakePostCtx(self._status, self._body, self._capture)


def _patch_aiohttp(monkeypatch, status: int, body: dict[str, Any]) -> dict:
    """Swap out aiohttp.ClientSession on the orchestrator module's import."""
    capture: dict[str, Any] = {}

    def _factory(*_args, **_kwargs):
        return _FakeSession(status, body, capture)

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", _factory)
    return capture


async def test_who_is_in_view_chains_to_face_rec_recognized(monkeypatch):
    capture = _patch_aiohttp(
        monkeypatch,
        status=200,
        body={
            "found": True,
            "name": "Matt",
            "person_id": "1955219f-425a-4a6a-8487-92b4530d0eed",
            "confidence": 0.93,
            "face_count": 1,
        },
    )

    orch = _make_orchestrator()
    tool_call_id = "call_face_001"
    blob_token = "tok_face_001"
    _put_blob(blob_token, b"FAKE_JPEG_BYTES_for_face", tool_call_id)

    tc = _make_tool_call("who_is_in_view", tool_call_id, {})

    async def _post_after_delay():
        await asyncio.sleep(0.05)
        fut = companion_module._pending_uploads.get(tool_call_id)
        assert fut is not None, "orchestrator did not register a future"
        fut.set_result({
            "image_url": f"http://agent:6002/api/companion/blob/{blob_token}",
            "mime": "image/jpeg",
            "captured_at": 1234567890.0,
            "device_id": "matts-s24",
            "blob_token": blob_token,
        })

    post_task = asyncio.create_task(_post_after_delay())
    result = await orch._execute_tool_call(tc)
    await post_task

    payload = json.loads(result)
    assert payload["status"] == "captured_and_recognized"
    assert payload["found"] is True
    assert payload["name"] == "Matt"
    assert payload["confidence"] == pytest.approx(0.93)
    assert payload["image_url"].endswith(blob_token)
    assert "captured_at" in payload

    # Multipart shape: posted to /api/identify on the configured face-rec base.
    assert capture["url"].endswith("/api/identify")
    # aiohttp.FormData carries fields internally; the public surface is the
    # _fields list. Each entry is (options, headers, value).
    form = capture["data"]
    assert form is not None
    fields = list(getattr(form, "_fields", []))
    assert any(
        opts.get("name") == "file" for opts, _h, _v in fields
    ), f"file field not present in form fields: {fields}"


async def test_who_is_in_view_unrecognized_passes_through(monkeypatch):
    """face-rec returning {found: false} flows through verbatim — the LLM
    phrases the reply, not the orchestrator."""
    _patch_aiohttp(
        monkeypatch,
        status=200,
        body={"found": False, "face_count": 1, "confidence": 0.42},
    )

    orch = _make_orchestrator()
    tool_call_id = "call_face_unknown"
    blob_token = "tok_unknown"
    _put_blob(blob_token, b"BYTES", tool_call_id)

    tc = _make_tool_call("who_is_in_view", tool_call_id, {})

    async def _post():
        await asyncio.sleep(0.05)
        fut = companion_module._pending_uploads.get(tool_call_id)
        fut.set_result({
            "image_url": f"http://agent:6002/api/companion/blob/{blob_token}",
            "mime": "image/jpeg",
            "captured_at": 1234567890.0,
            "device_id": "matts-s24",
            "blob_token": blob_token,
        })

    task = asyncio.create_task(_post())
    result = await orch._execute_tool_call(tc)
    await task

    payload = json.loads(result)
    assert payload["status"] == "captured_and_recognized"
    assert payload["found"] is False
    assert "name" not in payload
    assert payload["face_count"] == 1


async def test_who_is_in_view_face_rec_error_envelope(monkeypatch):
    """face-recognition returning 5xx → structured face_identify_error result."""
    _patch_aiohttp(
        monkeypatch,
        status=500,
        body={"detail": "embedder not ready"},
    )

    orch = _make_orchestrator()
    tool_call_id = "call_face_err"
    blob_token = "tok_err"
    _put_blob(blob_token, b"BYTES", tool_call_id)

    tc = _make_tool_call("who_is_in_view", tool_call_id, {})

    async def _post():
        await asyncio.sleep(0.05)
        fut = companion_module._pending_uploads.get(tool_call_id)
        fut.set_result({
            "image_url": f"http://agent:6002/api/companion/blob/{blob_token}",
            "mime": "image/jpeg",
            "captured_at": 1234567890.0,
            "device_id": "matts-s24",
            "blob_token": blob_token,
        })

    task = asyncio.create_task(_post())
    result = await orch._execute_tool_call(tc)
    await task

    payload = json.loads(result)
    assert payload["status"] == "face_identify_error"
    assert "embedder not ready" in payload["error"]
    assert payload["image_url"].endswith(blob_token)


async def test_who_is_in_view_blob_evicted_returns_error(monkeypatch):
    """If the blob is gone before the chain reads it, surface a chain
    error rather than silently fetching over self-HTTP."""
    # No aiohttp patch — the chain must fail before any HTTP call.

    orch = _make_orchestrator()
    tool_call_id = "call_face_evicted"

    tc = _make_tool_call("who_is_in_view", tool_call_id, {})

    async def _post():
        await asyncio.sleep(0.05)
        fut = companion_module._pending_uploads.get(tool_call_id)
        fut.set_result({
            "image_url": "http://agent:6002/api/companion/blob/missing",
            "mime": "image/jpeg",
            "captured_at": 1234567890.0,
            "device_id": "matts-s24",
            "blob_token": "missing",  # never inserted into BlobStore
        })

    task = asyncio.create_task(_post())
    result = await orch._execute_tool_call(tc)
    await task

    payload = json.loads(result)
    assert payload["status"] == "face_identify_error"
    assert "blob unavailable" in payload["error"]


def test_face_identify_tools_aligned_with_other_constants():
    """FACE_IDENTIFY_TOOLS must be a subset of COMPANION_UPLOAD_TOOLS and
    disjoint from VISION_CHAINED_TOOLS — the orchestrator picks at most one
    chain per tool, so overlapping membership would silently drop one
    branch."""
    assert FACE_IDENTIFY_TOOLS <= orch_module.COMPANION_UPLOAD_TOOLS
    assert FACE_IDENTIFY_TOOLS <= orch_module.PRE_EXECUTE_DEVICE_ACTION_TOOLS
    assert FACE_IDENTIFY_TOOLS.isdisjoint(set(VISION_CHAINED_TOOLS.keys()))
