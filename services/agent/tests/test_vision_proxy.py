"""Tests for the agent's vision proxy (`selene_agent.api.vision._call_vision`).

The vision-capable vLLM is the multimodal chat model by default — a reasoning
model — so the proxy forwards `chat_template_kwargs` (enable_thinking=false by
default) and refuses an empty `content` instead of handing the caller a blank
description. aiohttp is swapped for a fake session that records the posted
body, matching the style of test_companion_face_chain.py.
"""
from __future__ import annotations

from typing import Any

import aiohttp
import pytest
from fastapi import HTTPException

from selene_agent.api import vision as vision_module


class _FakeResponse:
    def __init__(self, status: int, body: Any):
        self.status = status
        self._body = body

    async def json(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, status: int, body: Any, capture: dict):
        self._status = status
        self._body = body
        self._capture = capture

    def post(self, url, *, json=None, **kwargs):
        self._capture["url"] = url
        self._capture["json"] = json
        return _FakeResponse(self._status, self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_aiohttp(monkeypatch, status: int, body: Any) -> dict:
    capture: dict[str, Any] = {}
    monkeypatch.setattr(
        aiohttp, "ClientSession", lambda *a, **kw: _FakeSession(status, body, capture)
    )
    return capture


def _chat_response(content):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


async def test_call_vision_includes_chat_template_kwargs_when_configured(monkeypatch):
    capture = _patch_aiohttp(monkeypatch, 200, _chat_response("a porch"))
    monkeypatch.setattr(
        vision_module.config, "vision_chat_template_kwargs",
        lambda: {"enable_thinking": False},
    )
    monkeypatch.setattr(vision_module.config, "VISION_SERVED_NAME", "gpt-3.5-turbo")
    monkeypatch.setattr(vision_module.config, "VISION_API_BASE", "http://vllm:8000/v1")

    content, latency_ms, usage = await vision_module._call_vision(
        _MESSAGES, max_tokens=64, temperature=0.2
    )

    assert content == "a porch"
    assert usage["completion_tokens"] == 5
    assert capture["url"] == "http://vllm:8000/v1/chat/completions"
    body = capture["json"]
    assert body["model"] == "gpt-3.5-turbo"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_tokens"] == 64 and body["stream"] is False


async def test_call_vision_omits_chat_template_kwargs_when_helper_returns_none(monkeypatch):
    capture = _patch_aiohttp(monkeypatch, 200, _chat_response("a porch"))
    monkeypatch.setattr(vision_module.config, "vision_chat_template_kwargs", lambda: None)

    await vision_module._call_vision(_MESSAGES, max_tokens=64, temperature=0.2)

    assert "chat_template_kwargs" not in capture["json"]


@pytest.mark.parametrize("empty", [None, ""])
async def test_call_vision_raises_502_on_empty_content(monkeypatch, empty):
    _patch_aiohttp(monkeypatch, 200, _chat_response(empty))
    monkeypatch.setattr(vision_module.config, "vision_chat_template_kwargs", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await vision_module._call_vision(_MESSAGES, max_tokens=64, temperature=0.2)

    assert excinfo.value.status_code == 502
    assert "returned no content" in excinfo.value.detail
    assert "enable_thinking=false" in excinfo.value.detail


async def test_call_vision_passes_through_upstream_http_error(monkeypatch):
    _patch_aiohttp(monkeypatch, 400, {"error": "image too large"})
    monkeypatch.setattr(vision_module.config, "vision_chat_template_kwargs", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await vision_module._call_vision(_MESSAGES, max_tokens=64, temperature=0.2)

    assert excinfo.value.status_code == 400
    assert "image too large" in excinfo.value.detail


# --- config helper -----------------------------------------------------------

def test_vision_chat_template_kwargs_parses_default(monkeypatch):
    from selene_agent.utils import config

    monkeypatch.setattr(config, "VISION_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
    assert config.vision_chat_template_kwargs() == {"enable_thinking": False}


@pytest.mark.parametrize("raw", ["", "   ", "[1, 2]", '"str"', "42"])
def test_vision_chat_template_kwargs_none_for_empty_or_non_dict(monkeypatch, raw):
    from selene_agent.utils import config

    monkeypatch.setattr(config, "VISION_CHAT_TEMPLATE_KWARGS", raw)
    assert config.vision_chat_template_kwargs() is None


def test_vision_chat_template_kwargs_tolerates_invalid_json(monkeypatch):
    from selene_agent.utils import config

    monkeypatch.setattr(config, "VISION_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": fals')
    assert config.vision_chat_template_kwargs() is None  # logs, doesn't raise
