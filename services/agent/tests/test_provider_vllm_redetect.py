"""VLLMProvider re-detects a stale model id on 404 and retries once.

Startup model detection can fall back (backend still loading after a host
reboot); the first chat call then 404s with "The model `X` does not exist".
The provider must re-read /v1/models, swap to the served id, retry, and tell
the app via ``on_model_change``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import NotFoundError

from selene_agent.providers.vllm import VLLMProvider


def _not_found(model: str) -> NotFoundError:
    req = httpx.Request("POST", "http://vllm/v1/chat/completions")
    resp = httpx.Response(404, request=req)
    return NotFoundError(
        f"The model `{model}` does not exist.",
        response=resp,
        body={"error": {"message": f"The model `{model}` does not exist."}},
    )


def _client(served: str, stale: str, completion):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[_not_found(stale), completion]
    )
    client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(id=served)])
    )
    return client


async def test_redetects_on_404_and_retries():
    completion = SimpleNamespace(choices=[], usage=None)
    client = _client("gpt-3.5-turbo", "llama", completion)
    provider = VLLMProvider.from_client(client=client, model="llama")
    seen = []
    provider.on_model_change = seen.append

    out = await provider.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert out is completion
    assert provider.model == "gpt-3.5-turbo"
    assert seen == ["gpt-3.5-turbo"]
    calls = client.chat.completions.create.call_args_list
    assert [c.kwargs["model"] for c in calls] == ["llama", "gpt-3.5-turbo"]


async def test_reraises_when_served_model_unchanged():
    client = _client("llama", "llama", SimpleNamespace(choices=[], usage=None))
    provider = VLLMProvider.from_client(client=client, model="llama")

    with pytest.raises(NotFoundError):
        await provider.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert client.chat.completions.create.call_count == 1


async def test_reraises_when_models_list_fails():
    client = _client("gpt-3.5-turbo", "llama", SimpleNamespace(choices=[], usage=None))
    client.models.list = AsyncMock(side_effect=RuntimeError("down"))
    provider = VLLMProvider.from_client(client=client, model="llama")

    with pytest.raises(NotFoundError):
        await provider.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert provider.model == "llama"
