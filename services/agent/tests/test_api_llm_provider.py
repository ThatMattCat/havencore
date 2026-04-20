"""Tests for /api/system/llm-provider — persistence + hot-swap semantics."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from selene_agent.api.agent import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_get_returns_current_provider_and_valid_list(client):
    now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    with patch(
        "selene_agent.api.agent.agent_state.get_llm_provider_name",
        AsyncMock(return_value="vllm"),
    ), patch(
        "selene_agent.api.agent.agent_state.get_state",
        AsyncMock(return_value=("vllm", now)),
    ), patch(
        "selene_agent.selene_agent.app", SimpleNamespace(state=SimpleNamespace(
            provider=SimpleNamespace(name="vllm", model="gpt-3.5-turbo")
        ))
    ):
        r = client.get("/api/system/llm-provider")

    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "vllm"
    assert body["model"] == "gpt-3.5-turbo"
    assert set(body["valid"]) == {"vllm", "anthropic", "openai"}
    assert body["since"].startswith("2026-04-20")


def test_post_invalid_provider_returns_400(client):
    r = client.post(
        "/api/system/llm-provider",
        json={"provider": "mystery-llm"},
    )
    assert r.status_code == 400
    assert "invalid provider" in r.json()["detail"]


def test_post_persists_and_hotswaps(client):
    now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    fake_state = SimpleNamespace(
        provider=SimpleNamespace(name="vllm", model="gpt-3.5-turbo"),
        model_name="gpt-3.5-turbo",
    )
    new_provider = SimpleNamespace(name="anthropic", model="claude-opus-4-7")

    with patch(
        "selene_agent.api.agent.agent_state.set_llm_provider_name",
        AsyncMock(return_value=now),
    ) as set_mock, patch(
        "selene_agent.selene_agent.app", SimpleNamespace(state=fake_state)
    ), patch(
        "selene_agent.providers.build_provider", return_value=new_provider
    ) as build_mock:
        r = client.post(
            "/api/system/llm-provider",
            json={"provider": "anthropic"},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-opus-4-7"
    # Persisted exactly once with the new value.
    set_mock.assert_awaited_once_with("anthropic")
    # Factory was asked to build the new provider with the known vLLM model id.
    build_mock.assert_called_once()
    assert build_mock.call_args.kwargs["vllm_model"] == "gpt-3.5-turbo"
    # And the hot-swap landed on app.state.
    assert fake_state.provider is new_provider
