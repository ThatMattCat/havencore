"""Tests for POST /api/conversations/{session_id}/resume.

The endpoint must return the post-hydrate orchestrator messages so /chat
can render what the model now sees, with the base system prompt filtered
out and the rolling-summary system message preserved.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from selene_agent.api import conversations as conv_api


def _build_app(orch_messages, session_id="sid-resume"):
    """Stand up a tiny FastAPI app with a fake pool that returns a stub orch."""
    fake_orch = SimpleNamespace(session_id=session_id, messages=list(orch_messages))
    pool = MagicMock()
    pool.get_or_create = AsyncMock(return_value=fake_orch)

    app = FastAPI()
    app.state.session_pool = pool
    app.include_router(conv_api.router, prefix="/api")
    return app, pool, fake_orch


def test_resume_filters_base_system_prompt_keeps_summary_system():
    messages = [
        {"role": "system", "content": "You are Selene, base prompt..."},
        {"role": "system", "content": "[Prior conversation summary]\nUser asked about lights."},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    app, _, _ = _build_app(messages, session_id="sid-resume")
    client = TestClient(app)

    r = client.post("/api/conversations/sid-resume/resume")
    assert r.status_code == 200
    body = r.json()
    # Filtered: leading base system stripped; summary system kept.
    assert body["session_id"] == "sid-resume"
    assert body["resumed"] is True
    assert body["message_count"] == 4  # unchanged: server-side count, not filtered count
    out = body["messages"]
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("[Prior conversation summary]")
    assert [m["role"] for m in out[1:]] == ["user", "assistant"]


def test_resume_strips_only_leading_base_system():
    """A session that doesn't have a rolling summary still gets its leading
    base system prompt stripped — but nothing else."""
    messages = [
        {"role": "system", "content": "You are Selene..."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    app, _, _ = _build_app(messages, session_id="sid-plain")
    client = TestClient(app)
    r = client.post("/api/conversations/sid-plain/resume")
    body = r.json()
    out = body["messages"]
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_resume_preserves_summary_system_when_first():
    """If the first message is already a [Prior conversation summary] system
    message (no base system at index 0), it must NOT be stripped."""
    messages = [
        {"role": "system", "content": "[Prior conversation summary]\nrecap"},
        {"role": "user", "content": "follow-up"},
    ]
    app, _, _ = _build_app(messages, session_id="sid-summary-only")
    client = TestClient(app)
    r = client.post("/api/conversations/sid-summary-only/resume")
    body = r.json()
    out = body["messages"]
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("[Prior conversation summary]")
    assert out[1]["role"] == "user"


def test_resume_unknown_session_reports_not_resumed():
    """Mint-on-miss path: pool returns an orch with only the system prompt.
    resumed is False; messages is filtered (system stripped → empty)."""
    messages = [
        {"role": "system", "content": "You are Selene..."},
    ]
    app, _, fake_orch = _build_app(messages, session_id="freshly-minted")
    # Different session_id requested — endpoint compares orch.session_id
    # against the path arg to set `resumed`. Simulate that by giving the
    # fake orch a different id from what we request.
    fake_orch.session_id = "freshly-minted-different"
    client = TestClient(app)
    r = client.post("/api/conversations/sid-not-stored/resume")
    body = r.json()
    assert body["resumed"] is False
    # Filter strips the base system → empty list.
    assert body["messages"] == []


def test_resume_includes_message_count():
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    app, _, _ = _build_app(messages, session_id="sid-count")
    client = TestClient(app)
    r = client.post("/api/conversations/sid-count/resume")
    body = r.json()
    assert body["message_count"] == 3
    assert len(body["messages"]) == 2  # base system filtered out


def test_filter_helper_unit():
    """Direct unit test of the helper — avoids the FastAPI overhead for
    edge cases."""
    f = conv_api._filter_messages_for_resume
    assert f([]) == []
    # Non-system first message — return as-is.
    assert f([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]
    # Base system stripped.
    out = f([{"role": "system", "content": "base"}, {"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "hi"}]
    # Summary system preserved.
    out = f([{"role": "system", "content": "[Prior conversation summary]\nx"}])
    assert out == [{"role": "system", "content": "[Prior conversation summary]\nx"}]
