"""Tests for the in-process AudioStore used by SpeakerNotifier."""
from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_put_returns_token_and_get_round_trips():
    from selene_agent.services.audio_store import AudioStore

    store = AudioStore(default_ttl_sec=60, max_entries=4)
    token = await store.put(b"hello", content_type="audio/mpeg")
    assert isinstance(token, str) and len(token) > 10
    result = await store.get(token)
    assert result == (b"hello", "audio/mpeg")


@pytest.mark.asyncio
async def test_get_is_single_fetch():
    from selene_agent.services.audio_store import AudioStore

    store = AudioStore()
    token = await store.put(b"once")
    assert await store.get(token) is not None
    # Second read returns None — the blob evicted after first fetch.
    assert await store.get(token) is None


@pytest.mark.asyncio
async def test_empty_data_rejected():
    from selene_agent.services.audio_store import AudioStore

    store = AudioStore()
    with pytest.raises(ValueError):
        await store.put(b"")


@pytest.mark.asyncio
async def test_ttl_expires(monkeypatch):
    from selene_agent.services import audio_store as mod

    store = mod.AudioStore(default_ttl_sec=1)
    token = await store.put(b"x")
    # Monkey-patch monotonic to jump past the TTL.
    future = [1e9]
    monkeypatch.setattr(mod.time, "monotonic", lambda: future[0])
    assert await store.get(token) is None


@pytest.mark.asyncio
async def test_max_entries_evicts_oldest():
    from selene_agent.services.audio_store import AudioStore

    store = AudioStore(default_ttl_sec=60, max_entries=2)
    t1 = await store.put(b"a")
    t2 = await store.put(b"b")
    t3 = await store.put(b"c")
    # t1 should have been evicted to make room for t3.
    assert await store.get(t1) is None
    # t2 and t3 should still be readable.
    assert (await store.get(t2))[0] == b"b"
    assert (await store.get(t3))[0] == b"c"


@pytest.mark.asyncio
async def test_unknown_token_returns_none():
    from selene_agent.services.audio_store import AudioStore

    store = AudioStore()
    assert await store.get("nope") is None
    assert await store.get("") is None
