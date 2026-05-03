"""Tests for the companion-app upload endpoint and BlobStore.

Covers:
- POST /api/companion/upload happy path resolves the waiting future and
  returns a fetchable image_url + expires_at.
- GET /api/companion/blob/{token} streams the bytes back.
- POST /api/companion/upload with no waiting future returns 410.
- BlobStore TTL expiry returns 404 from the blob fetch.
"""
from __future__ import annotations

import asyncio
import time
from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from selene_agent.api import companion as companion_module
from selene_agent.api.companion import (
    BlobStore,
    register_pending_upload,
    pop_pending_upload,
    reset_blob_store_for_testing,
    router as companion_router,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    reset_blob_store_for_testing()
    yield
    reset_blob_store_for_testing()


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(companion_router, prefix="/api")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_upload_happy_path_resolves_future_and_returns_image_url(client: TestClient):
    tool_call_id = "call_happy_001"

    async def _wait_and_post():
        loop = asyncio.get_event_loop()
        fut = register_pending_upload(tool_call_id)

        # Drive the upload in a thread (TestClient is sync) so the future
        # resolves while we await it on this loop.
        def _post():
            return client.post(
                "/api/companion/upload",
                data={"tool_call_id": tool_call_id, "device_id": "matts-s24"},
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0fakejpegbytes", "image/jpeg")},
            )

        post_task = loop.run_in_executor(None, _post)
        payload = await asyncio.wait_for(fut, timeout=5)
        resp = await post_task
        return payload, resp

    payload, resp = asyncio.run(_wait_and_post())

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["image_url"].endswith("/api/companion/blob/" + body["image_url"].rsplit("/", 1)[-1])
    assert body["expires_at"] > time.time()

    assert payload["mime"] == "image/jpeg"
    assert payload["device_id"] == "matts-s24"
    assert payload["image_url"] == body["image_url"]
    assert payload["size_bytes"] == len(b"\xff\xd8\xff\xe0fakejpegbytes")

    # Pending future entry should have been cleared off as part of the
    # caller-side cleanup contract once consumed.
    pop_pending_upload(tool_call_id)  # idempotent


def test_upload_with_no_waiting_future_returns_410(client: TestClient):
    resp = client.post(
        "/api/companion/upload",
        data={"tool_call_id": "call_unknown_999"},
        files={"file": ("a.jpg", b"x", "image/jpeg")},
    )
    assert resp.status_code == 410
    assert "waiting" in resp.json()["detail"].lower()


def test_uploaded_blob_can_be_fetched_via_blob_endpoint(client: TestClient):
    tool_call_id = "call_fetch_001"
    raw = b"\x89PNG\r\n\x1a\nfakepng"

    async def _go():
        loop = asyncio.get_event_loop()
        fut = register_pending_upload(tool_call_id)

        def _post():
            return client.post(
                "/api/companion/upload",
                data={"tool_call_id": tool_call_id},
                files={"file": ("a.png", raw, "image/png")},
            )

        post_task = loop.run_in_executor(None, _post)
        payload = await asyncio.wait_for(fut, timeout=5)
        resp = await post_task
        return payload, resp

    payload, resp = asyncio.run(_go())
    assert resp.status_code == 200
    image_url = payload["image_url"]
    token = image_url.rsplit("/", 1)[-1]

    fetched = client.get(f"/api/companion/blob/{token}")
    assert fetched.status_code == 200
    assert fetched.content == raw
    assert fetched.headers["content-type"].startswith("image/png")


def test_blob_ttl_expiry_returns_404(client: TestClient, monkeypatch):
    # Replace the singleton with a 0-TTL store so we can observe expiry
    # without sleeping. ``reset_blob_store_for_testing`` rebuilds it on next
    # call, so we install our own and patch the accessor.
    short_store = BlobStore(ttl_sec=0, max_bytes=1024 * 1024)
    monkeypatch.setattr(companion_module, "_blob_store", short_store)
    monkeypatch.setattr(companion_module, "get_blob_store", lambda: short_store)

    tool_call_id = "call_ttl_001"
    raw = b"expiring-bytes"

    async def _go():
        loop = asyncio.get_event_loop()
        fut = register_pending_upload(tool_call_id)

        def _post():
            return client.post(
                "/api/companion/upload",
                data={"tool_call_id": tool_call_id},
                files={"file": ("a.jpg", raw, "image/jpeg")},
            )

        post_task = loop.run_in_executor(None, _post)
        payload = await asyncio.wait_for(fut, timeout=5)
        resp = await post_task
        return payload, resp

    payload, _resp = asyncio.run(_go())
    token = payload["image_url"].rsplit("/", 1)[-1]

    # TTL is 0s, but allow a beat for the wall clock to tick past expires_at.
    time.sleep(0.05)

    fetched = client.get(f"/api/companion/blob/{token}")
    assert fetched.status_code == 404


def test_blob_store_byte_cap_evicts_oldest():
    store = BlobStore(ttl_sec=300, max_bytes=1024)
    # Insert two 600B blobs — second one should evict the first to fit the cap.
    t1, _ = store.put(b"a" * 600, "image/jpeg", "tc1", "dev")
    t2, _ = store.put(b"b" * 600, "image/jpeg", "tc2", "dev")

    assert store.get(t1) is None  # evicted
    assert store.get(t2) is not None
    assert store.get(t2).data == b"b" * 600


def test_oversized_upload_rejected(client: TestClient, monkeypatch):
    monkeypatch.setattr(companion_module.config, "COMPANION_BLOB_MAX_BYTES", 10)
    tool_call_id = "call_oversize_001"

    async def _go():
        register_pending_upload(tool_call_id)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: client.post(
                "/api/companion/upload",
                data={"tool_call_id": tool_call_id},
                files={"file": ("a.jpg", b"this is way more than ten bytes", "image/jpeg")},
            ),
        )

    resp = asyncio.run(_go())
    assert resp.status_code == 413
    pop_pending_upload(tool_call_id)
