"""Cross-origin access control for the agent's HTTP + WebSocket surface.

Regression tests for the wildcard-CORS hole: /api/*, /ws/* and /v1/* have no
auth gate, so with `allow_origins=["*"]` any web page a household member opened
could POST /api/chat cross-origin and drive the full tool-calling loop — and,
because WebSocket handshakes are exempt from CORS entirely, could also open
/ws/chat directly.

The absent-Origin cases below are the load-bearing part of the "zero client
changes" requirement: the ESP32 satellite firmware and the companion app's
OkHttp socket never send an Origin header, and must keep connecting.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Always in the derived default allowlist, whatever HOST_IP_ADDRESS is set to.
ALLOWED_ORIGIN = "http://localhost:6002"
EVIL_ORIGIN = "http://evil.example"


def _load_app():
    """Import the real app lazily.

    Deliberately not a module-level import: pytest imports every test module up
    front at collection time, and dragging the whole agent app (plus its
    background log threads) in that early measurably perturbs the timing of
    unrelated WS tests that race a client send against a server-side assert.
    """
    from selene_agent.selene_agent import app
    return app


@pytest.fixture
def client():
    # No `with`: the lifespan (vLLM model detection, DB pools, MCP) must not run.
    # /ws/chat bails out cleanly on a None pool, right after accepting — which is
    # exactly what the "handshake accepted" assertions need to observe.
    app = _load_app()
    app.state.session_pool = None
    return TestClient(app)


# --- HTTP CORS -------------------------------------------------------------

def test_allowed_origin_is_echoed_without_credentials(client):
    resp = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    # Nothing here is cookie-authenticated; allow_credentials must stay off.
    assert "access-control-allow-credentials" not in resp.headers


def test_disallowed_origin_gets_no_acao_header(client):
    resp = client.get("/health", headers={"Origin": EVIL_ORIGIN})
    # The request itself still runs (CORS is browser-enforced), but without an
    # Access-Control-Allow-Origin header the attacker page cannot read it.
    assert "access-control-allow-origin" not in resp.headers


def test_preflight_from_disallowed_origin_is_not_approved(client):
    resp = client.options(
        "/api/chat",
        headers={
            "Origin": EVIL_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_preflight_from_allowed_origin_keeps_dashboard_headers(client):
    resp = client.options(
        "/api/chat",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type, x-session-id",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    allow_headers = resp.headers.get("access-control-allow-headers", "").lower()
    assert "x-session-id" in allow_headers
    assert "content-type" in allow_headers
    assert "POST" in resp.headers.get("access-control-allow-methods", "")
    assert "access-control-allow-credentials" not in resp.headers


def test_allowlist_is_driven_by_the_config_var(monkeypatch):
    # Imported inside the test so the behavioural tests above still report a
    # real CORS failure (not a collection error) when the fix is reverted.
    from selene_agent.utils import config
    from selene_agent.utils.origins import install_origin_policy

    monkeypatch.setattr(config, "AGENT_CORS_ORIGINS", "http://dashboard.example, http://kiosk.example:8080")

    scoped = FastAPI()

    @scoped.get("/ping")
    async def ping():
        return {"ok": True}

    install_origin_policy(scoped)
    scoped_client = TestClient(scoped)

    for origin in ("http://dashboard.example", "http://kiosk.example:8080"):
        resp = scoped_client.get("/ping", headers={"Origin": origin})
        assert resp.headers.get("access-control-allow-origin") == origin

    # An explicit AGENT_CORS_ORIGINS replaces the HOST_IP-derived defaults.
    resp = scoped_client.get("/ping", headers={"Origin": ALLOWED_ORIGIN})
    assert "access-control-allow-origin" not in resp.headers


def test_default_origins_cover_agent_port_and_nginx(monkeypatch):
    from selene_agent.utils import config
    from selene_agent.utils.origins import allowed_origins

    monkeypatch.setattr(config, "AGENT_CORS_ORIGINS", "")
    monkeypatch.setattr(config, "HOST_IP_ADDRESS", "10.0.0.1")

    assert allowed_origins() == [
        "http://10.0.0.1:6002",
        "http://10.0.0.1",
        "http://localhost:6002",
        "http://localhost",
        "http://127.0.0.1:6002",
        "http://127.0.0.1",
    ]


# --- WebSocket handshake ---------------------------------------------------

def test_ws_handshake_with_disallowed_origin_is_rejected(client):
    # CORS never sees a WS handshake — without the guard this connects and the
    # attacker page gets the full tool-calling loop.
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/chat", headers={"Origin": EVIL_ORIGIN}):
            pass
    assert exc.value.code == 1008


def test_ws_handshake_without_origin_is_accepted(client):
    # The satellite firmware / companion app path: no Origin header at all.
    # Guard must be installed (else this proves nothing) *and* must let it in.
    from selene_agent.utils.origins import WebSocketOriginGuard
    assert any(mw.cls is WebSocketOriginGuard for mw in _load_app().user_middleware)

    with client.websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["type"] == "error"  # pool is None; handshake succeeded


def test_ws_handshake_with_allowed_origin_is_accepted(client):
    from selene_agent.utils.origins import WebSocketOriginGuard
    assert any(mw.cls is WebSocketOriginGuard for mw in _load_app().user_middleware)

    with client.websocket_connect("/ws/chat", headers={"Origin": ALLOWED_ORIGIN}) as ws:
        assert ws.receive_json()["type"] == "error"


# --- same-host auto-allow --------------------------------------------------
# TLS / hostname reverse-proxy fronts: the page's Origin names a host the agent
# can never derive from HOST_IP_ADDRESS, but it *is* the host the request was
# addressed to — the Host header proves it. TestClient sends `Host: testserver`.

def test_ws_handshake_from_own_host_origin_is_accepted(client):
    # http://testserver = origin port 80; portless Host implies 80/443.
    with client.websocket_connect(
        "/ws/chat", headers={"Origin": "http://testserver"}
    ) as ws:
        assert ws.receive_json()["type"] == "error"  # pool None; handshake OK


def test_ws_handshake_from_own_host_https_origin_is_accepted(client):
    # The real deployment shape: browser at https://<hostname> behind a proxy.
    with client.websocket_connect(
        "/ws/chat", headers={"Origin": "https://testserver"}
    ) as ws:
        assert ws.receive_json()["type"] == "error"


def test_ws_handshake_same_hostname_different_port_is_rejected(client):
    # A page served by *another* service on the same Docker host (e.g. ComfyUI
    # on :8188) must not inherit access — the match is port-exact.
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/ws/chat", headers={"Origin": "http://testserver:8188"}
        ):
            pass
    assert exc.value.code == 1008


@pytest.mark.parametrize(
    "origin,host,expected",
    [
        # The TLS-proxy front this exists for.
        ("https://selene.example", "selene.example", True),
        ("https://selene.example:443", "selene.example", True),
        ("http://selene.example", "selene.example", True),
        ("https://SELENE.EXAMPLE", "selene.example", True),
        # Direct-to-port access.
        ("http://10.0.0.134:6002", "10.0.0.134:6002", True),
        # Sibling service on the same host IP — port mismatch.
        ("http://10.0.0.134:8188", "10.0.0.134:6002", False),
        ("http://10.0.0.134:6002", "10.0.0.134", False),
        # Different host entirely.
        ("http://evil.example", "10.0.0.134:6002", False),
        # Sandboxed-iframe Origin, junk schemes, missing/garbage headers.
        ("null", "10.0.0.134:6002", False),
        ("file://selene.example", "selene.example", False),
        ("https://selene.example", None, False),
        ("https://selene.example", "selene.example:not-a-port", False),
        ("", "selene.example", False),
    ],
)
def test_is_same_host_origin_matrix(origin, host, expected):
    from selene_agent.utils.origins import is_same_host_origin
    assert is_same_host_origin(origin, host) is expected
