"""Sensor-event normalization for the autonomy engine.

Camera/sensor events arriving on MQTT (face-recognition, future vehicle/LPR,
motion, doorbell) are mostly raw payloads from heterogeneous publishers. This
module is the seam where they get translated into a single ``SensorEvent``
shape that the rest of autonomy reasons over — most importantly, raw camera
entity_ids get replaced with generic *zones* (front_door, backyard, driveway)
that generalize across deployments.

Topic contract — every domain that wants its events to flow through this
normalizer publishes on::

    haven/<domain>/<kind>

Where ``domain`` is a bounded vocabulary (``face``, ``vehicle``, ``motion``,
``doorbell``, ``presence``) and ``kind`` is domain-specific (e.g. ``identified``,
``unknown``, ``detected``, ``rang``). Topics that don't match this pattern
flow through untouched, preserving backward compatibility.

Adding a new domain is a two-line change here (register a normalizer) plus
publishing from the source service. No engine / handler changes needed.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from selene_agent.autonomy import db as autonomy_db
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger("loki")


# --- Constants ------------------------------------------------------------

TOPIC_PREFIX = "haven/"
ALLOWED_DOMAINS = {"face", "vehicle", "motion", "doorbell", "presence"}

# Where the face-rec snapshot stream lives. Same default as the rest of the
# agent uses to talk to itself; signal-api / HA push fetch attachments from
# this URL, so it has to be reachable from inside the docker network.
_AGENT_BASE_URL = (
    os.getenv("AGENT_INTERNAL_BASE_URL")
    or "http://agent:6002"
).rstrip("/")

# Cache TTL fallback for when LISTEN/NOTIFY isn't running (tests, dev).
_CACHE_TTL_SEC = 60


# --- Data shape -----------------------------------------------------------

@dataclass
class SensorEvent:
    """Canonical event flowing into ``engine.trigger_event``.

    Handlers see this dict in ``item._trigger_event``. ``raw`` always holds
    the unmodified original payload + topic so deterministic ``trigger_match``
    rules continue to work even after normalization.
    """

    domain: str
    kind: str
    event_id: Optional[str] = None
    camera_entity: Optional[str] = None
    zone: Optional[str] = None
    zone_label: Optional[str] = None
    subject: Optional[Dict[str, Any]] = None
    snapshot_url: Optional[str] = None
    captured_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_event_dict(self, *, source: str, topic: str) -> Dict[str, Any]:
        """Wrap the SensorEvent in the dict shape engine.trigger_event expects.

        ``source`` and ``topic`` mirror what the existing trigger_match matcher
        reads, so the matcher keeps working unchanged.
        """
        return {
            "source": source,
            "topic": topic,
            "payload": self.raw,
            "sensor_event": asdict(self),
        }


# --- Zone cache -----------------------------------------------------------

class _ZoneCache:
    """In-memory mirror of the ``camera_zones`` table.

    Refresh strategy:
      1. Fully reload on startup (``refresh()``).
      2. On NOTIFY ``camera_zones_ch`` from Postgres, reload the changed row.
      3. As a belt-and-braces backup, refresh fully every ``_CACHE_TTL_SEC``
         seconds — covers the case where the LISTEN connection drops.
    """

    def __init__(self) -> None:
        self._by_entity: Dict[str, Tuple[str, Optional[str]]] = {}
        self._loaded_at: float = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        async with self._lock:
            try:
                rows = await autonomy_db.list_camera_zones()
            except Exception as e:
                logger.warning(f"[sensor_events] zone refresh failed: {e}")
                return
            new = {
                r["camera_entity"]: (r["zone"], r.get("zone_label"))
                for r in rows
            }
            self._by_entity = new
            self._loaded_at = time.monotonic()

    async def lookup(self, camera_entity: str) -> Tuple[Optional[str], Optional[str]]:
        if not camera_entity:
            return None, None
        if (time.monotonic() - self._loaded_at) > _CACHE_TTL_SEC:
            await self.refresh()
        zone_tuple = self._by_entity.get(camera_entity)
        if zone_tuple:
            return zone_tuple
        # Some publishers send bare slugs ("front_door"); try a forgiving match
        # against the ``camera.<x>`` HA entity convention.
        if not camera_entity.startswith("camera."):
            zone_tuple = self._by_entity.get(f"camera.{camera_entity}")
            if zone_tuple:
                return zone_tuple
        return None, None

    def set_for_test(self, mapping: Dict[str, Tuple[str, Optional[str]]]) -> None:
        self._by_entity = dict(mapping)
        self._loaded_at = time.monotonic()


_zone_cache = _ZoneCache()


def get_zone_cache() -> _ZoneCache:
    return _zone_cache


async def start_zone_listener(shutdown: asyncio.Event) -> None:
    """Subscribe to the ``camera_zones_ch`` NOTIFY channel and refresh the
    cache on each notification. Reconnects with exponential backoff if the
    underlying connection drops; falls back to the TTL refresh in lookup()
    while disconnected.
    """
    await _zone_cache.refresh()
    from selene_agent.utils.conversation_db import conversation_db

    backoff = 1.0
    while not shutdown.is_set():
        pool = conversation_db.pool
        if pool is None:
            await asyncio.sleep(2)
            continue
        try:
            async with pool.acquire() as conn:
                async def _on_notify(conn_, pid, channel, payload):
                    # Cheap full refresh — table is tiny, simpler than diffing.
                    asyncio.create_task(_zone_cache.refresh())
                await conn.add_listener("camera_zones_ch", _on_notify)
                backoff = 1.0
                # Hold the connection until shutdown; conn is freed on exit.
                while not shutdown.is_set():
                    await asyncio.sleep(5)
                await conn.remove_listener("camera_zones_ch", _on_notify)
                return
        except Exception as e:
            logger.warning(f"[sensor_events] LISTEN camera_zones_ch failed: {e}")
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                backoff = min(60.0, backoff * 2)


# --- Topic parsing --------------------------------------------------------

def parse_topic(topic: str) -> Optional[Tuple[str, str]]:
    """Return (domain, kind) if topic matches ``haven/<domain>/<kind>``,
    else None. Sub-paths like ``haven/face/trigger/<camera>`` are ignored —
    triggers are inputs, not outputs.
    """
    if not topic.startswith(TOPIC_PREFIX):
        return None
    rest = topic[len(TOPIC_PREFIX):]
    parts = rest.split("/")
    if len(parts) != 2:
        return None
    domain, kind = parts
    if domain not in ALLOWED_DOMAINS:
        return None
    return domain, kind


# --- Normalizers ----------------------------------------------------------

Normalizer = Callable[[str, Dict[str, Any]], Awaitable[SensorEvent]]


async def _normalize_face(kind: str, payload: Dict[str, Any]) -> SensorEvent:
    """Translate face-recognition's ``haven/face/{identified,unknown}`` payload.

    The face-rec MQTT publish lives in
    ``services/face-recognition/app/mqtt_bridge.py:_publish_result``; this
    normalizer intentionally stays loose about missing fields so a partial
    payload still produces a usable event.
    """
    camera = payload.get("camera") or ""
    zone, zone_label = await _zone_cache.lookup(camera)

    subject: Optional[Dict[str, Any]] = None
    if kind == "identified":
        subject = {
            "type": "person",
            "identity": payload.get("person_name"),
            "person_id": payload.get("person_id"),
            "confidence": _float_or_none(payload.get("confidence")),
            "quality": _float_or_none(payload.get("quality_score")),
        }
    elif kind == "unknown":
        subject = {
            "type": "person",
            "identity": None,
            "person_id": None,
            "confidence": None,
            "quality": _float_or_none(payload.get("quality_score")),
        }
    elif kind == "no_face":
        # Person sensor tripped + frames captured but no face cleared the
        # quality floor. Could be a hidden face, bad angle, or wildlife —
        # the LLM (and, when wired, vision AI) decide on context.
        subject = {
            "type": "unknown",
            "identity": None,
            "person_id": None,
            "confidence": None,
            "quality": None,
            "no_face": True,
        }

    snapshot_url: Optional[str] = None
    detection_id = payload.get("detection_id")
    if detection_id:
        snapshot_url = (
            f"{_AGENT_BASE_URL}/api/face/detections/{detection_id}/snapshot"
        )

    return SensorEvent(
        domain="face",
        kind=kind,
        event_id=str(payload.get("event_id")) if payload.get("event_id") else None,
        camera_entity=camera or None,
        zone=zone,
        zone_label=zone_label,
        subject=subject,
        snapshot_url=snapshot_url,
        captured_at=payload.get("captured_at"),
        raw=dict(payload),
    )


async def _normalize_passthrough(domain: str, kind: str, payload: Dict[str, Any]) -> SensorEvent:
    """Default normalizer for domains that don't have a dedicated translator
    yet (vehicle, motion, doorbell). Preserves the payload verbatim and tries
    to enrich zone if the payload includes a ``camera`` field.
    """
    camera = payload.get("camera") if isinstance(payload, dict) else None
    zone, zone_label = await _zone_cache.lookup(camera or "")
    return SensorEvent(
        domain=domain,
        kind=kind,
        event_id=str(payload.get("event_id")) if isinstance(payload, dict) and payload.get("event_id") else None,
        camera_entity=camera or None,
        zone=zone,
        zone_label=zone_label,
        subject=None,
        snapshot_url=None,
        captured_at=payload.get("captured_at") if isinstance(payload, dict) else None,
        raw=payload if isinstance(payload, dict) else {"value": payload},
    )


async def _dispatch(domain: str, kind: str, payload: Any) -> SensorEvent:
    if not isinstance(payload, dict):
        # Non-JSON payload — wrap it so handlers always get a dict to read.
        payload = {"value": payload}
    if domain == "face":
        return await _normalize_face(kind, payload)
    return await _normalize_passthrough(domain, kind, payload)


# --- Public entrypoint ----------------------------------------------------

async def normalize(
    topic: str, payload: Any, *, source: str = "mqtt"
) -> Optional[Dict[str, Any]]:
    """If ``topic`` matches the haven sensor schema, return an enriched event
    dict for ``engine.trigger_event``. Otherwise return None and the caller
    falls back to the legacy raw event shape.
    """
    parsed = parse_topic(topic)
    if parsed is None:
        return None
    domain, kind = parsed
    try:
        event = await _dispatch(domain, kind, payload)
    except Exception as e:
        logger.warning(
            f"[sensor_events] normalize failed for {topic}: {e}; falling back to raw"
        )
        return None
    return event.to_event_dict(source=source, topic=topic)


# --- Helpers --------------------------------------------------------------

def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
