"""MQTT bridge for the face-recognition service.

Subscribes to ``haven/face/trigger/+`` and runs ``pipeline.process_event``
for each message. Publishes the result to ``haven/face/identified`` or
``haven/face/unknown`` (no-face / no-frames events are intentionally not
published — nothing actionable for downstream subscribers). Lifecycle
status (capturing → matching → idle) is published to ``haven/face/status``
via the pipeline's ``status_emitter`` callback.

Threading model mirrors ``selene_agent/autonomy/mqtt_listener.py``: paho's
reader runs on its own thread (``loop_start``), incoming messages are
bridged to asyncio via ``loop.call_soon_threadsafe`` + ``asyncio.Queue``.
The consumer ``await``s ``process_event`` directly, so the queue acts as
the natural single-flight serializer — only one event runs at a time,
which avoids concurrent CUDA contexts hitting the same FaceAnalysis model.

----------------------------------------------------------------------------
HA automation snippet (drop into configuration.yaml or add via the UI)
----------------------------------------------------------------------------
One template-driven automation covers every camera that follows the
``binary_sensor.<base>_person`` ↔ ``camera.<base>_fluent`` naming
convention. Adding a new camera is a one-line append to the entity_id
list.

    automation:
      - alias: "Face rec - person sensor → MQTT trigger"
        trigger:
          - platform: state
            entity_id:
              - binary_sensor.backyard_left_cam_person
              - binary_sensor.backyard_right_camera_person
              - binary_sensor.front_duo_3_person
            from: "off"
            to: "on"
        variables:
          base: >-
            {{ trigger.entity_id.replace('binary_sensor.', '')
                                 .replace('_person', '') }}
        action:
          - service: mqtt.publish
            data:
              topic: "haven/face/trigger/camera.{{ base }}_fluent"
              payload: >-
                {"source":"ha_person_sensor",
                 "sensor":"{{ trigger.entity_id }}",
                 "event_id":"{{ now().timestamp() | string }}",
                 "captured_at":"{{ now().isoformat() }}"}

The bridge mints a real UUID when ``event_id`` isn't a parseable UUID,
so the HA-side ``now().timestamp()`` value above works as-is.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import paho.mqtt.client as mqtt

import config
import pipeline
from ha_snapshot import HASnapshotError
from models import PipelineResult


logger = logging.getLogger("face-recognition.mqtt")


TRIGGER_TOPIC_PATTERN = "haven/face/trigger/+"
TRIGGER_TOPIC_PREFIX = "haven/face/trigger/"
TOPIC_IDENTIFIED = "haven/face/identified"
TOPIC_UNKNOWN = "haven/face/unknown"
TOPIC_STATUS = "haven/face/status"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_uuid(value: Any) -> Optional[uuid.UUID]:
    if value is None:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        return None


def _coerce_iso_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        # fromisoformat handles offset suffixes in 3.11+; HA emits "+00:00".
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class FaceMqttBridge:
    """paho-on-thread + asyncio.Queue bridge.

    Lifecycle:
      start() — connect with backoff, subscribe, kick the consumer task.
      stop()  — set shutdown flag, cancel tasks, disconnect paho cleanly.
    """

    def __init__(self) -> None:
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._connect_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self._backoff = 1.0

    # --- public lifecycle ---------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=256)
        self._shutdown.clear()

        # paho 2.x prefers an explicit callback API version. We use VERSION1
        # to match autonomy's listener and avoid the deprecation warning.
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
            client_id=config.MQTT_CLIENT_ID,
            clean_session=True,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        self._connect_task = asyncio.create_task(
            self._connect_loop(), name="face-mqtt-connect"
        )
        self._consumer_task = asyncio.create_task(
            self._consume(), name="face-mqtt-consume"
        )

    async def stop(self) -> None:
        self._shutdown.set()
        if self._client is not None:
            try:
                self._client.loop_stop()
            except Exception:
                pass
            try:
                self._client.disconnect()
            except Exception:
                pass
        for task in (self._consumer_task, self._connect_task):
            if task is not None:
                task.cancel()
        self._consumer_task = None
        self._connect_task = None
        self._client = None
        self._connected = False
        logger.info("MQTT bridge stopped")

    # --- introspection ------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    def subscribed_topics(self) -> list[str]:
        return [TRIGGER_TOPIC_PATTERN] if self._connected else []

    # --- paho callbacks (run on paho's reader thread) ------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            logger.error("MQTT connect failed rc=%s", rc)
            self._connected = False
            return
        logger.info("MQTT connected to %s:%d", config.MQTT_BROKER, config.MQTT_PORT)
        self._connected = True
        self._backoff = 1.0
        try:
            client.subscribe(TRIGGER_TOPIC_PATTERN, qos=0)
            logger.info("Subscribed to %s", TRIGGER_TOPIC_PATTERN)
        except Exception as e:
            logger.error("Subscribe failed: %s", e)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("MQTT disconnected rc=%s; paho will auto-reconnect", rc)

    def _on_message(self, client, userdata, msg):
        if self._loop is None or self._queue is None:
            return
        try:
            self._loop.call_soon_threadsafe(
                self._enqueue, (msg.topic, msg.payload)
            )
        except RuntimeError:
            # Loop closed during shutdown.
            pass

    def _enqueue(self, item: tuple[str, bytes]) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("MQTT queue full, dropping message on %s", item[0])

    # --- async tasks --------------------------------------------------

    async def _connect_loop(self) -> None:
        """Connect with exponential backoff; on success start paho's reader."""
        assert self._client is not None
        cap = float(config.MQTT_RECONNECT_MAX_SEC)
        while not self._shutdown.is_set():
            try:
                self._client.connect(
                    config.MQTT_BROKER, config.MQTT_PORT, keepalive=60
                )
                self._client.loop_start()
                return
            except Exception as e:
                delay = min(cap, self._backoff) + random.uniform(0, 1)
                logger.warning(
                    "MQTT connect to %s:%d failed: %s; retrying in %.1fs",
                    config.MQTT_BROKER, config.MQTT_PORT, e, delay,
                )
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
                    return  # shutdown fired during wait
                except asyncio.TimeoutError:
                    self._backoff = min(cap, self._backoff * 2)

    async def _consume(self) -> None:
        assert self._queue is not None
        while not self._shutdown.is_set():
            try:
                topic, raw = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                await self._dispatch(topic, raw)
            except Exception as e:
                # Never let a single message take down the consumer.
                logger.exception("MQTT dispatch error on %s: %s", topic, e)

    # --- dispatch -----------------------------------------------------

    async def _dispatch(self, topic: str, raw: bytes) -> None:
        if not topic.startswith(TRIGGER_TOPIC_PREFIX):
            logger.debug("Ignoring non-trigger topic: %s", topic)
            return
        camera = topic[len(TRIGGER_TOPIC_PREFIX):]
        if not camera:
            logger.warning("Trigger topic missing camera suffix: %s", topic)
            return

        payload: dict[str, Any] = {}
        try:
            decoded = json.loads(raw.decode("utf-8"))
            if isinstance(decoded, dict):
                payload = decoded
        except Exception:
            # An empty / malformed payload is fine — the topic suffix
            # carries the only required field (camera).
            pass

        event_id = _coerce_uuid(payload.get("event_id")) or uuid.uuid4()
        captured_at = _coerce_iso_dt(payload.get("captured_at")) or datetime.now(
            timezone.utc
        )
        source = payload.get("source") or "mqtt"

        logger.info(
            "trigger camera=%s event_id=%s source=%s",
            camera, event_id, source,
        )

        try:
            result = await pipeline.process_event(
                camera=camera,
                event_id=event_id,
                captured_at=captured_at,
                status_emitter=self._publish_status,
            )
        except HASnapshotError as e:
            logger.warning("HA capture failed for %s: %s", camera, e)
            return
        except Exception as e:
            logger.exception("Pipeline error for %s: %s", camera, e)
            return

        await self._publish_result(result)

    # --- publishers ---------------------------------------------------

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        if self._client is None or not self._connected:
            logger.debug("Skipping publish to %s — not connected", topic)
            return
        try:
            self._client.publish(topic, json.dumps(payload, default=str), qos=0)
        except Exception as e:
            logger.warning("Publish to %s failed: %s", topic, e)

    async def _publish_status(self, camera: str, mode: str) -> None:
        # Sync publish under the hood; wrapped in async so pipeline can await it.
        self._publish(
            TOPIC_STATUS,
            {"camera": camera, "mode": mode, "since": _utc_now_iso()},
        )

    async def _publish_result(self, result: PipelineResult) -> None:
        if result.outcome == "identified":
            self._publish(
                TOPIC_IDENTIFIED,
                {
                    "event_id": str(result.event_id),
                    "camera": result.camera,
                    "person_id": str(result.person_id) if result.person_id else None,
                    "person_name": result.person_name,
                    "confidence": result.confidence,
                    "quality_score": result.quality_score,
                    "snapshot_path": result.snapshot_path,
                    "captured_at": result.captured_at.isoformat(),
                },
            )
        elif result.outcome == "unknown":
            self._publish(
                TOPIC_UNKNOWN,
                {
                    "event_id": str(result.event_id),
                    "camera": result.camera,
                    "snapshot_path": result.snapshot_path,
                    "quality_score": result.quality_score,
                    "captured_at": result.captured_at.isoformat(),
                },
            )
        # no_face / no_frames are intentionally not published.


bridge = FaceMqttBridge()
