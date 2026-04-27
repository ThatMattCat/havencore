"""MQTT listener for reactive agenda items.

Subscribes to every topic declared on an enabled ``trigger_spec.source=mqtt``
agenda item, forwards inbound messages to ``AutonomyEngine.trigger_event``,
and diff-resubscribes whenever CRUD on ``agenda_items`` flips the refresh
event on the engine.

The listener uses ``paho-mqtt`` on its own thread (paho's internal reader
loop, started via ``loop_start``) and bridges into asyncio via
``loop.call_soon_threadsafe`` and an ``asyncio.Queue``.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import paho.mqtt.client as mqtt

from selene_agent.autonomy import db as autonomy_db
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

if TYPE_CHECKING:  # pragma: no cover
    from selene_agent.autonomy.engine import AutonomyEngine

logger = custom_logger.get_logger('loki')


MQTT_BROKER_HOST = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_BROKER_PORT = int(os.getenv("MQTT_PORT", "1883"))


class MqttListener:
    def __init__(self, *, engine: "AutonomyEngine") -> None:
        self.engine = engine
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._shutdown = asyncio.Event()
        self._queue: Optional[asyncio.Queue] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._refresher_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._subscriptions: Set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._backoff = 1.0

    # --- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=1024)
        self._shutdown.clear()

        client_id = getattr(config, "AUTONOMY_MQTT_CLIENT_ID", "") or "selene-autonomy"
        self._client = mqtt.Client(client_id=client_id, clean_session=True)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

        await self._connect_with_backoff()
        self._client.loop_start()

        self._consumer_task = asyncio.create_task(self._consume(), name="autonomy-mqtt-consume")
        self._refresher_task = asyncio.create_task(self._refresh_loop(), name="autonomy-mqtt-refresh")
        # Kick an initial subscribe once ``on_connect`` fires.
        self.engine.mqtt_refresh.set()

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
        for task in (self._consumer_task, self._refresher_task, self._reconnect_task):
            if task:
                task.cancel()
        self._consumer_task = None
        self._refresher_task = None
        self._reconnect_task = None
        self._client = None
        self._connected = False

    # --- status -------------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    def subscribed_topics(self) -> List[str]:
        return sorted(self._subscriptions)

    # --- paho callbacks (run on paho thread) --------------------------

    def _on_connect(self, client, userdata, flags, rc):  # noqa: D401
        self._connected = rc == 0
        if not self._connected:
            logger.error(f"[mqtt] connect failed rc={rc}")
            return
        logger.info("[mqtt] connected")
        self._backoff = 1.0
        # Re-subscribe everything on reconnect.
        if self._subscriptions:
            for topic in self._subscriptions:
                client.subscribe(topic)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning(f"[mqtt] disconnected rc={rc}; paho will auto-reconnect")

    def _on_message(self, client, userdata, msg):
        raw = msg.payload
        try:
            decoded: Any = json.loads(raw.decode("utf-8"))
        except Exception:
            try:
                decoded = raw.decode("utf-8")
            except Exception:
                decoded = raw
        if self._loop is None or self._queue is None:
            return
        payload = (msg.topic, decoded)
        try:
            self._loop.call_soon_threadsafe(self._enqueue, payload)
        except RuntimeError:
            # Event loop closed during shutdown.
            pass

    def _enqueue(self, payload: Tuple[str, Any]) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("[mqtt] queue full, dropping message")

    # --- async tasks --------------------------------------------------

    async def _connect_with_backoff(self) -> None:
        assert self._client is not None
        cap = float(getattr(config, "AUTONOMY_MQTT_RECONNECT_MAX_SEC", 60))
        while not self._shutdown.is_set():
            try:
                self._client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
                return
            except Exception as e:
                delay = min(cap, self._backoff) + random.uniform(0, 1)
                logger.warning(f"[mqtt] connect_async failed: {e}; retrying in {delay:.1f}s")
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    self._backoff = min(cap, self._backoff * 2)

    async def _consume(self) -> None:
        assert self._queue is not None
        while not self._shutdown.is_set():
            try:
                topic, payload = await self._queue.get()
            except asyncio.CancelledError:
                return
            await self._dispatch_message(topic, payload)

    async def _dispatch_message(self, topic: str, payload: Any) -> None:
        try:
            items = await autonomy_db.list_mqtt_items()
        except Exception as e:
            logger.error(f"[mqtt] failed to load agenda for dispatch: {e}")
            return
        from selene_agent.autonomy import sensor_events
        from selene_agent.autonomy.trigger_match import match  # local to avoid cycle

        # Normalize haven/<domain>/<kind> events into a SensorEvent envelope so
        # handlers can reason about zone/subject without each one re-parsing
        # the raw payload. Topics outside the schema fall back to the legacy
        # raw shape — pre-existing agenda items keep working.
        enriched = await sensor_events.normalize(topic, payload, source="mqtt")
        if enriched is None:
            event: Dict[str, Any] = {"source": "mqtt", "topic": topic, "payload": payload}
        else:
            event = enriched
        for item in items:
            spec = item.get("trigger_spec") or {}
            if spec.get("source") != "mqtt":
                continue
            if not match(spec, event):
                continue
            try:
                await self.engine.trigger_event(item["id"], source="mqtt", event=event)
            except Exception as e:
                logger.error(f"[mqtt] trigger_event failed for {item['id']}: {e}")

    async def _refresh_loop(self) -> None:
        """Diff-subscribe/unsubscribe when the engine's refresh event fires."""
        refresh_event: asyncio.Event = self.engine.mqtt_refresh
        while not self._shutdown.is_set():
            try:
                await refresh_event.wait()
            except asyncio.CancelledError:
                return
            refresh_event.clear()
            await self._resync_subscriptions()

    async def _resync_subscriptions(self) -> None:
        if self._client is None:
            return
        try:
            items = await autonomy_db.list_mqtt_items()
        except Exception as e:
            logger.error(f"[mqtt] refresh load failed: {e}")
            return
        desired: Set[str] = set()
        for item in items:
            topic = ((item.get("trigger_spec") or {}).get("match") or {}).get("topic")
            if topic:
                desired.add(topic)
        to_add = desired - self._subscriptions
        to_remove = self._subscriptions - desired
        for topic in to_add:
            try:
                self._client.subscribe(topic)
                logger.info(f"[mqtt] subscribed {topic}")
            except Exception as e:
                logger.warning(f"[mqtt] subscribe {topic} failed: {e}")
        for topic in to_remove:
            try:
                self._client.unsubscribe(topic)
                logger.info(f"[mqtt] unsubscribed {topic}")
            except Exception as e:
                logger.warning(f"[mqtt] unsubscribe {topic} failed: {e}")
        self._subscriptions = desired
