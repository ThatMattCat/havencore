"""AutonomyEngine — dispatcher loop and run orchestration.

Runs as an asyncio background task started from the FastAPI lifespan.
Owns the agenda_items/autonomy_runs lifecycle, kill-switch, rate limiting,
per-signature cooldowns, and (v3) reactive event dispatch + quiet hours.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from selene_agent.autonomy import db as autonomy_db
from selene_agent.autonomy import quiet_hours as quiet_hours_mod
from selene_agent.autonomy import schedule
from selene_agent.autonomy import trigger_match
from selene_agent.autonomy.event_rate_limit import limiter as event_rate_limiter
from selene_agent.autonomy.handlers import act as act_handler
from selene_agent.autonomy.handlers import anomaly as anomaly_handler
from selene_agent.autonomy.handlers import briefing as briefing_handler
from selene_agent.autonomy.handlers import memory_review as memory_review_handler
from selene_agent.autonomy.handlers import reminder as reminder_handler
from selene_agent.autonomy.handlers import routine as routine_handler
from selene_agent.autonomy.handlers import watch as watch_handler
from selene_agent.autonomy.handlers import watch_llm as watch_llm_handler
from selene_agent.autonomy.notifiers import (
    HAPushNotifier,
    NullNotifier,
    SignalNotifier,
    SpeakerNotifier,
)
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


_SEVERITY_RANK = {"none": 0, "low": 1, "med": 2, "high": 3}


def _severity_escalated(prev: Optional[str], current: str) -> bool:
    return _SEVERITY_RANK.get(current or "none", 0) > _SEVERITY_RANK.get(prev or "none", 0)


_HANDLERS = {
    "briefing": briefing_handler.handle,
    "anomaly_sweep": anomaly_handler.handle,
    "memory_review": memory_review_handler.handle,
    "reminder": reminder_handler.handle,
    "watch": watch_handler.handle,
    "watch_llm": watch_llm_handler.handle,
    "routine": routine_handler.handle,
    "act": act_handler.handle,
}


class AutonomyEngine:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        mcp_manager: MCPClientManager,
        model_name: str,
        base_tools: List[Dict[str, Any]],
    ):
        self.client = client
        self.mcp_manager = mcp_manager
        self.model_name = model_name
        self.base_tools = base_tools

        self._task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        self.paused: bool = False
        self.started_at: Optional[datetime] = None
        self.last_dispatch_at: Optional[datetime] = None
        self._running_items: set[str] = set()
        self._mqtt_listener = None  # wired in start() when enabled
        self.mqtt_refresh = asyncio.Event()

    # --- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        await autonomy_db.ensure_schema()
        await autonomy_db.ensure_default_agenda()
        self.started_at = datetime.now(timezone.utc)
        self._shutdown.clear()
        await self._start_mqtt_listener()
        self._task = asyncio.create_task(self._loop(), name="autonomy-engine")
        logger.info("AutonomyEngine started")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._shutdown.set()
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except asyncio.TimeoutError:
            self._task.cancel()
            logger.warning("AutonomyEngine stop: task did not exit cleanly, cancelled")
        self._task = None
        await self._stop_mqtt_listener()
        logger.info("AutonomyEngine stopped")

    async def _start_mqtt_listener(self) -> None:
        if not getattr(config, "AUTONOMY_MQTT_ENABLED", False):
            return
        try:
            from selene_agent.autonomy.mqtt_listener import MqttListener
        except Exception as e:
            logger.warning(f"[engine] MQTT listener unavailable: {e}")
            return
        self._mqtt_listener = MqttListener(engine=self)
        try:
            await self._mqtt_listener.start()
        except Exception as e:
            logger.error(f"[engine] MQTT listener failed to start: {e}")
            self._mqtt_listener = None

    async def _stop_mqtt_listener(self) -> None:
        if self._mqtt_listener is None:
            return
        try:
            await self._mqtt_listener.stop()
        except Exception as e:
            logger.warning(f"[engine] MQTT listener stop error: {e}")
        self._mqtt_listener = None

    def notify_agenda_changed(self) -> None:
        """Invoked by the REST layer on agenda CRUD so the MQTT listener
        can diff-subscribe/unsubscribe without a poll."""
        self.mqtt_refresh.set()

    async def _loop(self) -> None:
        interval = max(5, int(config.AUTONOMY_DISPATCH_INTERVAL_SECONDS))
        while not self._shutdown.is_set():
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"AutonomyEngine tick error: {e}")
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # --- status / control ----------------------------------------------

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def is_paused(self) -> bool:
        return self.paused or not config.AUTONOMY_ENABLED

    def pause(self) -> None:
        self.paused = True
        logger.info("AutonomyEngine paused")

    def resume(self) -> None:
        self.paused = False
        logger.info("AutonomyEngine resumed")

    async def status(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        runs_last_hour = await autonomy_db.count_runs_since(now - timedelta(hours=1))
        items = await autonomy_db.list_all_items()
        next_items = sorted(
            [i for i in items if i.get("enabled") and i.get("next_fire_at")],
            key=lambda i: i["next_fire_at"],
        )[:5]
        deferred = await autonomy_db.count_deferred_runs()
        return {
            "running": self.is_running(),
            "paused": self.is_paused(),
            "kill_switch_env": not config.AUTONOMY_ENABLED,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_dispatch_at": self.last_dispatch_at.isoformat() if self.last_dispatch_at else None,
            "runs_last_hour": runs_last_hour,
            "deferred_runs_pending": deferred,
            "rate_limit_per_hour": config.AUTONOMY_MAX_RUNS_PER_HOUR,
            "timezone": config.CURRENT_TIMEZONE or "UTC",
            "mqtt_connected": bool(self._mqtt_listener and self._mqtt_listener.is_connected()),
            "subscribed_topics": (
                self._mqtt_listener.subscribed_topics() if self._mqtt_listener else []
            ),
            "next_due": [
                {
                    "id": i["id"],
                    "kind": i["kind"],
                    "next_fire_at": i["next_fire_at"].isoformat() if i.get("next_fire_at") else None,
                }
                for i in next_items
            ],
        }

    # --- dispatch -------------------------------------------------------

    async def _tick(self) -> None:
        self.last_dispatch_at = datetime.now(timezone.utc)
        if self.is_paused():
            return
        due = await autonomy_db.list_due_items(self.last_dispatch_at)
        for item in due:
            if item["id"] in self._running_items:
                continue
            asyncio.create_task(self._fire_item(item, manual=False, trigger_source="cron"))

        # Deferred-run sweep (quiet-hours 'defer' placeholders due to fire).
        try:
            due_deferred = await autonomy_db.list_scheduled_runs_due(self.last_dispatch_at)
        except Exception as e:
            logger.error(f"[engine] deferred sweep failed: {e}")
            due_deferred = []
        for run_row in due_deferred:
            asyncio.create_task(self._dispatch_deferred(run_row))

        # Confirmation-timeout sweep (act-tier parked runs past their deadline).
        await self._sweep_confirmation_timeouts(self.last_dispatch_at)

    async def _dispatch_deferred(self, run_row: Dict[str, Any]) -> None:
        claimed = await autonomy_db.claim_scheduled_run(run_row["id"])
        if claimed is None:
            return  # another tick won
        item_id = claimed["agenda_item_id"]
        if not item_id:
            return
        item = await autonomy_db.get_item(item_id)
        if item is None or not item.get("enabled"):
            return
        await self._fire_item(
            item,
            manual=False,
            trigger_source=claimed.get("trigger_source") or "cron",
            trigger_event=claimed.get("trigger_event"),
            bypass_quiet=False,
        )

    async def trigger(self, item_id: str, *, bypass_quiet: bool = False) -> Dict[str, Any]:
        """Manually fire an agenda item regardless of schedule."""
        item = await autonomy_db.get_item(item_id)
        if item is None:
            return {"status": "not_found"}
        if item["id"] in self._running_items:
            return {"status": "already_running"}
        return await self._fire_item(
            item, manual=True, trigger_source="manual", bypass_quiet=bypass_quiet
        )

    async def trigger_event(
        self,
        item_id: str,
        *,
        source: str,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fire an agenda item in response to a reactive event (webhook/mqtt).

        Subset-matches the event against the item's ``trigger_spec`` first —
        the LLM never decides whether an inbound event 'counts'. Applies the
        per-item event rate limit before the global rate-limit gate.
        """
        item = await autonomy_db.get_item(item_id)
        if item is None:
            return {"status": "not_found"}
        if item["id"] in self._running_items:
            return {"status": "already_running"}

        # Stamp source on the event so the matcher can check.
        enriched = {**event, "source": source}
        spec = item.get("trigger_spec")
        if not spec or not trigger_match.match(spec, enriched):
            await autonomy_db.insert_run({
                "agenda_item_id": item_id,
                "kind": item["kind"],
                "triggered_at": datetime.now(timezone.utc),
                "completed_at": datetime.now(timezone.utc),
                "status": "skipped_trigger_mismatch",
                "summary": "event did not match trigger_spec",
                "trigger_source": source,
                "trigger_event": enriched,
            })
            return {"status": "skipped_trigger_mismatch"}

        rate_spec = (item.get("config") or {}).get("event_rate_limit") or getattr(
            config, "AUTONOMY_DEFAULT_EVENT_RATE_LIMIT", None
        )
        if not event_rate_limiter.try_consume(item_id, rate_spec):
            await autonomy_db.insert_run({
                "agenda_item_id": item_id,
                "kind": item["kind"],
                "triggered_at": datetime.now(timezone.utc),
                "completed_at": datetime.now(timezone.utc),
                "status": "rate_limited",
                "summary": f"per-item event rate limit ({rate_spec})",
                "trigger_source": source,
                "trigger_event": enriched,
            })
            return {"status": "rate_limited"}

        return await self._fire_item(
            item,
            manual=False,
            trigger_source=source,
            trigger_event=enriched,
        )

    async def _fire_item(
        self,
        item: Dict[str, Any],
        *,
        manual: bool,
        trigger_source: str = "cron",
        trigger_event: Optional[Dict[str, Any]] = None,
        bypass_quiet: bool = False,
    ) -> Dict[str, Any]:
        item_id = item["id"]
        self._running_items.add(item_id)
        triggered_at = datetime.now(timezone.utc)
        kind = item["kind"]
        agenda_fields = {
            "agenda_item_id": item_id,
            "kind": kind,
            "triggered_at": triggered_at,
            "trigger_source": trigger_source,
            "trigger_event": trigger_event,
        }
        cfg = item.get("config") or {}
        quiet_spec = cfg.get("quiet_hours") or _default_quiet_hours_spec()

        try:
            # Quiet-hours gate (honored even for event triggers; manual can bypass).
            if quiet_spec and not bypass_quiet and quiet_hours_mod.is_quiet(triggered_at, quiet_spec):
                policy = quiet_hours_mod.policy(quiet_spec)
                if policy == "defer":
                    next_end = quiet_hours_mod.next_end_at(triggered_at, quiet_spec)
                    await autonomy_db.insert_run({
                        **agenda_fields,
                        "status": "scheduled",
                        "summary": "deferred by quiet hours",
                        "scheduled_for": next_end,
                    })
                    await self._advance(item, triggered_at)
                    return {"status": "scheduled", "scheduled_for": next_end.isoformat() if next_end else None}
                await autonomy_db.insert_run({
                    **agenda_fields,
                    "completed_at": datetime.now(timezone.utc),
                    "status": "skipped_quiet_hours",
                    "summary": "dropped by quiet hours",
                })
                await self._advance(item, triggered_at)
                return {"status": "skipped_quiet_hours"}

            # Rate limit gate (skips manual triggers intentionally — operator override).
            if not manual:
                runs = await autonomy_db.count_runs_since(
                    triggered_at - timedelta(hours=1)
                )
                if runs >= config.AUTONOMY_MAX_RUNS_PER_HOUR:
                    await autonomy_db.insert_run({
                        **agenda_fields,
                        "completed_at": datetime.now(timezone.utc),
                        "status": "rate_limited",
                        "summary": f"global rate limit ({runs}/hr)",
                    })
                    await self._advance(item, triggered_at)
                    return {"status": "rate_limited"}

            # Dispatch handler.
            handler = _HANDLERS.get(kind)
            if handler is None:
                await autonomy_db.insert_run({
                    **agenda_fields,
                    "completed_at": datetime.now(timezone.utc),
                    "status": "error",
                    "error": f"unknown kind: {kind}",
                })
                await self._advance(item, triggered_at)
                return {"status": "error", "error": "unknown kind"}

            # Make the triggering event visible to the handler (watch reads this).
            if trigger_event is not None:
                item = {**item, "_trigger_event": trigger_event}

            result = await handler(
                item,
                client=self.client,
                mcp_manager=self.mcp_manager,
                model_name=self.model_name,
                base_tools=self.base_tools,
            )

            # Act-tier parked run: persist + notify user, don't advance schedule.
            if result.get("status") == "awaiting_confirmation":
                run_id = await autonomy_db.insert_run({
                    **agenda_fields,
                    "status": "awaiting_confirmation",
                    "summary": result.get("summary"),
                    "signature_hash": result.get("signature_hash"),
                    "messages": result.get("messages"),
                    "metrics": result.get("metrics"),
                    "action_audit": result.get("action_audit"),
                    "confirmation_token": result.get("confirmation_token"),
                })
                notified_via = await self._notify_confirmation(
                    item, run_id, result
                )
                if notified_via:
                    await autonomy_db.finalize_run(
                        run_id, {"notified_via": notified_via}
                    )
                return {
                    "status": "awaiting_confirmation",
                    "run_id": run_id,
                    "summary": result.get("summary"),
                }

            # Shared cooldown + notification path for anomaly_sweep, watch, watch_llm.
            if kind in ("anomaly_sweep", "watch", "watch_llm") and result.get("_unusual"):
                sig_hash = result.get("signature_hash")
                cooldown_min = int(cfg.get(
                    "cooldown_min", config.AUTONOMY_ANOMALY_COOLDOWN_MIN
                ))
                prev = await autonomy_db.last_run_for_signature(
                    sig_hash, triggered_at - timedelta(minutes=cooldown_min)
                )
                severity = result.get("severity") or "none"
                if prev and not _severity_escalated(prev.get("severity"), severity):
                    await autonomy_db.insert_run({
                        **agenda_fields,
                        "completed_at": datetime.now(timezone.utc),
                        "status": "skipped_cooldown",
                        "summary": result.get("summary"),
                        "severity": severity,
                        "signature_hash": sig_hash,
                        "notified_via": None,
                        "messages": result.get("messages"),
                        "metrics": result.get("metrics"),
                    })
                    await self._advance(item, triggered_at)
                    return {"status": "skipped_cooldown", "signature_hash": sig_hash}

                # Notifier selection: watch may pick its own channel/target;
                # anomaly falls back to HA push per the v1 convention.
                notify_channel = result.get("_notify_channel") or "ha_push"
                notify_to = result.get("_notify_to") or ""
                notify_cfg = {**cfg, **(result.get("_notify_cfg") or {})}
                notifier = _build_notifier(
                    self.mcp_manager, notify_channel, notify_to, notify_cfg
                )
                delivered = await notifier.send(
                    title=result.get("_notify_title") or "Selene",
                    body=result.get("_notify_body") or result.get("summary") or "",
                    severity=severity,
                )
                result["notified_via"] = notify_channel if delivered else None

            # Persist the completed run.
            await autonomy_db.insert_run({
                **agenda_fields,
                "completed_at": datetime.now(timezone.utc),
                "status": result.get("status", "ok"),
                "summary": result.get("summary"),
                "severity": result.get("severity"),
                "signature_hash": result.get("signature_hash"),
                "notified_via": result.get("notified_via"),
                "messages": result.get("messages"),
                "metrics": result.get("metrics"),
                "error": result.get("error"),
                "action_audit": result.get("action_audit"),
            })
            await self._advance(item, triggered_at)
            return {
                "status": result.get("status", "ok"),
                "summary": result.get("summary"),
                "severity": result.get("severity"),
                "notified_via": result.get("notified_via"),
            }
        except Exception as e:
            logger.error(f"[engine] unhandled error firing {kind} {item_id}: {e}")
            try:
                await autonomy_db.insert_run({
                    **agenda_fields,
                    "completed_at": datetime.now(timezone.utc),
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                })
                await self._advance(item, triggered_at)
            except Exception as e2:
                logger.error(f"[engine] failed to record error run: {e2}")
            return {"status": "error", "error": str(e)}
        finally:
            self._running_items.discard(item_id)

    async def _notify_confirmation(
        self,
        item: Dict[str, Any],
        run_id: str,
        result: Dict[str, Any],
    ) -> Optional[str]:
        """Send the 'approve this action?' notification for a parked act run.

        Channel is picked from ``cfg.deliver.channel`` (defaulting to signal).
        The token is included in the body via a deep-link so the user can
        approve without copy-pasting. Returns the channel name on success.
        """
        cfg = item.get("config") or {}
        deliver = cfg.get("deliver") or {}
        channel = deliver.get("channel") or "signal"
        to = deliver.get("to") or ""
        notifier = _build_notifier(self.mcp_manager, channel, to, {**cfg, **deliver})

        token = result.get("confirmation_token") or ""
        base_url = (
            getattr(config, "AGENT_BASE_URL", "")
            or getattr(config, "AGENT_INTERNAL_BASE_URL", "")
            or ""
        ).rstrip("/")
        audit = result.get("action_audit") or []
        executable = [a for a in audit if a.get("outcome") == "pending"]
        action_lines = [
            f"- {a.get('tool')}: {a.get('rationale') or ''}".strip()
            for a in executable[:5]
        ]

        body_lines = [
            f"{config.AGENT_NAME or 'Selene'} proposes an action:",
            "",
            *action_lines,
            "",
            f"Approve: {base_url}/autonomy?confirm={run_id}&token={token}"
            if base_url
            else f"Approve run {run_id} with token {token[:8]}…",
        ]
        title = f"{config.AGENT_NAME or 'Selene'}: confirm action"
        try:
            delivered = await notifier.send(title=title, body="\n".join(body_lines))
        except Exception as e:
            logger.error(f"[engine] confirmation notify failed: {e}")
            return None
        return channel if delivered else None

    async def resume_confirmed_run(
        self, run_id: str, *, approved: bool, token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Called by POST /api/autonomy/runs/{id}/confirm after the user
        decides. Validates state, runs or cancels, and updates the row.
        """
        run_row = await autonomy_db.get_run(run_id, include_messages=False)
        if run_row is None:
            return {"status": "not_found"}
        if run_row.get("status") != "awaiting_confirmation":
            return {"status": "invalid_state", "current": run_row.get("status")}
        if token is not None:
            import hmac
            stored = run_row.get("confirmation_token") or ""
            if not hmac.compare_digest(stored, token):
                return {"status": "invalid_token"}

        item_id = run_row.get("agenda_item_id")
        item = await autonomy_db.get_item(item_id) if item_id else None

        if not approved:
            await autonomy_db.finalize_run(run_id, {
                "status": "confirmation_denied",
                "confirmation_response": "denied",
                "completed_at": datetime.now(timezone.utc),
                "summary": "user denied action",
            })
            if item is not None:
                await self._advance(item, datetime.now(timezone.utc))
            return {"status": "confirmation_denied"}

        if item is None:
            await autonomy_db.finalize_run(run_id, {
                "status": "error",
                "completed_at": datetime.now(timezone.utc),
                "error": "agenda item missing at confirm time",
            })
            return {"status": "error", "error": "item_missing"}

        exec_result = await act_handler.execute_approved(
            run_row, item, self.mcp_manager
        )
        patch = {
            "status": exec_result.get("status", "ok"),
            "summary": exec_result.get("summary"),
            "error": exec_result.get("error"),
            "action_audit": exec_result.get("action_audit"),
            "confirmation_response": "approved",
            "completed_at": datetime.now(timezone.utc),
        }
        await autonomy_db.finalize_run(run_id, patch)
        await self._advance(item, datetime.now(timezone.utc))
        return {
            "status": exec_result.get("status", "ok"),
            "summary": exec_result.get("summary"),
            "action_audit": exec_result.get("action_audit"),
        }

    async def _sweep_confirmation_timeouts(self, now_utc: datetime) -> None:
        try:
            expired = await autonomy_db.list_expired_confirmations(now_utc)
        except Exception as e:
            logger.error(f"[engine] confirmation-timeout sweep failed: {e}")
            return
        for row in expired:
            try:
                claimed = await autonomy_db.claim_confirmation_timeout(
                    row["run_id"], now_utc
                )
            except Exception as e:
                logger.error(
                    f"[engine] claim_confirmation_timeout failed for "
                    f"{row['run_id']}: {e}"
                )
                continue
            if not claimed:
                continue
            logger.info(f"[engine] confirmation timed out: run {row['run_id']}")

    async def _advance(self, item: Dict[str, Any], fired_at: datetime) -> None:
        cron = item.get("schedule_cron")
        if not cron:
            return
        try:
            next_at = schedule.next_fire_at(cron, after=fired_at)
        except Exception as e:
            logger.error(f"[engine] cron advance failed for {item['id']}: {e}")
            return
        await autonomy_db.advance_item(item["id"], fired_at, next_at)


def _default_quiet_hours_spec() -> Optional[Dict[str, Any]]:
    start = getattr(config, "AUTONOMY_DEFAULT_QUIET_START", "") or ""
    end = getattr(config, "AUTONOMY_DEFAULT_QUIET_END", "") or ""
    policy = getattr(config, "AUTONOMY_DEFAULT_QUIET_POLICY", "defer") or "defer"
    if not start or not end:
        return None
    return {"start": start, "end": end, "policy": policy}


def _build_notifier(mcp_manager, channel: str, to: str, cfg: Dict[str, Any]):
    if channel in ("signal", "email"):  # "email" retained as legacy alias
        return SignalNotifier(
            mcp_manager,
            default_to=to or cfg.get("to") or config.AUTONOMY_BRIEFING_NOTIFY_TO,
        )
    if channel == "ha_push":
        target = (
            to
            or cfg.get("ha_notify_target")
            or cfg.get("to")
            or config.AUTONOMY_HA_NOTIFY_TARGET
        )
        return HAPushNotifier(mcp_manager, target=target)
    if channel == "speaker":
        speaker_cfg = cfg.get("speaker") or {}
        return SpeakerNotifier(
            mcp_manager,
            device=speaker_cfg.get("device") or to or cfg.get("device") or "",
            voice=speaker_cfg.get("voice") or cfg.get("voice") or "",
            volume=speaker_cfg.get("volume", cfg.get("volume")),
        )
    return NullNotifier()
