"""AutonomyEngine — dispatcher loop and run orchestration.

Runs as an asyncio background task started from the FastAPI lifespan.
Owns the agenda_items/autonomy_runs lifecycle, kill-switch, rate limiting,
and per-signature cooldowns.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from selene_agent.autonomy import db as autonomy_db
from selene_agent.autonomy import schedule
from selene_agent.autonomy.handlers import anomaly as anomaly_handler
from selene_agent.autonomy.handlers import briefing as briefing_handler
from selene_agent.autonomy.notifiers import HAPushNotifier
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


_SEVERITY_RANK = {"none": 0, "low": 1, "med": 2, "high": 3}


def _severity_escalated(prev: Optional[str], current: str) -> bool:
    return _SEVERITY_RANK.get(current or "none", 0) > _SEVERITY_RANK.get(prev or "none", 0)


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

    # --- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        await autonomy_db.ensure_schema()
        await autonomy_db.ensure_default_agenda()
        self.started_at = datetime.now(timezone.utc)
        self._shutdown.clear()
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
        logger.info("AutonomyEngine stopped")

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
        return {
            "running": self.is_running(),
            "paused": self.is_paused(),
            "kill_switch_env": not config.AUTONOMY_ENABLED,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_dispatch_at": self.last_dispatch_at.isoformat() if self.last_dispatch_at else None,
            "runs_last_hour": runs_last_hour,
            "rate_limit_per_hour": config.AUTONOMY_MAX_RUNS_PER_HOUR,
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
                # Prior fire still executing — skip this tick.
                continue
            asyncio.create_task(self._fire_item(item, manual=False))

    async def trigger(self, item_id: str) -> Dict[str, Any]:
        """Manually fire an agenda item regardless of schedule."""
        item = await autonomy_db.get_item(item_id)
        if item is None:
            return {"status": "not_found"}
        if item["id"] in self._running_items:
            return {"status": "already_running"}
        return await self._fire_item(item, manual=True)

    async def _fire_item(self, item: Dict[str, Any], *, manual: bool) -> Dict[str, Any]:
        item_id = item["id"]
        self._running_items.add(item_id)
        triggered_at = datetime.now(timezone.utc)
        kind = item["kind"]
        agenda_fields = {"agenda_item_id": item_id, "kind": kind, "triggered_at": triggered_at}

        try:
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
            handler = {
                "briefing": briefing_handler.handle,
                "anomaly_sweep": anomaly_handler.handle,
            }.get(kind)
            if handler is None:
                await autonomy_db.insert_run({
                    **agenda_fields,
                    "completed_at": datetime.now(timezone.utc),
                    "status": "error",
                    "error": f"unknown kind: {kind}",
                })
                await self._advance(item, triggered_at)
                return {"status": "error", "error": "unknown kind"}

            result = await handler(
                item,
                client=self.client,
                mcp_manager=self.mcp_manager,
                model_name=self.model_name,
                base_tools=self.base_tools,
            )

            # Anomaly-specific cooldown / notification path.
            if kind == "anomaly_sweep" and result.get("_unusual"):
                sig_hash = result.get("signature_hash")
                cooldown_min = int((item.get("config") or {}).get(
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

                # Send HA push.
                target = (item.get("config") or {}).get(
                    "ha_notify_target", config.AUTONOMY_HA_NOTIFY_TARGET
                )
                notifier = HAPushNotifier(self.mcp_manager, target=target)
                delivered = await notifier.send(
                    title=result.get("_notify_title") or "Selene",
                    body=result.get("_notify_body") or result.get("summary") or "",
                    severity=severity,
                )
                result["notified_via"] = "ha_push" if delivered else None

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
