"""
SessionOrchestratorPool — per-session AgentOrchestrators.

Each pool entry is a live orchestrator with its own `messages`, `session_id`,
and `last_query_time`. Singletons (`client`, `mcp_manager`, `model_name`,
`tools`) are shared across all entries.

Contract:
- `get_or_create(sid)` returns an orchestrator hydrated from memory, from
  `conversation_db` (cold resume), or freshly initialized with a minted UUID.
- `lock_for(sid)` returns a per-session `asyncio.Lock`. Callers must hold it
  around any `orchestrator.run()` invocation — `run()` mutates `messages` and
  is not reentrant.
- `idle_sweep()` opportunistically resets timed-out sessions by delegating to
  the orchestrator's existing `_check_session_timeout()`. It skips busy
  sessions (non-blocking lock acquire).
- `flush_all()` persists every non-empty session to `conversation_db`. Called
  from app shutdown.
- LRU-bounded at `max_size`; eviction flushes the dropped session.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from selene_agent.orchestrator import AgentOrchestrator
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


class SessionOrchestratorPool:
    def __init__(
        self,
        client: AsyncOpenAI,
        mcp_manager: MCPClientManager,
        model_name: str,
        tools: List[Dict[str, Any]],
        max_size: int = 64,
        mcp_failure_note: Optional[str] = None,
    ):
        self._client = client
        self._mcp_manager = mcp_manager
        self._model_name = model_name
        self._tools = tools
        self._max_size = max_size
        self._mcp_failure_note = mcp_failure_note

        self._sessions: "OrderedDict[str, AgentOrchestrator]" = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._pool_lock = asyncio.Lock()
        self._sweep_task: Optional[asyncio.Task] = None

    # ----- Construction helpers -------------------------------------------------

    def _build_orchestrator(self, session_id: str) -> AgentOrchestrator:
        return AgentOrchestrator(
            client=self._client,
            mcp_manager=self._mcp_manager,
            model_name=self._model_name,
            tools=self._tools,
            session_id=session_id,
        )

    def _maybe_append_mcp_note(self, orch: AgentOrchestrator) -> None:
        if self._mcp_failure_note:
            orch.messages.append({"role": "system", "content": self._mcp_failure_note})

    async def _hydrate_from_db(self, session_id: str) -> Optional[AgentOrchestrator]:
        """Cold-resume: load most recent stored history for this session_id.

        Uses `prepare()` (not `initialize()`) so the L4 block prepends without
        clobbering the restored messages.
        """
        try:
            histories = await conversation_db.get_conversation_history(session_id, limit=1)
        except Exception as e:
            logger.warning(f"Cold-resume lookup failed for session {session_id}: {e}")
            return None

        if not histories:
            return None

        messages = histories[0].get("messages") or []
        if not messages:
            return None

        orch = self._build_orchestrator(session_id)
        orch.messages = list(messages)
        # Rehydrate per-session idle-timeout override if it was persisted.
        metadata = histories[0].get("metadata") or {}
        raw_override = metadata.get("idle_timeout_override")
        if isinstance(raw_override, (int, float)):
            v = int(raw_override)
            lo, hi = config.CONVERSATION_TIMEOUT_MIN, config.CONVERSATION_TIMEOUT_MAX
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            orch.idle_timeout_override = v
        # Rehydrate device_name if it was persisted.
        raw_name = metadata.get("device_name")
        if isinstance(raw_name, str):
            cleaned = raw_name.strip()
            if cleaned:
                orch.device_name = cleaned[:64]
        # prepare() will prepend the L4 block to the existing system message
        # (or be a no-op if the first message isn't system).
        try:
            await orch.prepare()
        except Exception as e:
            logger.warning(f"prepare() failed during cold-resume for {session_id}: {e}")
            orch._l4_pending = False
        orch.last_query_time = time.time()
        logger.info(f"Cold-resumed session {session_id} with {len(messages)} messages")
        return orch

    # ----- Public API -----------------------------------------------------------

    async def get_or_create(self, session_id: Optional[str]) -> AgentOrchestrator:
        """Return the orchestrator for session_id, creating or hydrating as needed.

        Passing None (or an unknown sid absent from the DB) mints a new session.
        """
        async with self._pool_lock:
            # 1. In-pool → bump LRU and return.
            if session_id and session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]

            # 2. Known sid but not in pool → try to cold-resume from DB.
            resumed: Optional[AgentOrchestrator] = None
            if session_id:
                resumed = await self._hydrate_from_db(session_id)

            if resumed is not None:
                await self._admit(session_id, resumed)
                return resumed

            # 3. Mint a fresh session.
            new_sid = session_id or str(uuid.uuid4())
            orch = self._build_orchestrator(new_sid)
            await orch.initialize()
            self._maybe_append_mcp_note(orch)
            await self._admit(new_sid, orch)
            return orch

    async def _admit(self, session_id: str, orch: AgentOrchestrator) -> None:
        """Insert an orchestrator, evicting the LRU entry if at capacity.

        Caller must hold `_pool_lock`.
        """
        if len(self._sessions) >= self._max_size and session_id not in self._sessions:
            try:
                evict_sid, evict_orch = next(iter(self._sessions.items()))
                await self._flush_one(evict_orch, reason="lru_eviction")
                self._sessions.pop(evict_sid, None)
                self._locks.pop(evict_sid, None)
                logger.info(f"Evicted LRU session {evict_sid} (pool at max_size={self._max_size})")
            except Exception as e:
                logger.error(f"LRU eviction failed: {e}")

        self._sessions[session_id] = orch
        self._sessions.move_to_end(session_id)
        self._locks.setdefault(session_id, asyncio.Lock())

    def lock_for(self, session_id: str) -> asyncio.Lock:
        """Return the per-session lock. Idempotent."""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    def size(self) -> int:
        return len(self._sessions)

    def max_size(self) -> int:
        return self._max_size

    def health(self) -> Dict[str, Any]:
        return {
            "active_sessions": len(self._sessions),
            "max_size": self._max_size,
            "sweep_running": self._sweep_task is not None and not self._sweep_task.done(),
        }

    # ----- Background sweep -----------------------------------------------------

    async def start_idle_sweep(self, interval_sec: int = 30) -> None:
        if self._sweep_task and not self._sweep_task.done():
            return
        self._sweep_task = asyncio.create_task(self._sweep_loop(interval_sec))

    async def stop_idle_sweep(self) -> None:
        if self._sweep_task is None:
            return
        self._sweep_task.cancel()
        try:
            await self._sweep_task
        except (asyncio.CancelledError, Exception):
            pass
        self._sweep_task = None

    async def _sweep_loop(self, interval_sec: int) -> None:
        while True:
            try:
                await asyncio.sleep(interval_sec)
                await self.idle_sweep()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Idle sweep iteration failed: {e}")

    async def idle_sweep(self) -> None:
        """Reset timed-out sessions. Non-blocking: skips busy sessions.

        Each session's idle window is `orch.effective_timeout()` — the
        per-session override when set, otherwise the global default.
        """
        now = time.time()
        # Snapshot under pool lock to avoid mutation during iteration.
        async with self._pool_lock:
            candidates = [
                (sid, orch) for sid, orch in self._sessions.items()
                if orch.last_query_time
                and orch._user_turn_since_reset
                and (now - orch.last_query_time) > orch.effective_timeout()
            ]

        for sid, orch in candidates:
            lock = self.lock_for(sid)
            # Non-blocking try-acquire: skip busy sessions.
            try:
                await asyncio.wait_for(lock.acquire(), timeout=0.01)
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue
            try:
                # Re-check under the lock — a turn may have just started.
                if (
                    orch.last_query_time
                    and orch._user_turn_since_reset
                    and (time.time() - orch.last_query_time) > orch.effective_timeout()
                ):
                    await orch._summarize_and_reset(reason="idle_timeout_summarize")
            except Exception as e:
                logger.error(f"Idle-sweep reset failed for session {sid}: {e}")
            finally:
                lock.release()

    # ----- Shutdown flush -------------------------------------------------------

    async def _flush_one(self, orch: AgentOrchestrator, reason: str) -> None:
        msgs = orch.messages or []
        # Only persist meaningful conversations (more than the system message).
        if len(msgs) <= 1:
            return
        try:
            metadata = {
                "reset_reason": reason,
                "message_count": len(msgs),
                "last_query_time": orch.last_query_time,
                "agent_name": orch.agent_name,
                "idle_timeout_override": orch.idle_timeout_override,
                "device_name": orch.device_name,
            }
            await conversation_db.store_conversation_history(
                messages=msgs,
                session_id=orch.session_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Flush failed for session {orch.session_id}: {e}")

    async def flush_all(self) -> None:
        """Persist every non-empty session. Called from app shutdown."""
        async with self._pool_lock:
            items = list(self._sessions.items())

        for sid, orch in items:
            lock = self.lock_for(sid)
            try:
                await lock.acquire()
            except Exception:
                continue
            try:
                await self._flush_one(orch, reason="shutdown_flush")
            finally:
                lock.release()
        logger.info(f"Shutdown flush complete for {len(items)} sessions")
