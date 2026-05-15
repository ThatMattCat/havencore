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
from typing import Any, Callable, Dict, List, Optional, Set

from openai import AsyncOpenAI

from selene_agent.orchestrator import AgentOrchestrator
from selene_agent.providers import LLMProvider
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils.mcp_client_manager import MCPClientManager
from selene_agent.utils.tokens import (
    estimate_messages_tokens,
    resolve_context_limit_tokens,
)

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
        provider_getter: Optional[Callable[[], LLMProvider]] = None,
    ):
        self._client = client
        self._mcp_manager = mcp_manager
        self._model_name = model_name
        self._tools = tools
        self._max_size = max_size
        self._mcp_failure_note = mcp_failure_note
        self._provider_getter = provider_getter

        self._sessions: "OrderedDict[str, AgentOrchestrator]" = OrderedDict()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._pool_lock = asyncio.Lock()
        self._sweep_task: Optional[asyncio.Task] = None

        # Per-session subscriber queues for out-of-band server → client events
        # (e.g. summary_reset fired by the background idle sweep). Bounded queue
        # with drop-on-full semantics — these are UI notifications, not source
        # of truth; the DB is authoritative.
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

    # ----- Construction helpers -------------------------------------------------

    def _build_orchestrator(self, session_id: str) -> AgentOrchestrator:
        return AgentOrchestrator(
            client=self._client,
            mcp_manager=self._mcp_manager,
            model_name=self._model_name,
            tools=self._tools,
            session_id=session_id,
            provider_getter=self._provider_getter,
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
        # The -1 sentinel ("never auto-summarize") round-trips as-is.
        metadata = histories[0].get("metadata") or {}
        raw_override = metadata.get("idle_timeout_override")
        if isinstance(raw_override, (int, float)):
            v = int(raw_override)
            if v == -1:
                orch.idle_timeout_override = -1
            else:
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

    # ----- Subscriber pub/sub ---------------------------------------------------

    def subscribe(self, session_id: str) -> asyncio.Queue:
        """Return a queue that receives out-of-band events for `session_id`.

        Callers must call `unsubscribe(session_id, queue)` when done.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        self._subscribers.setdefault(session_id, set()).add(q)
        return q

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(session_id)
        if not subs:
            return
        subs.discard(queue)
        if not subs:
            self._subscribers.pop(session_id, None)

    def publish(self, session_id: str, event: Dict[str, Any]) -> None:
        """Fan out an event to every subscriber for `session_id`.

        Non-blocking: drops the event for any subscriber whose queue is full.
        """
        for q in list(self._subscribers.get(session_id, ())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Subscriber queue full for session {session_id}; dropping event")

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

    def _is_idle_candidate(self, orch: AgentOrchestrator, now: float) -> bool:
        return bool(
            orch.last_query_time
            and orch._user_turn_since_reset
            and orch.effective_timeout() > 0
            and (now - orch.last_query_time) > orch.effective_timeout()
        )

    async def _is_size_candidate(self, orch: AgentOrchestrator) -> bool:
        if not orch.context_size_check_enabled:
            return False
        if not orch.messages or len(orch.messages) <= 1:
            return False
        try:
            provider = self._provider_getter() if self._provider_getter else None
        except Exception:
            return False
        threshold = await resolve_context_limit_tokens(provider)
        if threshold is None:
            return False
        return estimate_messages_tokens(orch.messages) > threshold

    async def idle_sweep(self) -> None:
        """Reset sessions that exceed their idle window OR their token budget.

        Non-blocking: skips busy sessions. Each candidate is tagged with the
        reason it qualified so the right ``summary_reset`` event reaches live
        clients. Idle and size are independent axes — a dashboard session with
        ``idle_timeout=-1`` will never qualify on idle but can qualify on size,
        and that's the intended behavior.
        """
        now = time.time()
        # Snapshot under pool lock to avoid mutation during iteration.
        async with self._pool_lock:
            sessions_snapshot = list(self._sessions.items())

        # Compute size membership outside the pool lock — get_max_model_len()
        # may hit the provider's HTTP endpoint and we don't want to serialize
        # that behind the pool lock.
        candidates: List[tuple[str, AgentOrchestrator, str]] = []
        for sid, orch in sessions_snapshot:
            if self._is_idle_candidate(orch, now):
                candidates.append((sid, orch, "idle_timeout_summarize"))
            elif await self._is_size_candidate(orch):
                candidates.append((sid, orch, "context_size_summarize"))

        for sid, orch, reason in candidates:
            lock = self.lock_for(sid)
            # Non-blocking try-acquire: skip busy sessions.
            try:
                await asyncio.wait_for(lock.acquire(), timeout=0.01)
            except asyncio.TimeoutError:
                continue
            except Exception:
                continue
            try:
                # Re-check under the lock — a turn may have just started, or the
                # condition may have already been resolved by the orchestrator's
                # own per-turn checks.
                still_idle = (
                    reason == "idle_timeout_summarize"
                    and self._is_idle_candidate(orch, time.time())
                )
                still_oversized = (
                    reason == "context_size_summarize"
                    and await self._is_size_candidate(orch)
                )
                if not (still_idle or still_oversized):
                    continue
                summary = await orch._summarize_and_reset(reason=reason)
                if summary:
                    self.publish(sid, {
                        "type": "summary_reset",
                        "reason": reason,
                        "summary": summary,
                    })
            except Exception as e:
                logger.error(f"Idle-sweep reset failed for session {sid}: {e}")
            finally:
                lock.release()

    # ----- System-prompt rebuild (phase change) ---------------------------------

    async def rebuild_system_prompts(self) -> None:
        """Refresh the system message (messages[0]) of every active session.

        Called when the operational phase changes — the phase-specific prompt
        addendum differs between learning and operating, so in-flight sessions
        need their system prompt regenerated. Mid-turn sessions are skipped
        (non-blocking try-acquire); their next prepare() call will pick up the
        change if we arrange for it, but phase changes are rare enough that
        "updates next turn after any ongoing one" is acceptable.
        """
        async with self._pool_lock:
            items = list(self._sessions.items())

        refreshed = 0
        for sid, orch in items:
            lock = self.lock_for(sid)
            try:
                await asyncio.wait_for(lock.acquire(), timeout=0.05)
            except (asyncio.TimeoutError, Exception):
                continue
            try:
                if not orch.messages or orch.messages[0].get("role") != "system":
                    continue
                # Rebuild a fresh system prompt inline. Reuse the same logic
                # as initialize() without nuking the rest of the message list.
                from selene_agent.utils.agent_state import get_agent_phase
                from selene_agent.utils.l4_context import build_l4_block
                try:
                    system_prompt = config.SYSTEM_PROMPT
                    phase = await get_agent_phase()
                    if phase == "learning":
                        system_prompt = system_prompt + "\n" + config.SYSTEM_PROMPT_LEARNING_ADDENDUM
                    else:
                        system_prompt = system_prompt + "\n" + config.SYSTEM_PROMPT_OPERATING_ADDENDUM
                    if config.TTS_PROVIDER == "v2":
                        system_prompt = system_prompt + "\n" + config.SYSTEM_PROMPT_PARALINGUISTIC_ADDENDUM
                    block = await build_l4_block()
                    if block:
                        system_prompt = block + "\n\n" + system_prompt
                    orch.messages[0] = {"role": "system", "content": system_prompt}
                    refreshed += 1
                except Exception as e:
                    logger.warning(f"rebuild_system_prompts failed for {sid}: {e}")
            finally:
                lock.release()
        logger.info(f"rebuild_system_prompts: refreshed {refreshed}/{len(items)} sessions")

    # ----- Shutdown flush -------------------------------------------------------

    async def _flush_one(self, orch: AgentOrchestrator, reason: str) -> None:
        msgs = orch.messages or []
        # Only persist meaningful conversations (more than the system message).
        if len(msgs) <= 1:
            return
        # Size-aware bypass: if this session is being evicted/flushed while
        # oversized, route through _summarize_and_reset so the persisted row
        # carries metadata.rolling_summary instead of a bloated raw blob.
        # Otherwise cold-resume would just replay the bloat on the next visit.
        # _summarize_and_reset persists internally, so we can return early.
        if orch.context_size_check_enabled:
            try:
                provider = self._provider_getter() if self._provider_getter else None
                threshold = await resolve_context_limit_tokens(provider)
                if (
                    threshold is not None
                    and estimate_messages_tokens(msgs) > threshold
                ):
                    flush_reason = (
                        "lru_eviction_size" if reason == "lru_eviction"
                        else f"{reason}_size"
                    )
                    await orch._summarize_and_reset(reason=flush_reason)
                    return
            except Exception as e:
                logger.warning(
                    f"Size-aware flush check failed for {orch.session_id}: {e}"
                )
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
