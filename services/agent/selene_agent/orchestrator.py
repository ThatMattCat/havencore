"""
Agent Orchestrator - Event-based agent loop for streaming and tool visibility.

Separates the LLM interaction loop from the FastAPI server, yielding typed events
that can be consumed by both non-streaming (collect all) and streaming (SSE/WebSocket) endpoints.
"""

import asyncio
import contextvars
import json
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

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

TOOL_RESULT_MAX_CHARS = config.TOOL_RESULT_MAX_CHARS
MAX_TOOL_ITERATIONS = 20


class EventType(str, Enum):
    """Types of events emitted by the orchestrator"""
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESPONSE_CHUNK = "response_chunk"
    METRIC = "metric"
    DONE = "done"
    ERROR = "error"
    SUMMARY_RESET = "summary_reset"
    # Dashboard-only chain-of-thought surfaced from reasoning-capable models
    # (e.g. GLM-4.5 via vLLM's --reasoning-parser). Deliberately not consumed
    # by /api/chat, /v1/chat/completions, satellites, conversation_db, or
    # turn_metrics — see api/chat.py REST filter and orchestrator.run() where
    # this event is yielded but never appended to self.messages.
    REASONING = "reasoning"
    # Companion-app side-channel: emitted in addition to the normal
    # tool_call/tool_result pair when the LLM invokes a device-targeted tool
    # (see DEVICE_ACTION_TOOLS). The companion app fires the corresponding
    # platform intent (e.g. AlarmClock.ACTION_SET_ALARM); older app builds
    # without device_action support drop the event silently.
    DEVICE_ACTION = "device_action"


# Tools whose execution should fan out a DEVICE_ACTION event to the device
# that owns the session. Adding a new device-side action is one new entry
# here plus the matching MCP tool definition.
DEVICE_ACTION_TOOLS = frozenset({
    "set_alarm",
    "take_photo",
    "identify_object_in_photo",
    "read_text_from_image",
})

# Subset of DEVICE_ACTION_TOOLS whose ``device_action`` event must be emitted
# *before* the tool body runs. Camera-style tools fall in this bucket: their
# server-side handler awaits an upload from the phone, which can't arrive
# unless the wire event has already gone out. ``set_alarm`` stays on the
# original post-result emission (the LLM-visible result is the trigger; no
# round trip is needed).
PRE_EXECUTE_DEVICE_ACTION_TOOLS = frozenset({
    "take_photo",
    "identify_object_in_photo",
    "read_text_from_image",
})

# Subset of DEVICE_ACTION_TOOLS whose result is filled by the companion-app
# upload future, not by an MCP call. The orchestrator routes these through
# the local pending_uploads registry instead of ``mcp_manager.execute_tool``.
COMPANION_UPLOAD_TOOLS = frozenset({
    "take_photo",
    "identify_object_in_photo",
    "read_text_from_image",
})

# Vision-chained camera tools: after the upload future resolves, post the
# image_url to the in-process vision pipeline with the prompt the tool implies
# and surface the vision response to the LLM as the tool result. Mapping is
# tool name -> (prompt-builder(args) -> str, max_tokens, temperature). Prompts
# duplicated from ``mcp_vision_tools.server.DEFAULT_IDENTIFY_PROMPT`` /
# ``DEFAULT_OCR_PROMPT`` rather than imported because that module's import
# spins up an MCP stdio server as a side effect.
_DEFAULT_IDENTIFY_PROMPT = (
    "Identify the primary subject of this image. Give a concise name and a "
    "one-sentence description (material, model, species — whatever is most "
    "salient). If you cannot identify it confidently, say so and offer the "
    "closest plausible match."
)
_DEFAULT_OCR_PROMPT = (
    "Transcribe all visible text in this image. Preserve line breaks and "
    "rough layout where it carries meaning (receipts, forms, tables, code). "
    "If a region is illegible, mark it [illegible]. Do not paraphrase."
)


def _build_identify_prompt(args: Dict[str, Any]) -> str:
    base = _DEFAULT_IDENTIFY_PROMPT
    hint = (args.get("hint") or "").strip() if isinstance(args, dict) else ""
    if hint:
        base = f"{base}\n\nHint from the user about what this might be: {hint}"
    return base


def _build_ocr_prompt(_args: Dict[str, Any]) -> str:
    return _DEFAULT_OCR_PROMPT


VISION_CHAINED_TOOLS: Dict[str, Dict[str, Any]] = {
    "identify_object_in_photo": {
        "prompt": _build_identify_prompt,
        "max_tokens": 300,
        "temperature": 0.7,
        "result_key": "identification",
    },
    "read_text_from_image": {
        "prompt": _build_ocr_prompt,
        "max_tokens": 1024,
        "temperature": 0.1,
        "result_key": "text",
    },
}

# ContextVar carrying the in-flight tool_call_id for the duration of a single
# tool invocation. Set in the orchestrator loop right before _execute_tool_call
# so any in-process helper that looks it up (e.g. companion-upload waiter)
# can correlate without threading the id through every layer. Out-of-process
# MCP subprocess handlers cannot read this — they would receive the id via
# tool arguments instead.
current_tool_call_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_tool_call_id", default=None
)


@dataclass
class AgentEvent:
    """An event emitted during agent processing"""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


def truncate_tool_result(result: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Truncate a tool result to prevent unbounded context growth."""
    if len(result) <= max_chars:
        return result
    omitted = len(result) - max_chars
    return result[:max_chars] + f"\n[...truncated, {omitted} chars omitted]"


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> tuple[str, Optional[str]]:
    """Pull ``<think>…</think>`` blocks out of raw model content.

    Returns ``(clean_text, extracted_reasoning)``. ``extracted_reasoning`` is
    ``None`` when no blocks were present. Multiple blocks are concatenated
    with a blank line between them. Belt-and-suspenders for when vLLM's
    reasoning parser is unavailable or misses an edge case.
    """
    if not text or "<think>" not in text.lower():
        return text, None
    captured = [m.group(1).strip() for m in _THINK_BLOCK_RE.finditer(text)]
    captured = [c for c in captured if c]
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    reasoning = "\n\n".join(captured) if captured else None
    return cleaned, reasoning


class AgentOrchestrator:
    """
    Orchestrates the agent query loop, yielding events for each step.

    This separates the agent logic from transport (HTTP/WebSocket),
    enabling both streaming and non-streaming consumption.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        mcp_manager: MCPClientManager,
        model_name: str,
        tools: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        provider_getter: Optional[Callable[[], LLMProvider]] = None,
    ):
        self.client = client
        self.mcp_manager = mcp_manager
        self.model_name = model_name
        self.tools = tools
        # Every chat-completion call goes through ``provider_getter()`` so a
        # mid-session provider flip (set via /api/system/llm-provider) lands on
        # the next turn without a session rebuild. If no getter was supplied
        # (stateless /v1 compat path, tests with a mock client), wrap the given
        # ``client`` directly so the caller's AsyncOpenAI — real or mocked — is
        # what actually receives the call.
        if provider_getter is None:
            from selene_agent.providers.vllm import VLLMProvider
            _fallback = VLLMProvider.from_client(client=client, model=model_name)
            provider_getter = lambda: _fallback
        self.provider_getter: Callable[[], LLMProvider] = provider_getter

        self.messages: List[Dict[str, Any]] = []
        self.last_query_time: float = time.time()
        self.agent_name = config.AGENT_NAME
        self.session_id: str = session_id or str(uuid.uuid4())
        self._session_id_pinned: bool = session_id is not None

        self.temperature = 0.7
        self.top_p = 0.8
        self.max_tokens = 1024
        self._l4_pending = True

        self.idle_timeout_override: Optional[int] = None
        self.device_name: Optional[str] = None
        self._user_turn_since_reset: bool = False

        # Stash for a summary produced by _check_session_timeout or
        # _check_context_size at turn start; run() reads and clears this to
        # yield a SUMMARY_RESET event before the first thinking frame so the
        # client sees the compaction inline. Tuple is (reason, summary) so the
        # event carries the right reason for both idle and size triggers.
        self._pending_summary_reset: Optional[tuple[str, str]] = None

        # Per-turn retrieval injection. Session-pool sessions enable it; the
        # stateless /v1/chat/completions path overrides this to False.
        self.retrieval_enabled: bool = True

        # Context-size summarization gate. Pool-managed sessions leave this on;
        # autonomy turns flip it off so a one-shot orchestrator can never
        # trigger a summarize-and-reset on its tiny scratch state.
        self.context_size_check_enabled: bool = True

    def effective_timeout(self) -> int:
        """Idle window in seconds — per-session override or global default.

        Returns -1 as a sentinel meaning "never auto-summarize". Callers that
        schedule off this value (e.g. SessionOrchestratorPool.idle_sweep) must
        guard on `> 0` before comparing against elapsed time.
        """
        return int(self.idle_timeout_override or config.CONVERSATION_TIMEOUT)

    async def initialize(self):
        """Initialize with system prompt (prepends L4 persistent-memory block
        when present; appends phase-specific addendum)."""
        system_prompt = config.SYSTEM_PROMPT

        # Append phase-specific guidance so memory behavior shifts with the
        # operational phase (learning encourages aggressive memory creation).
        try:
            from selene_agent.utils.agent_state import get_agent_phase
            phase = await get_agent_phase()
            if phase == "learning":
                system_prompt = system_prompt + "\n" + config.SYSTEM_PROMPT_LEARNING_ADDENDUM
            else:
                system_prompt = system_prompt + "\n" + config.SYSTEM_PROMPT_OPERATING_ADDENDUM
        except Exception as e:
            logger.warning(f"phase addendum lookup failed: {e}")

        try:
            from selene_agent.utils.l4_context import build_l4_block
            block = await build_l4_block()
            if block:
                system_prompt = block + "\n\n" + system_prompt
        except Exception as e:
            logger.warning(f"L4 block build failed during initialize: {e}")
        self.messages = [{"role": "system", "content": system_prompt}]
        self._l4_pending = False
        if not getattr(self, "_session_id_pinned", False):
            self.session_id = str(uuid.uuid4())

    async def _check_session_timeout(self):
        """Route into summarize-and-reset if the session has been idle past its window.

        A non-positive timeout is the "never auto-summarize" sentinel — skip.
        """
        timeout = self.effective_timeout()
        if timeout <= 0:
            return
        if self.last_query_time and time.time() - self.last_query_time > timeout:
            summary = await self._summarize_and_reset(reason="idle_timeout_summarize")
            if summary:
                self._pending_summary_reset = ("idle_timeout_summarize", summary)

    async def _check_context_size(self):
        """Route into summarize-and-reset when message bytes exceed the budget.

        Threshold tracks the active provider's max_model_len (see
        ``utils.tokens.resolve_context_limit_tokens``) so a ``--max-model-len``
        bump in compose flows through automatically. ``idle_timeout=-1`` is
        unaffected — size is a separate axis from idle. Skipped on autonomy
        orchestrators (``context_size_check_enabled = False``).
        """
        if not self.context_size_check_enabled:
            return
        if not self.messages or len(self.messages) <= 1:
            return
        try:
            provider = self.provider_getter()
        except Exception as e:
            logger.warning(f"context-size check provider lookup failed: {e}")
            return
        threshold = await resolve_context_limit_tokens(provider)
        if threshold is None:
            return
        size = estimate_messages_tokens(self.messages)
        if size <= threshold:
            return
        logger.info(
            "context_size_threshold_exceeded",
            extra={
                "event": "context_size_threshold_exceeded",
                "session_id": self.session_id,
                "estimated_tokens": size,
                "threshold_tokens": threshold,
            },
        )
        summary = await self._summarize_and_reset(reason="context_size_summarize")
        if summary:
            self._pending_summary_reset = ("context_size_summarize", summary)

    async def _summarize_and_reset(self, reason: str) -> Optional[str]:
        """Persist full history, then reset `messages` to [system, summary, last N exchanges].

        Falls back to keep-tail-only if the summary LLM call fails or times out.
        Preserves `session_id` (pool orchestrators are always pinned).

        Returns the rolling summary string when one was produced (i.e. a real
        compaction happened), otherwise ``None``. Callers use the return value
        to emit a user-visible summary_reset event.
        """
        msgs = list(self.messages or [])
        msg_count = len(msgs)
        timeout = self.effective_timeout()
        idle = int(time.time() - self.last_query_time) if self.last_query_time else 0

        # No-op for fresh/empty sessions (system prompt only or less).
        if msg_count <= 1:
            await self.initialize()
            self.last_query_time = time.time()
            self._user_turn_since_reset = False
            return None

        # Count real user/assistant exchanges excluding the leading system msg.
        real_turns = [m for m in msgs[1:] if m.get("role") in ("user", "assistant")]
        if not real_turns:
            await self.initialize()
            self.last_query_time = time.time()
            self._user_turn_since_reset = False
            return None

        summary = await self._build_session_summary(msgs)
        summary_ok = summary is not None
        tail = self._tail_exchanges(msgs, config.SESSION_SUMMARY_TAIL_EXCHANGES)

        # Persist full pre-reset history with summary and override in metadata.
        try:
            metadata = {
                "reset_reason": reason,
                "message_count": msg_count,
                "last_query_time": self.last_query_time,
                "agent_name": self.agent_name,
                "idle_timeout_override": self.idle_timeout_override,
                "device_name": self.device_name,
                "idle_seconds": idle,
                "timeout_seconds": timeout,
                "rolling_summary": summary,
                "tail_exchanges_kept": len(tail),
            }
            await conversation_db.store_conversation_history(
                messages=msgs,
                session_id=self.session_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to persist history during summarize-reset ({self.session_id}): {e}")

        # Rebuild messages: system prompt (+ L4) + summary + tail.
        await self.initialize()
        if summary:
            self.messages.append({
                "role": "system",
                "content": f"[Prior conversation summary]\n{summary}",
            })
        self.messages.extend(tail)
        self.last_query_time = time.time()
        self._user_turn_since_reset = False

        logger.info(
            "session_summarize_reset",
            extra={
                "event": "session_summarize_reset",
                "session_id": self.session_id,
                "reason": reason,
                "prev_msg_count": msg_count,
                "idle_seconds": idle,
                "timeout_seconds": timeout,
                "summary_ok": summary_ok,
                "tail_kept": len(tail),
            },
        )
        return summary

    async def _build_session_summary(self, msgs: List[Dict[str, Any]]) -> Optional[str]:
        """One-shot LLM call that condenses `msgs` to a compact recap. Returns None on failure."""
        transcript_lines: List[str] = []
        for m in msgs[1:]:  # skip system prompt
            role = m.get("role", "")
            if role not in ("user", "assistant", "tool"):
                continue
            content = m.get("content") or ""
            if not isinstance(content, str):
                continue
            if role == "tool":
                tool_name = m.get("name") or "tool"
                transcript_lines.append(f"[{tool_name}]: {content}")
            else:
                transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)[:16000]
        if not transcript.strip():
            return None

        system_prompt = (
            f"You are summarizing a conversation between a user and an assistant named "
            f"{self.agent_name}. Produce a compact recap (<= {config.SESSION_SUMMARY_MAX_TOKENS} "
            "tokens) covering: (1) user intents and questions, (2) any side effects the "
            "assistant caused (device changes, media playback, memory writes, messages sent), "
            "(3) unresolved threads or open questions. Do NOT include pleasantries, role "
            "labels, or step-by-step tool traces. Plain text only, no markdown, no emojis."
        )

        try:
            resp = await asyncio.wait_for(
                self.provider_getter().chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcript},
                    ],
                    max_tokens=config.SESSION_SUMMARY_MAX_TOKENS,
                    temperature=0.2,
                ),
                timeout=config.SESSION_SUMMARY_LLM_TIMEOUT_SEC,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or None
        except asyncio.TimeoutError:
            logger.warning(f"Session summary timed out for session {self.session_id}")
            return None
        except Exception as e:
            logger.warning(f"Session summary LLM call failed for session {self.session_id}: {e}")
            return None

    def _tail_exchanges(self, msgs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        """Return up to the last `n` complete user→assistant exchanges (including
        associated tool_call / tool messages). Drops orphaned tool messages at the
        boundary to keep OpenAI schema valid.
        """
        if n <= 0 or len(msgs) <= 1:
            return []

        body = msgs[1:]  # drop system prompt
        # Find indices of each 'user' message.
        user_idxs = [i for i, m in enumerate(body) if m.get("role") == "user"]
        if not user_idxs:
            return []

        start = user_idxs[-n] if n <= len(user_idxs) else user_idxs[0]
        tail = body[start:]

        # Drop any trailing assistant message that declared tool_calls whose
        # tool responses aren't in the tail (or vice-versa) — iterate from the
        # end and trim dangling tool messages with no matching assistant.
        # Strategy: walk forward, keep only assistant tool_calls whose ids are
        # fully satisfied by subsequent tool messages.
        cleaned: List[Dict[str, Any]] = []
        pending_tool_ids: set = set()
        for m in tail:
            role = m.get("role")
            if role == "tool":
                # Only keep if we have a pending matching id from an assistant
                # we've already included; otherwise drop as orphan.
                tcid = m.get("tool_call_id")
                if tcid and tcid in pending_tool_ids:
                    cleaned.append(m)
                    pending_tool_ids.discard(tcid)
                continue
            if role == "assistant" and m.get("tool_calls"):
                pending_tool_ids.update(
                    tc.get("id") for tc in (m.get("tool_calls") or []) if tc.get("id")
                )
            cleaned.append(m)

        # Drop any trailing assistant whose tool_calls are still unsatisfied —
        # the model would otherwise expect tool responses that aren't here.
        while cleaned and cleaned[-1].get("role") == "assistant" and cleaned[-1].get("tool_calls"):
            ids = {tc.get("id") for tc in (cleaned[-1].get("tool_calls") or [])}
            if ids & pending_tool_ids:
                cleaned.pop()
                pending_tool_ids -= ids
                continue
            break

        return cleaned

    async def _build_retrieval_block(self, user_message: str) -> Optional[str]:
        """Fetch top-K L2/L3 for this user turn. Returns None when disabled or empty."""
        if not self.retrieval_enabled:
            return None
        try:
            from selene_agent.utils.retrieval import build_retrieval_block
            from selene_agent.utils.agent_state import get_agent_phase
            phase = await get_agent_phase()
            return await build_retrieval_block(user_message, phase=phase)
        except Exception as e:
            logger.warning(f"retrieval block build failed: {e}")
            return None

    def _messages_for_llm(self, retrieval_block: Optional[str]) -> List[Dict[str, Any]]:
        """Return self.messages with the retrieval block inserted before the
        most recent user message. Does not mutate self.messages.
        """
        if not retrieval_block:
            return self.messages
        msgs = list(self.messages)
        # Find the last user message and insert the retrieval block before it.
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs.insert(i, {"role": "system", "content": retrieval_block})
                return msgs
        return msgs

    async def prepare(self) -> None:
        """Lazily prepend L4 block if the orchestrator was set up without initialize()."""
        if not getattr(self, "_l4_pending", True):
            return
        try:
            from selene_agent.utils.l4_context import build_l4_block
            block = await build_l4_block()
            if block and self.messages and self.messages[0].get("role") == "system":
                self.messages[0]["content"] = block + "\n\n" + self.messages[0]["content"]
        except Exception as e:
            logger.warning(f"L4 block build failed during prepare: {e}")
        self._l4_pending = False

    async def run(self, user_message: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Process a user message, yielding events as the agent works.

        Events:
        - THINKING: Agent is calling the LLM
        - TOOL_CALL: Agent is calling a tool
        - TOOL_RESULT: Tool returned a result
        - RESPONSE_CHUNK: Part of the final response (for streaming)
        - DONE: Final response complete
        - ERROR: An error occurred
        """
        await self.prepare()
        wrapped_message = f"""
### System Context
- Current date and time: {datetime.now(ZoneInfo(config.CURRENT_TIMEZONE)).strftime('%A, %Y-%m-%d %H:%M:%S %Z')}

### User Message
{user_message}
"""
        unique_id = f"query_{int(time.time())}"

        await self._check_session_timeout()
        await self._check_context_size()
        if self._pending_summary_reset:
            reason, summary_text = self._pending_summary_reset
            yield AgentEvent(
                type=EventType.SUMMARY_RESET,
                data={
                    "reason": reason,
                    "summary": summary_text,
                },
            )
            self._pending_summary_reset = None
        self.last_query_time = time.time()
        self._user_turn_since_reset = True

        turn_start = time.perf_counter()
        llm_ms_total = 0.0
        tool_calls_timing: List[Dict[str, Any]] = []
        cache_read_total = 0
        cache_create_total = 0

        # Per-turn ephemeral retrieval block. Built once at turn start from the
        # raw user message; passed to the LLM on every iteration; never stored
        # in self.messages, so it can't compound across turns or survive into
        # a cold-resumed session.
        retrieval_block = await self._build_retrieval_block(user_message)

        try:
            self.messages.append({"role": "user", "content": wrapped_message})
            logger.info(f"Query: {user_message}")

            iteration = 0

            while iteration < MAX_TOOL_ITERATIONS:
                iteration += 1
                logger.debug(f"Iteration {iteration} of tool calling loop")

                yield AgentEvent(type=EventType.THINKING, data={"iteration": iteration})

                llm_start = time.perf_counter()
                provider = self.provider_getter()
                response = await provider.chat_completion(
                    messages=self._messages_for_llm(retrieval_block),
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                llm_ms_total += (time.perf_counter() - llm_start) * 1000
                try:
                    cache_stats = provider.pop_last_cache_stats()
                    cache_read_total += int(cache_stats.get("read", 0) or 0)
                    cache_create_total += int(cache_stats.get("create", 0) or 0)
                except AttributeError:
                    # Provider predates the protocol extension — skip silently.
                    pass

                assistant_message = response.choices[0].message
                logger.debug(f"Assistant response: {assistant_message}")

                # Reasoning capture (dashboard-only). Pulls from two sources:
                #   1. Provider's reasoning_content hook — populated when vLLM's
                #      --reasoning-parser split <think>…</think> server-side.
                #   2. Defensive strip of the raw content — catches cases where
                #      the parser missed or isn't available (wrong vLLM version,
                #      different model). Both sources are combined so no CoT is
                #      lost. Neither is appended to self.messages — after this
                #      block, assistant_message.content is the cleaned answer.
                try:
                    provider_reasoning = provider.pop_last_reasoning()
                except AttributeError:
                    provider_reasoning = None
                stripped_reasoning: Optional[str] = None
                if assistant_message.content:
                    cleaned, stripped_reasoning = strip_think_blocks(
                        assistant_message.content
                    )
                    if stripped_reasoning is not None:
                        assistant_message.content = cleaned
                reasoning_parts = [r for r in (provider_reasoning, stripped_reasoning) if r]
                if reasoning_parts:
                    yield AgentEvent(
                        type=EventType.REASONING,
                        data={
                            "content": "\n\n".join(reasoning_parts),
                            "iteration": iteration,
                        },
                    )

                # Handle models that embed tool calls in content tags
                if assistant_message.content and not assistant_message.tool_calls:
                    tool_calls_extracted = self._extract_tool_calls_from_content(
                        assistant_message.content
                    )
                    if tool_calls_extracted:
                        logger.debug(f"Extracted {len(tool_calls_extracted)} tool calls from content")
                        formatted_tool_calls = []
                        for idx, tool_data in enumerate(tool_calls_extracted):
                            tool_call = ChatCompletionMessageToolCall(
                                id=f"call_{unique_id}_{iteration}_{idx}",
                                type="function",
                                function=Function(
                                    name=tool_data["name"],
                                    arguments=json.dumps(tool_data["arguments"]),
                                ),
                            )
                            formatted_tool_calls.append(tool_call)
                        assistant_message.tool_calls = formatted_tool_calls
                        assistant_message.content = None

                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls == []:
                    assistant_message.tool_calls = None
                dumped_message = assistant_message.model_dump()
                # Normalize reasoning into the single field GLM-4.5-Air's
                # chat_template.jinja reads (``reasoning_content``). vLLM's
                # glm45 parser writes the legacy alias ``reasoning``; drop it.
                # The template renders <think>…</think> only for assistant
                # messages newer than the most recent user message — i.e. the
                # in-progress agentic tool-call loop, where the model expects
                # to see its own prior reasoning before the next call. For
                # already-completed turns it auto-emits empty <think></think>
                # regardless of what's stored, so retaining the field across
                # turns is harmless.
                dumped_message.pop("reasoning", None)
                combined_reasoning = "\n\n".join(
                    p.strip() for p in reasoning_parts if p and p.strip()
                )
                if combined_reasoning:
                    dumped_message["reasoning_content"] = combined_reasoning
                else:
                    dumped_message.pop("reasoning_content", None)
                self.messages.append(dumped_message)

                # Execute tool calls
                if assistant_message.tool_calls:
                    logger.debug(f"Model requested {len(assistant_message.tool_calls)} tool calls")

                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        yield AgentEvent(
                            type=EventType.TOOL_CALL,
                            data={"tool": function_name, "args": function_args, "id": tool_call.id},
                        )

                        # Pre-execute device_action: camera tools need the
                        # phone to act before their handler can return, so
                        # the wire event has to ship before _execute_tool_call.
                        if function_name in PRE_EXECUTE_DEVICE_ACTION_TOOLS:
                            yield AgentEvent(
                                type=EventType.DEVICE_ACTION,
                                data={
                                    "action": function_name,
                                    "args": function_args,
                                    "id": tool_call.id,
                                    "device_id": self.device_name,
                                },
                            )

                        tool_start = time.perf_counter()
                        token = current_tool_call_id.set(tool_call.id)
                        try:
                            result = await self._execute_tool_call(tool_call)
                        finally:
                            current_tool_call_id.reset(token)
                        tool_ms = (time.perf_counter() - tool_start) * 1000
                        tool_calls_timing.append({"name": function_name, "ms": int(tool_ms)})

                        yield AgentEvent(
                            type=EventType.TOOL_RESULT,
                            data={
                                "tool": function_name,
                                "result": result,
                                "id": tool_call.id,
                                "ms": int(tool_ms),
                            },
                        )

                        # Post-result device_action: pre-execute tools already
                        # emitted theirs above, skip the duplicate.
                        if (
                            function_name in DEVICE_ACTION_TOOLS
                            and function_name not in PRE_EXECUTE_DEVICE_ACTION_TOOLS
                        ):
                            yield AgentEvent(
                                type=EventType.DEVICE_ACTION,
                                data={
                                    "action": function_name,
                                    "args": function_args,
                                    "id": tool_call.id,
                                    "device_id": self.device_name,
                                },
                            )

                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })

                    continue

                # Final text response
                if assistant_message.content:
                    logger.info(f"Got final response after {iteration} iteration(s)")
                    total_ms = (time.perf_counter() - turn_start) * 1000
                    tool_ms_total = sum(tc["ms"] for tc in tool_calls_timing)
                    metric_payload = {
                        "llm_ms": int(llm_ms_total),
                        "tool_ms_total": int(tool_ms_total),
                        "total_ms": int(total_ms),
                        "iterations": iteration,
                        "tool_calls": tool_calls_timing,
                        "cache_read_tokens": cache_read_total,
                        "cache_creation_tokens": cache_create_total,
                    }
                    yield AgentEvent(type=EventType.METRIC, data=metric_payload)
                    yield AgentEvent(
                        type=EventType.DONE,
                        data={"content": assistant_message.content},
                    )
                    return

                logger.warning("Response had neither tool calls nor content")
                break

            if iteration >= MAX_TOOL_ITERATIONS:
                error_msg = (
                    f"ERROR: Maximum tool calling iterations ({MAX_TOOL_ITERATIONS}) "
                    "reached. The model may be stuck in a loop."
                )
                logger.error(
                    f"Hit maximum iterations ({MAX_TOOL_ITERATIONS}) in tool "
                    f"calling loop (session_id={self.session_id})"
                )
                yield AgentEvent(
                    type=EventType.ERROR,
                    data={"error": error_msg, "iterations": iteration},
                )
                return

            yield AgentEvent(type=EventType.ERROR, data={"error": "ERROR: No valid response generated"})

        except Exception as e:
            logger.error(f"Error in query: {e}\n{traceback.format_exc()}")
            yield AgentEvent(type=EventType.ERROR, data={"error": f"ERROR: {str(e)}"})

    async def _execute_tool_call(self, tool_call) -> str:
        """Execute a single tool call via MCP (or the companion-upload path)."""
        try:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.debug(f"Executing tool: {function_name} with args: {function_args}")

            if function_name in COMPANION_UPLOAD_TOOLS:
                return await self._handle_companion_camera(
                    function_name, tool_call.id, function_args
                )

            result = await self.mcp_manager.execute_tool(function_name, function_args)
            return truncate_tool_result(str(result))

        except Exception as e:
            logger.error(f"Error executing tool {tool_call.function.name}: {e}")
            return f"ERROR executing tool: {str(e)}"

    async def _handle_companion_camera(
        self,
        function_name: str,
        tool_call_id: str,
        function_args: Dict[str, Any],
    ) -> str:
        """End-to-end driver for companion-app camera tools.

        Awaits the upload future, then optionally chains to the in-process
        vision pipeline if the tool is in ``VISION_CHAINED_TOOLS``. Returns
        the JSON string the LLM will see as the tool result.
        """
        payload = await self._await_companion_upload_payload(
            function_name, tool_call_id
        )
        if isinstance(payload, str):
            # Already a structured error JSON string — bail before chaining.
            return payload

        captured_at_iso = self._unix_to_iso(payload.get("captured_at"))

        if function_name in VISION_CHAINED_TOOLS:
            spec = VISION_CHAINED_TOOLS[function_name]
            prompt = spec["prompt"](function_args or {})
            try:
                vision_text = await self._ask_vision(
                    image_url=payload.get("image_url"),
                    prompt=prompt,
                    max_tokens=spec["max_tokens"],
                    temperature=spec["temperature"],
                )
            except Exception as e:
                logger.warning(
                    f"vision chain failed: tool={function_name} "
                    f"tool_call_id={tool_call_id}: {e}"
                )
                return json.dumps({
                    "status": "vision_error",
                    "error": str(e),
                    "image_url": payload.get("image_url"),
                    "captured_at": captured_at_iso,
                })
            result_key = spec["result_key"]
            return json.dumps({
                "status": "captured_and_analyzed",
                "image_url": payload.get("image_url"),
                "captured_at": captured_at_iso,
                result_key: vision_text,
            })

        return json.dumps({
            "status": "captured",
            "image_url": payload.get("image_url"),
            "mime": payload.get("mime"),
            "captured_at": captured_at_iso,
            "device_id": payload.get("device_id"),
        })

    async def _await_companion_upload_payload(
        self,
        function_name: str,
        tool_call_id: str,
    ):
        """Wait for the companion-app upload future. Returns the dict payload
        on success, or a JSON-string error envelope on timeout (caller can
        return that directly to the LLM)."""
        # Local import — the api package depends on selene_agent at startup,
        # so an import-time pull would cycle.
        from selene_agent.api.companion import (
            register_pending_upload,
            pop_pending_upload,
        )

        timeout = float(getattr(config, "COMPANION_PHOTO_UPLOAD_TIMEOUT_SEC", 25))
        fut = register_pending_upload(tool_call_id)
        try:
            payload = await asyncio.wait_for(fut, timeout=timeout)
            return payload
        except asyncio.TimeoutError:
            pop_pending_upload(tool_call_id)
            logger.warning(
                f"companion upload timeout: tool={function_name} "
                f"tool_call_id={tool_call_id} timeout={timeout}s"
            )
            return json.dumps({
                "status": "timeout",
                "error": (
                    f"No upload received within {int(timeout)}s. The companion "
                    "app may be offline, the user may have cancelled the "
                    "capture, or the camera permission may be denied."
                ),
            })
        except asyncio.CancelledError:
            pop_pending_upload(tool_call_id)
            raise
        finally:
            # Clear on success too — payload already consumed.
            pop_pending_upload(tool_call_id)

    @staticmethod
    def _unix_to_iso(unix_ts) -> str:
        ts = float(unix_ts) if unix_ts is not None else time.time()
        return (
            datetime.fromtimestamp(ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    async def _ask_vision(
        self,
        image_url: Optional[str],
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Direct in-process call to the vllm-vision pipeline. Reuses the
        same ``_call_vision`` helper that powers ``/api/vision/ask_url`` so
        the chokepoint shape (model, message format) stays in one place."""
        if not image_url:
            raise ValueError("vision chain requires image_url")
        from selene_agent.api.vision import _call_vision  # local import — see _await
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
        content, latency_ms, _usage = await _call_vision(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        logger.info(
            f"vision chain ok: image_url={image_url} latency_ms={latency_ms} "
            f"prompt_chars={len(prompt)} response_chars={len(content)}"
        )
        return content

    @staticmethod
    def _extract_tool_calls_from_content(content: str) -> Optional[list]:
        """Extract tool calls from content wrapped in <tool_call> tags."""
        if not content:
            return None

        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            return None

        tool_calls = []
        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                if "name" in tool_data and "arguments" in tool_data:
                    tool_calls.append(tool_data)
                else:
                    logger.warning(f"Invalid tool call structure: {tool_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON: {e}\nContent: {match}")
                continue

        return tool_calls if tool_calls else None


async def collect_response(orchestrator: AgentOrchestrator, user_message: str) -> str:
    """
    Helper to run the orchestrator and collect the final text response.
    Used by non-streaming endpoints for backward compatibility.
    """
    final_content = ""
    async for event in orchestrator.run(user_message):
        if event.type == EventType.DONE:
            final_content = event.data.get("content", "")
        elif event.type == EventType.ERROR:
            final_content = event.data.get("error", "ERROR: Unknown error")
    return final_content
