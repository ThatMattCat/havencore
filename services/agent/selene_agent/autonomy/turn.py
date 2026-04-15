"""AutonomousTurn — single-use orchestrator for autonomy runs.

Each autonomous run constructs a fresh ``AgentOrchestrator`` with a custom
system prompt and a tier-filtered tool list, drives its event generator to
completion, captures the full message trace + metrics, and is discarded.
User conversation state is never touched.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from openai import AsyncOpenAI

from selene_agent.autonomy import tool_gating
from selene_agent.orchestrator import AgentOrchestrator, EventType
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


@dataclass
class TurnResult:
    status: str  # 'ok' | 'error' | 'timeout'
    content: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AutonomousTurn:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        mcp_manager: MCPClientManager,
        model_name: str,
        base_tools: List[Dict[str, Any]],
        autonomy_level: str,
        system_prompt: str,
        timeout_sec: int = 60,
        temperature: float = 0.3,
        max_tokens: int = 800,
        tools_override: Optional[Iterable[str]] = None,
    ):
        self.client = client
        self.mcp_manager = mcp_manager
        self.model_name = model_name
        self.base_tools = base_tools
        self.autonomy_level = autonomy_level
        self.system_prompt = system_prompt
        self.timeout_sec = timeout_sec
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools_override = list(tools_override) if tools_override is not None else None
        if autonomy_level not in ("observe", "notify", "speak", "act"):
            raise ValueError(
                f"unknown autonomy tier {autonomy_level!r}; "
                "expected observe|notify|speak|act"
            )
        # Eager validation: a misconfigured tools_override must fail at
        # construction so handlers can surface the error cleanly.
        self._tools = tool_gating.filter_tools(
            self.base_tools, self.autonomy_level, override=self.tools_override
        )

    def _filtered_tools(self) -> List[Dict[str, Any]]:
        return self._tools

    async def run(self, user_message: str) -> TurnResult:
        """Drive one autonomous turn to completion and return a captured trace."""
        tools = self._filtered_tools()
        orch = AgentOrchestrator(
            client=self.client,
            mcp_manager=self.mcp_manager,
            model_name=self.model_name,
            tools=tools,
        )
        # Replace system prompt + override sampling for this one-shot run.
        from selene_agent.utils.l4_context import build_l4_block
        _l4 = await build_l4_block()
        _sys = (_l4 + "\n\n" + self.system_prompt) if _l4 else self.system_prompt
        orch.messages = [{"role": "system", "content": _sys}]
        # Autonomous turn already handled L4 injection — skip prepare()'s path.
        orch._l4_pending = False
        orch.temperature = self.temperature
        orch.max_tokens = self.max_tokens

        content = ""
        metrics: Dict[str, Any] = {}
        error: Optional[str] = None
        status = "ok"
        start = time.perf_counter()

        async def _drive() -> None:
            nonlocal content, metrics, error
            async for event in orch.run(user_message):
                if event.type == EventType.METRIC:
                    metrics = dict(event.data)
                elif event.type == EventType.DONE:
                    content = event.data.get("content", "") or ""
                elif event.type == EventType.ERROR:
                    error = event.data.get("error")

        try:
            await asyncio.wait_for(_drive(), timeout=self.timeout_sec)
        except asyncio.TimeoutError:
            status = "timeout"
            error = f"Turn exceeded {self.timeout_sec}s timeout"
            logger.warning(f"[AutonomousTurn] {error}")
        except Exception as e:
            status = "error"
            error = f"{type(e).__name__}: {e}"
            logger.error(f"[AutonomousTurn] unhandled: {error}")

        if error and status == "ok":
            status = "error"

        # Always capture whatever the orchestrator accumulated for audit.
        messages = list(orch.messages)
        if not metrics:
            metrics = {
                "total_ms": int((time.perf_counter() - start) * 1000),
                "llm_ms": 0,
                "tool_ms_total": 0,
                "iterations": 0,
                "tool_calls": [],
            }
        metrics["autonomy_level"] = self.autonomy_level
        metrics["tools_allowed"] = len(tools)

        return TurnResult(
            status=status,
            content=content,
            messages=messages,
            metrics=metrics,
            error=error,
        )
