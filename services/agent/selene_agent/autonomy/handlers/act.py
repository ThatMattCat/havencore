"""Actuation handler — two-phase LLM plan, deterministic execute.

Phase 1 — the LLM runs at the ``observe`` tool surface (no actuator tools
exposed) and emits a strict JSON action plan::

    {"steps": [{"tool": "...", "args": {...}, "rationale": "..."}], ...}

Phase 2 — the engine (not the LLM) validates each step against the item's
``action_allow_list`` and either:
 - executes inline when ``require_confirmation=false``, or
 - persists ``awaiting_confirmation`` + a confirmation token, notifies the
   user, and waits for ``POST /api/autonomy/runs/{id}/confirm`` to resume.

The handler itself returns the Phase 1 decision. ``execute_approved`` is
the Phase 3 helper invoked by the engine's resume path.

Config shape::

    {
      "prompt": "Turn on the living room lamp.",
      "action_allow_list": ["ha_control_light", "ha_activate_scene"],
      "require_confirmation": true,
      "confirmation_timeout_sec": 300,
      "strict_execute": false,        # when true, any step error => status='error'
      "deliver": {"channel": "signal", "to": "..."},
      "quiet_hours": {...},
      "event_rate_limit": "..."
    }
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from selene_agent.autonomy.handlers.anomaly import _extract_json
from selene_agent.autonomy.turn import AutonomousTurn
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


ACT_PLAN_SYSTEM_PROMPT = (
    "You are {agent_name}'s autonomous actuation planner. The user has "
    "defined a goal; you decide WHICH tools to call and WITH WHAT ARGUMENTS "
    "to achieve it. You are NOT running tools yourself — you are producing "
    "a plan that the engine will execute after validation.\n\n"
    "You may call read-only tools (observe tier) to gather entity_ids and "
    "state. You MUST NOT call actuator tools; they are not in your surface. "
    "After gathering, respond with ONE JSON object and NOTHING else — no "
    "prose, no code fences. Schema:\n"
    "{{"
    "\"steps\": ["
    "  {{\"tool\": string, \"args\": object, \"rationale\": string}}"
    "], "
    "\"reasoning\": string (<=200 chars)"
    "}}.\n\n"
    "Allowed actuator tools (the engine will reject any others): {allow_list}.\n"
    "Keep the plan minimal — the fewest steps that achieve the goal. "
    "If the goal is impossible or ambiguous, return steps=[] and explain in "
    "reasoning."
)


def _signature(item_id: str) -> str:
    raw = f"act:{item_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _validate_steps(
    steps: List[Dict[str, Any]], allow_list: List[str]
) -> List[Dict[str, Any]]:
    """Annotate each step with an ``outcome`` field: ``pending`` for
    allow-listed calls, ``skipped_not_allowed`` otherwise. Steps are NOT
    dropped — the audit trail keeps every proposed call for visibility.
    """
    allow_set = set(allow_list or [])
    audit: List[Dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, dict):
            audit.append(
                {"tool": None, "args": {}, "rationale": "", "outcome": "malformed"}
            )
            continue
        tool = str(step.get("tool") or "").strip()
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        entry = {
            "tool": tool,
            "args": args,
            "rationale": str(step.get("rationale") or "")[:200],
        }
        if not tool:
            entry["outcome"] = "malformed"
        elif tool not in allow_set:
            entry["outcome"] = "skipped_not_allowed"
        else:
            entry["outcome"] = "pending"
        audit.append(entry)
    return audit


def _summary_from_plan(audit: List[Dict[str, Any]]) -> str:
    executable = [a for a in audit if a.get("outcome") == "pending"]
    if not executable:
        return "no executable steps"
    names = [a.get("tool") for a in executable[:3]]
    extra = len(executable) - 3
    suffix = f" (+{extra} more)" if extra > 0 else ""
    return f"{len(executable)} step(s): {', '.join(names)}{suffix}"


async def handle(
    item: Dict[str, Any],
    *,
    client,
    mcp_manager: MCPClientManager,
    model_name: str,
    base_tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = item.get("config") or {}
    prompt = str(cfg.get("prompt") or "").strip()
    allow_list = list(cfg.get("action_allow_list") or [])
    if not prompt:
        return {
            "status": "error",
            "summary": "act item has no prompt",
            "error": "empty prompt",
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }
    if not allow_list:
        return {
            "status": "error",
            "summary": "act item has empty action_allow_list",
            "error": "action_allow_list must be non-empty",
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }
    if not getattr(config, "AUTONOMY_ACT_ENABLED", False):
        return {
            "status": "error",
            "summary": "act tier disabled",
            "error": "AUTONOMY_ACT_ENABLED is false",
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    # Phase 1 — plan. LLM runs at observe tier so it can read state but
    # NEVER issue actuator calls from inside the turn.
    try:
        turn = AutonomousTurn(
            client=client,
            mcp_manager=mcp_manager,
            model_name=model_name,
            base_tools=base_tools,
            autonomy_level="observe",
            system_prompt=ACT_PLAN_SYSTEM_PROMPT.format(
                agent_name=config.AGENT_NAME or "Selene",
                allow_list=", ".join(allow_list),
            ),
            timeout_sec=int(cfg.get("timeout_sec") or config.AUTONOMY_TURN_TIMEOUT_SEC),
            temperature=float(cfg.get("temperature", 0.1)),
            max_tokens=int(cfg.get("max_tokens", 800)),
        )
    except ValueError as e:
        return {
            "status": "error",
            "summary": "act turn setup failed",
            "error": str(e),
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    result = await turn.run(prompt)

    parsed = _extract_json(result.content)
    if parsed is None or not isinstance(parsed, dict):
        return {
            "status": "error",
            "summary": "act: invalid plan JSON",
            "error": result.error or f"unparseable: {result.content[:200]}",
            "messages": result.messages,
            "metrics": result.metrics,
            "notified_via": None,
        }

    steps = parsed.get("steps")
    if not isinstance(steps, list):
        steps = []
    audit = _validate_steps(steps, allow_list)
    executable = [a for a in audit if a.get("outcome") == "pending"]

    if not executable:
        return {
            "status": "ok",
            "summary": f"act: no executable steps ({_summary_from_plan(audit)})",
            "signature_hash": _signature(item["id"]),
            "messages": result.messages,
            "metrics": result.metrics,
            "notified_via": None,
            "action_audit": audit,
            "reasoning": str(parsed.get("reasoning") or "")[:200],
        }

    require_confirm = bool(cfg.get("require_confirmation", True))

    if require_confirm:
        # Phase 2 — park the run awaiting user confirmation.
        confirmation_token = secrets.token_urlsafe(24)
        return {
            "status": "awaiting_confirmation",
            "summary": _summary_from_plan(audit),
            "signature_hash": _signature(item["id"]),
            "messages": result.messages,
            "metrics": result.metrics,
            "notified_via": None,
            "action_audit": audit,
            "confirmation_token": confirmation_token,
            "reasoning": str(parsed.get("reasoning") or "")[:200],
        }

    # Phase 3 — execute inline.
    audit = await execute_audit(audit, mcp_manager, strict=bool(cfg.get("strict_execute")))
    return _build_executed_result(item, audit, result.messages, result.metrics)


def _build_executed_result(
    item: Dict[str, Any],
    audit: List[Dict[str, Any]],
    messages: Optional[List[Dict[str, Any]]],
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = item.get("config") or {}
    strict = bool(cfg.get("strict_execute"))
    executed = sum(1 for a in audit if a.get("outcome") == "executed")
    errored = sum(1 for a in audit if a.get("outcome") == "error")
    skipped = sum(
        1 for a in audit if a.get("outcome") in ("skipped_not_allowed", "skipped_denied", "malformed")
    )
    if strict and errored:
        status = "error"
    elif executed == 0 and errored == 0:
        status = "ok"  # nothing needed to run
    else:
        status = "ok"
    summary_parts = []
    if executed:
        summary_parts.append(f"{executed} executed")
    if errored:
        summary_parts.append(f"{errored} errored")
    if skipped:
        summary_parts.append(f"{skipped} skipped")
    summary = ", ".join(summary_parts) or "no steps"
    return {
        "status": status,
        "summary": summary[:200],
        "signature_hash": _signature(item["id"]),
        "messages": messages or [],
        "metrics": metrics or {},
        "notified_via": None,
        "action_audit": audit,
        "error": f"{errored} step(s) errored" if (strict and errored) else None,
    }


async def execute_audit(
    audit: List[Dict[str, Any]],
    mcp_manager: MCPClientManager,
    *,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Execute every audit entry whose ``outcome`` is ``pending``. Mutates
    and returns the same list with outcomes set to ``executed`` / ``error``.
    Steps already marked ``skipped_*`` are left alone.
    """
    for entry in audit:
        if entry.get("outcome") != "pending":
            continue
        tool = entry.get("tool")
        args = entry.get("args") or {}
        try:
            result = await mcp_manager.execute_tool(tool, args)
            entry["outcome"] = "executed"
            entry["result"] = _summarize_tool_result(result)
        except Exception as e:
            entry["outcome"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            if strict:
                # Subsequent pending steps still run — the engine decides
                # overall status based on error count. Strict only changes
                # the final status, not the execution order.
                continue
    return audit


def _summarize_tool_result(result: Any) -> str:
    """Keep audit payloads small — raw MCP results can be large."""
    if result is None:
        return ""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text[:300]


async def execute_approved(
    run_row: Dict[str, Any],
    item: Dict[str, Any],
    mcp_manager: MCPClientManager,
) -> Dict[str, Any]:
    """Phase 3 re-entry from the confirm endpoint. The run row already has
    an ``action_audit`` list from Phase 1; we execute its pending entries
    and return the finalized result dict in the same shape as ``handle()``.
    """
    cfg = item.get("config") or {}
    audit = list(run_row.get("action_audit") or [])
    # Clear any stale "skipped_denied" flag from a prior denial attempt
    # (shouldn't happen with single-use tokens, but defensive).
    audit = await execute_audit(
        audit, mcp_manager, strict=bool(cfg.get("strict_execute"))
    )
    return _build_executed_result(item, audit, None, None)
