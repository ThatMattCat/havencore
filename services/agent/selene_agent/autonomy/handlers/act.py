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
    "HOW EXECUTION WORKS — read carefully:\n"
    "1. THIS turn is your ONLY chance to gather information. Call read-only "
    "   tools NOW (in this turn, via tool_calls) to discover entity_ids, "
    "   current states, area memberships, etc. The engine will NOT give you "
    "   another LLM turn after this.\n"
    "2. After you have all the information you need, emit ONE final JSON "
    "   object (schema below). That JSON is your plan.\n"
    "3. The engine then executes the plan's steps LITERALLY and IN ORDER, "
    "   with NO further LLM involvement. Each step's ``args`` is passed to "
    "   the tool exactly as written. Steps CANNOT reference each other's "
    "   output. There are NO placeholders, NO variables, NO templates — "
    "   every value must be a concrete literal you resolved during THIS "
    "   turn.\n\n"
    "Therefore:\n"
    "- If you need an entity_id, CALL an observe tool NOW to look it up. "
    "  Do NOT put a reconnaissance call into your plan's steps — read-only "
    "  tools belong in this turn, not in the plan.\n"
    "- Do NOT write placeholder strings like ``<entity_id>``, "
    "  ``{{previous.result}}``, ``<from_step_1>``, ``PLACEHOLDER``, etc. "
    "  The engine will reject any step whose args contain such patterns.\n"
    "- If you cannot resolve a concrete value (e.g., the entity genuinely "
    "  does not exist), return ``steps: []`` and explain in ``reasoning``.\n\n"
    "Plan JSON schema — respond with ONE JSON object, no prose, no code "
    "fences:\n"
    "{{"
    "\"steps\": ["
    "  {{\"tool\": string, \"args\": object, \"rationale\": string}}"
    "], "
    "\"reasoning\": string (<=200 chars)"
    "}}.\n\n"
    "Allowed actuator tools (the engine will reject any others): {allow_list}.\n\n"
    "CRITICAL — actuator tool schemas. Each step's ``args`` object MUST use "
    "EXACTLY the parameter names shown below. Do NOT invent parameter names "
    "from your general knowledge of Home Assistant or other APIs; use these "
    "schemas verbatim:\n"
    "{tool_schemas}\n\n"
    "Keep the plan minimal — the fewest steps that achieve the goal."
)


_PLACEHOLDER_PATTERNS = (
    re.compile(r"<[^>]*>"),           # <entity_id>, <from_step_1>, etc.
    re.compile(r"\{\{[^}]*\}\}"),     # {{previous.result}}, {{var}}
)
_PLACEHOLDER_TOKENS = (
    "previous_step",
    "previous step",
    "from_previous",
    "from previous",
    "placeholder",
    "todo",
)


def _placeholder_reason(args: Dict[str, Any]) -> Optional[str]:
    """Return a human-readable reason if ``args`` contains a value that
    looks like an unresolved template placeholder. The planner must emit
    concrete literals — any ``<...>`` / ``{{...}}`` / "previous_step" text
    means the LLM intended a follow-up resolution pass that does not exist.
    """
    def _scan(value: Any, path: str) -> Optional[str]:
        if isinstance(value, str):
            for pat in _PLACEHOLDER_PATTERNS:
                m = pat.search(value)
                if m:
                    return f"{path}={value!r} contains placeholder {m.group(0)!r}"
            lowered = value.lower()
            for tok in _PLACEHOLDER_TOKENS:
                if tok in lowered:
                    return f"{path}={value!r} contains placeholder token {tok!r}"
        elif isinstance(value, dict):
            for k, v in value.items():
                r = _scan(v, f"{path}.{k}" if path else str(k))
                if r:
                    return r
        elif isinstance(value, list):
            for i, v in enumerate(value):
                r = _scan(v, f"{path}[{i}]")
                if r:
                    return r
        return None

    return _scan(args, "args")


def _format_actuator_schemas(
    base_tools: List[Dict[str, Any]], allow_list: List[str]
) -> str:
    """Return a compact schema block for each allow-listed actuator tool, so
    the planner LLM can plan with the correct parameter names instead of
    guessing from training data.
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for tool in base_tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name") or tool.get("name")
        if name:
            by_name[name] = fn or tool
    blocks: List[str] = []
    for name in allow_list:
        fn = by_name.get(name)
        if fn is None:
            blocks.append(f"- {name}: (schema not found — tool may be unavailable)")
            continue
        desc = (fn.get("description") or "").strip()
        if len(desc) > 400:
            desc = desc[:400].rstrip() + "…"
        params = fn.get("parameters") or {}
        blocks.append(
            f"- {name}\n"
            f"  description: {desc}\n"
            f"  parameters: {json.dumps(params, separators=(',', ':'))}"
        )
    return "\n".join(blocks) if blocks else "(no actuator schemas available)"


def _tool_result_error(result: Any) -> Optional[str]:
    """Detect failure encoded inside a tool's *successful* return value.

    MCP tools often swallow HA errors and return a dict/JSON with
    ``success: false`` or an ``error`` key instead of raising. Those cases
    would otherwise be stamped as ``executed`` in the audit, hiding real
    failures. Returns an error message when the result indicates failure,
    or ``None`` when it looks successful.
    """
    if result is None:
        return None
    payload: Any = result
    if isinstance(result, str):
        text = result.strip()
        if not text:
            return None
        # The MCP client manager emits this exact prefix for isError results.
        if text.startswith("MCP tool error:") or text.startswith("Error:"):
            return text[:300]
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return None  # Non-JSON plain text is treated as success.
    if isinstance(payload, dict):
        if payload.get("success") is False:
            err = payload.get("error") or "tool returned success=false"
            return str(err)[:300]
        if "error" in payload and payload.get("success") is not True:
            err = payload.get("error")
            if err:
                return str(err)[:300]
    return None


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
            ph = _placeholder_reason(args)
            if ph:
                entry["outcome"] = "malformed"
                entry["error"] = f"unresolved placeholder: {ph}"
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
                tool_schemas=_format_actuator_schemas(base_tools, allow_list),
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
        except Exception as e:
            entry["outcome"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"
            if strict:
                # Subsequent pending steps still run — the engine decides
                # overall status based on error count. Strict only changes
                # the final status, not the execution order.
                continue
            continue
        err = _tool_result_error(result)
        if err is not None:
            entry["outcome"] = "error"
            entry["error"] = err
            entry["result"] = _summarize_tool_result(result)
        else:
            entry["outcome"] = "executed"
            entry["result"] = _summarize_tool_result(result)
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
