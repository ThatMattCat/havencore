"""LLM-judged reactive triage handler.

Fires from an MQTT / webhook event (same trigger_spec matcher as ``watch``),
but where ``watch`` is deterministic ("if payload.state == 'open' notify"),
``watch_llm`` hands the event + a bounded state gather to the LLM and asks
for a single JSON judgment. Intended for cases where "unusual enough to
notify" can't be expressed as a simple condition.

Config::

    {
      "subject": "front-door-after-midnight",
      "gather": {
          "entities": ["binary_sensor.front_door", "person.matt"],
          "memories_k": 3,
          "presence": true,             # gather ha_get_presence
          "recent_visitors_hours": 6    # gather face_recent_visitors
      },
      "notify": {
          "channel": "signal" | "ha_push" | "speaker",
          "to": "...",
          "device": "...",      # speaker only
          "voice": "...",       # speaker only
          "volume": 0.5         # speaker only
      },
      "severity_floor": "low" | "med" | "high",
      "cooldown_min": 15,
      "attach_snapshot": true   # forward sensor_event.snapshot_url to notifier
    }

The LLM may override the configured ``notify.channel`` per-event by emitting
``"channel": ...`` in its JSON response. Useful when urgency depends on
context the agenda item can't predict — e.g. "stranger at door + you're
home" → speaker; same event + you're away → signal with snapshot.

Returns the same ``_unusual`` / ``_notify_*`` / ``signature_hash`` shape as
``anomaly_sweep`` so the engine's shared cooldown + notification path picks
it up without special-casing.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from selene_agent.autonomy.handlers.anomaly import _extract_json, _VALID_SEVERITY
from selene_agent.autonomy.turn import AutonomousTurn
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


WATCH_LLM_SYSTEM_PROMPT = (
    "You are {agent_name}'s autonomous event triage. You will receive an "
    "inbound event plus a short state gather of relevant entities and "
    "memories. Decide whether this event warrants notifying the resident. "
    "You are running autonomously. Do not ask questions. Do not call tools. "
    "Respond with ONE JSON object and NOTHING else — no prose, no code "
    "fences. Schema: "
    "{{"
    "\"unusual\": boolean, "
    "\"severity\": \"low\"|\"med\"|\"high\" (or \"none\" when unusual=false), "
    "\"summary\": string (<=120 chars, terse; '' when nominal), "
    "\"signature\": string (stable slug for dedup; 'nominal' when unusual=false), "
    "\"evidence\": array of up to 3 short bullet strings, "
    "\"channel\": \"signal\"|\"ha_push\"|\"speaker\"|\"silent\" (optional; pick "
    "based on urgency + presence — speaker only when residents are plausibly "
    "home and awake; signal for time-sensitive when away; ha_push for "
    "low-priority log; silent forces unusual=false), "
    "\"urgency\": \"info\"|\"warn\"|\"alert\" (optional)"
    "}}. "
    "Be conservative. Routine events that match the expected pattern "
    "should be marked unusual=false. When the event involves a sensor_event "
    "block, prefer reasoning about subject/zone/quality over raw camera "
    "entity_ids — the same logic should generalize across deployments."
)


_SEVERITY_RANK = {"none": 0, "low": 1, "med": 2, "high": 3}
_VALID_CHANNELS = {"signal", "ha_push", "speaker", "silent"}
_VALID_URGENCY = {"info", "warn", "alert"}


_DEFAULT_SCENE_PROMPT = (
    "Briefly describe what is visible in this camera image. Note: people "
    "(clothing, posture, what they're holding), animals, vehicles, packages, "
    "and anything that looks unusual for a residential property. 2-3 sentences. "
    "No speculation about intent."
)


async def _safe_tool(mcp: MCPClientManager, name: str, args: Dict[str, Any]) -> Any:
    try:
        return await mcp.execute_tool(name, args)
    except Exception as e:
        logger.warning(f"[watch_llm] tool {name} failed: {e}")
        return f"<tool {name} failed: {e}>"


def _extract_snapshot_url(event: Dict[str, Any]) -> Optional[str]:
    """Pull the camera snapshot URL out of a trigger event.

    Prefers the normalized ``sensor_event`` block (which already resolves
    face/* topics to the agent-internal ``/api/face/detections/{id}/snapshot``
    URL). Falls back to a raw-payload ``snapshot_url`` for events that
    didn't go through the sensor_events normalizer.
    """
    if not isinstance(event, dict):
        return None
    se = event.get("sensor_event")
    if isinstance(se, dict) and se.get("snapshot_url"):
        return str(se["snapshot_url"])
    payload = event.get("payload")
    if isinstance(payload, dict) and payload.get("snapshot_url"):
        return str(payload["snapshot_url"])
    return None


async def _gather(
    mcp: MCPClientManager,
    item_config: Dict[str, Any],
    event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    gather_cfg = item_config.get("gather") or {}
    entities: List[str] = [e for e in (gather_cfg.get("entities") or []) if e]
    memories_k = int(gather_cfg.get("memories_k") or 3)
    subject = str(item_config.get("subject") or "event context").strip()

    calls: Dict[str, Any] = {}
    for ent in entities:
        # Prefer a bounded history window when available, else fall back to
        # a single current-state read via get_domain_entity_states.
        domain = ent.split(".", 1)[0] if "." in ent else ""
        if domain:
            calls[f"state_{ent}"] = _safe_tool(
                mcp,
                "ha_get_entity_history",
                {"entity_id": ent, "hours": 6, "limit": 20},
            )
    if memories_k > 0:
        calls["memories"] = _safe_tool(
            mcp, "search_memories", {"query": subject, "limit": memories_k}
        )
    if gather_cfg.get("presence"):
        calls["presence"] = _safe_tool(mcp, "ha_get_presence", {})
    visitor_hours = gather_cfg.get("recent_visitors_hours")
    if visitor_hours:
        try:
            hours = float(visitor_hours)
        except (TypeError, ValueError):
            hours = 6.0
        calls["recent_visitors"] = _safe_tool(
            mcp, "face_recent_visitors", {"hours": hours}
        )
    if gather_cfg.get("scene_description"):
        snapshot_url = _extract_snapshot_url(event or {})
        if snapshot_url:
            scene_prompt = (
                gather_cfg.get("scene_description_prompt") or _DEFAULT_SCENE_PROMPT
            )
            calls["scene_description"] = _safe_tool(
                mcp,
                "query_multimodal_api",
                {"image_url": snapshot_url, "text": scene_prompt},
            )

    if not calls:
        return {}
    results = await asyncio.gather(*calls.values(), return_exceptions=False)
    state = dict(zip(calls.keys(), results))

    # The face_recent_visitors tool returns up to 50 entries; that's enough
    # to balloon the prompt past the LLM's effective input budget on a
    # busy household. Cap to the most-recent 8, plus a tail summary so the
    # LLM still knows the volume.
    visitors = state.get("recent_visitors")
    if isinstance(visitors, dict) and isinstance(visitors.get("visitors"), list):
        all_v = visitors["visitors"]
        if len(all_v) > 8:
            visitors = dict(visitors)
            visitors["visitors"] = all_v[:8]
            visitors["truncated"] = len(all_v) - 8
            state["recent_visitors"] = visitors
    return state


def _format_sensor_event(se: Dict[str, Any]) -> str:
    """Render the normalized sensor_event block as a compact, LLM-friendly
    summary instead of the raw dict. Designed so the LLM can reason in terms
    of zone/subject/quality without parsing JSON itself.
    """
    parts: List[str] = []
    parts.append(f"domain={se.get('domain')!r} kind={se.get('kind')!r}")
    if se.get("zone"):
        parts.append(
            f"zone={se['zone']!r}"
            + (f" ({se['zone_label']})" if se.get('zone_label') else "")
        )
    elif se.get("camera_entity"):
        parts.append(f"camera_entity={se['camera_entity']!r} (no zone mapped)")
    subj = se.get("subject")
    if subj:
        ident = subj.get("identity") or "<unknown>"
        sline = f"subject={ident}"
        if subj.get("type"):
            sline += f" type={subj['type']}"
        if subj.get("confidence") is not None:
            sline += f" confidence={subj['confidence']:.2f}"
        if subj.get("quality") is not None:
            sline += f" quality={subj['quality']:.2f}"
        parts.append(sline)
    if se.get("captured_at"):
        parts.append(f"captured_at={se['captured_at']}")
    if se.get("snapshot_url"):
        parts.append(f"snapshot_url={se['snapshot_url']}")
    return "\n".join(f"- {p}" for p in parts)


def _render_user_prompt(
    subject: str, event: Dict[str, Any], state: Dict[str, Any]
) -> str:
    from zoneinfo import ZoneInfo

    tz = config.CURRENT_TIMEZONE or "UTC"
    local = datetime.now(ZoneInfo(tz))
    lines = [
        f"Subject: {subject or '(none)'}",
        f"Current local time: {local.strftime('%A %Y-%m-%d %H:%M %Z')}",
        "",
        "## Inbound event",
        f"source={event.get('source')!r} topic={event.get('topic')!r} "
        f"name={event.get('name')!r}",
    ]
    sensor_event = event.get("sensor_event")
    if isinstance(sensor_event, dict):
        lines += ["", "### Normalized sensor event", _format_sensor_event(sensor_event)]
        # Still expose raw payload for handlers that need granular fields.
        lines += [f"raw_payload={sensor_event.get('raw')!r}"]
    else:
        lines += [f"payload={event.get('payload')!r}"]
    for key, value in state.items():
        if key == "memories":
            lines += ["", "## Relevant memories", str(value)]
        else:
            lines += ["", f"## {key}", str(value)]
    lines += ["", "Output the JSON object only. No other text."]
    return "\n".join(lines)


def _signature_hash(item_id: str, sig: str) -> str:
    raw = f"{item_id}:{sig}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _deterministic_signature(sensor_event: Dict[str, Any]) -> Optional[str]:
    """Derive a stable dedup signature from a normalized sensor_event.

    Shape: ``{domain}:{kind}:{zone}:{subject_key}`` where ``subject_key`` is
    the subject identity (e.g. resident name) when known, else the subject
    type (e.g. ``person``, ``unknown``). Returns None when domain/kind are
    missing — caller should fall back to the LLM-emitted signature.

    Why we override the LLM here: the engine's signature cooldown only fires
    when repeat events hash to the same key, but the LLM's free-form
    ``signature`` field drifts across near-identical events ("backyard_motion"
    vs "person_backyard_evening"), defeating dedup. The sensor_event already
    carries zone + subject after enrichment; using those directly keeps
    cooldown stable per (zone, subject) without depending on LLM consistency.
    """
    domain = str(sensor_event.get("domain") or "").strip()
    kind = str(sensor_event.get("kind") or "").strip()
    if not domain or not kind:
        return None
    zone = str(sensor_event.get("zone") or "no_zone").strip() or "no_zone"
    subj = sensor_event.get("subject") or {}
    if isinstance(subj, dict):
        subject_key = (
            str(subj.get("identity") or "").strip()
            or str(subj.get("type") or "").strip()
            or "unknown"
        )
    else:
        subject_key = "unknown"
    return f"{domain}:{kind}:{zone}:{subject_key}"


def _anyone_home(presence_payload: Any) -> Optional[bool]:
    """Return True if any person.* entity is in state 'home', False if at
    least one is known to be away and none home, None if presence data is
    unavailable / unparseable. Conservative: if we can't tell, return None
    and let the caller decide.
    """
    if not isinstance(presence_payload, dict):
        return None
    persons = presence_payload.get("persons")
    if not isinstance(persons, list) or not persons:
        return None
    states = [str(p.get("state") or "").lower() for p in persons]
    if any(s == "home" for s in states):
        return True
    if all(s and s != "home" for s in states):
        return False
    return None


def _resolve_channel(
    llm_choice: Optional[str],
    cfg_default: str,
    presence_payload: Any,
) -> Tuple[str, Optional[str]]:
    """Pick the final notification channel given LLM output + safety rails.

    Returns ``(channel, downgrade_reason)``. ``channel`` is always one of
    signal/ha_push/speaker/silent. ``downgrade_reason`` is non-None when we
    chose something different than the LLM asked for (logged for ops visibility).
    """
    requested = (llm_choice or "").strip().lower()
    if requested not in _VALID_CHANNELS:
        return (cfg_default, None) if cfg_default in _VALID_CHANNELS else ("signal", None)
    if requested == "speaker":
        # Speaker only fires when somebody could plausibly hear it. If presence
        # data says nobody is home, fall back to signal so the alert reaches them.
        home = _anyone_home(presence_payload)
        if home is False:
            return "signal", "no one home — speaker downgraded to signal"
    return requested, None


async def handle(
    item: Dict[str, Any],
    *,
    client,
    mcp_manager: MCPClientManager,
    model_name: str,
    base_tools: List[Dict[str, Any]],
    provider_getter=None,
) -> Dict[str, Any]:
    cfg = item.get("config") or {}
    event = item.get("_trigger_event") or {}
    subject = str(cfg.get("subject") or item.get("name") or "watch_llm").strip()
    severity_floor = str(cfg.get("severity_floor") or "low").lower()
    if severity_floor not in _SEVERITY_RANK:
        severity_floor = "low"

    state = await _gather(mcp_manager, cfg, event)
    user_prompt = _render_user_prompt(subject, event, state)

    try:
        turn = AutonomousTurn(
            client=client,
            mcp_manager=mcp_manager,
            model_name=model_name,
            base_tools=base_tools,
            autonomy_level=item.get("autonomy_level", "notify"),
            system_prompt=WATCH_LLM_SYSTEM_PROMPT.format(
                agent_name=config.AGENT_NAME or "Selene"
            ),
            timeout_sec=int(cfg.get("timeout_sec") or config.AUTONOMY_TURN_TIMEOUT_SEC),
            temperature=0.2,
            # GLM-4.5's reasoning parser consumes the generation budget on
            # CoT before producing the JSON answer; 400 tokens is enough for
            # a non-reasoning model but leaves content empty here. 2000 is
            # plenty for a strict short-JSON response.
            max_tokens=2000,
            provider_getter=provider_getter,
        )
    except ValueError as e:
        return {
            "status": "error",
            "summary": "tier invalid for watch_llm",
            "error": str(e),
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    result = await turn.run(user_prompt)
    parsed = _extract_json(result.content)
    if parsed is None:
        logger.warning(
            f"[watch_llm] LLM did not produce valid JSON for {item.get('id')}"
        )
        return {
            "status": "error",
            "summary": "watch_llm: invalid JSON output",
            "severity": None,
            "signature_hash": None,
            "notified_via": None,
            "messages": result.messages,
            "metrics": result.metrics,
            "error": result.error or f"unparseable: {result.content[:200]}",
        }

    unusual = bool(parsed.get("unusual"))
    severity = str(parsed.get("severity") or "none").lower()
    if severity not in _VALID_SEVERITY:
        severity = "none"
    summary_text = str(parsed.get("summary") or "").strip()
    llm_signature = str(parsed.get("signature") or "").strip() or "unnamed"
    sensor_event = event.get("sensor_event") if isinstance(event, dict) else None
    deterministic = (
        _deterministic_signature(sensor_event)
        if isinstance(sensor_event, dict)
        else None
    )
    signature = deterministic or llm_signature
    sig_hash = _signature_hash(item["id"], signature)

    llm_channel = str(parsed.get("channel") or "").strip().lower() or None
    urgency = str(parsed.get("urgency") or "").strip().lower()
    if urgency not in _VALID_URGENCY:
        urgency = ""

    # ``silent`` is the LLM's way of saying "I judged this notable but on
    # reflection don't notify" — collapse it to nominal so the cooldown
    # bookkeeping treats it consistently.
    if llm_channel == "silent":
        unusual = False

    if not unusual:
        return {
            "status": "ok",
            "summary": "nominal",
            "severity": "none",
            "signature_hash": _signature_hash(item["id"], "nominal"),
            "notified_via": None,
            "messages": result.messages,
            "metrics": result.metrics,
            "error": None,
            "_unusual": False,
        }

    # Severity floor gate.
    if _SEVERITY_RANK[severity] < _SEVERITY_RANK[severity_floor]:
        return {
            "status": "ok",
            "summary": f"below severity floor ({severity} < {severity_floor})",
            "severity": severity,
            "signature_hash": sig_hash,
            "notified_via": None,
            "messages": result.messages,
            "metrics": result.metrics,
            "error": None,
            "_unusual": False,
        }

    evidence = parsed.get("evidence") or []
    body_lines = [summary_text or "Unusual event detected."]
    if evidence:
        body_lines.append("")
        for e in evidence[:3]:
            body_lines.append(f"- {str(e)[:200]}")

    notify = cfg.get("notify") or {}
    cfg_channel = str(notify.get("channel") or "signal").lower()
    chosen_channel, downgrade = _resolve_channel(
        llm_channel, cfg_channel, state.get("presence")
    )
    if downgrade:
        logger.info(f"[watch_llm] {item.get('id')}: {downgrade}")
    to = notify.get("to") or ""

    # Snapshot attachment passthrough — only when the source event carries a
    # synthesized URL and the channel actually supports binary attachments.
    attachments: Optional[List[str]] = None
    snapshot_url = (
        sensor_event.get("snapshot_url") if isinstance(sensor_event, dict) else None
    )
    attach_snapshot = bool(cfg.get("attach_snapshot")) or bool(
        (cfg.get("notify") or {}).get("attach_snapshot")
    )
    if attach_snapshot and snapshot_url and chosen_channel == "signal":
        attachments = [snapshot_url]

    # The notification title is user-facing — keep it short. The LLM's
    # summary field is already a terse <=120-char description of the event,
    # so we derive the title from that. ``subject`` was historically used
    # here but it is LLM-prompt context (long), not a notification title.
    title_text = (summary_text or "event")[:100]
    return {
        "status": "ok",
        "summary": (summary_text or "unusual event")[:200],
        "severity": severity,
        "signature_hash": sig_hash,
        "signature_raw": llm_signature,
        "notified_via": None,
        "messages": result.messages,
        "metrics": result.metrics,
        "error": None,
        "_unusual": True,
        "_notify_title": f"{config.AGENT_NAME or 'Selene'}: {title_text}",
        "_notify_body": "\n".join(body_lines)[:1000],
        "_notify_channel": chosen_channel,
        "_notify_to": to,
        "_notify_attachments": attachments,
        "_notify_urgency": urgency or None,
        # Pass speaker sub-config through so the engine's _build_notifier
        # can hand device/voice/volume into SpeakerNotifier.
        "_notify_cfg": {
            "speaker": {
                "device": notify.get("device"),
                "voice": notify.get("voice"),
                "volume": notify.get("volume"),
            }
        },
    }
