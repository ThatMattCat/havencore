"""
Home Assistant proxy API router — thin proxies to HA REST API for the dashboard.
"""

import requests as http_requests
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()


def _ha_request(path: str, timeout: int = 10) -> dict:
    """Make an authenticated request to the Home Assistant REST API"""
    if not config.HAOS_URL or not config.HAOS_TOKEN:
        raise HTTPException(status_code=503, detail="Home Assistant not configured")

    url = f"{config.HAOS_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {config.HAOS_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        resp = http_requests.get(url, headers=headers, timeout=timeout, verify=bool(config.HAOS_USE_SSL))
        resp.raise_for_status()
        return resp.json()
    except http_requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="Cannot connect to Home Assistant")
    except http_requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Home Assistant request timed out")
    except http_requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Home Assistant error: {e}")
    except Exception as e:
        logger.error(f"HA proxy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ha/entities")
async def get_entities(domain: Optional[str] = Query(None, description="Filter by domain (e.g. light, switch, media_player)")):
    """Get Home Assistant entity states, optionally filtered by domain"""
    states = _ha_request("states")

    if domain:
        states = [s for s in states if s.get("entity_id", "").startswith(f"{domain}.")]

    # Summarize for the dashboard
    entities = []
    for state in states:
        entities.append({
            "entity_id": state.get("entity_id"),
            "state": state.get("state"),
            "friendly_name": state.get("attributes", {}).get("friendly_name", ""),
            "domain": state.get("entity_id", "").split(".")[0] if "." in state.get("entity_id", "") else "",
            "last_changed": state.get("last_changed"),
        })

    return {"entities": entities, "count": len(entities)}


@router.get("/ha/entities/summary")
async def get_entity_summary():
    """Get a summary of entity counts by domain"""
    states = _ha_request("states")

    domain_counts = {}
    domain_active = {}
    for state in states:
        entity_id = state.get("entity_id", "")
        if "." not in entity_id:
            continue
        domain = entity_id.split(".")[0]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Count "active" entities (on, playing, open, etc.)
        s = state.get("state", "").lower()
        if s in ("on", "playing", "open", "home", "active"):
            domain_active[domain] = domain_active.get(domain, 0) + 1

    summary = []
    for domain in sorted(domain_counts.keys()):
        summary.append({
            "domain": domain,
            "total": domain_counts[domain],
            "active": domain_active.get(domain, 0),
        })

    return {"domains": summary}


@router.get("/ha/automations")
async def get_automations():
    """Get Home Assistant automations"""
    states = _ha_request("states")
    automations = [
        {
            "entity_id": s.get("entity_id"),
            "state": s.get("state"),
            "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
            "last_triggered": s.get("attributes", {}).get("last_triggered"),
        }
        for s in states
        if s.get("entity_id", "").startswith("automation.")
    ]
    return {"automations": automations, "count": len(automations)}


@router.get("/ha/scenes")
async def get_scenes():
    """Get Home Assistant scenes"""
    states = _ha_request("states")
    scenes = [
        {
            "entity_id": s.get("entity_id"),
            "friendly_name": s.get("attributes", {}).get("friendly_name", ""),
        }
        for s in states
        if s.get("entity_id", "").startswith("scene.")
    ]
    return {"scenes": scenes, "count": len(scenes)}
