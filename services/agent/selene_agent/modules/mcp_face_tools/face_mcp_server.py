"""
Face Recognition MCP Server — proxies the agent to the face-recognition service.

Exposes 5 tools to the LLM:
  face_who_is_at, face_recent_visitors, face_list_known_people,
  face_enroll_person, face_set_access_level.

Every call hits the face-recognition service over HTTP at FACE_REC_API_BASE.
Person and camera arguments are fuzzy-matched on the client side against
`GET /api/people` and `GET /api/cameras` so the LLM can pass natural names
("front door", "matt") instead of strict entity_ids — the canonical strings
live in face-recognition, not the prompt.
"""

import asyncio
import difflib
import json
import os
import sys
from typing import Any, Dict, List, Optional

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from selene_agent.utils.logger import get_logger

logger = get_logger('loki')


FACE_REC_API_BASE = os.getenv("FACE_REC_API_BASE", "http://face-recognition:6006").rstrip("/")
HTTP_TIMEOUT_SEC = float(os.getenv("FACE_REC_HTTP_TIMEOUT_SEC", "20"))
URL_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("FACE_REC_URL_DOWNLOAD_TIMEOUT_SEC", "15"))
URL_DOWNLOAD_MAX_BYTES = int(os.getenv("FACE_REC_URL_DOWNLOAD_MAX_BYTES", str(10 * 1024 * 1024)))

ACCESS_LEVELS = ("unknown", "resident", "guest", "blocked")


def _fuzzy_pick(query: str, choices: List[str]) -> List[str]:
    """Resolve `query` against `choices`. Returns the list of candidates
    surfaced by the first stage that matches anything.

    Caller interprets:
      []              → no match (tell the LLM the query failed)
      [one]           → unambiguous match (use it)
      [two_or_more]   → ambiguous (surface candidates, let the LLM disambiguate)

    Stages, in order:
      1. exact
      2. case-insensitive exact
      3. case-insensitive substring (natural-language tolerance:
         "front" → "camera.front_duo_3_fluent")
      4. difflib close-match at cutoff 0.3 (catches "front door" / "frontdoor"
         / "front_door" — empirically all need <=0.3 against the
         entity_id naming convention).

    Each stage short-circuits, so substring matches never compete with
    fuzzy matches; the more specific stage wins.
    """
    if not choices:
        return []

    if query in choices:
        return [query]

    q_lower = query.lower()
    lower_to_orig = {c.lower(): c for c in choices}
    if q_lower in lower_to_orig:
        return [lower_to_orig[q_lower]]

    substring_hits = [c for c in choices if q_lower in c.lower()]
    if substring_hits:
        return substring_hits

    close_lower = difflib.get_close_matches(
        q_lower, [c.lower() for c in choices], n=5, cutoff=0.3,
    )
    if close_lower:
        return [lower_to_orig[c] for c in close_lower]

    return []


class FaceMCPServer:
    def __init__(self):
        self.server = Server("havencore-face-tools")
        self._setup_handlers()

    # --- HTTP helpers --------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        r = requests.get(f"{FACE_REC_API_BASE}{path}", params=params, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        return r.json()

    def _post_json(self, path: str, body: Dict[str, Any]) -> Any:
        r = requests.post(f"{FACE_REC_API_BASE}{path}", json=body, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        return r.json()

    def _patch_json(self, path: str, body: Dict[str, Any]) -> Any:
        r = requests.patch(f"{FACE_REC_API_BASE}{path}", json=body, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        return r.json()

    def _post_multipart(self, path: str, files: Dict[str, Any], data: Dict[str, Any]) -> Any:
        r = requests.post(
            f"{FACE_REC_API_BASE}{path}",
            files=files,
            data=data,
            timeout=HTTP_TIMEOUT_SEC,
        )
        r.raise_for_status()
        return r.json()

    # --- resolution helpers --------------------------------------------

    def _resolve_person(self, name: str) -> Dict[str, Any]:
        """Fuzzy-match a person by name.

        Returns {person_id, name} on a single hit, otherwise {error,
        candidates?, known_people?} so the LLM can disambiguate or retry.
        """
        people = self._get("/api/people")
        names = [p["name"] for p in people]
        hits = _fuzzy_pick(name, names)
        if not hits:
            return {
                "error": f"no person matched '{name}'",
                "known_people": names,
            }
        if len(hits) > 1:
            return {
                "error": f"multiple people matched '{name}'; specify which",
                "candidates": hits,
            }
        person = next(p for p in people if p["name"] == hits[0])
        return {"person_id": person["id"], "name": hits[0]}

    def _resolve_camera(self, camera_arg: str) -> Dict[str, Any]:
        """Fuzzy-match a camera entity_id against /api/cameras.

        Tries the canonical `camera.*_fluent` entity_ids first; falls back to
        the corresponding `binary_sensor.*_person` ids (so a person-sensor
        name still resolves to its camera). Returns {camera} on a single
        hit, otherwise {error, candidates?, known_cameras?}.
        """
        cams = self._get("/api/cameras")
        entity_ids = [c["camera_entity"] for c in cams]
        sensor_ids = [c["sensor_entity"] for c in cams]
        hits = _fuzzy_pick(camera_arg, entity_ids)
        if not hits:
            sensor_hits = _fuzzy_pick(camera_arg, sensor_ids)
            hits = [
                next(c["camera_entity"] for c in cams if c["sensor_entity"] == s)
                for s in sensor_hits
            ]
        if not hits:
            return {
                "error": f"no camera matched '{camera_arg}'",
                "known_cameras": entity_ids,
            }
        if len(hits) > 1:
            return {
                "error": f"multiple cameras matched '{camera_arg}'; specify which",
                "candidates": hits,
            }
        return {"camera": hits[0]}

    # --- tool implementations ------------------------------------------

    def _who_is_at(self, args: Dict[str, Any]) -> Dict[str, Any]:
        camera_arg = args["camera"]
        resolved = self._resolve_camera(camera_arg)
        if "error" in resolved:
            return resolved
        camera = resolved["camera"]
        rows = self._get(
            "/api/detections",
            params={"camera": camera, "since_seconds_ago": 60, "limit": 1},
        )
        if not rows:
            return {"camera": camera, "found": False, "message": "no detection in last 60 seconds"}
        d = rows[0]
        return {
            "camera": camera,
            "found": True,
            "name": d.get("person_name") or "unknown",
            "person_id": d.get("person_id"),
            "confidence": d.get("confidence"),
            "captured_at": d.get("captured_at"),
        }

    def _recent_visitors(self, args: Dict[str, Any]) -> Dict[str, Any]:
        hours = float(args.get("hours", 24))
        if hours <= 0:
            return {"error": "hours must be positive"}
        params: Dict[str, Any] = {
            "since_seconds_ago": int(hours * 3600),
            "limit": 50,
        }
        camera_arg = args.get("camera")
        if camera_arg:
            resolved = self._resolve_camera(camera_arg)
            if "error" in resolved:
                return resolved
            params["camera"] = resolved["camera"]
        rows = self._get("/api/detections", params=params)
        return {
            "hours": hours,
            "camera": params.get("camera"),
            "count": len(rows),
            "visitors": [
                {
                    "name": r.get("person_name") or "unknown",
                    "camera": r.get("camera"),
                    "captured_at": r.get("captured_at"),
                    "confidence": r.get("confidence"),
                }
                for r in rows
            ],
        }

    def _list_known_people(self, _args: Dict[str, Any]) -> Dict[str, Any]:
        people = self._get("/api/people")
        return {
            "count": len(people),
            "people": [
                {
                    "name": p["name"],
                    "image_count": p.get("image_count", 0),
                    "access_level": p.get("access_level"),
                }
                for p in people
            ],
        }

    def _download_url_to_bytes(self, url: str) -> bytes:
        resp = requests.get(url, stream=True, timeout=URL_DOWNLOAD_TIMEOUT_SEC)
        resp.raise_for_status()
        buf = bytearray()
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > URL_DOWNLOAD_MAX_BYTES:
                resp.close()
                raise ValueError(f"url payload exceeds {URL_DOWNLOAD_MAX_BYTES} bytes")
        return bytes(buf)

    def _enroll_person(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args["name"].strip()
        source = args["source"].strip()
        if not name or not source:
            return {"error": "name and source are both required"}

        people = self._get("/api/people")
        names = [p["name"] for p in people]
        hits = _fuzzy_pick(name, names)
        if len(hits) > 1:
            return {
                "error": (
                    f"'{name}' is ambiguous — clarify which person to enroll "
                    f"against, or pass the exact stored name to create a new one"
                ),
                "candidates": hits,
            }
        if hits:
            person = next(p for p in people if p["name"] == hits[0])
            person_id = person["id"]
            display_name = hits[0]
            created = False
        else:
            created_row = self._post_json("/api/people", {"name": name})
            person_id = created_row["id"]
            display_name = created_row["name"]
            created = True

        if source.startswith("camera:"):
            camera_arg = source[len("camera:"):].strip()
            if not camera_arg:
                return {"error": "camera source missing entity, expected 'camera:<entity_id>'"}
            resolved = self._resolve_camera(camera_arg)
            if "error" in resolved:
                return resolved
            result = self._post_json(
                f"/api/people/{person_id}/enroll-from-camera",
                {"camera": resolved["camera"]},
            )
            return {
                "success": True,
                "person_id": person_id,
                "name": display_name,
                "person_created": created,
                "source": f"camera:{resolved['camera']}",
                "face_image_id": result.get("id"),
                "quality_score": result.get("quality_score"),
                "frames_processed": result.get("frames_processed"),
                "faces_kept": result.get("faces_kept"),
            }

        if source.startswith(("http://", "https://")):
            try:
                payload = self._download_url_to_bytes(source)
            except Exception as e:
                return {"error": f"failed to download '{source}': {e}"}
            files = {"file": ("upload.jpg", payload, "application/octet-stream")}
            data = {"source": "agent_enroll", "is_primary": "false"}
            result = self._post_multipart(
                f"/api/people/{person_id}/images", files=files, data=data,
            )
            return {
                "success": True,
                "person_id": person_id,
                "name": display_name,
                "person_created": created,
                "source": source,
                "face_image_id": result.get("id"),
                "quality_score": result.get("quality_score"),
                "faces_detected": result.get("faces_detected"),
            }

        return {
            "error": (
                f"unsupported source '{source}'. Use 'camera:<entity_id>' for a "
                f"live snapshot or an http(s) URL pointing at an image."
            )
        }

    def _set_access_level(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args["name"]
        level = args["level"]
        if level not in ACCESS_LEVELS:
            return {"error": f"level must be one of {list(ACCESS_LEVELS)}"}
        resolved = self._resolve_person(name)
        if "error" in resolved:
            return resolved
        result = self._patch_json(
            f"/api/people/{resolved['person_id']}",
            {"access_level": level},
        )
        return {
            "success": True,
            "name": result["name"],
            "access_level": result["access_level"],
        }

    # --- MCP wiring ----------------------------------------------------

    def _setup_handlers(self):

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="face_who_is_at",
                    description=(
                        "Most recent face detection on `camera` within the last 60 seconds. "
                        "Returns the matched person's name (or 'unknown' if a face was seen but "
                        "not identified), confidence, and timestamp. `camera` is fuzzy-matched "
                        "against the configured camera entity_ids — pass a friendly name like "
                        "'front door' or the full entity_id."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "camera": {"type": "string", "description": "Camera entity_id or alias"},
                        },
                        "required": ["camera"],
                    },
                ),
                Tool(
                    name="face_recent_visitors",
                    description=(
                        "List face detections from the last `hours` hours, newest first. "
                        "Optionally restrict to one camera. Returns up to 50 entries."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hours": {"type": "number", "default": 24, "minimum": 0.1},
                            "camera": {"type": "string", "description": "Optional camera filter"},
                        },
                    },
                ),
                Tool(
                    name="face_list_known_people",
                    description=(
                        "List every enrolled person along with the number of face images on "
                        "file and their access level. Use this before enrollment to check "
                        "whether someone is already known."
                    ),
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="face_enroll_person",
                    description=(
                        "Add a face to the gallery. If `name` matches an existing person "
                        "(fuzzy), that person gets a new face image; otherwise a new person is "
                        "created first. `source` must be either 'camera:<entity_id>' to capture "
                        "a snapshot from a live camera, or an http(s) URL pointing at an image."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "source": {
                                "type": "string",
                                "description": "'camera:<entity_id>' or an http(s) image URL",
                            },
                        },
                        "required": ["name", "source"],
                    },
                ),
                Tool(
                    name="face_set_access_level",
                    description=(
                        "Set a person's access_level to one of: unknown, resident, guest, "
                        "blocked. Stored for future automation policies — v1 has no enforcer."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "level": {
                                "type": "string",
                                "enum": list(ACCESS_LEVELS),
                            },
                        },
                        "required": ["name", "level"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                if name == "face_who_is_at":
                    result = self._who_is_at(arguments)
                elif name == "face_recent_visitors":
                    result = self._recent_visitors(arguments)
                elif name == "face_list_known_people":
                    result = self._list_known_people(arguments)
                elif name == "face_enroll_person":
                    result = self._enroll_person(arguments)
                elif name == "face_set_access_level":
                    result = self._set_access_level(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}
            except requests.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", e.response.text[:300])
                except Exception:
                    detail = e.response.text[:300] if e.response is not None else str(e)
                logger.warning(f"face tool {name} HTTP error: {e} {detail}")
                result = {"error": f"face-recognition service error: {detail}"}
            except requests.RequestException as e:
                logger.warning(f"face tool {name} network error: {e}")
                result = {"error": f"face-recognition service unreachable: {e}"}
            except Exception as e:
                logger.exception(f"face tool {name} failed")
                result = {"error": str(e)}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def run(self):
        options = self.server.create_initialization_options()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, options, raise_exceptions=True)


async def main():
    logger.info("Starting Face Recognition MCP Server...")
    server = FaceMCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
