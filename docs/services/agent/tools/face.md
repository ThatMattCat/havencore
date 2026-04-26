# MCP Server: Face Recognition (`mcp_face_tools`)

Reference doc for the face-recognition MCP server. Surfaces the
`face-recognition` microservice (port 6006) to the LLM as five
function-calling tools.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_face_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_face_tools` |
| Transport | MCP stdio |
| Server name | `havencore-face-tools` |
| Backing service | [face-recognition (port 6006)](../../face-recognition/README.md) |
| Tool count | 5 |

The MCP server is a thin shim over the face-recognition HTTP API. It uses
sync `requests` calls — there's no MCP-side state — and includes a fuzzy
resolver for human-typed person and camera names so the LLM doesn't have
to guess the exact entity_id.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `face_who_is_at(camera)` | Most recent detection on `camera` in the last 60 s. Returns `{name, confidence, captured_at}` (`name="unknown"` if a face was seen but didn't match). |
| `face_recent_visitors(hours=24, camera=None)` | List of detections newest-first; up to 50 entries. Optional camera filter. |
| `face_list_known_people()` | Every enrolled person with `image_count` + `access_level`. Use before enrolling to avoid duplicates. |
| `face_enroll_person(name, source)` | Add a face image to the gallery. `source` is either `"camera:<entity_id>"` (live snapshot) or an `http(s)://` URL. New people are created on the fly when `name` doesn't fuzzy-match an existing one. |
| `face_set_access_level(name, level)` | `level ∈ {unknown, resident, guest, blocked}`. Persisted on `people.access_level`. v1 has no enforcer — the field is reserved for later automation policies. |

## Fuzzy resolution

People and cameras are tiny lists, so resolution lives MCP-side rather
than in face-recognition (avoids an extra HTTP round-trip per call). The
resolver runs four short-circuiting stages against the candidate list:

1. Exact match
2. Case-insensitive exact
3. Case-insensitive substring
4. `difflib.get_close_matches` at cutoff 0.3

Each stage returns a **list** so the caller can distinguish unambiguous
from ambiguous matches. When more than one candidate survives, the tool
returns `{"error": ..., "candidates": [...]}` and the LLM is expected to
disambiguate with the user instead of guessing. (An earlier
implementation broke ties with `min(matches, key=len)` and would have
silently returned the wrong camera for queries like `"backyard"` —
ambiguity now surfaces explicitly.)

The substring + `get_close_matches` cutoff is intentionally generous:
empirically every natural-language LLM query (`"front door"`,
`"frontdoor"`, `"front_door"`, `"front"`) needed cutoff ≤ 0.3 to resolve
against entity_ids like `camera.front_duo_3_fluent`.

## Configuration

| Var | Default | Purpose |
|-----|---------|---------|
| `FACE_REC_API_BASE` | `http://face-recognition:6006` | Base URL the MCP shim hits. The default works for the standard compose layout. |

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "face",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_face_tools"],
  "enabled": true
}
```

If you're updating an existing deployment, merge this entry into your
`.env` `MCP_SERVERS` JSON and bounce the agent
(`docker compose down agent && up -d agent`).

## Internals worth knowing

- **Enrollment from URL is downloaded MCP-side**, not by face-recognition
  itself, with a 10 MB cap. Keeps the face-recognition service from making
  outbound HTTP to arbitrary URLs.
- **`face_enroll_person` with `source="camera:<entity_id>"`** routes to
  `POST /api/people/{id}/enroll-from-camera` on the face-recognition
  service. That endpoint deliberately does NOT create a `face_detections`
  row or publish MQTT — enrollment is not detection.
- **`face_who_is_at` returns the most recent matching detection within
  60 s**, not "is there a face on the camera right now." There's no
  separate "look now" tool because the LLM can compose
  `face_who_is_at` after asking the user to wait — and live triggering
  belongs on the camera-side automation, not on demand from chat.

## Usage patterns

The system prompt encourages composing these tools naturally:

- **"Who's at the front door?"** → `face_who_is_at("front door")`
- **"Who's been around today?"** → `face_recent_visitors(hours=24)`
- **"Add this person — they're called Sam"** → check
  `face_list_known_people()`, then `face_enroll_person("Sam",
  "camera:camera.front_duo_3_fluent")`
- **"Mark Sam as a resident"** → `face_set_access_level("Sam", "resident")`

## Troubleshooting

### Tool calls fail with `face-recognition unreachable`

The `face-recognition` container isn't responding. Check
`docker compose ps face-recognition` and `curl http://localhost:6006/health`.
The agent and face-recognition are both on the compose network — the
default `FACE_REC_API_BASE` should resolve unless the service name
changed.

### `face_enroll_person("Sam", "camera:...")` returns "no face cleared quality floor"

The burst-capture worked but no frame had a face above
`FACE_REC_QUALITY_FLOOR`. This is most often a lighting / angle issue
on outdoor wide-angle cameras at the moment of capture. Try again, or
enroll from a URL / uploaded photo.

### Fuzzy resolver returns "ambiguous candidates"

Working as intended — the resolver refuses to pick when more than one
person/camera matches. The tool's response includes `candidates`; the
LLM should re-ask the user with that list.

## Related files

- `services/agent/selene_agent/modules/mcp_face_tools/face_mcp_server.py` —
  MCP server + fuzzy resolver
- `services/face-recognition/app/api/people.py` and
  `services/face-recognition/app/api/detections.py` — backing endpoints

## See also

- [Face Recognition Service](../../face-recognition/README.md) — backing service docs
- [API Reference → Face recognition (agent proxy)](../../../api-reference.md#face-recognition-agent-proxy)
