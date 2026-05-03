# face-recognition

InsightFace-backed face detection + identification service for HavenCore.
Loads the `buffalo_l` pack (RetinaFace detector + ArcFace R100 embedder) on a
dedicated GPU and exposes an HTTP API consumed by the agent's
`mcp_face_tools` MCP module, the companion-app `who_is_in_view` flow, the
SvelteKit `/people/*` dashboard, and an MQTT bridge that turns Home
Assistant person-sensor triggers into identified / unknown events.

## HTTP API

All endpoints are served on port `6006` (`FACE_REC_PORT`). Routers:
`api/people.py`, `api/detections.py`, `api/face_images.py`, `api/identify.py`,
`api/cameras.py`, `api/admin.py`. The `/health` endpoint is defined directly
in `app/main.py`.

### Health

`GET /health` — readiness + subsystem status. Returns:

```json
{
  "ready": true,
  "enabled": true,
  "model": {
    "pack": "buffalo_l",
    "ctx_id": 0,
    "det_size": 1280,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "load_seconds": 4.2
  },
  "gpu_device_label": "3",
  "db": "ok",
  "qdrant": {"status": "ok", "collection": "faces", "dim": 512},
  "mqtt": {
    "enabled": true,
    "connected": true,
    "broker": "mosquitto:1883",
    "subscribed_topics": ["haven/face/trigger/+"]
  },
  "retention": { "...": "..." }
}
```

`ready: true` requires the embedder loaded **and** Postgres reachable **and**
Qdrant reachable. CPU-only fallback (no `CUDAExecutionProvider`) is logged as
a warning at startup but still answers `ready: true`.

### Identify (one-shot)

`POST /api/identify` (multipart `file`) — runs detect+embed on a single image
and returns the nearest gallery match above `FACE_REC_MATCH_THRESHOLD`. Does
not write a `face_detections` row, store the snapshot, or publish MQTT — it
is the agent's companion-camera side-channel asking "who is this?" once.

Returns 200 in all face-not-recognized cases:

```json
{"found": true, "name": "Matt", "person_id": "...", "confidence": 0.71, "face_count": 1}
{"found": false, "face_count": 0}
{"found": false, "face_count": 1, "confidence": 0.34}
```

### People + enrollment

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/people` | Create person (`{name, access_level?, notes?}`) |
| `GET` | `/api/people` | List all people with `image_count` and `access_level` |
| `GET` | `/api/people/{id}` | Person detail + face_image rows |
| `PATCH` | `/api/people/{id}` | Update `access_level` and/or `notes` |
| `DELETE` | `/api/people/{id}` | Delete person, unlink JPEGs, drop Qdrant points (detection rows survive with `person_id` NULL'd) |
| `POST` | `/api/people/{id}/images` | Multipart upload — adds one face image to the gallery |
| `POST` | `/api/people/{id}/enroll-from-camera` | Burst-capture from an HA camera, pick the best face, enroll it |
| `DELETE` | `/api/people/{id}/images/{image_id}` | Remove one face image (refuses to delete `is_primary`) |
| `POST` | `/api/people/{id}/images/{image_id}/set-primary` | Atomically swap the primary marker |

`access_level` is one of `unknown`, `resident`, `guest`, `blocked` (stored
for downstream policy use; the service itself does not gate on it).

### Detections

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/trigger?camera=...` | Run the full pipeline once on demand for an HA camera entity |
| `GET` | `/api/detections` | List detections; filters: `camera`, `since_seconds_ago`, `person_id`, `review_state`, `unknowns_only`, `limit` |
| `GET` | `/api/detections/{id}/snapshot` | Stream the captured JPEG |
| `POST` | `/api/detections/{id}/confirm` | Confirm an unknown detection — re-embeds from disk and attaches to a person (`person_id` or `name`) |
| `POST` | `/api/detections/{id}/reject` | Mark an unknown detection rejected so it stops showing up in the queue |
| `POST` | `/api/detections/bulk-delete` | Mass-delete unknowns by scope |

### Face images

`GET /api/face_images/{id}/bytes` — stream a face image JPEG by id (identity
URL; no on-disk path leaks to the client).

### Cameras

`GET /api/cameras` — discovers `binary_sensor.*_person` entities in Home
Assistant and pairs each with its snapshot camera entity (from the `cameras`
Postgres table when registered, otherwise falling back to the Reolink
`_clear` naming convention). Returns `sensor_entity`, `camera_entity`,
`camera_exists`, `current_state`, `registered`, `fov_type`,
`native_resolution`. Requires `HAOS_URL` + `HAOS_TOKEN`.

### Admin / maintenance

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/admin/retention/sweep` | Run one retention pass right now |
| `POST` | `/api/admin/rebuild-embeddings` | Re-embed every `face_images` row from disk; reconciles drift between Postgres and Qdrant. Returns a `job_id` |
| `POST` | `/api/admin/rescan-unknowns` | Re-match every unknown detection against the current Qdrant index; matches above `FACE_REC_MATCH_THRESHOLD` flip to `confirmed` |
| `GET` | `/api/admin/jobs/{job_id}` | Poll a long-running admin job |
| `GET` | `/api/admin/jobs?limit=20` | Most-recent jobs first |

Both `rebuild-embeddings` and `rescan-unknowns` are single-flight per
process (a second concurrent request returns 409 with the in-flight
`job_id`).

## Consumers

### `mcp_face_tools` (agent MCP module)

`services/agent/selene_agent/modules/mcp_face_tools/face_mcp_server.py`
proxies five tools to the LLM, all over HTTP at `FACE_REC_API_BASE`
(default `http://face-recognition:6006`):

| Tool | Endpoint(s) used |
|---|---|
| `face_who_is_at` | `GET /api/cameras` (fuzzy resolve), `GET /api/detections?camera=…&since_seconds_ago=60&limit=1` |
| `face_recent_visitors` | `GET /api/detections?since_seconds_ago=…&limit=50` |
| `face_list_known_people` | `GET /api/people` |
| `face_enroll_person` | `POST /api/people` (get-or-create), then either `POST /api/people/{id}/enroll-from-camera` (`source: camera:<entity>`) or `POST /api/people/{id}/images` (http(s) URL) |
| `face_set_access_level` | `GET /api/people` (fuzzy resolve), `PATCH /api/people/{id}` |

Person and camera arguments are fuzzy-matched on the agent side so the LLM
can pass natural names like `"front door"` or `"matt"` instead of canonical
entity_ids.

### `who_is_in_view` (companion-app camera path)

`mcp_device_action_tools` exposes `who_is_in_view` to the LLM. The
orchestrator captures a photo from the user's companion app, then POSTs the
image as multipart to `/api/identify` and returns the matched name (or
`unknown`) to the LLM.

### SvelteKit dashboard

`/people`, `/people/{id}`, `/people/detections`, `/people/unknowns` consume
the same endpoints listed above to render the gallery, enrollment UI, and
unknown-detection review queue.

## MQTT bridge + autonomy

The bridge (`app/mqtt_bridge.py`) subscribes to `haven/face/trigger/+`. Each
message's topic suffix is the HA camera entity_id; the payload may include
`event_id`, `captured_at`, and `source`. For each trigger the pipeline
captures a burst from HA, scores faces, persists a `face_detections` row,
and publishes one of:

| Topic | Outcome |
|---|---|
| `haven/face/identified` | A face matched a known person above threshold |
| `haven/face/unknown` | A face was found but no gallery match |
| `haven/face/no_face` | Frames captured but no face cleared `FACE_REC_QUALITY_FLOOR` |
| `haven/face/status` | Lifecycle ticks: `capturing` → `matching` → `idle` |

(The `no_frames` outcome — capture failed before InsightFace ran — is
intentionally not published; nothing actionable for downstream subscribers.)

The agent's autonomy engine subscribes to the three result topics via
`watch_llm` agenda items seeded in
`services/agent/selene_agent/autonomy/seeds/camera_events.py`. The
`sensor_events` normalizer translates each payload into a `SensorEvent` and
maps the `camera_entity` to a generic zone slug via the `camera_zones`
Postgres table (LISTEN/NOTIFY-refreshed in-memory mirror). The LLM then
decides whether the event is unusual enough to notify and which channels to
use.

A representative HA automation that fans person-sensor state changes into
the right `haven/face/trigger/<camera>` topic is documented in the
`mqtt_bridge.py` module docstring.

## GPU pinning

Pinned via `compose.yaml` to `CUDA_VISIBLE_DEVICES=3`. Inside the container
the GPU appears as `cuda:0`, so `FACE_REC_CTX_ID=0`. To move to a different
host GPU, change `CUDA_VISIBLE_DEVICES` in the compose block — `ctx_id` does
not change.

## Smoke test

```bash
docker compose build face-recognition
docker compose up -d face-recognition
docker compose logs -f face-recognition       # expect "InsightFace buffalo_l loaded on CUDA"
curl http://localhost:6006/health
nvidia-smi                                    # face-recognition process on GPU 3
```

`/health` should return `ready: true` and `model.providers` containing
`CUDAExecutionProvider`. CPU-only fallback is logged as a warning but the
service still starts.

## Configuration

All env vars are optional; defaults are below. `.env` changes require
`docker compose down && up -d face-recognition` (env vars are read at
container start).

### Service / model

| Var | Default | Notes |
|---|---|---|
| `FACE_REC_ENABLED` | `true` | Set false to skip model load (debugging only) |
| `FACE_REC_PORT` | `6006` | |
| `FACE_REC_MODEL_PACK` | `buffalo_l` | InsightFace pack name |
| `FACE_REC_CTX_ID` | `0` | Local CUDA index after `CUDA_VISIBLE_DEVICES` |
| `FACE_REC_DET_SIZE` | `1280` | RetinaFace input edge length; `640` for legacy low-res cameras, `1280` calibrated for high-res Reolink `_clear` feeds and panoramic dual-lens cameras (tiled per-half before detection) |
| `FACE_REC_GPU_DEVICE` | `3` | Informational label surfaced in `/health`; the actual pin is in compose |

### Pipeline thresholds

| Var | Default | Notes |
|---|---|---|
| `FACE_REC_MATCH_THRESHOLD` | `0.50` | Cosine-sim cutoff for "this is a known person" |
| `FACE_REC_IMPROVEMENT_THRESHOLD` | `0.65` | Match must clear this to additionally feed the gallery |
| `FACE_REC_QUALITY_FLOOR` | `0.40` | Faces below this quality are discarded outright |
| `FACE_REC_IMPROVEMENT_QUALITY_FLOOR` | `0.65` | Quality cutoff for adding a detection to the gallery |
| `FACE_REC_MAX_EMBEDDINGS_PER_PERSON` | `50` | FIFO cap on stored face images per person |

### Capture (burst)

| Var | Default | Notes |
|---|---|---|
| `FACE_REC_TRIGGER_MODE` | `ha_person_detected` | Reserved for future trigger sources |
| `FACE_REC_BURST_FRAMES` | `6` | Frames per trigger |
| `FACE_REC_BURST_INTERVAL_MS` | `500` | Spacing between burst frames |

### Snapshots / retention

| Var | Default | Notes |
|---|---|---|
| `FACE_REC_SNAPSHOT_DIR` | `/data/snapshots` | Bind-mounted from `./volumes/face_snapshots` |
| `FACE_SNAPSHOT_RETENTION_UNKNOWN_DAYS` | `30` | How long unknown detections + their JPEGs are kept |
| `FACE_SNAPSHOT_RETENTION_KNOWN_DAYS` | `7` | How long identified detections are kept |
| `FACE_REC_RETENTION_SWEEP_INTERVAL_MIN` | `60` | Sweep cadence; `0` disables periodic sweeps |
| `FACE_REC_RETENTION_SWEEP_ON_STARTUP` | `true` | Run one sweep at startup |

### MQTT

| Var | Default | Notes |
|---|---|---|
| `FACE_REC_MQTT_ENABLED` | `true` | Set false to skip the bridge entirely |
| `MQTT_BROKER` | `mosquitto` | Shared with the agent's autonomy listener |
| `MQTT_PORT` | `1883` | |
| `FACE_REC_MQTT_CLIENT_ID` | `havencore-face-recognition` | |
| `FACE_REC_MQTT_RECONNECT_MAX_SEC` | `60` | Cap on exponential backoff between reconnects |

### Home Assistant (used by `/api/cameras` and the burst capture)

`HAOS_URL` and `HAOS_TOKEN` — same env vars as the agent's
`mcp_homeassistant_tools`. The trailing `/api` is stripped if present, so
the example `.env` value works unchanged. `FACE_REC_HA_TIMEOUT_SEC`
(default `10`) bounds the HA HTTP calls.
