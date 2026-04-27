# Face Recognition Service

Identifies known people seen by Reolink cameras (or any camera Home Assistant exposes), persists detection history, and publishes results to MQTT for HavenCore and other subscribers. v1 is **log-only** — no access-control enforcement; the architecture is shaped so a "welcome home" or smart-lock action can subscribe to the same MQTT topics later without schema changes.

## Purpose

- Identify *who* a camera is seeing on demand or in response to HA person-detection events.
- Maintain a per-person gallery of reference images plus auto-improvement crops.
- Surface a review queue of unknown faces so the operator can teach the system.
- Publish identified / unknown results to MQTT for downstream subscribers.

## Architecture

```
Reolink ─► HA person binary_sensor flips
           │
           ▼  HA automation (one MQTT publish, template per-camera)
   haven/face/trigger/{camera}   ◄── trigger topic
           │
           ▼
┌──────────────────────────────────────────────┐
│  face-recognition (port 6006)                │
│  • subscribes to trigger topic               │
│  • bursts N frames via HA camera_proxy REST  │
│  • detect (RetinaFace) → quality-score       │
│    → top-K → embed (ArcFace) → mean-norm     │
│  • Qdrant knn against `faces` collection     │
│  • persists detection + snapshot             │
│  • publishes result to MQTT                  │
│  • exposes HTTP API for people CRUD,         │
│    enrollment, detection history, review     │
└──────────────────────────────────────────────┘
           │                            │
           ▼                            ▼
   MQTT result topics           Postgres + Qdrant
```

## Configuration

```yaml
face-recognition:
  build: { context: ./services/face-recognition }
  ports: ["6006:6006"]
  environment:
    - CUDA_VISIBLE_DEVICES=3
  volumes:
    - ./services/face-recognition/app:/app
    - ./volumes/face_snapshots:/data/snapshots
    - ./volumes/insightface_models:/root/.insightface
  depends_on:
    postgres:  { condition: service_healthy }
    qdrant:    { condition: service_started }
    mosquitto: { condition: service_started }
```

| | |
|---|---|
| Service port | 6006 (HTTP, no nginx route — accessed by the agent + operator on the host) |
| Model | InsightFace `buffalo_l` (RetinaFace detect + ArcFace R100 embed, 512-d, ONNXRuntime-GPU on cuDNN 9) |
| GPU | Pinned to host GPU 3 via `CUDA_VISIBLE_DEVICES`; `ctx_id=0` inside the container |
| Vector store | Qdrant collection `faces`, 512-d cosine |
| DB | Postgres tables `people`, `face_images`, `face_detections` (idempotent migration on startup; same DDL appended to `services/postgres/init.sql` for fresh deployments) |
| MQTT | `paho-mqtt` 2.x, subscribes to `haven/face/trigger/+`, publishes results + status |

The full env reference lives in [`docs/configuration.md` → Face recognition](../../configuration.md#face-recognition).

## Data model

| Table | Purpose |
|---|---|
| `people` | Identity records — name (unique), `access_level` (`unknown\|resident\|guest\|blocked`), notes |
| `face_images` | Per-person gallery — JPEG path, Qdrant point id, source (`upload\|detection_confirmed\|detection_auto\|agent_enroll`), is_primary, quality_score |
| `face_detections` | Every event seen — camera, captured_at, person_id (NULL for unknowns), confidence, quality_score, snapshot_path, review_state (`auto\|confirmed\|rejected\|pending`), embedding_contributed, plus InsightFace genderage estimates (`age`, `sex`) for display only — never used in matching, gating, or autonomy |

`face_images.path` and `face_detections.snapshot_path` are stored relative to `SNAPSHOT_DIR` so rows survive container or mount-point moves.

## Pipeline (per trigger event)

1. Mint an `event_id` (UUID); publish `capturing` to `haven/face/status`.
2. Burst-capture `FACE_REC_BURST_FRAMES` frames via HA `camera_proxy` REST (default 6 frames at 500 ms).
3. Run RetinaFace on each frame; quality-score every face (4 factors: bbox area, Laplacian sharpness, pose alignment, brightness).
4. Drop anything below `FACE_REC_QUALITY_FLOOR`; keep top-3 by quality across all frames.
5. ArcFace-embed each, mean over the L2-normalized embeddings, re-normalize.
6. Qdrant knn (k=3); top hit > `FACE_REC_MATCH_THRESHOLD` → identified, otherwise unknown.
7. Persist a `face_detections` row + best-frame JPEG to `{SNAPSHOT_DIR}/{yyyy/mm/dd}/{event_id}.jpg`.
8. **Continuous improvement** (only when identified): if quality ≥ `FACE_REC_IMPROVEMENT_QUALITY_FLOOR`, confidence ≥ `FACE_REC_IMPROVEMENT_THRESHOLD`, and the person has fewer than `FACE_REC_MAX_EMBEDDINGS_PER_PERSON` gallery embeddings, contribute the new crop. FIFO-evicts the oldest non-primary embedding if the cap is hit.
9. Publish to `haven/face/identified` or `haven/face/unknown`; status → `idle` (in `try/finally` so it always emits).

If no face cleared `FACE_REC_QUALITY_FLOOR` at step 4 (frames were captured but nothing identifiable came back — hidden face, bad angle, wildlife), the pipeline still saves the *middle* frame as a snapshot, inserts a `face_detections` row with `person_id=NULL`, `confidence=NULL`, `quality_score=0.0`, `age=NULL`, `sex=NULL`, and publishes to `haven/face/no_face`. This gives downstream subscribers (autonomy + a future vision LLM) a chance to evaluate the snapshot for context — see [autonomy/cameras.md](../agent/autonomy/cameras.md).

InsightFace inference runs in a worker thread (`asyncio.to_thread`) so a burst doesn't peg the FastAPI event loop while the MQTT bridge is consuming triggers.

## MQTT contract

| Topic | Direction | Payload |
|---|---|---|
| `haven/face/trigger/{camera}` | HA → face-rec | `{source, event_id, captured_at}` (camera comes from the topic suffix; HA's `event_id` is a timestamp string and is logged but not persisted — see deferred follow-up in [docs/todo.md](../../todo.md)) |
| `haven/face/identified` | face-rec → world | `{event_id, detection_id, camera, person_id, person_name, confidence, quality_score, snapshot_path, captured_at}` |
| `haven/face/unknown` | face-rec → world | `{event_id, detection_id, camera, snapshot_path, quality_score, captured_at}` |
| `haven/face/no_face` | face-rec → world | `{event_id, detection_id, camera, snapshot_path, frames_processed, captured_at}` — person sensor tripped + frames captured, but no face cleared `FACE_REC_QUALITY_FLOOR`. Middle frame is saved so a vision-LLM gather step can still inspect it. |
| `haven/face/status` | face-rec → world | `{camera, mode: "idle"\|"capturing"\|"matching", since}` (single global topic; not retained) |

`detection_id` on the result topics is the UUID of the inserted `face_detections` row. Downstream consumers (e.g. the agent's autonomy `sensor_events` normalizer) use it to synthesize the snapshot URL `{AGENT_INTERNAL_BASE_URL}/api/face/detections/{detection_id}/snapshot` for notification attachments.

`no_frames` outcomes (HA `camera_proxy` returned zero decoded frames — capture itself failed) are still intentionally not published; there's nothing to attach.

The HA-side automation that fires the trigger is documented in [`docs/integrations/home-assistant.md` → Face recognition camera triggers](../../integrations/home-assistant.md#face-recognition-camera-triggers).

## HTTP API

All paths below are served on port 6006 directly. The agent at port 6002 mirrors them under `/api/face/*` for the SvelteKit `/people` UI — see [API reference](../../api-reference.md#face-recognition-agent-proxy).

### People

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/people` | List with `image_count` per person |
| `POST` | `/api/people` | Create — body `{name, access_level?, notes?}` |
| `GET` | `/api/people/{id}` | Detail with image gallery (primary first) |
| `PATCH` | `/api/people/{id}` | Partial update of `access_level` and/or `notes` |
| `DELETE` | `/api/people/{id}` | Cascade: face_images rows (FK), JPEG files, Qdrant points by payload filter. Detection rows survive with `person_id=NULL` |

### Person images

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/people/{id}/images` | Multipart upload — picks the highest-`det_score` face, persists file → Qdrant → DB; supports `is_primary` toggle |
| `POST` | `/api/people/{id}/enroll-from-camera` | Burst-capture from HA, picks the highest-quality face. Body `{camera, is_primary?}`. Does NOT create a detection row or publish MQTT |
| `DELETE` | `/api/people/{id}/images/{img_id}` | Deletes file + Qdrant point + DB row. Refuses to delete `is_primary` rows — set another primary first |
| `POST` | `/api/people/{id}/images/{img_id}/set-primary` | Atomic primary swap |

### Face image bytes

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/face_images/{id}/bytes` | Streams the JPEG. Identity URL — no filesystem path leaks to the client |

### Detections

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/trigger?camera=<entity_id>` | Manually fire the pipeline for a camera (operator/debug) |
| `GET` | `/api/detections` | Filters: `camera`, `person_id`, `since_seconds_ago`, `review_state`, `unknowns_only=true` shorthand (`person_id IS NULL AND review_state != 'rejected'`), `limit` (≤ 200, default 20) |
| `GET` | `/api/detections/{id}/snapshot` | Streams the snapshot JPEG |
| `POST` | `/api/detections/{id}/confirm` | Body must specify exactly one of `{person_id}` or `{name}` (new person get-or-create). Re-runs detect+embed on the saved snapshot, persists to that person's gallery, marks the row `review_state='confirmed', embedding_contributed=true` |
| `POST` | `/api/detections/{id}/reject` | Marks `review_state='rejected'` so it stops appearing in the unknowns queue |
| `POST` | `/api/detections/bulk-delete` | Body `{scope: "rejected"\|"all_unknowns"}`. Mass-deletes unknown (`person_id IS NULL`) detection rows + their snapshot files. `rejected` narrows to `review_state='rejected'`; `all_unknowns` is the heavy hammer. Mirrors the retention sweeper's list → unlink → batch-delete pattern; file unlinks are best-effort. Returns `{rows_deleted, files_unlinked, scope}` |

### Cameras

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/cameras` | Discovery — queries HA `/api/states`, filters person sensors, joins the `cameras` registry table to surface `camera_entity`, `fov_type`, `native_resolution`, and `registered`. Falls back to deriving `camera.<base>_clear` for sensors not yet seeded into the registry. `camera_exists` flags naming-convention drift before the bridge fails on it |

### Admin / operator

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/admin/retention/sweep` | Triggers a single retention sweep right now and returns the result (rows examined, files unlinked, rows deleted) |
| `POST` | `/api/admin/rebuild-embeddings` | Returns `{job_id, status: "running"}`; full re-embed of every `face_images` row from disk in the background. Refreshes Qdrant points in place + updates `face_images.quality_score` with the freshly-computed value. Cleans orphan Qdrant points (payload references no DB row). One job at a time per process — concurrent POST returns 409 with the in-flight `job_id` |
| `POST` | `/api/admin/rescan-unknowns` | Returns `{job_id, status: "running"}`; walks every unknown (`person_id IS NULL AND review_state != 'rejected'`) detection, re-embeds the saved snapshot, queries Qdrant. Matches above `MATCH_THRESHOLD` flip to `review_state='confirmed'`. High-quality matches that also clear the live pipeline's `IMPROVEMENT_QUALITY_FLOOR` + `IMPROVEMENT_THRESHOLD` additionally feed the gallery (FIFO-bounded by `MAX_EMBEDDINGS_PER_PERSON`). Single-flight; concurrent POST returns 409 |
| `GET` | `/api/admin/jobs/{job_id}` | Polls a job. Job-type-specific phases — rebuild: `loading → embedding → cleaning_orphans → done`; rescan: `scanning → done`. Per-row failures accumulate in `errors[]` without aborting the run. Totals shape varies by job type: rebuild reports `{images, re_embedded, missing_files, no_face, embed_failed, orphan_points_removed}`; rescan reports `{examined, matched, no_match, contributed, skipped_missing_snapshot, errors}` |
| `GET` | `/api/admin/jobs?limit=20` | Most-recent jobs first |

### Health

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Reports model providers + load time, db status, qdrant collection + dim, mqtt broker + subscribed topics, retention `{last_sweep, interval_min, unknown_days, known_days}` |

## Cameras registry & FOV-aware detection

The `cameras` Postgres table is the per-camera metadata source. Each row maps
the HA snapshot entity (e.g. `camera.front_duo_3_clear`) to the matching
person sensor and to a `fov_type`:

| `fov_type` | Behavior |
|---|---|
| `standard` | Single-pass detection. The full frame goes to InsightFace at `FACE_REC_DET_SIZE` (default 1280) |
| `panoramic_dual_lens` | Frame is split into two horizontally overlapping halves (10% overlap on each side of the seam). Detection runs per tile; bboxes are translated back to full-frame coordinates and overlap-deduped by IoU > 0.5 |

Tiling exists because Reolink dual-lens cameras stitch two ~90° views into
one ~180° image. A 7680×2160 panorama hitting a 1280-wide detector is a 6×
downscale; tiling halves that loss before the resize, so faces near either
lens see ~3× the pixels they would in a single-pass run.

A camera that isn't registered falls back to `fov_type=standard` so adding
new cameras to HA doesn't require a DB write before face-rec works on them
— it just won't get tiling until you seed a row. Use `GET /api/cameras` to
see which sensors are registered and which are running on convention
defaults.

Adding or correcting a camera:

```sql
-- Register a new panoramic camera
INSERT INTO cameras (entity_id, sensor_entity, fov_type, native_width, native_height)
VALUES ('camera.driveway_clear', 'binary_sensor.driveway_person',
        'panoramic_dual_lens', 4096, 1440);

-- Flip an existing camera's FOV type
UPDATE cameras SET fov_type = 'standard' WHERE entity_id = 'camera.porch_clear';
```

## Retention

A periodic asyncio task started by the FastAPI lifespan prunes old detection rows and their snapshot files:

- `FACE_SNAPSHOT_RETENTION_UNKNOWN_DAYS` (default 30) — unknowns are kept longer because they're the review-queue source of truth.
- `FACE_SNAPSHOT_RETENTION_KNOWN_DAYS` (default 7) — identified events age out faster; the gallery is the durable record.
- `FACE_REC_RETENTION_SWEEP_INTERVAL_MIN` (default 60) — set to 0 to disable the periodic loop.
- `FACE_REC_RETENTION_SWEEP_ON_STARTUP` (default true) — runs one sweep when the service starts.

Auto-improvement face_images are NOT touched here — they're gallery state, capped by the FIFO eviction the pipeline does.

Operators can force a sweep via `POST /api/admin/retention/sweep` (useful right after lowering a threshold).

## Operational notes

- **Idle GPU footprint** is just the loaded model (~600 MB VRAM on GPU 3); active inference is <100 ms per frame.
- **Snapshots stay on the host** — nothing leaves the LAN.
- **Volume-mounted code**: edits under `services/face-recognition/app/` go live with `docker compose restart face-recognition`. `.env` changes still need `down && up -d`.
- **Cold start downloads the buffalo_l pack** (~280 MB) into `./volumes/insightface_models`; subsequent restarts skip the download.
- **Single-instance assumption**: the in-memory job registry for `/api/admin/jobs` and the rebuild lock assume one process. Don't scale horizontally.

### Quality scoring

Every detected face is scored 0–1 from four signals: bbox area, sharpness
(Laplacian variance of the crop), pose (eye symmetry around the nose),
and brightness (mean grayscale). The composite uses fixed weights —
0.30 area, 0.30 sharpness, 0.25 pose, 0.15 brightness — and the per-face
breakdown is logged at INFO so tuning is observable:

```
quality score=0.83 (area=0.75 sharp=1.00 pose=0.97 bright=0.40) face=156x191 det=0.82
```

`area` saturates at ~200×200 pixels in the face crop (`AREA_SATURATION_PX`,
40000 px), an *absolute* count rather than a fraction of the frame —
ArcFace resizes everything to 112×112 internally, and frame-relative
ratios punish wide-angle/panoramic cameras unfairly. A close-up face on
a 7680×2160 panorama saturates area to 1.0; a small distant face still
scores low.

Two thresholds gate the score: `FACE_REC_QUALITY_FLOOR` (default 0.40) is
the bar to be considered for matching; `FACE_REC_IMPROVEMENT_QUALITY_FLOOR`
(default 0.65) is the higher bar to feed back into the gallery via
continuous improvement. Tune both by watching the score breakdown logs
across a few weeks of real events.

## Troubleshooting

### `/health` shows `mqtt.connected: false`

Mosquitto isn't reachable. Check `docker compose ps mosquitto`. Set `FACE_REC_MQTT_ENABLED=false` in `.env` to run as HTTP-only (manual `/api/trigger` and the dashboard still work; HA triggers won't).

### Trigger fires but pipeline returns `no_frames`

HA's `camera_proxy` returned 0 frames. Check `HAOS_URL` / `HAOS_TOKEN` and that the camera entity is `camera.<base>_clear` (matches the discovery convention) or that the camera is registered in the `cameras` table with the snapshot entity HA actually exposes. `GET /api/cameras` will show whether the bridge can derive a camera from each person sensor.

### Identified events have `embedding_contributed: false` consistently

One of the three improvement gates is blocking. Most common culprit on outdoor wide-angle cameras: `quality_score` < `FACE_REC_IMPROVEMENT_QUALITY_FLOOR`. The default was lowered to 0.65 specifically for the front_duo_3 install; per-camera overrides are tracked in [docs/todo.md](../../todo.md).

### Unknowns queue is bloated

After confirming several unknowns as the same person, kick `POST /api/admin/rescan-unknowns` to re-match every still-unknown detection against the now-richer index. The dashboard `/people/unknowns` page exposes this as a "Rescan against current index" button. High-quality matches contribute to the gallery on the same pass; the rest stay unknown for manual review.

For raw cleanup, the same page has "Clear rejected" (deletes `review_state='rejected'` rows and their snapshot files) and "Delete ALL unknowns" (deletes every `person_id IS NULL` row, with a typed-`DELETE` confirmation). Both call `POST /api/detections/bulk-delete` with the appropriate scope.

### Postgres + Qdrant drift suspected

`POST /api/admin/rebuild-embeddings`, then poll `GET /api/admin/jobs/{job_id}`. The orphan-cleanup pass removes Qdrant points whose payload references no DB row. Per-row failures land in `errors[]`.

## Related files

- `services/face-recognition/app/main.py` — FastAPI lifespan + `/health`
- `services/face-recognition/app/pipeline.py` — detection + matching + continuous improvement
- `services/face-recognition/app/mqtt_bridge.py` — paho-on-thread → asyncio.Queue
- `services/face-recognition/app/retention.py` — periodic sweeper
- `services/face-recognition/app/api/{people,detections,face_images,cameras,admin}.py` — HTTP routes
- `services/postgres/init.sql` — schema for fresh deployments

## See also

- [MCP Face Tools](../agent/tools/face.md) — agent-side LLM tool reference
- [Face recognition camera triggers (HA)](../../integrations/home-assistant.md#face-recognition-camera-triggers) — the single-template MQTT automation
- [Configuration → Face recognition](../../configuration.md#face-recognition) — env var reference
- [API Reference → Face recognition (agent proxy)](../../api-reference.md#face-recognition-agent-proxy) — same surface served by the agent for the dashboard
