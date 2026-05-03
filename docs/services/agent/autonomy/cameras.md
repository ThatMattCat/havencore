# Camera & Sensor Events

Generic, multi-source proactive notifications for camera/sensor events. Built
on top of the autonomy engine — no new handler kind, no new dispatcher.
Face-recognition is the first source; vehicles, motion, doorbell rings, and
license-plate reads plug in by publishing on the same topic schema.

## Topic contract

Every camera/sensor source that wants its events triaged by autonomy
publishes on:

```
haven/<domain>/<kind>
```

| Field | Allowed values | Notes |
|-------|----------------|-------|
| `domain` | `face`, `vehicle`, `motion`, `doorbell`, `presence` | Vocabulary is bounded — see `sensor_events.ALLOWED_DOMAINS`. |
| `kind`   | domain-specific (`identified`, `unknown`, `detected`, `rang`, …) | Free-form; flows through to the LLM as-is. |

Sub-paths like `haven/face/trigger/<camera>` are *inputs* to the
face-recognition service from HA automations — those are intentionally
ignored by the autonomy normalizer.

### Required + recommended payload fields

The normalizer is forgiving: missing fields downgrade gracefully rather than
dropping the event. Recommended payload shape, per domain:

**Face** (already published by `services/face-recognition/`; full payload
contract documented in
[face-recognition/README.md → MQTT contract](../../face-recognition/README.md#mqtt-contract)):

| topic | when |
|-------|------|
| `haven/face/identified` | A face matched a known person above MATCH_THRESHOLD. |
| `haven/face/unknown`    | A face was detected but didn't match any enrolled person. |
| `haven/face/no_face`    | The person sensor tripped + frames were captured but no face cleared QUALITY_FLOOR. The middle frame is saved as a snapshot which the watch_llm gather step then sends to the vision LLM for a scene description. |

```json
{
  "event_id": "uuid",
  "detection_id": "uuid",
  "camera": "camera.front_duo_3_clear",
  "person_id": "uuid | null",
  "person_name": "Matt | null",
  "confidence": 0.91,
  "quality_score": 0.78,
  "snapshot_path": "2026/04/27/abc.jpg",
  "captured_at": "2026-04-27T18:00:00+00:00"
}
```

**Vehicle** (when LPR / Frigate ALPR is added):
```json
{
  "event_id": "uuid",
  "camera": "camera.driveway",
  "plate": "ABC123",
  "plate_known": true,
  "make_model": "Honda Civic gray",
  "captured_at": "..."
}
```

**Motion / doorbell**:
```json
{
  "event_id": "uuid",
  "camera": "camera.front_duo_3_clear",
  "captured_at": "..."
}
```

## How an event flows

```
publisher  ──►  mosquitto  ──►  agent autonomy/mqtt_listener
                                        │
                                        ▼
                                sensor_events.normalize()
                                        │
                                        ▼
                          { source, topic, payload, sensor_event:{
                              domain, kind, zone, zone_label,
                              subject, snapshot_url, captured_at, raw }}
                                        │
                                        ▼
                          engine.trigger_event() → trigger_match
                                        │
                                        ▼
                            watch_llm handler (per matched item)
                                        │
                                        ▼
                          gather: HA entities + L4 memory
                                  + ha_get_presence
                                  + face_recent_visitors
                                  + query_multimodal_api (scene description
                                    of sensor_event.snapshot_url)
                                        │
                                        ▼
                            LLM judgment (JSON)
                                        │
                                        ▼
                  cooldown / severity / channel rails
                                        │
                                        ▼
              SignalNotifier | HAPushNotifier | SpeakerNotifier
```

The normalizer sits in `services/agent/selene_agent/autonomy/sensor_events.py`.
It maps the raw camera entity_id (e.g. `camera.front_duo_3_clear`) to a
generic *zone* (e.g. `front_door`) using the `camera_zones` Postgres table —
the LLM downstream reasons about zones, not entity_ids, so the same logic
generalizes across deployments.

## Configuring zones

Two ways:

1. **Dashboard** (recommended): visit `http://<host>:6002/cameras`. List of
   discovered HA cameras with a free-text zone field per row. Saves go
   straight to the `camera_zones` table. The autonomy engine refreshes its
   in-memory zone cache via `LISTEN/NOTIFY` so changes apply immediately —
   no agent restart needed.

2. **Direct SQL**:
   ```sql
   INSERT INTO camera_zones (camera_entity, zone, zone_label)
   VALUES ('camera.front_duo_3_clear', 'front_door', 'Front Door');
   ```

Common zone slugs: `front_door`, `backyard`, `driveway`, `side_yard`,
`garage`. Pick whatever vocabulary fits your home — the LLM is told to
reason about the *role* of a zone, not match a fixed list.

## Watch_llm handler — extended schema

The `watch_llm` handler that triages camera events accepts a few extra
config flags:

```jsonc
{
  "subject": "camera event triage prompt context",
  "gather": {
    "entities": [],                  // optional HA entities to pull history for
    "memories_k": 3,                 // L2/L3 memory hits
    "presence": true,                // adds ha_get_presence to the gather
    "recent_visitors_hours": 6,      // adds face_recent_visitors(hours=N)
    "scene_description": true,       // run the vision LLM on the trigger snapshot
    "scene_description_prompt": ""   // optional override of the default prompt
  },
  "notify": { "channel": "signal" },  // default channel; LLM may override
  "severity_floor": "low",
  "cooldown_min": 10,
  "attach_snapshot": true             // forward sensor_event.snapshot_url
                                      // to the notifier (signal-only)
}
```

The LLM JSON output is also extended:

```json
{
  "unusual": true,
  "severity": "med",
  "summary": "...",
  "signature": "...",
  "evidence": ["...", "..."],
  "channel": "signal | ha_push | speaker | silent",
  "urgency": "info | warn | alert"
}
```

`channel` and `urgency` are optional; when omitted the agenda item's
configured `notify.channel` is used. **Safety rails** still apply on top of
the LLM's choice:

- `silent` collapses to `unusual=false` (no notification, no cooldown row).
- `speaker` is downgraded to `signal` if `ha_get_presence` shows nobody is
  home — there's no point speaking to an empty house.
- Quiet hours are evaluated *before* the handler runs, so a `speaker` choice
  during quiet hours never reaches the speaker; the run is deferred or
  dropped per the agenda item's `quiet_hours.policy`.
- The dedup `signature` for cooldown is **derived from the normalized
  `sensor_event`** (`{domain}:{kind}:{zone}:{subject_identity_or_type}`)
  rather than the LLM's `signature` field — so repeat detections of the
  same person in the same zone reliably hash to the same key regardless of
  how the LLM phrases its output. The LLM-emitted `signature` is preserved
  on the run as `signature_raw` for observability and is the fallback when
  the trigger has no `sensor_event` (generic webhooks).

## Vision-AI scene description

When `gather.scene_description: true` is set, `_gather()` extracts a snapshot
URL from the trigger event and calls `query_multimodal_api` against it. The
vision LLM's reply lands in the LLM's user prompt as a `## scene_description`
block alongside `presence`, `recent_visitors`, and the rest — so the triage
LLM sees a sentence like *"a person in a maroon shirt holding a small dark
dog on a grassy lawn"* instead of just `subject=Matt confidence=0.95
zone=backyard`.

Snapshot URL resolution order (first hit wins):

1. `event.sensor_event.snapshot_url` — already populated by the
   `sensor_events` normalizer for face/* topics (e.g.
   `http://agent:6002/api/face/detections/{id}/snapshot`).
2. `event.payload.snapshot_url` — fallback for events that didn't pass
   through the normalizer.

If neither is present, the gather step silently skips — the LLM still
runs, just without a scene description.

The default prompt (`watch_llm._DEFAULT_SCENE_PROMPT`) is intentionally
generic ("describe what's visible, note people, animals, vehicles, packages,
unusual things"). A seed can override it per-item with
`gather.scene_description_prompt`. The shipped `face_no_face_triage` seed
uses an override biased toward *resident vs. pet vs. delivery driver vs.
wildlife* disambiguation, since that's the hard call for triggers where no
face was visible.

The vision call is wrapped in `_safe_tool` — a vision-LLM failure (model
unreachable, snapshot 404) won't block the triage. The state block falls
back to "tool failed: …" and the LLM still gets to judge on the rest of
the gather.

## Default seeded items

The agent seeds four `watch_llm` agenda items at first startup
(`autonomy/seeds/camera_events.py`):

| name | enabled | trigger topic | notes |
|------|---------|---------------|-------|
| `face_identified_triage` | ✅ | `haven/face/identified` | Known-resident path. Usually nominal unless context is off (late hour while away, etc.). Default `cooldown_min=30`. Ships with `scene_description: true`. |
| `face_unknown_triage`    | ✅ | `haven/face/unknown`    | Face detected but not matched — primary stranger-at-the-door path. Default `cooldown_min=15` (kept short — high signal). Ships with `scene_description: true`. |
| `face_no_face_triage`    | ✅ | `haven/face/no_face`    | Person sensor tripped, no face visible. Higher severity floor (`med`) since these are noisier (wildlife, shadows). Default `cooldown_min=45`. Ships with `scene_description: true` plus a tailored prompt biased toward resident/pet/delivery/wildlife disambiguation. |
| `vehicle_event_triage`   | ❌ | `haven/vehicles/+`      | Off by default; flip on once an LPR / vehicle source publishes. |

Both are `created_by='system_camera'`. Re-running the seed only inserts
missing rows — your dashboard tweaks survive restarts. The exception is a
narrow one-time migration that adds `scene_description` (and the no_face
prompt override) to existing system-seeded rows that don't already have the
key set; rows where the operator has already explicitly set or unset
`scene_description` via the dashboard are skipped.

## Adding a new sensor source

1. Decide on a `<domain>/<kind>` (avoid colliding with the existing
   vocabulary above; if `domain` isn't in `ALLOWED_DOMAINS`, normalization
   is skipped and the event flows through with the legacy raw shape).
2. Publish a JSON payload that includes a `camera` field (so zone mapping
   works) and `captured_at` ISO timestamp. Include a domain-specific subject
   block where applicable.
3. Optionally add a dedicated normalizer in `sensor_events.py` to put the
   subject info into a `subject:{type, identity, confidence, quality}`
   block — that lets the LLM reason about it without parsing the raw
   payload.
4. Either enable the matching seeded `watch_llm` item, or POST a new agenda
   item via `/api/autonomy/items` with `trigger_spec.match.topic` set to
   your topic pattern.

## Worked examples

### Stranger at front door

1. `face-recognition` publishes `haven/face/unknown` with
   `camera="camera.front_duo_3_clear"`, `quality_score=0.7`,
   `detection_id="d1"`.
2. Normalizer maps the camera → zone `front_door`, attaches
   `snapshot_url=http://agent:6002/api/face/detections/d1/snapshot`.
3. `face_unknown_triage` matches; gather pulls presence + last 6h of visitors
   + 3 memories + the vision LLM's scene description of the snapshot
   ("a man in a hooded jacket carrying a clipboard at the front porch").
4. LLM sees: subject is unknown, zone is front_door, quality is moderate,
   no recent visitors, presence shows resident is home, scene looks like a
   delivery courier → returns `{unusual: true, severity: med,
   channel: speaker, urgency: alert, summary: "Unknown person at front door"}`.
5. Engine cooldown + severity floor passes; `SpeakerNotifier` synthesizes
   "Selene: front door. Unknown person at front door." through the default
   speaker.

### Resident in backyard at low quality (the cat-walking case)

1. Backyard camera fires `haven/face/unknown` because the angle was bad —
   no face match. `quality_score=0.45`, zone `backyard`.
2. Gather sees: presence has resident `home`, recent_visitors over 6 hours
   shows two prior detections of the resident at the backyard zone, and the
   vision LLM's scene description reads "a person in a maroon shirt holding
   a small dark dog on a grassy lawn near a wicker patio set."
3. L4 memory contains "Resident walks the cat in the backyard most
   evenings, often at unusual angles" (you wrote this once, now it
   propagates into every autonomy turn).
4. LLM returns `{unusual: false, channel: silent, summary: ""}`. No
   notification fires; one nominal row in `autonomy_runs`.

If the same event hits with presence showing everyone away, the LLM should
flip `unusual=true` and pick `channel: signal` — alert reaches the user
even though the face wasn't matched.
