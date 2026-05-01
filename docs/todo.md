# TODO

Forward-looking items that aren't in-flight. Each bullet is a seed for a later pass; details to be fleshed out when the work is picked up.

## LLM provider work — follow-ups

- **Wire the stubbed OpenAI provider.** `providers/openai.py` currently raises `NotImplementedError` and the factory falls back to vLLM when it's selected. Shape should mirror vLLM (`AsyncOpenAI` with a different base_url + key + model). The System-page button is already rendered disabled with "soon"; un-disable once the provider lands. Translation layer is not needed — OpenAI tool-calling is already OpenAI-shaped.
- **Consider caching the summarize-and-reset call too.** The `_summarize_session` path uses a different system prompt than normal chat so it's always a cache miss (`read=0 create=N`). Tiny cost per compaction, but trivial to add — one `cache_control` breakpoint on the summary's system block. Only worth doing if a profile shows compactions are fireing often enough to matter.

## Agent tool surface — additions

- **Todo / shopping list tool over HA's `todo.*` services.** *Coordinate with the companion app's list-screen phase* — the LLM-facing tool and the human-facing list display are joint work, no point shipping one without the other. Ship them in the same pass: dedicated tool (or small tool family) in `mcp_homeassistant_tools/` with list-name + item-text params over `todo.add_item` / `todo.remove_item` / `todo.get_items`, alongside the app screen that renders/checks-off the same lists. Check which `todo.*` list entities exist in the target HA instance before committing to the param shape.
- **`web_quick_answer` — additive convenience over `brave_search` + `fetch`.** Keep both primitives; add a convenience tool that runs a search and auto-fetches the top result's body in one call. Saves a round trip on the common "look something up" path, which matters on a local model. Not a replacement — when the model wants result #3 or wants to inspect titles before fetching, it still uses the primitives.
- **Direct CalDav access for calendar edit/delete.** `ha_create_calendar_event` works against HA's CalDav integration, but `ha_update_calendar_event` / `ha_delete_calendar_event` are currently hidden because HA's CalDav integration doesn't declare the `UPDATE_EVENT` / `DELETE_EVENT` features — the WS commands always fail. The underlying CalDav protocol fully supports PUT/DELETE on events by `uid`; the limitation is HA's wrapper. Path forward: bypass HA and talk to the CalDav server directly via the [`caldav` Python lib](https://github.com/python-caldav/caldav), with URL + creds added to the agent's env (likely the same ones HA already uses). Handler methods `_update_calendar_event` / `_delete_calendar_event` and their dispatch branches in `mcp_homeassistant_tools/mcp_server.py` are kept as dead code so re-enabling is just a Tool() restore once the backing transport is swapped. Decide alongside the Android companion app — that work also has to pick a calendar transport, and using one shared CalDav client would be cleanest.

## Android companion app

In flight in the sibling repo
[`havencore-companion-app`](https://github.com/ThatMattCat/havencore-companion-app)
— native Kotlin, LAN-only for v1. Surfaces being delivered there:
in-app chat over `/ws/chat`, registration as Android's default-assistant app
(`VoiceInteractionService`), push notifications via UnifiedPush + ntfy, and
the todo/shopping-list view that pairs with the deferred `todo.*` MCP work
above. iOS is explicitly out of scope. Server-side changes the app needs
(e.g. a push registration endpoint and an `NtfyNotifier` alongside
`SignalNotifier`) land in this repo when their phase reaches them; consult
the companion repo for the current phase.

## Face recognition — deferred polish

Three carry-overs from step 8 of the face-recognition rollout. Each was
consciously skipped because the trigger condition hasn't shown up in
practice. **Trigger first, implement second** — don't pre-build any of
these unless the trigger is actually firing.

- **Per-camera quality thresholds.** `FACE_REC_IMPROVEMENT_QUALITY_FLOOR`
  is a single global. Outdoor wide-angle (front_duo_3) consistently
  scores 0.66–0.69; an indoor doorbell will likely score much higher.
  *Trigger:* more than one camera class is in production AND the global
  default is causing one camera to either spam improvements or never
  contribute. *Sketch:* `face_cameras` table (camera, quality_floor) or
  a JSON env knob; pipeline reads the per-camera value with fallback
  to the global. ~30 min of work.

- **Per-camera `haven/face/status/{camera}` topic with retain.** Status
  is currently a single global `haven/face/status` topic, not retained
  (because retain on a global topic surfaces stale per-camera state).
  *Trigger:* the dashboard wants persistent per-camera mode display
  (e.g. show which cameras are mid-capture across page reloads), or an
  external subscriber (HA card, Grafana) needs persistent state.
  *Sketch:* `mqtt_bridge._emit_status` publishes to
  `haven/face/status/{camera}` with `retain=True`; dashboard subscribes
  per-camera. ~20 min.

- **`face_detections.source_event_id` for HA↔HavenCore correlation.**
  HA's MQTT trigger payload has an `event_id` (timestamp string); the
  bridge mints a fresh UUID for the persisted event. The two are not
  linked beyond an INFO log line. *Trigger:* "HA logged a person
  detection at 5:42pm but I don't see a corresponding HavenCore
  detection event" becomes a debugging pattern. Today the only way
  to correlate is by approximate `captured_at` + camera, which is
  fuzzy. *Sketch:* add `source_event_id TEXT` column to
  `face_detections` (idempotent migration in db.py); pipeline writes
  the original HA id alongside the minted UUID. ~15 min.

- **Per-camera trigger cooldown.** Outdoor wide-angle cameras can fire
  person-detection events every few seconds when someone walks around
  the yard, generating a flood of `face_detections` rows (mostly
  low-quality side/back views that get rejected during review).
  *Trigger:* the unknowns queue + DB are still bloating after rescan
  + bulk-delete are in active use. *Sketch:* per-camera in-memory
  last-fire dict in `mqtt_bridge.FaceMqttBridge`, guarded by a new
  `FACE_REC_TRIGGER_COOLDOWN_SECONDS` env (default 0 = off, ~15 in
  production). Skip enqueue if `monotonic() - last < cooldown`. Place
  the check in `_dispatch` before the HA snapshot fetch so the round-
  trip is also dropped on cooldown. Pair with HA-side `for: '00:00:02'`
  on the `binary_sensor.*_person` trigger as a complement. Trade-off:
  presence/last-camera signal is preserved at coarser (cooldown-window)
  granularity. ~30 min.

## Stretch goals

Post-MVP / "v2"-class features — multi-modal perception, identity awareness, richer output surfaces. To work on **after** the items above are cleared. Scoping, feasibility, effort, and suggested sequencing live in [`stretch-goals.md`](./stretch-goals.md).
