# TODO

Forward-looking items that aren't in-flight. Each bullet is a seed for a later pass; details to be fleshed out when the work is picked up.

## Resume-from-history UX

- **Resume should repopulate `/chat` with the orchestrator's actual `messages`, not an empty pane.** Today the Resume button hydrates the session server-side and navigates to `/chat`, but the chat transcript doesn't visually reflect what the LLM will see on the next turn. After resume, the Chat pane should clear any stale transcript and render exactly the post-hydrate `messages` — i.e. `[system prompt + L4] + [Prior conversation summary] + tail exchanges`, with the base system prompt hidden and the summary shown in the same distinct styling we use on `/history`. The user should see what the model sees.

## Context-size-triggered summarization

- **Auto-summarize any session when its context size crosses a threshold.** With the dashboard's `idle_timeout=-1` sentinel, dashboard sessions now never auto-reset on idle — they live until "New Chat" or LRU eviction, and their message list grows unbounded. Pucks/satellites also risk this if a single exchange balloons (long tool outputs, big multimodal payloads). Add a size-based trigger parallel to the idle one: when total token/char count of `messages` exceeds some limit (TBD — probably a fraction of the model's context window, e.g. 75%), fire `_summarize_and_reset(reason="context_size_summarize")`. Hook points: likely in `AgentOrchestrator.run()` alongside the existing `_check_session_timeout()` call (`orchestrator.py:411`), and/or as a second gate in the pool sweep. Needs: (a) a cheap token-count estimator (reuse whatever the metrics path uses, or a char/4 approximation), (b) a new config var `CONVERSATION_CONTEXT_LIMIT_TOKENS` or similar, (c) a way to bypass/override for dashboard sessions if the user wants truly unbounded, or to set a higher ceiling there. Keep the `idle_timeout=-1` sentinel path unaffected — this is a separate axis.

## History detail parity with what the LLM received

- **`/history` detail should mirror the LLM's view of each flush, not the raw pre-flush buffer.** When a session is summarized, the stored flush includes the pre-summary messages (captured for auditing) plus `metadata.rolling_summary`. The dashboard currently renders the pre-summary messages too, which misrepresents what context the LLM actually had on subsequent turns. Flip the default: for rows where `metadata.rolling_summary` is set, show the summary (and any post-reset exchanges, if we begin storing those as separate flushes) rather than the pre-reset transcript. Keep the raw pre-reset transcript accessible — maybe via a "show raw" toggle — since it's still useful for debugging.

## Broken tests on main

- **`tests/test_memory_review_handler.py::test_prune_respects_source_protection` asserts `stats["l2_pruned"] == 1` but gets `2`.** The `_prune_l2` flow is deleting the protected L2 point even though an L3 still references it via `source_ids=["protected"]`. Either the protection lookup isn't being consulted or the L3 scroll filter is missing in the code path the test exercises. Pre-dates the LLM-provider branch (confirmed via `git stash`). Fix the handler or update the test to match the intended semantics — whichever is correct.
- **`tests/test_integration_memory_review.py::test_end_to_end_consolidation` fails with `requests.exceptions.MissingSchema: Invalid URL '/api/memory/runs/trigger'`.** The test uses bare `requests` with a relative path; it needs either a `base_url` or a FastAPI `TestClient`. Probably broke when the endpoint was moved and nobody re-ran this integration test. Also pre-dates the LLM-provider branch.

## LLM provider work — follow-ups

- **Wire the stubbed OpenAI provider.** `providers/openai.py` currently raises `NotImplementedError` and the factory falls back to vLLM when it's selected. Shape should mirror vLLM (`AsyncOpenAI` with a different base_url + key + model). The System-page button is already rendered disabled with "soon"; un-disable once the provider lands. Translation layer is not needed — OpenAI tool-calling is already OpenAI-shaped.
- **Consider caching the summarize-and-reset call too.** The `_summarize_session` path uses a different system prompt than normal chat so it's always a cache miss (`read=0 create=N`). Tiny cost per compaction, but trivial to add — one `cache_control` breakpoint on the summary's system block. Only worth doing if a profile shows compactions are fireing often enough to matter.

## Replace iav-to-text (image/vision analysis)

- **Find a new multimodal backend for image/audio/video → text.** The `iav-to-text` service (Qwen2.5-Omni on CUDA device 2) was effectively evicted when vLLM moved to GLM-4.5-Air-AWQ-FP16Mix at `-tp 4 --gpu-memory-utilization 0.77` — GLM now spans all four GPUs and there's no room for a second resident vision model. The compose entry is still present (`compose.yaml:163`) but unusable in practice. Callers still referencing it: `selene_agent/api/vision.py`, `selene_agent/modules/mcp_general_tools/mcp_server.py` (the `describe_image` / camera-snapshot path used by MQTT camera events and the autonomy engine). Options to weigh: (a) a smaller quantized VLM that fits alongside GLM (Qwen2-VL-2B-AWQ, InternVL2-2B, MiniCPM-V-2.6-int4) with reduced memory util on one of the vLLM GPUs; (b) on-demand load/unload around GLM (slow, ugly); (c) route to a cloud VLM for image turns only (Claude/Gemini/OpenAI via the soon-to-land OpenAI provider); (d) offload to Frigate/DeepStack style CPU inference for object/face detection and reserve VLM-quality description for rare cases. Needs a decision on whether identity matters (face rec is better served by Frigate — see stretch goal #2) vs. scene description (needs a real VLM). Once chosen: swap the service image/command, keep the `:8100/v1/chat/completions` OpenAI-compat shape so callers don't change, and update `docs/services/iav-to-text/`.

## Agent tool surface — additions

- **`ha_get_state(entity_id)` — cheap single-entity state lookup.** Today, checking one entity forces either `ha_list_entities` (returns the full domain) or an `ha_evaluate_template` detour. A direct single-entity call would get heavy use ("is the porch light on", "what's the thermostat set to") and keeps the LLM off the template escape hatch for trivial reads. Lives alongside the existing HA tools in `mcp_homeassistant_tools/`; thin wrapper over `/api/states/{entity_id}`.
- **Todo / shopping list tool over HA's `todo.*` services.** Very common voice ask, and a smaller model won't intuit the `todo.add_item` / `todo.remove_item` / `todo.get_items` shape from `ha_execute_service`. Ship a dedicated tool (or small tool family) in `mcp_homeassistant_tools/` with list-name + item-text params and explicit descriptions. Check which `todo.*` list entities exist in the target HA instance before committing to the param shape.
- **`web_quick_answer` — additive convenience over `brave_search` + `fetch`.** Keep both primitives; add a convenience tool that runs a search and auto-fetches the top result's body in one call. Saves a round trip on the common "look something up" path, which matters on a local model. Not a replacement — when the model wants result #3 or wants to inspect titles before fetching, it still uses the primitives.

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

## Stretch goals

Post-MVP / "v2"-class features — multi-modal perception, identity awareness, richer output surfaces. To work on **after** the items above are cleared. Scoping, feasibility, effort, and suggested sequencing live in [`stretch-goals.md`](./stretch-goals.md).
