# Stretch Goals

Post-MVP / "v2"-class features: multi-modal perception, identity awareness, richer output surfaces. Work on these *after* the real TODOs in `todo.md` are cleared. Each entry captures current-state hook points, approach, effort, and dependencies.

Effort scale: **S** = weekend (4–12 hrs), **M** = 1–2 weeks spare-time, **L** = 3–6 weeks spare-time, **XL** = 2+ months.

---

## 1. Speaker Recognition (owner vs others)

**Plausibility: High.** Clean hook points, proven models, no architectural fight.

**Approach:**
- Run a speaker-embedding model (SpeechBrain ECAPA-TDNN or pyannote, ~50 MB, 192-dim vector, CPU-fast) on each inbound utterance.
- Best location: **satellite-side** if ESP32 can afford the cycles; otherwise add a parallel step inside `services/speech-to-text/` that returns `{transcript, embedding}` — the STT service currently discards audio after Whisper runs (`services/speech-to-text/app/main.py`), so adding a second pass on the same buffer is the natural fit.
- Enrollment flow: dashboard page records 3–5 utterances per household member, stores centroid embeddings in a new `speakers` table.
- At turn time: cosine-similarity against enrolled profiles, attach `speaker_id` + confidence to the turn. Fallback label: `unknown`.
- Storage: `turn_metrics.metadata` JSONB is already flexible — no migration needed. A `speakers` table is additive.

**Effort: S–M.**
- S: owner-only binary classifier, single enrolled profile, no UI.
- M: multi-user enrollment UI, per-user memory namespacing in the Qdrant L1–L4 tiers, autonomy gating ("only owner can arm/disarm").

**Unlocks:** per-user memory, owner-only privileged actions, personalized tone, emotion detection (same pipeline, different head).

**Gotchas:**
- STT endpoint at `services/agent/selene_agent/api/stt.py` currently passes through a multipart file; would need to either forward it to STT with a richer response or do embeddings upstream.
- No audio retention in Whisper — if satellite doesn't do embeddings, STT service must do it before discarding the buffer.

---

## 2. Face Recognition (doorbell + general cameras)

**Plausibility: High** — if offloaded to Frigate rather than rolling our own.

**Approach:**
- **Recommended**: stand up **Frigate** as an HA add-on, point it at camera streams, enable its built-in face recognition (or plug in `deepstack` / `double-take`). Frigate publishes detection + identity events to MQTT — a surface HavenCore already consumes.
- Extend `selene_agent/modules/mcp_mqtt_tools/` with a `camera_face_events` subscriber; wire detections into the autonomy engine (`autonomy/mqtt_listener.py`) so "unknown person at front door" becomes a trigger.
- Alternative (DIY, not recommended first pass): add `face_recognition` Python lib or InsightFace to a new service; camera snapshots already work via `get_camera_snapshots()` in `mcp_mqtt_tools/mcp_server.py`.

**Effort: S (with Frigate) | L (rolling own).**

**Unlocks:** doorbell greeter, "who's home" context, autonomy rules keyed to identity, per-person notifications.

**Gotchas:**
- Frigate wants its own GPU slice; vLLM + iav-to-text + ComfyUI + embeddings are already coexisting. May need a second GPU or CPU-only detection.
- Qwen2.5-Omni (iav-to-text) can *describe* faces but is unreliable for identity matching — don't use it as a face-rec primitive.

---

## 3. Larger Displays + Generated Avatar

**Plausibility: Moderate for good UX; hardware path is the rate limiter.**

Three sub-problems: hardware, rendering, pushing frames. The last one is mostly solved.

### 3a. Hardware options

| Option | Cost/unit | Pros | Cons |
|---|---|---|---|
| Wall-mounted Fire tablet / old iPad running dashboard in kiosk | $30–100 | Works today, no new code | Mic quality varies, charging logistics |
| Raspberry Pi 4/5 + 7" touchscreen + ReSpeaker mic HAT | $150–200 | Full control, good mic | DIY assembly |
| ESP32-S3 + larger display (WT32-SC01, Cheap Yellow Display 4.3"/5") | $40–80 | Same firmware lineage as current satellites | Limited rendering; avatar must be pre-rendered sprites |
| HA Voice PE / Onju Voice / similar appliance | $60–100 | Off-the-shelf | Small screens, limited extensibility |

**Recommendation:** Fire tablet wall-mounts are the fastest path. Treat ESP32 satellites as "audio-only" and tablets as "audio+visual." A single SvelteKit route served by the agent can be the kiosk UI.

### 3b. Avatar rendering approaches

| Approach | Latency | Effort | Quality |
|---|---|---|---|
| Static image + CSS mouth-animation driven by TTS amplitude | Instant | S | Cute but obviously fake |
| **Live2D / VTuber model** (open-source rigging + amplitude-driven lip-sync) | ~real-time | M | Anime-ish but expressive, cheap |
| Talking-head neural models (MuseTalk, EchoMimic, SadTalker) | 0.5–2× realtime on good GPU | L | Photorealistic but uncanny and slow |
| ComfyUI per-utterance generation | 10–30 s | S to wire, unusable for speech | N/A |
| Pre-rendered emotion frames cycled by state | Instant | S–M | Sprite-sheet feel |

**Recommendation:** **Live2D** is the sweet spot. Mature open-source tooling, amplitude-driven lip-sync, emotion states, runs without eating vLLM's GPU budget. Idle loops are free.

### 3c. Pushing frames to displays

- WebSocket `/ws/chat` already streams typed events and knows `device_name`. Add `{"type":"avatar_state", ...}` frames; kiosk subscribes by device name.
- For frame-level streaming (neural talking-head): MQTT works, but WebSocket is already ordered and there.

**Effort total:**
- S: kiosk route + static avatar + amplitude mouth animation.
- M: Live2D rig + emotion state transitions + idle loops + TTS-synced lip-sync.
- L: neural talking head + real-time streaming TTS coordination.

**Unlocks:** visual presence, idle ambient info display, natural home for face-recognition visitor previews.

---

## 4. Streaming TTS + Barge-in (HIGH leverage)

**Effort: M.** Current TTS is whole-utterance; Kokoro internally yields chunks but concatenates before responding (`services/text-to-speech/app/main.py`). Refactor to stream chunks over `/ws/chat`, let satellite play as they arrive, wire a "user started talking" signal that cancels in-flight LLM + TTS. Dramatically improves conversational feel and is a prerequisite for the avatar feeling alive during speech.

## 5. Presence Awareness ("who's home, which room")

**Effort: S.** HA already tracks phones/BT/wifi. Add an MCP tool that reads HA presence state; combine with speaker-ID + satellite `device_name` for room-level confidence.

## 6. Emotion-from-Voice

**Effort: S** alongside speaker ID. SpeechBrain has pretrained emotion classifiers running on the same audio buffer. Attach `emotion` label to turn metadata; system prompt / autonomy rules adapt.

## 7. Doorbell Visitor Greeter

**Effort: S** given #2 and bi-directional doorbell audio. Unknown face → autonomy trigger → TTS through doorbell speaker → capture reply → transcribe → decide: notify owner, take message, open gate, etc.

## 8. Personal Wake-Word Trained on Owner

**Effort: S–M**, mostly in the satellite firmware repo. microwakeword on-device training pipeline. Combine with on-device voice-print check to reject imposter wake attempts.

## 9. End-to-End Speech-to-Speech Mode (Moshi / Qwen-Audio)

**Effort: XL.** Replaces STT+LLM+TTS with one model for latency (~200 ms). Would need fallback to classical pipeline for tool-calling turns. Defer until open models close the quality gap with Qwen2.5-72B tool-calling.

## 10. Learned Routines / Predictive Automation

**Effort: L.** Nightly job over HA history + conversation logs proposes automations. Fits naturally into the autonomy v2 L1–L4 memory-tier direction.

## 11. Follow-Me Audio Across Satellites

**Effort: M.** Presence (#5) + per-room satellite awareness already in place. Music/TTS hops rooms as owner moves. Requires coordination in `mcp_music_assistant_tools/` and agent state.

## 12. Ambient Info on Idle Displays

**Effort: S** once kiosk exists. Idle large display shows weather, calendar, last-seen camera snapshot, recent autonomy events — smart-home home-screen. Natural complement to the avatar.

---

## Suggested Sequencing

Ordered by leverage and dependency:

1. **Streaming TTS + barge-in (#4, M)** — unblocks natural conversation; prerequisite for non-janky avatar.
2. **Speaker recognition (#1, S–M)** — unlocks per-user memory, security; feeds emotion/presence.
3. **Face rec via Frigate (#2, S)** — cheap given Frigate does the heavy lifting.
4. **Presence + emotion (#5 + #6, S each)** — small additions on top of #1.
5. **Kiosk display + Live2D avatar (#3, M)** — the visible payoff.
6. **Doorbell greeter (#7, S)** — composes #2, #4, #11.
7. **Personal wake-word (#8, S–M)** — satellite-repo work, can parallelize.
8. **Learned routines (#10, L)** — longer horizon, plugs into autonomy v2.
9. **Speech-to-speech (#9, XL)** — defer; revisit when open models catch up.

**Total for items 1–7:** ~2–3 months of spare-time work, mostly independent chunks.

---

## Notes

- GPU budget is the real constraint. vLLM, iav-to-text (Qwen-Omni), ComfyUI, embeddings already coexist. Adding Frigate + neural avatar would likely need a second GPU. Live2D sidesteps this.
- Satellite firmware lives in a separate repo (`havencore-satellite-firmware`); anything touching audio capture, wake-word, or on-device embeddings straddles both repos.
- Multi-user / speaker-aware memory is already flagged in the autonomy v2 direction — speaker ID (#1) is effectively the prerequisite for that roadmap, not a side quest.
