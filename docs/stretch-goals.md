# Stretch Goals

Post-MVP / longer-horizon features: multi-modal perception, identity awareness, richer output surfaces. Work on these *after* the real TODOs in `todo.md` are cleared. Each entry captures current-state hook points, approach, effort, and dependencies.

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
- M: multi-user enrollment UI, per-user memory namespacing in the Qdrant L1–L4 tiers, autonomy gating ("only the owner can arm/disarm").

**Unlocks:** per-user memory, owner-only privileged actions, personalized tone, emotion detection (same pipeline, different head).

**Gotchas:**
- STT endpoint at `services/agent/selene_agent/api/stt.py` currently passes through a multipart file; would need to either forward it to STT with a richer response or do embeddings upstream.
- No audio retention in Whisper — if satellite doesn't do embeddings, STT service must do it before discarding the buffer.

---

## 2. Larger Displays + Generated Avatar

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

## 3. Streaming TTS + Barge-in (HIGH leverage)

**Effort: M.** Current TTS is whole-utterance; Kokoro internally yields chunks but concatenates before responding (`services/text-to-speech/app/main.py`). Refactor to stream chunks over `/ws/chat`, let satellite play as they arrive, wire a "user started talking" signal that cancels in-flight LLM + TTS. Dramatically improves conversational feel and is a prerequisite for the avatar feeling alive during speech.

## 4. Presence Awareness ("who's home, which room")

**Effort: S.** HA already tracks phones/BT/wifi. Add an MCP tool that reads HA presence state; combine with speaker-ID + satellite `device_name` for room-level confidence.

## 5. Emotion-from-Voice

**Effort: S** alongside speaker ID. SpeechBrain has pretrained emotion classifiers running on the same audio buffer. Attach `emotion` label to turn metadata; system prompt / autonomy rules adapt.

## 6. Doorbell Visitor Greeter

**Effort: S** given the shipped face-recognition service plus bi-directional doorbell audio. Unknown face → autonomy trigger → TTS through doorbell speaker → capture reply → transcribe → decide: notify owner, take message, open gate, etc.

## 7. Personal Wake-Word Trained on Owner

**Effort: S–M**, mostly in the satellite firmware repo. microwakeword on-device training pipeline. Combine with on-device voice-print check to reject imposter wake attempts.

## 8. End-to-End Speech-to-Speech Mode (Moshi / Qwen-Audio)

**Effort: XL.** Replaces STT+LLM+TTS with one model for latency (~200 ms). Would need fallback to classical pipeline for tool-calling turns. Defer until open models close the quality gap with Qwen2.5-72B tool-calling.

## 9. Learned Routines / Predictive Automation

**Effort: L.** Nightly job over HA history + conversation logs proposes automations. Fits naturally on top of the L1–L4 memory consolidation pipeline.

## 10. Follow-Me Audio Across Satellites

**Effort: M.** Presence (#5) + per-room satellite awareness already in place. Music/TTS hops rooms as owner moves. Requires coordination in `mcp_music_assistant_tools/` and agent state.

## 11. Ambient Info on Idle Displays

**Effort: S** once kiosk exists. Idle large display shows weather, calendar, last-seen camera snapshot, recent autonomy events — smart-home home-screen. Natural complement to the avatar.

---

## Suggested Sequencing

Ordered by leverage and dependency.

1. **Streaming TTS + barge-in (#3, M)** — unblocks natural conversation; prerequisite for a non-janky avatar.
2. **Speaker recognition (#1, S–M)** — unlocks per-user memory, security; feeds emotion/presence.
3. **Presence + emotion (#4 + #5, S each)** — small additions on top of #1.
4. **Kiosk display + Live2D avatar (#2, M)** — the visible payoff.
5. **Doorbell greeter (#6, S)** — composes the shipped face-recognition MQTT events with bi-directional doorbell audio.
6. **Personal wake-word (#7, S–M)** — satellite-repo work, can parallelize.
7. **Learned routines (#9, L)** — longer horizon, plugs into the L1–L4 memory consolidation tier.
8. **Speech-to-speech (#8, XL)** — defer; revisit when open models catch up.

**Total for items 1–6:** ~2–3 months of spare-time work, mostly independent chunks.

---

## Notes

- GPU budget is the real constraint. vLLM (`-tp 4`) spans the four main GPUs at `--gpu-memory-utilization 0.79`; face-recognition co-tenants GPU 3 with vLLM (small ~600 MB resident model). ComfyUI and embeddings co-tenant on the lower GPUs. Vision (`vllm-vision`, Qwen3-VL) lives on a dedicated 5th RTX 3090 (GPU 4). A neural-avatar talking head would still need new headroom. Live2D sidesteps this.
- Satellite firmware lives in a separate repo (`havencore-satellite-firmware`); anything touching audio capture, wake-word, or on-device embeddings straddles both repos.
- The Android companion app lives in a separate repo (`havencore-companion-app`); anything touching mobile chat, the default-assistant slot, or push notifications straddles both repos.
- Multi-user / speaker-aware memory pairs naturally with the L1–L4 memory consolidation tiers — speaker ID (#1) is effectively the prerequisite for that direction, not a side quest.
