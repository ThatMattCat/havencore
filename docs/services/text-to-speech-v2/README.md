# Text-to-Speech v2 Service (Chatterbox-Turbo)

Expressive, zero-shot speech synthesis using [Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) from Resemble AI (MIT). Runs in parallel to the v1 Kokoro-based `text-to-speech` service and exposes the same OpenAI-compatible surface, so callers can swap engines by URL alone.

## Purpose

- Higher emotional range than Kokoro via native paralinguistic tags (`[laugh]`, `[sigh]`, `[chuckle]`, etc.)
- Zero-shot voice cloning from a short reference clip (no fine-tuning, no per-voice training)
- Drop-in replacement for `text-to-speech` — identical `/v1/audio/speech` schema, identical `X-Visemes` header format

## Architecture

```
Text Input → (pronunciation substitution) → Chatterbox-Turbo → WAV samples
                                              ↑                    ↓
                                   audio_prompt_path           Rhubarb Lip Sync
                                   (reference clip)                 ↓
                                                            X-Visemes header
                                                                    ↓
                                                             HTTP response
```

Chatterbox-Turbo is a 350M-parameter model with a distilled 1-step decoder. Sub-200 ms inference on a 3090 after warm-up; ~3 GB resident VRAM for single-user inference.

## Service endpoints

| Port | Purpose | Features |
|------|---------|----------|
| 6015 | OpenAI-compatible API | `/v1/audio/speech` (buffered), `/v1/audio/speech/stream` (NDJSON per-sentence), `/v1/voices`, `/v1/voices/upload`, `/v1/voices/{name}` (DELETE), `/health` |

For an interactive UI, use the agent dashboard's TTS playground at `/playgrounds/tts` — same page as v1, but the controls and labels switch automatically when `TTS_PROVIDER=v2`. The Voices card on that page is where voice uploads and the runtime-default voice are managed.

## Choosing the active engine

The agent reads `TTS_PROVIDER` from `.env` and routes its TTS client + `/api/tts/*` proxy to whichever engine is selected:

- `TTS_PROVIDER=v1` → `text-to-speech:6005` (Kokoro)
- `TTS_PROVIDER=v2` → `text-to-speech-v2:6015` (Chatterbox-Turbo)

Both services run in parallel; switching providers and `docker compose up -d agent` is the rollback path in either direction. Companion app, satellites, and dashboard playground all follow the active provider automatically because they reach TTS through the agent.

## Voices

Chatterbox is **zero-shot** — there are no preset voices baked into the model. A "voice" is a short reference WAV that the model clones the timbre of on every request via `audio_prompt_path`. The service ships with 20 bundled clips from [devnen/Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server) (MIT, attribution in `services/text-to-speech-v2/NOTICE`) and accepts uploads of additional clips.

### Voice resolution

When a request arrives with `voice: "<name>"`:

1. Exact match against any clip in `/app/voices/` (operator uploads) or `/opt/chatterbox-voices/` (bundled) → use that
2. OpenAI-compat alias (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) → resolves to the configured default
3. Unknown name → falls back to the configured default with a warning log

Custom uploads take precedence over bundled voices with the same name, so `Selene.wav` placed in the volume-mounted dir overrides any bundled `Selene` without rebuilding the image.

### Uploading a voice (cloning)

Via the dashboard at `/playgrounds/tts`:
- Upload form takes a name (1–40 chars, `[A-Za-z0-9_-]`) and an audio file.
- Accepted formats: WAV, FLAC, OGG. Convert MP3 first.
- Recommended clip: **10–30 seconds** of clean speech from a single speaker, minimal background noise.
- Hard limits: 3 s minimum, 120 s maximum.
- After upload the UI offers to make the new voice the runtime default (see below).

Via direct API:

```bash
curl -X POST http://localhost:6015/v1/voices/upload \
  -F "name=Selene" \
  -F "file=@./selene_reference.wav"
```

Returns `{name, path, duration_sec, original_sample_rate, stored_sample_rate}`. The clip is converted to mono and resampled to 24 kHz at storage time so subsequent requests skip re-encoding.

### Deleting a voice

```bash
curl -X DELETE http://localhost:6015/v1/voices/Selene
```

Bundled voices return `403 Forbidden` — they live in the image, not the mounted volume. Only uploaded clips are deletable.

### Runtime default voice

The agent persists a runtime-override default in the `agent_state.tts_default_voice` row. When set, it overrides `CHATTERBOX_VOICE` (and `TTS_VOICE` for v1) for every caller that doesn't send an explicit `voice`: chat dashboard, autonomy speaker announcements, companion app, satellites — all one knob.

Set/clear via the dashboard's Voices card or directly:

```bash
# Set
curl -X POST http://localhost:6002/api/tts/voices/default \
  -H "Content-Type: application/json" -d '{"voice":"Selene"}'

# Clear → falls back to the engine's configured default
curl -X POST http://localhost:6002/api/tts/voices/default \
  -H "Content-Type: application/json" -d '{"voice":null}'
```

Deleting a voice that's currently set as the runtime default automatically clears the override.

## Paralinguistic tags

Chatterbox-Turbo natively renders inline non-speech reactions when they appear in the input text. The full set:

```
[laugh] [chuckle] [sigh] [gasp] [groan] [cough] [sniff] [clear throat] [shush]
```

The agent's system prompt teaches the LLM to use these sparingly and only in spoken-reply text — but **only when `TTS_PROVIDER=v2`**. Under v1 (Kokoro) the same addendum is omitted because Kokoro would read the brackets aloud. See `selene_agent/utils/config.py:SYSTEM_PROMPT_PARALINGUISTIC_ADDENDUM` for the exact wording, applied in both `orchestrator.py:initialize` and `session_pool.py:rebuild_system_prompts`.

There is **no** `exaggeration` slider on Turbo — that knob is exclusive to the standard 500M Chatterbox model. Expressiveness comes from where the LLM places tags, not a numeric per-utterance setting.

## API usage

```bash
# Generate speech (same shape as v1)
curl -X POST http://localhost:6015/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Welcome home. [chuckle] The lights are already on.",
    "model": "tts-1",
    "voice": "Olivia",
    "response_format": "wav"
  }' \
  --output speech.wav
```

`response_format` accepts `wav`, `flac`, `ogg`, `opus`, `pcm`. Requests for `mp3`/`aac` fall back to WAV (libsndfile can't encode them without extra codec libs); content-type reflects the actual bytes.

### Voice catalog (`GET /v1/voices`)

```json
{
  "language": "chatterbox-turbo",
  "default": "Olivia",
  "native": ["Olivia", "Abigail", "..."],
  "user": ["Selene"],
  "bundled": ["Olivia", "Abigail", "..."],
  "aliases": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
}
```

`user` and `bundled` are v2-only additions that the dashboard uses to render delete buttons. The agent's `/api/tts/voices` proxy adds a `default_override` field with the current runtime-override voice (or `null`).

## Streaming (`POST /v1/audio/speech/stream`)

Buffered synthesis returns the whole utterance in one response — fine for short
replies, but a 30 s reply means ~5 s of dead air before any audio plays, and
the base64 Rhubarb timeline ends up in a single `X-Visemes` response header
that overflows aiohttp's default 8190-byte `max_field_size` past ~30 s. The
streaming endpoint splits the input by sentence, synthesizes each in turn, and
emits NDJSON events over chunked HTTP. Clients queue audio chunks gaplessly
and append visemes at the cumulative `offset_ms` the server reports.

Same request schema as `/v1/audio/speech`; `response_format` and `speed` are
ignored on this path (audio is always WAV-framed; Chatterbox has no speed
knob). Response is `Content-Type: application/x-ndjson` with
`X-Accel-Buffering: no` + `Cache-Control: no-store` so reverse proxies flush
each event as it's written.

### Event schema

One JSON object per line, newline-terminated:

```json
{"type":"start","stream_id":"<uuid>","sample_rate":24000,"audio_format":"wav","audio_mime":"audio/wav","total_sentences":N,"voice":"Olivia"}
{"type":"audio","seq":0,"offset_ms":0,"duration_ms":2100,"data":"<base64 WAV bytes for sentence 0>"}
{"type":"visemes","seq":0,"offset_ms":0,"duration_ms":2100,"cues":[{"start":0.00,"end":0.05,"value":"X"}, ...]}
{"type":"audio","seq":1,"offset_ms":2100,"duration_ms":1800,"data":"..."}
{"type":"visemes","seq":1,"offset_ms":2100,"duration_ms":1800,"cues":[...]}
{"type":"done","total_ms":29960,"sentences":N}
```

Errors are terminal: `{"type":"error","seq":N,"detail":"..."}` — clients
surface and stop. Per-sentence failures emit the error event and bail (no
partial-audio gaps). Empty / whitespace-only input emits a single
`error` event with `detail:"empty input"`. Each audio chunk is a
self-contained WAV file (full RIFF header), so a streaming `ConcatenatingMediaSource`
on Android or a sequential `Audio` element on the web can play them
without bitstream splicing.

### Sentence splitting

`app/sentences.py` (pure, unit-testable). Rules:

1. Split on `re.split(r'(?<=[.!?])\s+', text)` — keeps terminal punctuation on
   the preceding sentence so prosody falls naturally.
2. Merge sub-30-char fragments into the next chunk. Avoids 200 ms tail chunks
   between full sentences. The trailing fragment, having no "next chunk" to
   merge into, is emitted as its own event.
3. Hard cap at 280 chars per chunk; oversize chunks split on the nearest
   whitespace inside the window.

Known limitation: `"Dr. Smith said hello."` splits wrong. Acceptable today
(assistant output doesn't lean on medical/honorific titles); a heavier
splitter (NLTK Punkt, spaCy) is overkill for this scope.

### Performance — voice-prompt hoist

`chatterbox.tts_turbo.ChatterboxTurboTTS.generate(audio_prompt_path=...)`
calls `prepare_conditionals()` on every invocation — librosa load, loudness
normalization, `s3gen.embed_ref()`, voice-encoder forward pass, and S3
tokenizer all rerun per call. Naively re-passing `audio_prompt_path` per
sentence would amortize that pipeline N times. Streaming instead calls
`model.prepare_conditionals(audio_prompt_path)` once at stream start, then
calls `model.generate(text)` (no `audio_prompt_path`) per sentence, which
reuses the cached `model.conds`. TTFA on a 3-sentence input is ~1.1 s vs
~5 s for the buffered path on the same text.

### Concurrency

The model's reference-voice conditionals live in `model.conds` as mutable
process-global state. A module-level `threading.Lock` in `app/streaming.py`
guards both the `prepare_conditionals` call and every per-sentence
`generate` call for the duration of a stream. The buffered endpoint does
not acquire this lock — on this single-user host, concurrent TTS is rare
enough that the pre-existing race between two buffered callers (or one
buffered + one streaming) is accepted rather than retrofit.

Rhubarb runs serially with the next sentence's synthesis today (one
sentence's mouth-cue subprocess blocks the next sentence's synth from
starting). Overlapping is a clean follow-up but isn't wired yet.

### When to use which endpoint

- **`/v1/audio/speech` (buffered)** — autonomy speaker (Music Assistant
  needs a single URL it can play), external OpenAI-SDK clients pointed at
  port 80, debugging, regression tests, or any caller that wants a single
  blob back. v1 (Kokoro) only exposes this surface.
- **`/v1/audio/speech/stream` (streaming)** — dashboard chat + playground,
  companion app, and any future Live2D pipeline. TTFA win matters whenever
  the reply is more than one sentence long. v2 only.

### Curl smoke test

```bash
curl -N -X POST http://localhost:6015/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello there. This is a test of streaming. The mouth should move smoothly.","voice":"Olivia"}' \
  | head -20
```

Expect: a `start` event, then alternating `audio` / `visemes` events keyed
by `seq`, then `done`. The same request through the agent's
[`POST /api/tts/speak/stream`](../../api-reference.md#tts) proxy applies
the runtime-override default voice and adds the same NDJSON pass-through
framing.

## Configuration

```bash
# Which engine the agent uses. v1 keeps the existing Kokoro path; v2 routes
# to this service. Companion app and satellites reach TTS through the agent
# so they follow this switch automatically.
TTS_PROVIDER="v2"

# Engine base URLs (defaults work for the in-compose hostnames).
TTS_V1_BASE_URL="http://text-to-speech:6005"
TTS_V2_BASE_URL="http://text-to-speech-v2:6015"

# GPU pinning. Chatterbox has no tensor-parallel support — it runs on a
# single GPU. CHATTERBOX_GPU sets CUDA_VISIBLE_DEVICES on the container
# (same pattern as vllm-vision/face-recognition); inside the container the
# model always uses cuda:0.
CHATTERBOX_GPU="2"
CHATTERBOX_DEVICE="cuda:0"

# Default voice. Must match a clip name in /app/voices/ or
# /opt/chatterbox-voices/. The runtime-override set via /api/tts/voices/default
# takes precedence when present.
CHATTERBOX_VOICE="Olivia"

# Optional text substitutions applied before synthesis. Chatterbox has no
# lexicon-injection hook (resemble-ai/chatterbox#115), so the workaround
# is to rewrite the spelling. Distinct from v1's TTS_PRONUNCIATIONS which
# is IPA — v2 expects plain pseudo-phonetic English. Empty by default;
# Chatterbox handles most proper nouns reasonably without help.
#TTS_V2_PRONUNCIATIONS='{"Selene":"Suh-leen"}'
```

VRAM headroom matters: the service allocates ~3 GB on the selected GPU at startup, and per-request activations can grow another few hundred MB. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the compose env to ease fragmentation on a shared GPU.

## Audio formats

Same as v1: `wav`, `flac`, `ogg`, `opus`, `pcm` encode directly via libsndfile; `mp3` and `aac` fall back to WAV with an honest `audio/wav` content-type.

## Lip-sync viseme timeline (`X-Visemes` header)

**Byte-compatible with v1.** Every `/v1/audio/speech` response carries an `X-Visemes` header containing the same base64-encoded Rhubarb Lip Sync JSON timeline that the v1 service emits. The companion app's `VisemeScheduler` / `VisemeTimeline` decoder (see `havencore-companion-app/.../voice/avatar/`) is unchanged — switching providers requires zero client-side work.

```
HTTP/1.1 200 OK
Content-Type: audio/wav
X-Visemes: eyJtZXRhZGF0YSI6...    ← base64 JSON, optional
Access-Control-Expose-Headers: X-Visemes
```

Decoded JSON shape and tunables are identical to v1 — see [text-to-speech/README.md](../text-to-speech/README.md#lip-sync-viseme-timeline-x-visemes-header) for the full reference.

**Soft-fail behavior**: if `rhubarb` is missing, hits the timeout, or exits non-zero, the response omits the header and the body is unchanged. Same env knobs as v1 (`RHUBARB_BIN`, `RHUBARB_TIMEOUT_SEC`, `RHUBARB_RECOGNIZER`).

The header has a structural ceiling: past ~30 s of speech the base64-encoded
viseme JSON exceeds aiohttp's default 8190-byte `max_field_size`. The agent
bumps that limit to 65536 on its buffered TTS sessions (`api/tts.py:speak`
and `services/tts_client.py:synth`) to push the ceiling out, but the
permanent fix is the [streaming endpoint](#streaming-post-v1audiospeechstream) —
viseme cues ride the response body there, not a header, so there is no
upper bound.

## Dashboard playground

`http://localhost/playgrounds/tts` works against whichever engine is active. The page's voice dropdown labels reflect the engine (`Olivia (Chatterbox-Turbo)` vs `af_heart (Kokoro)`). The page also exposes a **Voices** card with:

- Current default voice + override badge when the runtime override is set
- "Reset to engine default" (clears the override)
- Upload form with 10–30 s guidance and accepted formats
- List of every voice with "Set as default" and "Delete" (only on uploaded clones)

### Delivery toggle (Stream / Buffered)

The Input card surfaces a **Delivery** toggle for A/B-ing the two TTS surfaces:

- **Stream** — calls `/api/tts/speak/stream`, decodes per-sentence NDJSON
  audio events, and plays them as they arrive. The Output card switches
  from `<audio controls>` to a live status panel showing TTFA (time to
  first audio chunk in ms), chunk count, and total synthesized speech ms
  once `done` lands. A red **Stop streaming** button replaces the Speak
  button while a stream is in flight. WAV is forced on this path, so the
  Format and Speed inputs are disabled (Chatterbox-Turbo has no speed knob
  regardless).
- **Buffered** — calls `/api/tts/speak`, waits for the whole audio blob,
  and renders an `<audio controls>` element plus a download link. Useful
  for comparing TTFA against the streaming path, and for grabbing a
  single-file render to play outside the dashboard.

The toggle defaults to Stream when v2 is active; under v1 (Kokoro) it
collapses to Buffered-only because the upstream has no streaming endpoint.
The page detects engine support automatically from the voice-list label
embedded in `GET /api/tts/voices`.

### /chat auto-speak

The Chat page's speaker icon (auto-speak) uses the streaming endpoint
unconditionally — there is no toggle there. When enabled, each assistant
turn that completes is synthesized and played sentence-by-sentence
through the same Blob-queue/`'ended'`-advance pattern the playground
uses. Stop is wired to the existing playback indicator: navigating away,
sending a new message, or toggling auto-speak off cancels the in-flight
stream and revokes every queued blob URL.

## Troubleshooting

```bash
# Health
curl http://localhost:6015/health

# Verify voice catalog
curl http://localhost:6015/v1/voices | python3 -m json.tool

# Quick synthesis with viseme header
curl -sS -D /tmp/h.txt -X POST http://localhost:6015/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, this is a test.","voice":"Olivia","response_format":"wav"}' \
  -o /tmp/say.wav
grep -i '^X-Visemes:' /tmp/h.txt | sed 's/^X-Visemes: //;s/\r$//' \
  | base64 -d | python3 -m json.tool

# GPU usage
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv

# Service logs
docker compose logs -f text-to-speech-v2
```

**Cold start**: model load takes 30–60 s on a 3090. Healthcheck `start_period` is set to 120 s.

**OOM on the configured GPU**: bump `CHATTERBOX_GPU` to a less-busy card in `.env` and `docker compose up -d text-to-speech-v2`. The model itself needs ~3 GB; activations can push peak higher.

**Voice sounds wrong / inconsistent**: a too-short or noisy reference clip is the usual cause. Re-record at 10–30 s of clean single-speaker audio.

## Related

- [text-to-speech (v1, Kokoro)](../text-to-speech/README.md) — parallel service, fallback engine
- [Configuration](../../configuration.md) — full env-var reference including `TTS_PROVIDER`
- [API Reference](../../api-reference.md) — full endpoint contracts including the new voice-management routes
