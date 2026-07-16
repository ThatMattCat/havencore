# Text-to-Speech Service

Speech synthesis exposed as an OpenAI-compatible `/v1/audio/speech` endpoint,
plus an `X-Visemes` header for avatar lip-sync. **Two engines are available and
mutually exclusive** — exactly one runs at a time, chosen by config:

| Engine | Default | Strengths | Limits |
|--------|:------:|-----------|--------|
| **Kokoro** (`text-to-speech-kokoro`) | ✓ | Small (82M), fast, low latency; fixed model voices | No streaming or voice cloning; does not render `[laugh]`/`[sigh]` tags |
| **Chatterbox-Turbo** (`text-to-speech`) | | Expressive, zero-shot voice cloning, per-sentence streaming, inline paralinguistic tags | Higher latency (~350M model) |

Both emit 24 kHz mono audio and the same `X-Visemes` Rhubarb timeline
(provider-independent), so avatar lip-sync works identically on either.

## Engine selection

Two vars must agree — one selects the container, one tells the agent which
engine is live:

```bash
COMPOSE_PROFILES="kokoro"   # which container docker compose starts. REQUIRED —
                            # both engines are profile-gated; unset => no TTS starts.
TTS_PROVIDER="kokoro"       # which engine the agent assumes. Gates streaming,
                            # voice cloning, and the [laugh]/[sigh] prompt tags.
```

To switch to Chatterbox, set **both** to `chatterbox` and rebuild:
`docker compose down && docker compose up -d --build`.

Both services answer at the same `text-to-speech` network alias (Kokoro via an
explicit compose alias), so the agent (`TTS_BASE_URL`, default
`http://text-to-speech:6005`) and the nginx `/v1/audio/speech` upstream reach
whichever engine is active with no per-engine config. nginx resolves the alias
lazily per-request so it survives the active engine restarting.

### Behavior differences the agent handles

- **Streaming** (`/v1/audio/speech/stream`) is a Chatterbox-only upstream.
  Under Kokoro the agent's `/api/tts/speak/stream` proxy transparently degrades
  to a **single-shot** NDJSON stream built from the buffered synth (same
  `start`/`audio`/`visemes`/`done` schema, one "sentence"), so streaming
  clients — the companion app's voice flow, the dashboard — keep working
  unchanged.
- **Voice cloning** (`/v1/voices/upload`, `DELETE /v1/voices/{name}`) is
  Chatterbox-only. Under Kokoro the agent proxy returns **501** (Kokoro has
  fixed model voices).
- **Paralinguistic tags** (`[laugh]`, `[sigh]`, ...): the agent appends the
  teaching addendum to the system prompt only under Chatterbox; Kokoro would
  speak the brackets aloud. The `<<EXPRESSION>>` avatar cue is engine-agnostic
  and always taught.

## Kokoro (default engine)

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) is an 82M-parameter model
(Apache-2.0). Runs as `text-to-speech-kokoro` on port 6005 (host-mapped to
6015), downloading its weights into `~/.cache/huggingface` at first start. It
uses **fixed model voices** rather than cloning — a "voice" is a preset name
like `af_heart` (default), not a reference clip.

```
Text Input → (misaki G2P + pronunciation override) → Kokoro → WAV samples
                                                                  ↓
                                                          Rhubarb Lip Sync
                                                                  ↓
                                                         X-Visemes header
```

- Voices are language-scoped: `GET /v1/voices` returns only voices whose prefix
  matches `TTS_LANGUAGE` (`a` = American English → `af_*`/`am_*`, `b` = British,
  etc.). Unknown or OpenAI-alias voices fall back to `TTS_VOICE` (default
  `af_heart`) with a warning.
- Endpoints: `POST /v1/audio/speech` (buffered), `GET /v1/voices`, `GET /health`.
  There is **no** streaming, upload, or delete endpoint (see the agent-handled
  behavior differences above).
- Pronunciation of the agent name is forced via a misaki phoneme override
  (`TTS_AGENT_NAME_PHONEMES`, IPA — default `səˈlin`).
- Config vars: `TTS_KOKORO_GPU` (host GPU), `TTS_DEVICE` (in-container device,
  `cuda:0` or `cpu`), `TTS_LANGUAGE`, `TTS_VOICE`, `TTS_AGENT_NAME_PHONEMES`.

---

## Chatterbox-Turbo (opt-in engine)

Expressive, zero-shot speech synthesis using [Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) from Resemble AI (MIT). The sections below detail this engine; enable it with `COMPOSE_PROFILES=chatterbox` + `TTS_PROVIDER=chatterbox`.

- Higher emotional range via native paralinguistic tags (`[laugh]`, `[sigh]`, `[chuckle]`, etc.)
- Zero-shot voice cloning from a short reference clip (no fine-tuning, no per-voice training)
- OpenAI-compatible `/v1/audio/speech` surface plus an `X-Visemes` header for avatar lip-sync

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
| 6005 | OpenAI-compatible API | `/v1/audio/speech` (buffered), `/v1/audio/speech/stream` (NDJSON per-sentence), `/v1/voices`, `/v1/voices/upload`, `/v1/voices/{name}` (DELETE), `/health` |

For an interactive UI, use the agent dashboard's TTS playground at `/playgrounds/tts` — it proxies through the agent service. The Voices card on that page is where voice uploads and the runtime-default voice are managed.

The agent reaches this service at `text-to-speech:6005` (the `TTS_BASE_URL` default). The companion app, satellites, and dashboard playground all reach TTS through the agent's `/api/tts/*` proxy.

## Voices

Chatterbox is **zero-shot** — there are no preset voices baked into the model. A "voice" is a short reference WAV that the model clones the timbre of on every request via `audio_prompt_path`. The service ships with 20 bundled clips from [devnen/Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server) (MIT, attribution in `services/text-to-speech/NOTICE`) and accepts uploads of additional clips.

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
curl -X POST http://localhost:6005/v1/voices/upload \
  -F "name=Selene" \
  -F "file=@./selene_reference.wav"
```

Returns `{name, path, duration_sec, original_sample_rate, stored_sample_rate}`. The clip is converted to mono and resampled to 24 kHz at storage time so subsequent requests skip re-encoding.

### Deleting a voice

```bash
curl -X DELETE http://localhost:6005/v1/voices/Selene
```

Bundled voices return `403 Forbidden` — they live in the image, not the mounted volume. Only uploaded clips are deletable.

### Runtime default voice

The agent persists a runtime-override default in the `agent_state.tts_default_voice` row. When set, it overrides `CHATTERBOX_VOICE` for every caller that doesn't send an explicit `voice`: chat dashboard, autonomy speaker announcements, companion app, satellites — all one knob.

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

The agent's system prompt teaches the LLM to use these sparingly and only in spoken-reply text. See `selene_agent/utils/config.py:SYSTEM_PROMPT_PARALINGUISTIC_ADDENDUM` for the exact wording; it is applied by `orchestrator.build_system_prompt()`, the shared assembly used for session creation, cold-resume, and the phase-change refresh.

There is **no** `exaggeration` slider on Turbo — that knob is exclusive to the standard 500M Chatterbox model. Expressiveness comes from where the LLM places tags, not a numeric per-utterance setting.

## API usage

```bash
# Generate speech
curl -X POST http://localhost:6005/v1/audio/speech \
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

`user` and `bundled` split the catalog so the dashboard can render delete buttons. The agent's `/api/tts/voices` proxy adds a `default_override` field with the current runtime-override voice (or `null`).

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

### Performance — conds cache (cross-stream)

The hoist above only saves work *within* a stream. Across streams the
same reference voice still re-ran the full prompt pipeline. A module-level
`_conds_cache` in `app/streaming.py` now memoizes the `Conditionals`
object produced by `prepare_conditionals`, keyed by
`(absolute audio_prompt_path, mtime)`. On a cache hit the prep step is a
pointer-swap into `model.conds` and the stream goes straight to the first
sentence's `generate`. Re-uploading the same voice file gets a fresh
mtime and naturally invalidates the entry; `DELETE /v1/voices/{name}`
calls `streaming.invalidate_conds_cache(...)` to purge explicitly. No
LRU — bundled (~20) + user-uploaded (typically 0-5) voices fit
comfortably at tens of MB each, and the single-user host re-uses the
same voice turn after turn. Reads + writes happen under the same
`_model_lock` that guards `prepare_conditionals` and per-sentence
`generate`, so a cache-hit assignment can't race a concurrent prep.

Measured directly against the localhost endpoint with a 3-sentence
input on `Olivia` (Chatterbox-Turbo, GPU 0):

| Scenario               | TTFA   |
|------------------------|--------|
| Cold (first after restart, miss) | ~2.4 s |
| Warm (same voice, hit) | ~1.0 s |
| New voice (miss, GPU warm)       | ~1.1 s |

End-to-end through the agent proxy + companion app, warm TTFA on a real
chat turn drops from ~5.4 s to ~1.5 s. The buffered `/v1/audio/speech`
endpoint calls `prepare_conditionals` *inside* `model.generate(...)` and
does not consult this cache — that path is autonomy / Music Assistant /
external SDK clients only, none of which are TTFA-sensitive in the same
way. Extending the cache there is a clean follow-up if the buffered TTFA
ever matters.

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
  blob back.
- **`/v1/audio/speech/stream` (streaming)** — dashboard chat + playground,
  companion app, and any future Live2D pipeline. The TTFA win matters
  whenever the reply is more than one sentence long.

### Curl smoke test

```bash
curl -N -X POST http://localhost:6005/v1/audio/speech/stream \
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
# --- Engine selection (see "Engine selection" above) ---
COMPOSE_PROFILES="kokoro"      # REQUIRED: which TTS container starts (kokoro|chatterbox)
TTS_PROVIDER="kokoro"          # which engine the agent assumes; keep in sync with the profile

# Where the agent reaches the TTS service. Both engines share the
# `text-to-speech` alias, so this default works for either — only set it for a
# non-standard layout.
#TTS_BASE_URL="http://text-to-speech:6005"

# --- Kokoro (default engine) ---
TTS_KOKORO_GPU="0"             # host GPU index for the Kokoro container
TTS_DEVICE="cuda:0"            # in-container device Kokoro loads on ("cpu" also works)
TTS_LANGUAGE="a"               # Kokoro language code (a=American English, b=British, ...)
TTS_VOICE="af_heart"           # default Kokoro voice, matched to TTS_LANGUAGE
#TTS_AGENT_NAME_PHONEMES="səˈlin"   # IPA override for the agent name (misaki G2P)

# --- Chatterbox-Turbo (opt-in engine) ---
# GPU pinning. Chatterbox has no tensor-parallel support — it runs on a
# single GPU. CHATTERBOX_GPU sets CUDA_VISIBLE_DEVICES on the container
# (same pattern as vllm-vision/face-recognition); inside the container the
# model always uses cuda:0.
CHATTERBOX_GPU="0"
CHATTERBOX_DEVICE="cuda:0"

# Default voice. Must match a clip name in /app/voices/ or
# /opt/chatterbox-voices/. The runtime-override set via /api/tts/voices/default
# takes precedence when present.
CHATTERBOX_VOICE="Olivia"

# Optional text substitutions applied before synthesis. Chatterbox has no
# lexicon-injection hook (resemble-ai/chatterbox#115), so the workaround
# is to rewrite the spelling. Empty by default; Chatterbox handles most
# proper nouns reasonably without help.
#TTS_PRONUNCIATIONS='{"Selene":"Suh-leen"}'
```

VRAM headroom matters: the service allocates ~3 GB on the selected GPU at startup, and per-request activations can grow another few hundred MB. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the compose env to ease fragmentation on a shared GPU.

## Audio formats

`wav`, `flac`, `ogg`, `opus`, `pcm` encode directly via libsndfile; `mp3` and `aac` fall back to WAV with an honest `audio/wav` content-type.

## Lip-sync viseme timeline (`X-Visemes` header)

Every `/v1/audio/speech` response carries an `X-Visemes` header containing a base64-encoded [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync) JSON timeline, for consumers that want to lip-sync a Live2D / 3D avatar against the TTS audio. The agent's `/api/tts/speak` proxy forwards this header unchanged.

```
HTTP/1.1 200 OK
Content-Type: audio/wav
Content-Length: 150044
X-Visemes: eyJtZXRhZGF0YSI6...    ← base64 JSON, optional
Access-Control-Expose-Headers: X-Visemes
```

Decoded JSON (Rhubarb `--machineReadable` `-f json`):

```json
{
  "metadata": {"duration": 3.12},
  "mouthCues": [
    {"start": 0.00, "end": 0.25, "value": "X"},
    {"start": 0.25, "end": 0.39, "value": "C"},
    ...
  ]
}
```

Values use Preston-Blair 9-shape phonemes: `A B C D E F G H X` (X is silence). The companion app's `VisemeScheduler` plays these against playback position — see `havencore-companion-app/docs/avatar-overlay.md`.

**Soft-fail behavior**: if `rhubarb` is missing, hits the timeout, or exits non-zero, the response omits the header and the audio body is unchanged. Clients should treat absent `X-Visemes` as "no lip-sync data — use closed-mouth playback".

Tunables (env vars, all optional):
- `RHUBARB_BIN` — binary path, default `rhubarb`
- `RHUBARB_TIMEOUT_SEC` — kill subprocess after N seconds, default `10`
- `RHUBARB_RECOGNIZER` — `phonetic` (fast) or `pocketSphinx`, default `phonetic`

The buffered-path header has a structural ceiling: past ~30 s of speech the
base64-encoded viseme JSON exceeds aiohttp's default 8190-byte
`max_field_size`. The agent bumps that limit to 65536 on its buffered TTS
sessions (`api/tts.py:speak` and `services/tts_client.py:synth`) to push the
ceiling out, but the permanent fix is the
[streaming endpoint](#streaming-post-v1audiospeechstream) — viseme cues ride
the response body there, not a header, so there is no upper bound. The
companion app and the dashboard chat / playground-stream paths consume cues
out of the streaming NDJSON body (one `visemes` event per chunk, shifted by
`offset_ms`); the header path remains for buffered-path callers (autonomy
speaker, external OpenAI-SDK clients, the dashboard's Buffered toggle).

Quick smoke test:

```bash
curl -sS -D /tmp/h.txt -X POST http://localhost:6005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, this is a test.","voice":"Olivia","response_format":"wav"}' \
  -o /tmp/say.wav
grep -i '^X-Visemes:' /tmp/h.txt | sed 's/^X-Visemes: //;s/\r$//' \
  | base64 -d | python3 -m json.tool
```

## Dashboard playground

`http://localhost/playgrounds/tts` proxies through the agent service for in-browser testing (text input, voice/format selection, inline playback, synthesis latency). The page also exposes a **Voices** card with:

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

The toggle defaults to Stream.

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
curl http://localhost:6005/health

# Verify voice catalog
curl http://localhost:6005/v1/voices | python3 -m json.tool

# Quick synthesis with viseme header
curl -sS -D /tmp/h.txt -X POST http://localhost:6005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, this is a test.","voice":"Olivia","response_format":"wav"}' \
  -o /tmp/say.wav
grep -i '^X-Visemes:' /tmp/h.txt | sed 's/^X-Visemes: //;s/\r$//' \
  | base64 -d | python3 -m json.tool

# GPU usage
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv

# Service logs
docker compose logs -f text-to-speech
```

**Cold start**: model load takes 30–60 s on a 3090. Healthcheck `start_period` is set to 120 s.

**OOM on the configured GPU**: bump `CHATTERBOX_GPU` to a less-busy card in `.env` and `docker compose up -d text-to-speech`. The model itself needs ~3 GB; activations can push peak higher.

**Voice sounds wrong / inconsistent**: a too-short or noisy reference clip is the usual cause. Re-record at 10–30 s of clean single-speaker audio.

## Related

- [Configuration](../../configuration.md) — full env-var reference
- [API Reference](../../api-reference.md) — full endpoint contracts including the voice-management routes
