# Text-to-Speech Service (v1, Kokoro)

High-quality speech synthesis using Kokoro TTS, exposed as an OpenAI-compatible endpoint.

> **Two TTS engines run in parallel.** This is the v1 service (Kokoro, port 6005). A second engine — [text-to-speech-v2](../text-to-speech-v2/README.md) (Chatterbox-Turbo, port 6015) — runs alongside it and exposes the same `/v1/audio/speech` surface and `X-Visemes` header. The agent picks one via `TTS_PROVIDER` (`v1` or `v2`) in `.env`; companion app and satellites follow that switch automatically because they reach TTS through the agent. v2 is the current default for most users; v1 stays as the always-on fallback.

## Purpose

- Neural speech synthesis via Kokoro TTS
- Multiple voice/language options
- OpenAI-compatible API
- Web interface via the agent dashboard playground

## Architecture

```
Text Input → Kokoro TTS → Audio Generation → HTTP response body
     ↓            ↓             ↓
Preprocessing  Neural TTS   WAV output
```

## Service endpoints

| Port | Purpose | Features |
|------|---------|----------|
| 6005 | OpenAI-compatible API | `/v1/audio/speech`, `/v1/voices`, `/health` |

For an interactive UI, use the agent dashboard's TTS playground at `/playgrounds/tts` — it proxies through the agent service. The legacy Gradio UI (6004) and static audio-file server (6003) were removed.

## Voice options

Two kinds of voice identifiers are accepted:

- **Native Kokoro voice ids** (e.g. `af_heart`, `af_bella`, `am_michael`) — passed straight through to the pipeline. The catalog is filtered to the voices whose prefix matches the configured `TTS_LANGUAGE` (Kokoro's G2P is language-specific), so only compatible voices are exposed.
- **OpenAI-compat aliases** (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) — resolve to the configured default voice (`TTS_VOICE`, fallback `af_heart`), so OpenAI-wired devices keep working.

Unknown names fall back to the default voice with a warning in the service log. `GET /v1/voices` returns the full catalog (`{language, default, native[], aliases[]}`); the agent's `/api/tts/voices` proxies this and is what the dashboard playground renders.

## API usage

```bash
# Generate speech
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of text to speech.",
    "model": "tts-1",
    "voice": "alloy",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Configuration

```bash
TTS_DEVICE="cuda:0"      # GPU device
TTS_LANGUAGE="a"         # Language option
TTS_VOICE="af_heart"     # Voice model
```

### Pronunciation overrides

Kokoro's G2P (misaki) sometimes reads proper nouns differently than intended
— e.g., it pronounces "Selene" as 3-syllable "Sell-uh-nee" instead of the
intended 2-syllable "Suh-LEEN" (homophone with "Celine"). Misaki has **no
inline `{phoneme}` syntax**; the supported override path is to inject the
word + phoneme string into the lexicon dict the G2P consults.

The service exposes one env knob and wires the rest automatically:

```bash
# Misaki phonemes Kokoro will use for AGENT_NAME. Must be characters from
# misaki's US (or GB) IPA-like inventory. Stress markers ˈ (primary) and
# ˌ (secondary) are supported. Default = "səˈlin".
TTS_AGENT_NAME_PHONEMES="səˈlin"
```

On startup, `app/main.py`'s `_apply_pronunciation_overrides()` injects the
entry into `pipeline.g2p.lexicon.golds` for `AGENT_NAME` plus its
lowercase/uppercase/capitalised variants. Adding more words requires editing
`TTS_PRONUNCIATIONS` in `services/text-to-speech/app/config.py`:

```python
TTS_PRONUNCIATIONS = {
    _agent_name: os.getenv("TTS_AGENT_NAME_PHONEMES", "səˈlin"),
    "Tomato": "təˈmɑdoʊ",
}
```

Misaki validates phoneme strings against `US_VOCAB` (or `GB_VOCAB`) — any
character outside that set will fail the assert at G2P time. To inspect or
test a candidate string interactively:

```bash
docker compose exec -T text-to-speech python3 -c "
import main
result = next(iter(main.pipeline('Hi, I am Selene.', voice='af_heart')))
print('phonemes:', result.phonemes)
"
```

## Audio formats

**Request formats** (accepted but all output as WAV): mp3, opus, aac, flac, wav, pcm

**Actual output**: always WAV regardless of request.

## Lip-sync viseme timeline (`X-Visemes` header)

For consumers that want to lip-sync a Live2D / 3D avatar against the
TTS audio, every `/v1/audio/speech` response carries an `X-Visemes`
header containing a base64-encoded Rhubarb Lip Sync JSON timeline.
The agent's `/api/tts/speak` proxy forwards this header unchanged.

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

Values use Preston-Blair 9-shape phonemes: `A B C D E F G H X` (X is
silence). The companion app's `VisemeScheduler` plays these against
ExoPlayer playback position — see `havencore-companion-app/docs/avatar-overlay.md`.

**Soft-fail behavior**: if `rhubarb` is missing, hits the 10s timeout,
or exits non-zero, the response omits the header and the body is
unchanged. Clients should treat absent `X-Visemes` as "no lip-sync
data — use closed-mouth playback".

Tunables (env vars, all optional):
- `RHUBARB_BIN` — binary path, default `rhubarb`
- `RHUBARB_TIMEOUT_SEC` — kill subprocess after N seconds, default `10`
- `RHUBARB_RECOGNIZER` — `phonetic` (fast) or `pocketSphinx`, default `phonetic`

Quick smoke test:

```bash
curl -sS -D /tmp/h.txt -X POST http://<host>:6005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, this is a test.","voice":"af_heart","response_format":"wav"}' \
  -o /tmp/say.wav
grep -i '^X-Visemes:' /tmp/h.txt | sed 's/^X-Visemes: //;s/\r$//' \
  | base64 -d | python3 -m json.tool
```

## Dashboard playground

Use the agent dashboard at `http://localhost/playgrounds/tts` for in-browser testing (text input, voice/format selection, inline playback, synthesis latency). The dashboard proxies to the TTS service.

The dashboard Chat page (`/chat`) also consumes this service: when the header speaker toggle is on, each completed assistant turn is synthesized via `/api/tts/speak` (default voice `af_heart`, mp3) and auto-played inline.

## File management

Generated audio is written to `/app/output/` inside the container, read back into the HTTP response body, and is not served via a separate static endpoint.

## Troubleshooting

```bash
# Check TTS service
curl http://localhost:6005/health

# Test audio generation
curl -X POST http://localhost:6005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "voice": "alloy"}'

# Check GPU usage
nvidia-smi

# View service logs
docker compose logs -f text-to-speech
```
