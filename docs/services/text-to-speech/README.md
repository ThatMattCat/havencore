# Text-to-Speech Service

High-quality speech synthesis using Kokoro TTS, exposed as an OpenAI-compatible endpoint.

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
| 6005 | OpenAI-compatible API | `/v1/audio/speech`, `/health` |

For an interactive UI, use the agent dashboard's TTS playground at `/playgrounds/tts` — it proxies through the agent service. The legacy Gradio UI (6004) and static audio-file server (6003) were removed.

## Voice options

OpenAI voice aliases (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) are accepted and all currently map to the `af_heart` Kokoro voice. The agent dashboard's TTS playground pulls the same alias list via `/api/tts/voices`.

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

## Audio formats

**Request formats** (accepted but all output as WAV): mp3, opus, aac, flac, wav, pcm

**Actual output**: always WAV regardless of request.

## Dashboard playground

Use the agent dashboard at `http://localhost/playgrounds/tts` for in-browser testing (text input, voice/format selection, inline playback, synthesis latency). The dashboard proxies to the TTS service.

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
