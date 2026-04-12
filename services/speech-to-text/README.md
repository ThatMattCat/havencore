# Speech-to-Text

Faster-Whisper-based transcription service. Supports two interfaces:

- **HTTP (`:6001`)** — OpenAI-compatible `POST /v1/audio/transcriptions`. Accepts any audio format Whisper can load (WAV, MP3, M4A, FLAC, OGG, WebM/Opus, …). Used by edge devices in the voice pipeline and by the agent dashboard's STT playground (record-then-upload mode).
- **WebSocket (`:6000`)** — streaming interface for clients that want to push audio live.

## WebSocket protocol (`ws://HOST:6000/`)

Bidirectional. The client sends binary frames of raw **int16 PCM at 16 kHz, mono** interleaved with JSON control messages.

### Control → server

```json
{"type": "CONTROL", "message": "start", "trace_id": "...", "source_ip": "..."}
{"type": "CONTROL", "message": "stop",  "trace_id": "...", "source_ip": "..."}
```

Between `start` and `stop`, send raw PCM audio frames as binary WS messages.

### Server → client

When the stream completes (after `stop`), the service emits:

```json
{"text": "<final transcript>", "final": true, "trace_id": "..."}
```

This replaced an older protocol that forwarded the transcript to the agent via an HTTP Gradio client and emitted only a playback URL — the service no longer has a Gradio dependency.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_DEVICE` | `0` | CUDA device index |
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large` |
| `SRC_LAN` | `en` | Source language hint |

## Health

`GET http://HOST:6001/health` → `{"status": "healthy"}`.
