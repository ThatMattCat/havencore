# Speech-to-Text

Faster-Whisper-based transcription service.

Exposes an OpenAI-compatible HTTP endpoint on port **6001**: `POST /v1/audio/transcriptions`. Accepts any audio format Whisper can load (WAV, MP3, M4A, FLAC, OGG, WebM/Opus, …). Used by edge devices in the voice pipeline and by the agent dashboard's STT playground (record-then-upload mode).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_DEVICE` | `0` | CUDA device index |
| `WHISPER_MODEL` | `base` | Whisper model size: `tiny` / `base` / `small` / `medium` / `large` |
| `SRC_LAN` | `en` | Source language hint |

## Health

`GET http://HOST:6001/health` → `{"status": "healthy"}`.
