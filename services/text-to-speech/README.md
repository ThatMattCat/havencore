# Text-To-Speech AI

Use [Kokoro TTS](https://github.com/hexgrad/kokoro) to convert text to audio. This model performs excellently, with speed and accuracy, despite the minimal implementation here.

For an interactive browser UI, use the agent dashboard's TTS playground at `http://localhost/playgrounds/tts` — it proxies to this service.

## OpenAI-like API - Port 6005

Hosts just the bare minimum needed for the project, more can be added pretty easily. Requests fronted by the nginx service.

Notes on OpenAI Parameters:

- `input` (required): The text to convert to speech
- `model`: OpenAI model name (can be anything, doesn't affect output in this implementation)
- `voice`: OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) — all map to `af_heart` in this implementation
- `response_format`: Audio format (mp3, opus, aac, flac, wav, pcm) — though **the code always generates WAV**
- `speed`: Playback speed from 0.25 to 4.0 (parameter is accepted but not actually used)

### POST /v1/audio/speech

Request:

```
curl -X POST http://localhost:6005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of the text to speech system.",
    "model": "tts-1",
    "voice": "alloy",
    "response_format": "wav",
    "speed": 1.0
  }'
```

Response:

- Status: 200 OK
- Content-Type: audio/wav (or audio/mpeg for mp3, etc.)
- Body: Raw audio file binary data

Example Error:

```
{
  "error": {
    "message": "Missing required parameter 'input'",
    "type": "invalid_request_error"
  }
}
```

### GET /health

Request:

`curl http://localhost:6005/health`

Response:

`{"status": "healthy"}`
