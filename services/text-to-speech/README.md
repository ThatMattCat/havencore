# Text-To-Speech AI

Use [Kokoro TTS](https://github.com/hexgrad/kokoro) to convert text to audio. This model performs excellently, with speed and accuracy, despite the minimal implementation here.


## OpenAI-like API - Port 6005

Hosts just the bare minimum needed for the project, more can be added pretty easily. Requests fronted by nginx service.

Notes on OpenAI Parameters:

- input (required): The text to convert to speech
- model: OpenAI model name (can be anything, doesn't affect output in this implementation)
- voice: OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) - all map to "af_heart" in this implementation
- response_format: Audio format (mp3, opus, aac, flac, wav, pcm) - though **the code always generates WAV**
- speed: Playback speed from 0.25 to 4.0 (parameter is accepted but not actually used)

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



## Gradio UI/API - Port 6004

### Gradio UI

`http://localhost:6004`


### Gradio API

`POST http://localhost:6004/api/predict`

Request:

```
curl -X POST http://localhost:6004/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      "Hello, this is a test of the text to speech system.",
      "af_heart",
      false
    ],
    "fn_index": 0
  }'
```

Response:

```
{
  "data": [
    "http://localhost:6003/1734567890-1234.wav",
    {
      "name": "/path/to/audio_files/1734567890-1234.wav",
      "data": "data:audio/wav;base64,UklGRi...",
      "is_file": true,
      "orig_name": "1734567890-1234.wav"
    }
  ],
  "is_generating": false,
  "duration": 2.345,
  "average_duration": 2.5
}
```



## Static Audio File Server - Port 6003

Request:

`curl http://localhost:6003/1734567890-1234.wav`

Response:

- Status: 200 OK
- Content-Type: Determined by file extension
- Body: Raw audio file