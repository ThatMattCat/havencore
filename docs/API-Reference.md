# API Reference

HavenCore exposes two surfaces:

1. **OpenAI-compatible APIs** — `/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/models`. These ride through the Nginx gateway at `http://localhost` and are used by the voice pipeline (edge devices, external integrations).
2. **Agent dashboard APIs** — `/api/*` and `/ws/*`, served by the agent on port 6002 (also available through Nginx). Used by the SvelteKit dashboard at `http://localhost:6002/` for chat, metrics, service playgrounds, Home Assistant state, and live logs.

## Authentication

The OpenAI-compatible endpoints forwarded to vLLM use the `DEV_CUSTOM_API_KEY` configured in `.env`:

```bash
DEV_CUSTOM_API_KEY="your_secret_key"
curl -H "Authorization: Bearer your_secret_key" http://localhost/v1/chat/completions ...
```

The `/api/*` dashboard endpoints are unauthenticated — the dashboard is intended for a private/home network. Do not expose port 6002 to the public internet without adding your own auth in front of it.

## Chat Completions API

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint for conversational AI.

**Endpoint**: `POST http://localhost/v1/chat/completions`

#### Request Headers
```
Content-Type: application/json
Authorization: Bearer your_api_key
```

#### Request Body
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user", 
      "content": "Hello! How are you today?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model identifier (can be any value) |
| `messages` | array | Yes | - | Array of message objects |
| `temperature` | number | No | 0.7 | Randomness (0.0-2.0) |
| `max_tokens` | number | No | 1024 | Maximum response length |
| `stream` | boolean | No | false | Enable streaming responses |

#### Message Object
```json
{
  "role": "user|assistant|system",
  "content": "Message content"
}
```

#### Response
```json
{
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 31,
    "total_tokens": 87
  }
}
```

#### Tool Calling

HavenCore supports tool calling for external integrations:

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in San Francisco?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather_forecast",
        "description": "Get weather forecast for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name or coordinates"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

#### Available Tools

Tools are grouped into MCP servers. Each server has its own reference
doc with the full tool list, arguments, config, and troubleshooting.

- **Home Assistant** — 21 tools (domain state / services, light & climate
  helpers, scenes, scripts, automations, notifications, areas, presence,
  timers, Jinja templates, history, calendar, media transport). See
  [MCP-HomeAssistant](MCP-HomeAssistant.md).
- **Plex** — library search + cloud-relay playback
  (`plex_search`, `plex_list_recent`, `plex_list_on_deck`,
  `plex_list_clients`, `plex_play`). See
  [MCP-Plex](MCP-Plex.md) and [Media Control](Media-Control.md).
- **General Tools** — `get_weather_forecast`, `brave_search`,
  `search_wikipedia`, `wolfram_alpha`, `generate_image`, `send_email`,
  `query_multimodal_api`. See [MCP-General](MCP-General.md).
- **Qdrant** — semantic memory (`create_memory`, `search_memories`). See
  [MCP-Qdrant](MCP-Qdrant.md).
- **MQTT / Cameras** — `get_camera_snapshots`. See [MCP-MQTT](MCP-MQTT.md).

#### Example Request
```bash
curl -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Turn on the living room lights"}
    ]
  }'
```

## Audio APIs

### Speech Synthesis (Text-to-Speech)

#### POST /v1/audio/speech

Convert text to spoken audio using Kokoro TTS.

**Endpoint**: `POST http://localhost/v1/audio/speech`

#### Request Body
```json
{
  "input": "Hello, this is a test of the text to speech system.",
  "model": "tts-1",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | string | Yes | - | Text to convert to speech |
| `model` | string | No | "tts-1" | TTS model (any value accepted) |
| `voice` | string | No | "alloy" | Voice selection (alloy, echo, fable, onyx, nova, shimmer) |
| `response_format` | string | No | "mp3" | Audio format (mp3, wav, opus, aac, flac, pcm) |
| `speed` | number | No | 1.0 | Playback speed (0.25-4.0) |

**Note**: OpenAI voice aliases (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) all currently map to the Kokoro `af_heart` voice. Output is always WAV format regardless of `response_format`.

#### Response
Returns raw audio binary data with appropriate Content-Type header.

#### Example Request
```bash
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, HavenCore is working perfectly!",
    "model": "tts-1",
    "voice": "alloy"
  }' \
  --output speech.wav
```

### Speech Recognition (Speech-to-Text)

#### POST /v1/audio/transcriptions

Transcribe audio to text using Whisper models.

**Endpoint**: `POST http://localhost/v1/audio/transcriptions`

#### Request Format
Multipart form data with audio file:

```bash
curl -X POST http://localhost/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "language=en"
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | Yes | - | Audio file to transcribe |
| `model` | string | No | "whisper-1" | Model identifier |
| `language` | string | No | auto | Language code (en, es, fr, etc.) |
| `prompt` | string | No | - | Optional prompt to guide transcription |
| `response_format` | string | No | "json" | Response format (json, text, srt, vtt) |
| `temperature` | number | No | 0 | Sampling temperature |

#### Supported Audio Formats
- WAV
- MP3
- MP4
- MPEG
- MPGA
- M4A
- WEBM

#### Response
```json
{
  "text": "Hello, this is a transcription of the audio file."
}
```

## Model Management API

### GET /v1/models

List available models in the system.

**Endpoint**: `GET http://localhost/v1/models`

#### Response
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1677610602,
      "owned_by": "openai"
    },
    {
      "id": "whisper-1",
      "object": "model", 
      "created": 1677610602,
      "owned_by": "openai"
    }
  ]
}
```

## System Management APIs

### Health Check Endpoints

#### GET /health
System-wide health check through the gateway.

```bash
curl http://localhost/health
```

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "agent": "healthy",
    "tts": "healthy", 
    "stt": "healthy",
    "llm": "healthy"
  }
}
```

#### Individual Service Health Checks

| Service | Endpoint | Description |
|---------|----------|-------------|
| Agent | `GET http://localhost:6002/health` | Agent service status |
| TTS | `GET http://localhost:6005/health` | Text-to-speech service |
| STT | `GET http://localhost:6001/health` | Speech-to-text service |
| LLM | `GET http://localhost:8000/health` | LLM backend status |

### Tool Management APIs

#### GET /tools/status
Get status of all registered tools.

**Endpoint**: `GET http://localhost:6002/tools/status`

#### Response
```json
{
  "total_tools": 12,
  "legacy_tools": 8,
  "mcp_tools": 4,
  "conflicts": [],
  "tool_preference": "legacy",
  "tools": [
    {
      "name": "get_weather_forecast",
      "source": "legacy",
      "description": "Get weather forecast for a location"
    }
  ]
}
```

#### POST /tools/preference
Set tool preference when conflicts exist between legacy and MCP tools.

```bash
curl -X POST http://localhost:6002/tools/preference \
  -H "Content-Type: application/json" \
  -d '{"prefer_mcp": true}'
```

### MCP Management APIs

#### GET /mcp/status
Get status of MCP (Model Context Protocol) connections.

**Endpoint**: `GET http://localhost:6002/mcp/status`

#### Response
```json
{
  "mcp_enabled": true,
  "active_servers": 2,
  "servers": [
    {
      "name": "filesystem",
      "status": "connected",
      "tools": 5,
      "last_ping": "2024-01-15T10:30:00Z"
    }
  ]
}
```

## Agent Dashboard APIs

The agent service at `http://localhost:6002` serves both the SvelteKit dashboard SPA and a set of JSON/WebSocket endpoints under `/api/*` and `/ws/*`. Full endpoint detail lives in [`services/agent/README.md`](../services/agent/README.md); this section is a quick reference.

### REST (`/api/*`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/chat` | One-shot message with tool event log |
| `GET`  | `/api/status` | Agent, MCP, DB, vLLM health |
| `GET`  | `/api/tools` | Registered tools grouped by MCP server |
| `GET`  | `/api/conversations` | Paginated conversation history |
| `GET`  | `/api/conversations/{session_id}` | Full messages for a session |
| `GET`  | `/api/ha/entities` | HA entities (optionally `?domain=light`) |
| `GET`  | `/api/ha/entities/summary` | Entity counts per domain |
| `GET`  | `/api/ha/automations` | HA automations |
| `GET`  | `/api/ha/scenes` | HA scenes |
| `GET`  | `/api/metrics/turns` | Recent per-turn timings |
| `GET`  | `/api/metrics/summary` | Daily aggregates, p95 |
| `GET`  | `/api/metrics/top-tools` | Tool invocation counts + avg latency |
| `POST` | `/api/tts/speak` | Synthesize speech (returns audio binary) |
| `GET`  | `/api/tts/voices` | Voice alias list |
| `POST` | `/api/stt/transcribe` | Multipart transcription proxy |
| `POST` | `/api/vision/ask` | Multipart image + prompt to vision LLM |
| `POST` | `/api/comfy/generate` | Queue ComfyUI workflow |
| `GET`  | `/api/comfy/status/{prompt_id}` | Poll generation status |
| `GET`  | `/api/comfy/view` | Stream a generated image |
| `GET`  | `/api/{tts,stt,vision,comfy}/health` | Per-service health proxies |

### WebSockets

| URL | Direction | Purpose |
|-----|-----------|---------|
| `/ws/chat` | bidirectional | Streaming chat with `thinking`, `tool_call`, `tool_result`, `metric`, `done`, `error` events |
| `/ws/logs` | server → client | Live tail of the agent's in-process log ring buffer |

## Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "error": {
    "message": "Missing required parameter 'input'",
    "type": "invalid_request_error",
    "param": "input",
    "code": "missing_parameter"
  }
}
```

#### 401 Unauthorized
```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "authentication_error"
  }
}
```

#### 429 Too Many Requests
```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error"
  }
}
```

#### 500 Internal Server Error
```json
{
  "error": {
    "message": "Internal server error",
    "type": "internal_error"
  }
}
```

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `missing_parameter` | Required parameter not provided | Check request body |
| `invalid_parameter` | Parameter value is invalid | Verify parameter format |
| `authentication_error` | API key invalid or missing | Check Authorization header |
| `rate_limit_error` | Too many requests | Wait and retry |
| `internal_error` | Server-side error | Check service logs |

## Rate Limiting

HavenCore implements rate limiting to prevent abuse:

- **Chat Completions**: 60 requests/minute per API key
- **Audio Processing**: 100 requests/minute per API key
- **System APIs**: 120 requests/minute per IP

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1640995200
```

## SDK Examples

### Python
```python
import requests

# Chat completion
response = requests.post(
    "http://localhost/v1/chat/completions",
    headers={
        "Authorization": "Bearer your_api_key",
        "Content-Type": "application/json"
    },
    json={
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json())
```

### JavaScript
```javascript
// Using fetch API
const response = await fetch('http://localhost/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your_api_key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'gpt-3.5-turbo',
    messages: [
      { role: 'user', content: 'Hello!' }
    ]
  })
});

const data = await response.json();
console.log(data);
```

### cURL
```bash
# Chat completion
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Text-to-speech
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "alloy"}' \
  --output audio.wav

# Speech-to-text
curl -X POST http://localhost/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

## WebSocket APIs

The agent dashboard already uses WebSockets for real-time features:

- **`/ws/chat`** — streaming conversation with inline tool-call visibility, per-turn timing metrics, and thinking indicators. See the Agent Dashboard APIs section above and `services/agent/README.md` for the event schema.
- **`/ws/logs`** — one-way stream of structured log records from the agent's in-process ring buffer (the last 500 records are replayed on connect).

---

**Next Steps**:
- [Tool Development](Tool-Development.md) - Creating custom tools
- [Integration Guides](Home-Assistant-Integration.md) - External service setup
- [Troubleshooting](Troubleshooting.md) - API debugging and issues