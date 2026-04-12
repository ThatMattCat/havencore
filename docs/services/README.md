# Services

HavenCore is a collection of containerized services orchestrated via Docker Compose. Each service has its own folder in this directory with a dedicated README covering purpose, configuration, endpoints, and troubleshooting.

## Service overview

| Service | Ports | Purpose | Technology |
|---------|-------|---------|------------|
| [Nginx Gateway](nginx/README.md) | 80 | API gateway / reverse proxy | Nginx Alpine |
| [Agent Service](agent/README.md) | 6002 | AI logic, tool calling, SvelteKit dashboard | Python, FastAPI, SvelteKit |
| [Speech-to-Text](speech-to-text/README.md) | 6001 | Audio transcription | Python, Faster Whisper, CUDA |
| [Text-to-Speech](text-to-speech/README.md) | 6005 | Speech synthesis | Python, Kokoro TTS, CUDA |
| [IAV-to-Text](iav-to-text/README.md) | 8100, 8110 | Image/audio/video understanding | Python, vision LLM |
| [Text-to-Image](text-to-image/README.md) | 8188 | Image generation | ComfyUI |
| [vLLM](vllm/README.md) | 8000 | Primary LLM inference | vLLM, CUDA |
| [LlamaCPP](llamacpp/README.md) | 8000 | Alternative LLM backend | llama.cpp |
| [PostgreSQL](postgres/README.md) | 5432 | Conversation + metrics storage | PostgreSQL 15 Alpine |
| [Qdrant](qdrant/README.md) | 6333, 6334 | Vector DB for semantic memory | Qdrant |
| [Embeddings](embeddings/README.md) | 3000 | Text embeddings | HuggingFace TEI |
| [Mosquitto](mosquitto/README.md) | 1883, 9001 | MQTT broker | Eclipse Mosquitto |

## Service communication

### Internal network

Services reach each other over the Docker Compose network:

```
agent → postgres:5432
agent → vllm:8000
agent → qdrant:6333
agent → embeddings:3000
agent → text-to-speech:6005     (TTS playground proxy)
agent → speech-to-text:6001     (STT playground proxy)
agent → iav-to-text:8100        (Vision playground proxy)
agent → text-to-image:8188      (ComfyUI playground proxy)
nginx → agent:6002
nginx → text-to-speech:6005
nginx → speech-to-text:6001
```

### Health check chain

```
nginx → agent/health
nginx → tts/health
nginx → stt/health
agent → postgres (connection test)
agent → vllm/v1/models
```

### Startup dependencies

```yaml
depends_on:
  agent:
    - postgres
    - vllm
  nginx:
    - agent
    - text-to-speech
    - speech-to-text
```

## See also

- [Architecture](../architecture.md) — how the services compose into the full system
- [Configuration](../configuration.md) — environment variables and `.env`
- [Troubleshooting](../troubleshooting.md) — service-level debugging
- [Development](../development.md) — editing and testing services locally
