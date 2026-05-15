# Services

HavenCore is a collection of containerized services orchestrated via Docker Compose. Each service has its own folder in this directory with a dedicated README covering purpose, configuration, endpoints, and troubleshooting.

## Service overview

| Service | Ports | Purpose | Technology |
|---------|-------|---------|------------|
| [Nginx Gateway](nginx/README.md) | 80 | API gateway / reverse proxy | Nginx Alpine |
| [Agent Service](agent/README.md) | 6002 | AI logic, tool calling, SvelteKit dashboard | Python, FastAPI, SvelteKit |
| [Speech-to-Text](speech-to-text/README.md) | 6001 | Audio transcription | Python, Faster Whisper, CUDA |
| [Text-to-Speech (v1)](text-to-speech/README.md) | 6005 | Speech synthesis — Kokoro engine, fallback path | Python, Kokoro TTS, CUDA |
| [Text-to-Speech v2](text-to-speech-v2/README.md) | 6015 | Speech synthesis — Chatterbox-Turbo, expressive + voice cloning | Python, Chatterbox-TTS, CUDA |
| [vLLM Vision](vllm-vision/README.md) | 8001 | Image / short-video understanding (Qwen3-VL on a dedicated GPU) | vLLM, CUDA |
| [Text-to-Image](text-to-image/README.md) | 8188 | Image generation | ComfyUI |
| [vLLM](vllm/README.md) | 8000 | Primary LLM inference | vLLM, CUDA |
| [LlamaCPP](llamacpp/README.md) | 8000 | Alternative LLM backend (inactive — compose stanza commented out) | llama.cpp |
| [PostgreSQL](postgres/README.md) | 5432 | Conversation + metrics storage | PostgreSQL 15 Alpine |
| [Qdrant](qdrant/README.md) | 6333, 6334 | Vector DB for semantic memory | Qdrant |
| [Embeddings](embeddings/README.md) | 3000 | Text embeddings | HuggingFace TEI |
| [Mosquitto](mosquitto/README.md) | 1883, 9001 | MQTT broker | Eclipse Mosquitto |
| [Face Recognition](face-recognition/README.md) | 6006 | Identity for HA cameras | Python, InsightFace `buffalo_l`, ONNXRuntime-GPU |

## Service communication

### Internal network

Services reach each other over the Docker Compose network:

```
agent → postgres:5432
agent → vllm:8000
agent → qdrant:6333
agent → embeddings:3000
agent → text-to-speech:6005     (TTS v1 — Kokoro, when TTS_PROVIDER=v1)
agent → text-to-speech-v2:6015  (TTS v2 — Chatterbox-Turbo, when TTS_PROVIDER=v2)
agent → speech-to-text:6001     (STT playground proxy)
agent → vllm-vision:8000        (Vision playground proxy + query_multimodal_api chokepoint)
agent → text-to-image:8188      (ComfyUI playground proxy)
agent → face-recognition:6006   (/people dashboard proxy + mcp_face_tools)
face-recognition → postgres:5432
face-recognition → qdrant:6333
face-recognition → mosquitto:1883
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
