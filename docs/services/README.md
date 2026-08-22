# Services

HavenCore is a collection of containerized services orchestrated via Docker Compose. Each service has its own folder in this directory with a dedicated README covering purpose, configuration, endpoints, and troubleshooting.

## Service overview

| Service | Ports | Purpose | Technology |
|---------|-------|---------|------------|
| [Nginx Gateway](nginx/README.md) | 80, 443 | API gateway / reverse proxy / TLS termination | Nginx Alpine |
| [Certbot](nginx/README.md#tls-termination-selenerenmanwtf) | — | Let's Encrypt cert issuance/renewal (Cloudflare DNS-01) for the gateway | certbot/dns-cloudflare |
| [Agent Service](agent/README.md) | 6002 | AI logic, tool calling, SvelteKit dashboard | Python, FastAPI, SvelteKit |
| [MCP Tools](mcp-tools/README.md) | 6010 (host IP only) | MCP Streamable HTTP tool host — serves the agent's 11 tool modules at `/mcp/<name>` behind per-mount bearer tokens | Python, Starlette, mcp SDK (reuses the agent image) |
| [Speech-to-Text](speech-to-text/README.md) | 6001 | Audio transcription | Python, Faster Whisper, CUDA |
| [Text-to-Speech](text-to-speech/README.md) | 6005 | Speech synthesis — selectable engine: Kokoro (default, fast) or Chatterbox-Turbo (cloning + streaming) | Python, Kokoro / Chatterbox-TTS, CUDA |
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
agent → mcp-tools:6010          (MCP Streamable HTTP tool sessions)
agent → text-to-speech:6005     (TTS — Kokoro or Chatterbox, active engine)
agent → speech-to-text:6001     (STT playground proxy)
agent → vllm-vision:8000        (Vision playground proxy + query_multimodal_api chokepoint)
agent → text-to-image:8188      (ComfyUI playground proxy)
agent → face-recognition:6006   (/people dashboard proxy)
mcp-tools → agent:6002          (reminder module → autonomy API; vision tools → /api/vision/ask_url)
mcp-tools → qdrant:6333, embeddings:3000, mosquitto:1883,
            face-recognition:6006, text-to-image:8188, signal-api:8080
face-recognition → postgres:5432
face-recognition → qdrant:6333
face-recognition → mosquitto:1883
nginx → agent:6002
nginx → mcp-tools:6010          (/mcp/ for external MCP clients)
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
    - mcp-tools   # service_healthy — the tool host must be up before the agent connects
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
