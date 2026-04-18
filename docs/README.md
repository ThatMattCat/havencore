# HavenCore Documentation

Welcome to the docs for **HavenCore** — a self-hosted AI smart home assistant with voice control.

## Top-level guides

- [Getting Started](getting-started.md) — install and run HavenCore
- [Architecture](architecture.md) — system design and how the services fit together
- [Configuration](configuration.md) — environment variables and `.env` reference
- [API Reference](api-reference.md) — OpenAI-compatible APIs and agent dashboard APIs
- [Development](development.md) — contributing and local development workflow
- [Troubleshooting](troubleshooting.md) — common issues and fixes
- [FAQ](faq.md)
- [TODO](todo.md) — forward-looking items not yet scheduled

## Services

Each service has its own folder under [`services/`](services/README.md):

- [Agent](services/agent/README.md) — core AI agent, dashboard, and MCP tool servers
  - [Agent tools (MCP servers)](services/agent/tools/README.md)
- [vLLM](services/vllm/README.md) — primary LLM backend
- [LlamaCPP](services/llamacpp/README.md) — alternative LLM backend
- [Speech-to-Text](services/speech-to-text/README.md) — Faster Whisper STT
- [Text-to-Speech](services/text-to-speech/README.md) — Kokoro TTS
- [IAV-to-Text](services/iav-to-text/README.md) — image/audio/video understanding
- [Text-to-Image](services/text-to-image/README.md) — ComfyUI image generation
- [Postgres](services/postgres/README.md) — conversation + metrics storage
- [Qdrant](services/qdrant/README.md) — vector DB for semantic memory
- [Embeddings](services/embeddings/README.md) — text-embeddings-inference
- [Nginx](services/nginx/README.md) — API gateway
- [Mosquitto](services/mosquitto/README.md) — MQTT broker

## Integrations

- [Home Assistant](integrations/home-assistant.md) — end-user setup and voice examples
- [Media Control](integrations/media-control.md) — Plex, Music Assistant, and Home Assistant for TV/speaker playback

## Project overview

HavenCore is a fully containerized, self-hosted AI assistant system:

- Voice activation with wake-word detection
- Natural-language conversation backed by a local LLM
- Home Assistant integration for smart-home control
- High-quality text-to-speech (Kokoro) and speech-to-text (Whisper)
- OpenAI-compatible APIs for external integrations
- Web search, computation, and image generation tools
- Docker Compose deployment with GPU support

## Getting help

- **Something broken?** See [Troubleshooting](troubleshooting.md).
- **Bug reports:** GitHub Issues on the [main repository](https://github.com/ThatMattCat/havencore).
- **Contributing:** see [Development](development.md).
