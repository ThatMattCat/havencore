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
- [vLLM Vision](services/vllm-vision/README.md) — Qwen3-VL image-understanding backend on a dedicated GPU
- [Text-to-Image](services/text-to-image/README.md) — ComfyUI image generation
- [Face Recognition](services/face-recognition/README.md) — InsightFace identity for HA cameras
- [Postgres](services/postgres/README.md) — conversation + metrics storage
- [Qdrant](services/qdrant/README.md) — vector DB for semantic memory
- [Embeddings](services/embeddings/README.md) — text-embeddings-inference
- [Nginx](services/nginx/README.md) — API gateway
- [Mosquitto](services/mosquitto/README.md) — MQTT broker
- ntfy — UnifiedPush server for companion-app push notifications. No dedicated service README; the [companion-app integration](integrations/companion-app.md) doc covers setup, ports, and the redirect convenience at `/ntfy` → `:8585`

## Integrations

- [Home Assistant](integrations/home-assistant.md) — end-user setup and voice examples
- [Media Control](integrations/media-control.md) — Plex, Music Assistant, and Home Assistant for TV/speaker playback
- [Companion App (push notifications)](integrations/companion-app.md) — UnifiedPush + ntfy: how the agent wakes the phone with autonomy briefings, anomaly alerts, and reminders

## Related repositories

HavenCore's clients live in their own repos so they can ship and version
independently from the agent stack:

- [`havencore-satellite-firmware`](https://github.com/ThatMattCat/havencore-satellite-firmware) — ESP-IDF firmware for ESP32-S3-BOX-3 voice satellites (wake-word, mic capture, OTA via the agent's `/firmware/` route).
- [`havencore-companion-app`](https://github.com/ThatMattCat/havencore-companion-app) — native Kotlin Android app: in-app chat, voice/assistant-slot, and push notifications. Under active development; consult its own README for current scope.

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
