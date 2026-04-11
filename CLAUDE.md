# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is HavenCore

A self-hosted AI smart home assistant with voice control. Microservices architecture running via Docker Compose on Linux with NVIDIA GPUs. The assistant (named "Selene") processes voice input through STT, runs an LLM agent with tool-calling, and responds via TTS.

## Common Commands

```bash
# Validate compose config
docker compose config --quiet

# Build all services (takes 60-90 min first time — NEVER cancel)
docker compose build --parallel

# Start services (model loading takes 10-15 min)
docker compose up -d

# Restart a single service after code changes
docker compose restart agent

# View logs
docker compose logs -f [service_name]

# Health checks
curl http://localhost/health
curl http://localhost:6002/          # agent
curl http://localhost:8000/v1/models # vLLM
```

Python services have their code mounted as Docker volumes, so file edits are live — just restart the service. Changes to `.env` require `docker compose down && docker compose up -d`.

## Architecture

### Services (compose.yaml)

| Service | Ports | Role |
|---------|-------|------|
| **agent** | 6002, 6006 | Core AI agent — Gradio UI (6002), OpenAI-compat API (6006) |
| **vllm** | 8000 | LLM inference (Qwen2.5-72B-AWQ, served as `gpt-3.5-turbo`) |
| **speech-to-text** | 6000, 6001, 5999 | Faster Whisper STT |
| **text-to-speech** | 6003, 6004, 6005 | Kokoro TTS |
| **iav-to-text** | 8100, 8110 | Image/audio/video to text (vision LLM) |
| **text-to-image** | 8188 | ComfyUI image generation |
| **nginx** | 80 | API gateway routing to services |
| **postgres** | 5432 | Conversation storage |
| **qdrant** | 6333, 6334 | Vector DB for semantic memory |
| **embeddings** | 3000 | HuggingFace text-embeddings-inference (bge-large-en-v1.5) |
| **mosquitto** | 1883, 9001 | MQTT broker |

LlamaCPP is available as an alternative LLM backend (commented out in compose.yaml).

### Agent Service (`services/agent/`)

The agent is a Python package (`selene-agent`) installed via `pyproject.toml`. Entry point: `selene_agent.selene_agent:main`.

Key components:
- `selene_agent/selene_agent.py` — Main agent loop, FastAPI/Gradio server, OpenAI-compat chat endpoint, tool-call orchestration via `AsyncToolExecutor`
- `selene_agent/utils/config.py` — Agent-specific config
- `selene_agent/utils/mcp_client_manager.py` — MCP client that discovers and executes tools from MCP servers, with `UnifiedTool` abstraction supporting both legacy and MCP tool sources
- `selene_agent/utils/conversation_db.py` — PostgreSQL conversation persistence

### MCP Tool Modules (`selene_agent/modules/`)

Each module is a self-contained MCP server with `__main__.py` entry point and `mcp_server.py`:
- `mcp_general_tools/` — Web search (Brave), Wolfram Alpha, weather, Wikipedia, ComfyUI image gen, email
- `mcp_homeassistant_tools/` — Home Assistant device control, media playback (`ha_media_controller.py`)
- `mcp_qdrant_tools/` — Semantic memory store/query/delete via Qdrant
- `mcp_mqtt_tools/` — MQTT publish/subscribe

### Shared Code (`shared/`)

- `shared/configs/shared_config.py` — Central config loaded from env vars (DB creds, API keys, LLM endpoint, device assignments, system prompt). All services reference this.
- `shared/libs/logger.py` — Logging with optional Grafana Loki push
- `shared/libs/trace_id.py` — Request tracing

### Configuration

All config flows through `.env` → `shared/configs/shared_config.py` → services. Key vars:
- `HOST_IP_ADDRESS` — Docker host IP (used to construct `LLM_API_BASE`)
- `HAOS_URL` / `HAOS_TOKEN` — Home Assistant connection
- `TTS_DEVICE` / `STT_DEVICE` — GPU assignments
- `AGENT_NAME` — Assistant persona name (default: "Selene")
- API keys: `BRAVE_SEARCH_API_KEY`, `WOLFRAM_ALPHA_API_KEY`, `WEATHER_API_KEY`

## Key Patterns

- The agent talks to vLLM via the OpenAI Python SDK, treating vLLM as an OpenAI-compatible endpoint at `HOST_IP_ADDRESS:8000`
- Tools are surfaced to the LLM as OpenAI function-calling format, converted from MCP tool definitions via `UnifiedTool.to_openai_format()`
- The agent runs Gradio for the chat UI and a separate FastAPI app for the OpenAI-compat API, both served by uvicorn
- Voice responses must avoid special characters/emojis (TTS limitation, enforced in system prompt)
