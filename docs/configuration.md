# Configuration Guide

This guide covers all configuration options for HavenCore, including environment variables, service settings, and integration parameters.

## Overview

HavenCore configuration is managed through:
1. **Environment Variables** (`.env` file) - Primary configuration
2. **Docker Compose** (`compose.yaml`) - Service orchestration
3. **Service Configs** - Individual service settings
4. **Runtime Configuration** - Dynamic settings via API

## Environment Variables Reference

### Core System Configuration

#### Host and Network Settings
```bash
# Required: Docker host IP address
HOST_IP_ADDRESS="192.168.1.100"  # Find with: ip route get 1.1.1.1 | awk '{print $7}'

# Required: API access key
LLM_API_KEY="your_secret_key"  # Set to any value for API access

# Debug and logging
DEBUG_LOGGING=0  # 0 = INFO, 1 = DEBUG
LOKI_URL="http://localhost:3100/loki/api/v1/push"  # Loki logging endpoint
```

#### Agent Configuration
```bash
# Agent identity
AGENT_NAME="Selene"  # AI assistant name

# Location and timezone
CURRENT_LOCATION="San Francisco, CA, USA"
CURRENT_TIMEZONE="America/Los_Angeles" 
CURRENT_ZIPCODE="94102"

# Language settings
SRC_LAN="en"  # Source language code

# Agent LLM provider (seed value — dashboard toggle wins at runtime)
LLM_PROVIDER="vllm"                # vllm | anthropic
ANTHROPIC_API_KEY=""               # required when LLM_PROVIDER=anthropic
ANTHROPIC_MODEL="claude-opus-4-7"  # default Anthropic model
```

### AI Model Configuration

#### GPU Settings
```bash
# Text-to-Speech GPU allocation
TTS_DEVICE="cuda:0"  # GPU index for TTS model

# Speech-to-Text GPU allocation  
STT_DEVICE="0"       # GPU index for STT model

# TTS Voice Settings
TTS_LANGUAGE="a"     # Kokoro TTS language option
TTS_VOICE="af_heart" # Voice model selection
```

#### Model Tokens
```bash
# Hugging Face access token
HF_HUB_TOKEN="hf_your_token_here"  # For model downloads
HUGGING_FACE_HUB_TOKEN="${HF_HUB_TOKEN}"  # Alternative name
```

### Database Configuration

#### PostgreSQL Settings
```bash
POSTGRES_HOST="postgres"           # Service name in Docker network
POSTGRES_PORT=5432                 # Database port
POSTGRES_DB="havencore"           # Database name
POSTGRES_USER="havencore"         # Database user
POSTGRES_PASSWORD="havencore_password"  # Database password
```

#### Database Usage
- **Conversation History**: Automatic storage after timeouts
- **User Sessions**: Session state persistence
- **Configuration**: Runtime settings storage
- **Analytics**: Usage metrics and logs

### External Service Integration

#### Home Assistant
```bash
# Home Assistant API configuration
HAOS_URL="https://homeassistant.local:8123"  # Your HA URL (with or without trailing /api)
HAOS_TOKEN="eyJ0eXAiOiJKV1QiLCJ..."          # Long-lived access token
```

**Getting Home Assistant Token**:
1. Go to Home Assistant → Profile → Long-Lived Access Tokens
2. Click "Create Token"
3. Copy the token and set as `HAOS_TOKEN`

#### Plex (video / TV playback)
```bash
# Plex server (reached via plexapi cloud relay — LAN URL is fine)
PLEX_URL="http://10.0.50.110:32400"
PLEX_TOKEN="xxxxxxxxxxxxxxxxxxxx"

# Optional: wake/launch mapping per Plex client — JSON object keyed by Plex client name.
# See docs/integrations/media-control.md for the full schema.
PLEX_CLIENT_HA_MAP='{
  "BRAVIA 4K VH21": {
    "state_entity": "media_player.living_room_tv_bravia_4k_vh21",
    "adb_entity":   "media_player.living_room_bravia_adb"
  }
}'
```

#### Music Assistant (audio / speaker playback)
```bash
# Music Assistant WebSocket endpoint — LAN URL of the MA server / HA add-on
MASS_URL="http://10.0.50.101:8095"

# Long-lived access token from MA web UI > Settings > Users > Create long-lived token.
# MA schema v28+ requires authenticated WS connections.
MASS_TOKEN="eyJhbGciOi..."
```

If `MASS_URL` / `MASS_TOKEN` are unset, the Music Assistant MCP server
starts in degraded mode and every `mass_*` tool returns a structured
"not configured" error — the rest of the agent stays healthy.

#### External APIs
```bash
# Weather service (weatherapi.com)
WEATHER_API_KEY="your_weather_api_key"

# Web search (Brave Search API)
BRAVE_SEARCH_API_KEY="your_brave_api_key"

# Computational queries (WolframAlpha)
WOLFRAM_ALPHA_API_KEY="your_wolfram_api_key"
```

Each key is credential-gated: tools that need a missing key simply don't
register, so the agent stays healthy without them.

**API Key Setup**:
- **Weather**: Register at [weatherapi.com](https://www.weatherapi.com/)
- **Brave Search**: Get key from [Brave Search API](https://api.search.brave.com/)
- **WolframAlpha**: Register at [Wolfram Developer Portal](https://developer.wolframalpha.com/)

### Semantic memory (Qdrant + embeddings)

```bash
# Qdrant vector DB (service on the compose network)
QDRANT_HOST="qdrant"
QDRANT_PORT=6333

# Text embeddings service (HuggingFace TEI)
EMBEDDINGS_URL="http://embeddings:3000"
EMBEDDING_DIM=1024   # Match the model: bge-large-en-v1.5 = 1024, MiniLM-L6 = 384, mpnet = 768
```

### Agent runtime tuning

```bash
# Seconds of inactivity before the session pool summarizes the conversation
# and resets it in place (same session_id, compact recap preserved as a
# system message, last 2 user/assistant exchanges kept verbatim).
CONVERSATION_TIMEOUT=90

# Bounds for the per-session override (see below). Values outside this range
# are clamped rather than rejected.
CONVERSATION_TIMEOUT_MIN=10
CONVERSATION_TIMEOUT_MAX=3600

# Tuning for the summarize-on-timeout LLM call.
SESSION_SUMMARY_MAX_TOKENS=400       # cap on the recap length
SESSION_SUMMARY_TAIL_EXCHANGES=2     # raw user/assistant pairs kept after reset
SESSION_SUMMARY_LLM_TIMEOUT_SEC=15   # fall back to tail-only if the call hangs

# Cap on tool-result text before it's summarized back to the LLM
TOOL_RESULT_MAX_CHARS=8000
```

Per-session override: clients can pass `X-Idle-Timeout: <seconds>` on
`POST /api/chat`, or include an `idle_timeout` field on any
`{"type":"session", ...}` WebSocket frame (first frame or mid-stream), to
widen or tighten the idle window for that session. The value sticks for
the session's life (or until another turn sends a new value) and is
persisted alongside the conversation for cold resume. Passing `-1` is a
sentinel meaning "never auto-summarize" — the idle sweep and the
turn-start check both skip the session. The dashboard sends `-1` on every
WS open so interactive tabs live until the user hits "New Chat"; pucks
and satellites omit the field and inherit the global default. The same surfaces
also accept an optional `X-Device-Name` header (REST) or `device_name`
field (WS) carrying a human-readable satellite/client label (e.g.
`"Kitchen Speaker"`); see
[services/agent/conversation-history.md](services/agent/conversation-history.md#device-attribution)
for validation rules and persistence behavior.

### Autonomy engine

Proactive background behaviors — morning briefings, ambient anomaly sweeps.
Full reference: [services/agent/autonomy/README.md](services/agent/autonomy/README.md).

```bash
# Master switch (set false to disable dispatch at startup)
AUTONOMY_ENABLED=true

# Dispatcher tick interval (seconds) — how often the engine checks for due items
AUTONOMY_DISPATCH_INTERVAL_SECONDS=30

# Cron schedules (interpreted in CURRENT_TIMEZONE, stored as UTC)
AUTONOMY_BRIEFING_CRON="0 8 * * *"
AUTONOMY_ANOMALY_CRON="*/15 * * * *"

# Anomaly-sweep cooldown: minutes before re-notifying on the same signature
AUTONOMY_ANOMALY_COOLDOWN_MIN=30

# Global ceiling on scheduled dispatches per rolling hour
AUTONOMY_MAX_RUNS_PER_HOUR=20

# Per-turn hard timeout (seconds)
AUTONOMY_TURN_TIMEOUT_SEC=60

# Notification targets
AUTONOMY_BRIEFING_NOTIFY_TO=""          # Signal recipient for the morning briefing
AUTONOMY_HA_NOTIFY_TARGET=""            # e.g. notify.mobile_app_pixel_8

# Handler inputs
AUTONOMY_BRIEFING_CAMERA_ENTITIES=""    # comma-separated camera entity_ids
AUTONOMY_ANOMALY_WATCH_DOMAINS="binary_sensor,lock,cover"
```

Notes:
- `send_signal_message` sends via the `signal-api` container (signal-cli-rest-api).
  Recipient precedence: per-notification `to` → `AUTONOMY_BRIEFING_NOTIFY_TO` →
  `SIGNAL_DEFAULT_RECIPIENT` → `SIGNAL_PHONE_NUMBER` (Note to Self). See
  `docs/services/agent/tools/general.md` for the one-time QR-link setup.
- `AUTONOMY_HA_NOTIFY_TARGET` accepts `notify.mobile_app_<device>` or
  `mobile_app_<device>`; the leading `notify.` is stripped.

### Memory retrieval & agent phase

Per-turn retrieval injection and phase-aware system prompts. Full
reference: [services/agent/autonomy/memory/README.md](services/agent/autonomy/memory/README.md).

```bash
# Master switch for per-turn retrieval injection into pool-backed chats
MEMORY_RETRIEVAL_ENABLED=true

# Top-K memories injected per user turn; switches on the current agent phase
MEMORY_RETRIEVAL_TOPK_LEARNING=5
MEMORY_RETRIEVAL_TOPK_OPERATING=3

# Hits below this cosine score are dropped (0.0–1.0)
MEMORY_RETRIEVAL_MIN_SCORE=0.3

# Seed phase on fresh installs (ignored once the agent_state row exists)
AGENT_PHASE_DEFAULT="learning"
```

Notes:
- Retrieval injection runs on `/api/chat` + `/ws/chat`. `/v1/chat/completions`
  and the autonomy engine skip it — they manage their own context.
- Phase is persisted in Postgres (`agent_state` table) and can be flipped
  at runtime via the `/memory` dashboard toggle or `POST /api/agent/phase`.

### MCP (Model Context Protocol) Configuration

The agent's tool surface is delivered by MCP servers bundled in the agent
image (Home Assistant, Plex, Music Assistant, general, Qdrant, MQTT).
They are spawned as subprocesses and advertise tools over stdio — no
separate container.

```bash
# Master switch for the MCP client manager
MCP_ENABLED=true

# Whether MCP-registered tools win over any same-named legacy registration
MCP_PREFER_OVER_LEGACY=true

# JSON array of MCP server definitions to spawn
MCP_SERVERS='[
  {"name": "homeassistant",    "command": "python", "args": ["-m", "selene_agent.modules.mcp_homeassistant_tools"],    "enabled": true},
  {"name": "plex",             "command": "python", "args": ["-m", "selene_agent.modules.mcp_plex_tools"],             "enabled": true},
  {"name": "music_assistant",  "command": "python", "args": ["-m", "selene_agent.modules.mcp_music_assistant_tools"],  "enabled": true},
  {"name": "general",          "command": "python", "args": ["-m", "selene_agent.modules.mcp_general_tools"],          "enabled": true},
  {"name": "qdrant",           "command": "python", "args": ["-m", "selene_agent.modules.mcp_qdrant_tools"],           "enabled": true},
  {"name": "mqtt",             "command": "python", "args": ["-m", "selene_agent.modules.mcp_mqtt_tools"],             "enabled": true}
]'
```

Per-server reference docs live under
[`services/agent/tools/`](services/agent/tools/README.md).

## Service-Specific Configuration

### LLM Backend Configuration

#### vLLM Configuration (Default)
Defined in `compose.yaml` (GLM-4.5-Air-AWQ-FP16Mix, a MoE reasoning
model served under the OpenAI-compat name `gpt-3.5-turbo` for client
convenience):

```yaml
command: >
  --model QuantTrio/GLM-4.5-Air-AWQ-FP16Mix
  --served-model-name gpt-3.5-turbo
  --tensor-parallel-size 4
  --enable-expert-parallel
  --max-model-len 32768
  --max-num-seqs 2
  --gpu-memory-utilization 0.77
  --tool-call-parser glm45
  --reasoning-parser glm45
  --enable-auto-tool-choice
  --trust-remote-code
```

`--reasoning-parser glm45` splits the model's `<think>…</think>`
chain-of-thought into a separate `reasoning` field on the response,
keeping `message.content` clean for voice satellites. The agent surfaces
the CoT as a dashboard-only `REASONING` event on `/ws/chat` — see
[Agent → WebSocket event schema](api-reference.md#websockets).

**Key Parameters**:
- `--model`: HuggingFace model path
- `--served-model-name`: Name clients use in `model`; lets an OpenAI SDK
  that hardcodes `gpt-3.5-turbo` talk to this backend.
- `--gpu-memory-utilization`: GPU memory usage (0.0-1.0)
- `--max-model-len`: Maximum sequence length
- `--dtype`: Data type (auto, float16, bfloat16)

#### LlamaCPP Configuration (Alternative)
Commented out in `compose.yaml`. Uncomment the `llamacpp` service block to
swap it in for vLLM. It uses `ghcr.io/ggml-org/llama.cpp:server-cuda` and
reads a GGUF model mounted at `/models/`.

#### Hosted Anthropic Claude (Alternative)
The agent can route its text-to-text LLM calls to Anthropic Claude instead
of the local vLLM, for benchmarking the agent harness against a frontier
model. STT/TTS stay local; the external OpenAI-compat endpoint
(`/v1/chat/completions`) is pinned to vLLM regardless of this setting.

Set:
```bash
LLM_PROVIDER=anthropic                # seed only; dashboard toggle wins at runtime
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-7       # default
```

Then flip providers live from **Dashboard → System → Agent LLM Provider**
(`vLLM` ↔ `Anthropic`). The swap takes effect on the next turn — no
session rebuild, no restart. The selected provider is persisted in the
`agent_state` table, so it survives container restarts. The env var is
only used as the first-boot seed; after that the DB value wins.

Server-side prompt caching is applied automatically when the Anthropic
provider is active (system block, tools array, and last conversation
message get `cache_control: ephemeral` breakpoints). Cache hits/writes
are logged at INFO: `[anthropic] cache read=N create=N input=N output=N`.

### Nginx Gateway Configuration

Located in `services/nginx/nginx.conf`:
```nginx
upstream agent_backend {
    server agent:6002;
}

upstream tts_backend {
    server text-to-speech:6005;
}

upstream stt_backend {
    server speech-to-text:6001;
}
```

**Customization Options**:
- Load balancing algorithms
- SSL/TLS termination
- Rate limiting rules
- CORS policies

### Speech Services Configuration

#### Speech-to-Text
The Whisper model is pinned in `services/speech-to-text/app/config.py`
(currently `distil-large-v3`); it is not driven by `.env`. Set the GPU
via `STT_DEVICE` and the source language via `SRC_LAN` above.

#### Text-to-Speech
Kokoro voice and language come from `TTS_VOICE` / `TTS_LANGUAGE` above.
The model files are baked into the image. GPU is selected by `TTS_DEVICE`.

## Advanced Configuration

### Resource Allocation

#### Memory Limits
```yaml
services:
  agent:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

#### GPU Allocation
```yaml
speech-to-text:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Health Check Configuration

#### Service Health Checks
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:6002/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

#### Custom Health Check Endpoints
- **Agent**: `GET /health`
- **TTS**: `GET /health` 
- **STT**: `GET /health`
- **vLLM**: `GET /v1/models`

### Network Configuration

#### Internal Networks
```yaml
networks:
  havencore-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### Port Mapping
```yaml
ports:
  - "80:80"          # Nginx gateway
  - "6002:6002"      # Agent web interface
  - "8000:8000"      # LLM API (optional external access)
```

### Volume Configuration

#### Persistent Storage
```yaml
volumes:
  postgres_data:    # Database persistence
  model_cache:      # AI model storage
  audio_cache:      # Generated audio files
```

#### Bind Mounts
```yaml
volumes:
  - ./services/agent/app:/app              # Live code reload
  - ./shared:/app/shared:ro                # Shared configuration
  - ./models:/models                       # Local model storage
```

## Environment Profiles

### Development Profile
```bash
# .env.dev
DEBUG_LOGGING=1
HOST_IP_ADDRESS="127.0.0.1"
LLM_API_KEY="dev123"
POSTGRES_PASSWORD="dev_password"
```

### Production Profile
```bash
# .env.prod
DEBUG_LOGGING=0
HOST_IP_ADDRESS="your.production.ip"
LLM_API_KEY="secure_random_key"
POSTGRES_PASSWORD="secure_database_password"
HAOS_TOKEN="production_ha_token"
```

### Testing Profile
```bash
# .env.test
DEBUG_LOGGING=1
HOST_IP_ADDRESS="127.0.0.1"
POSTGRES_DB="havencore_test"
MCP_ENABLED=true  # Test MCP features
```

## Configuration Validation

### Validation Commands
```bash
# Validate Docker Compose configuration
docker compose config --quiet

# Test database connection
docker compose exec postgres psql -U havencore -d havencore -c "SELECT 1;"

# Test GPU access
docker compose exec agent nvidia-smi

# Validate API keys
curl -H "Authorization: Bearer ${LLM_API_KEY}" http://localhost/health
```

### Common Configuration Issues

#### GPU Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

# Install NVIDIA Container Toolkit if needed
```

#### Network Connectivity Issues
```bash
# Check internal service connectivity
docker compose exec agent curl http://postgres:5432
docker compose exec agent curl http://text-to-speech:6005/health

# Check external API connectivity
docker compose exec agent curl https://api.weatherapi.com
```

#### Model Download Failures
```bash
# Check HuggingFace token
echo $HF_HUB_TOKEN

# Pre-download models
huggingface-cli download QuantTrio/GLM-4.5-Air-AWQ-FP16Mix

# Check disk space
df -h
```

## Security Configuration

### API Security
```bash
# Use strong API keys
LLM_API_KEY="$(openssl rand -base64 32)"

# Enable rate limiting in nginx.conf
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
```

### Network Security
```bash
# Restrict external access
# Only expose necessary ports
ports:
  - "127.0.0.1:80:80"  # Only local access

# Use internal networks
networks:
  havencore-internal:
    internal: true
```

### Secret Management
```bash
# Use Docker secrets for sensitive data
secrets:
  haos_token:
    external: true
  api_keys:
    external: true
```

## Backup and Recovery Configuration

### Database Backup
```bash
# Configure automatic backups
docker compose exec postgres pg_dump -U havencore havencore > backup.sql

# Scheduled backup script
#!/bin/bash
docker compose exec postgres pg_dump -U havencore havencore | gzip > "backup_$(date +%Y%m%d_%H%M%S).sql.gz"
```

### Configuration Backup
```bash
# Backup essential configurations
tar -czf havencore_config_backup.tar.gz .env compose.yaml services/nginx/nginx.conf
```

---

**Next Steps**:
- [API Reference](api-reference.md) - Complete API documentation
- [Troubleshooting](troubleshooting.md) - Common configuration issues
- [Home Assistant Integration](integrations/home-assistant.md) - Smart home setup