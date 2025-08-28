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
DEV_CUSTOM_API_KEY="your_secret_key"  # Set to any value for API access

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
HAOS_URL="https://homeassistant.local:8123/api"  # Your HA URL
HAOS_TOKEN="eyJ0eXAiOiJKV1QiLCJ..."              # Long-lived access token

# Legacy settings (still supported)
SOURCE_IP="10.0.0.100"  # Edge device IP (deprecated)
```

**Getting Home Assistant Token**:
1. Go to Home Assistant → Profile → Long-Lived Access Tokens
2. Click "Create Token"
3. Copy the token and set as `HAOS_TOKEN`

#### External APIs
```bash
# Weather service (weatherapi.com)
WEATHER_API_KEY="your_weather_api_key"

# Web search (Brave Search API)
BRAVE_SEARCH_API_KEY="your_brave_api_key"

# Computational queries (WolframAlpha)
WOLFRAM_ALPHA_API_KEY="your_wolfram_api_key"
```

**API Key Setup**:
- **Weather**: Register at [weatherapi.com](https://www.weatherapi.com/)
- **Brave Search**: Get key from [Brave Search API](https://api.search.brave.com/)
- **WolframAlpha**: Register at [Wolfram Developer Portal](https://developer.wolframalpha.com/)

### MCP (Model Context Protocol) Configuration

#### MCP Enable/Disable
```bash
# Enable MCP support (experimental)
MCP_ENABLED=false  # Set to true to enable MCP tools

# Tool preference when conflicts exist
MCP_PREFER_OVER_LEGACY=false  # false = use legacy, true = use MCP
```

#### MCP Server Configuration
```bash
# JSON array of MCP server configurations
MCP_SERVERS='[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
    "enabled": true
  },
  {
    "name": "brave-search",
    "command": "npx", 
    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
    "enabled": false
  }
]'
```

#### Individual MCP Server Environment Variables
```bash
# Alternative to JSON configuration
MCP_SERVER_EXAMPLE_ENABLED=false
MCP_SERVER_EXAMPLE_COMMAND="node"
MCP_SERVER_EXAMPLE_ARGS="server.js,--port,3000"  # Comma-separated
```

## Service-Specific Configuration

### LLM Backend Configuration

#### vLLM Configuration (Default)
Located in `compose.yaml`:
```yaml
command: [
  "--model", "TechxGenus/Mistral-Large-Instruct-2411-AWQ",
  "--gpu-memory-utilization", "0.9",
  "--max-model-len", "32768",
  "--dtype", "auto",
  "--api-key", "${DEV_CUSTOM_API_KEY}"
]
```

**Key Parameters**:
- `--model`: HuggingFace model path
- `--gpu-memory-utilization`: GPU memory usage (0.0-1.0)
- `--max-model-len`: Maximum sequence length
- `--dtype`: Data type (auto, float16, bfloat16)

#### LlamaCPP Configuration (Alternative)
```yaml
llamacpp:
  command: [
    "python", "-m", "llama_cpp.server",
    "--model", "/models/model.gguf",
    "--n_gpu_layers", "33",
    "--host", "0.0.0.0",
    "--port", "8000"
  ]
```

### Nginx Gateway Configuration

Located in `services/nginx/nginx.conf`:
```nginx
upstream agent_backend {
    server agent:6006;
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

#### Speech-to-Text Settings
```bash
# In speech-to-text service environment
WHISPER_MODEL="base"  # tiny, base, small, medium, large
DEVICE="cuda:0"       # GPU device
BATCH_SIZE=1          # Processing batch size
```

#### Text-to-Speech Settings
```bash
# In text-to-speech service environment
KOKORO_MODEL_PATH="/models/kokoro"
VOICE_SELECTION="af_heart"
SAMPLE_RATE=22050
```

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

#### Persistent Storage with Bind Mounts

HavenCore uses bind mounts for persistent data storage, providing direct access to data files on the host system:

```yaml
volumes:
  # PostgreSQL database storage
  - ./volumes/postgres_data/data:/var/lib/postgresql/data
  
  # Vector database storage  
  - ./volumes/qdrant_storage:/qdrant/storage
  
  # AI model cache
  - ./volumes/models:/data
```

**Setting Up Volume Directories:**

1. **Automated Setup (Recommended):**
   ```bash
   ./scripts/setup-volumes.sh
   ```

2. **Manual Setup:**
   ```bash
   mkdir -p ./volumes/postgres_data/data
   mkdir -p ./volumes/qdrant_storage
   mkdir -p ./volumes/models
   
   # Set PostgreSQL permissions (Linux only)
   sudo chown -R 999:999 ./volumes/postgres_data/data
   ```

**Permission Requirements:**
- **PostgreSQL**: Requires UID 999 (postgres user in container)
- **Qdrant**: Uses default Docker user permissions
- **Models**: Uses default Docker user permissions

**Why Bind Mounts?**
- Direct file system access for backups
- Easier data migration between environments
- Better performance than named volumes
- Simplified debugging and troubleshooting

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
DEV_CUSTOM_API_KEY="dev123"
POSTGRES_PASSWORD="dev_password"
```

### Production Profile
```bash
# .env.prod
DEBUG_LOGGING=0
HOST_IP_ADDRESS="your.production.ip"
DEV_CUSTOM_API_KEY="secure_random_key"
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
curl -H "Authorization: Bearer ${DEV_CUSTOM_API_KEY}" http://localhost/health
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
huggingface-cli download TechxGenus/Mistral-Large-Instruct-2411-AWQ

# Check disk space
df -h
```

## Security Configuration

### API Security
```bash
# Use strong API keys
DEV_CUSTOM_API_KEY="$(openssl rand -base64 32)"

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
- [API Reference](API-Reference.md) - Complete API documentation
- [Troubleshooting](Troubleshooting.md) - Common configuration issues
- [Home Assistant Integration](Home-Assistant-Integration.md) - Smart home setup