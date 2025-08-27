# HavenCore AI Smart Home Assistant

HavenCore is a Docker-based microservices architecture for an AI-powered smart home assistant with voice control, speech-to-text, text-to-speech, and Home Assistant integration.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Prerequisites and Environment Setup
- Ensure NVIDIA GPU is available with CUDA support
- Verify Docker and Docker Compose V2 are installed
- Install NVIDIA Container Toolkit for GPU access in containers
- Install required system dependencies:
  ```bash
  # Test GPU access first
  nvidia-smi
  
  # Test Docker GPU access
  docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
  ```

### Bootstrap, Build, and Test Repository
- **CRITICAL**: Build times are 45-90 minutes. NEVER CANCEL. Set timeout to 120+ minutes.
- **CRITICAL**: Model downloads can take 30+ minutes. NEVER CANCEL.
- Clone and configure:
  ```bash
  git clone https://github.com/ThatMattCat/havencore.git
  cd havencore
  cp .env.tmpl .env
  # Edit .env with your specific settings (see Configuration section)
  ```
- Validate configuration first:
  ```bash
  # Test docker compose configuration
  docker compose config --quiet
  echo "✓ Docker compose configuration is valid"
  ```
- Build all services (takes 60-90 minutes):
  ```bash
  # NEVER CANCEL: Build takes 60-90 minutes. Set timeout to 120+ minutes.
  docker compose build --parallel
  ```
- Start services (takes 10-15 minutes for model loading):
  ```bash
  # NEVER CANCEL: Service startup takes 10-15 minutes. Set timeout to 30+ minutes.
  docker compose up -d
  ```
- Monitor startup:
  ```bash
  # Watch logs during startup
  docker compose logs -f
  
  # Check service health (wait 5-10 minutes after startup)
  docker compose ps
  ```

### Validation
- **ALWAYS run these validation steps after making changes**:
  ```bash
  # Test gateway health
  curl http://localhost/health
  
  # Test text-to-speech endpoint
  curl -X POST http://localhost/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, HavenCore is working!", "model": "tts-1", "voice": "alloy"}'
  
  # Test chat completion (requires LLM to be loaded)
  curl -X POST http://localhost/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 1234" \
    -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'
  ```
- **UI Validation**: Access web interfaces:
  - Agent Interface: http://localhost:6002
  - TTS Interface: http://localhost:6004/
  - System Gateway: http://localhost/
- **End-to-End Scenario Testing**: 
  - Always test complete voice workflow when modifying speech services
  - Test Home Assistant integration when modifying agent tools
  - Verify all API endpoints respond correctly after changes

### Key Configuration Requirements
Edit `.env` file with these REQUIRED settings:
```env
# REQUIRED: Set to your Docker host IP
HOST_IP_ADDRESS="192.168.1.100"

# REQUIRED: Set to something for API access
DEV_CUSTOM_API_KEY="1234"

# REQUIRED for Home Assistant integration
HAOS_URL="https://homeassistant.local:8123/api"
HAOS_TOKEN="your_long_lived_token_here"

# GPU configuration (adjust based on your setup)
TTS_DEVICE="cuda:0"  # GPU for text-to-speech
STT_DEVICE="0"       # GPU for speech-to-text

# Optional API keys for enhanced functionality
WEATHER_API_KEY=""         # From weatherapi.com
BRAVE_SEARCH_API_KEY=""    # For web search
WOLFRAM_ALPHA_API_KEY=""   # For computational queries
```

## Service Architecture

### Core Services and Ports
| Service | Ports | Purpose | Build Time |
|---------|-------|---------|------------|
| **nginx** | 80 | API Gateway & Load Balancer | 2-5 minutes |
| **agent** | 6002, 6006 | LLM Logic & Tool Calling | 10-15 minutes |
| **speech-to-text** | 6000, 6001, 5999 | Audio Transcription | 30-45 minutes |
| **text-to-speech** | 6003, 6004, 6005 | Audio Generation | 15-25 minutes |
| **postgres** | 5432 | Database & Conversation Storage | 2-5 minutes |
| **vLLM** | 8000 | LLM Inference Backend | Pre-built image |
| **qdrant** | 6333, 6334 | Vector Database | Pre-built image |
| **embeddings** | 3000 | Text Embeddings Service | Pre-built image |

### Build Dependencies and Timing
- **NEVER CANCEL builds or model downloads**
- GPU services require NVIDIA base images (2-4GB downloads)
- Model downloads happen during first startup (10-30 minutes)
- Total first-time setup: 90-120 minutes
- Subsequent builds: 15-30 minutes (with cached layers)

## Common Development Tasks

### Making Code Changes
1. **Python Services** (agent, speech-to-text, text-to-speech):
   ```bash
   # Code is mounted as volumes - changes are live
   # Restart specific service to reload
   docker compose restart agent
   ```

2. **Shared Configuration**:
   ```bash
   # Changes to shared/configs/shared_config.py require restart
   docker compose restart agent speech-to-text text-to-speech
   ```

3. **Environment Changes**:
   ```bash
   # Changes to .env require full restart
   docker compose down && docker compose up -d
   ```

### Testing and Validation
- **Always run validation after changes**:
  ```bash
  # Quick health check
  curl http://localhost/health
  
  # Test specific service
  docker compose logs -f [service_name]
  
  # Check GPU usage
  nvidia-smi
  ```

### Troubleshooting Common Issues
- **Services won't start**:
  ```bash
  # Check Docker configuration first
  docker compose config --quiet
  
  # Check GPU access (if using GPU services)
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
  
  # Check logs for specific service
  docker compose logs [service_name]
  
  # Check all service status
  docker compose ps
  ```

- **Out of memory errors**:
  - Reduce model size in vLLM configuration
  - Adjust `--gpu-memory-utilization` parameter in compose.yaml
  - Switch to LlamaCPP backend for lower memory usage
  - Check disk space: `docker system df`

- **Model download failures**:
  ```bash
  # Pre-download models manually
  huggingface-cli download TechxGenus/Mistral-Large-Instruct-2411-AWQ
  
  # Check Hugging Face token
  echo $HF_HUB_TOKEN
  
  # Check network connectivity
  curl -I https://huggingface.co
  ```

- **Build failures due to disk space**:
  ```bash
  # Clean up Docker
  docker system prune -f
  docker image prune -f
  
  # Check available space
  df -h
  ```

### Development Workflow
1. **Make changes** to Python files (live reload with mounted volumes)
2. **Restart affected services**: `docker compose restart [service_name]`
3. **Validate functionality** using health checks and API tests
4. **Test end-to-end scenarios** through web interfaces
5. **Check logs** for errors: `docker compose logs -f [service_name]`

### Development Without Full GPU Setup
For development work that doesn't require GPU inference:
```bash
# Start only non-GPU services for faster development
docker compose up -d postgres qdrant nginx

# Test basic functionality
curl http://localhost/health

# Build and test individual services
cd services/agent && python3 -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt

# Test Python imports and configuration
PYTHONPATH=/path/to/havencore python3 -c "import shared.configs.shared_config as config; print(f'Agent: {config.AGENT_NAME}')"
```

## Repository Navigation

### Key Directories
```
havencore/
├── .env.tmpl                 # Environment configuration template
├── compose.yaml              # Docker services orchestration
├── services/                 # All microservices
│   ├── nginx/               # API gateway configuration
│   ├── agent/               # Main AI agent logic
│   ├── speech-to-text/      # STT service (Whisper-based)
│   ├── text-to-speech/      # TTS service (Kokoro TTS)
│   ├── postgres/            # Database initialization
│   └── vllm/                # LLM configuration
├── shared/                  # Shared configuration and utilities
│   ├── configs/            # Common configuration
│   └── scripts/            # Shared utilities
└── docs/                   # Documentation
```

### Important Files
- **compose.yaml**: Service definitions and GPU configuration
- **shared/configs/shared_config.py**: Central configuration management
- **services/agent/app/selene_agent.py**: Main agent logic and tool calling
- **services/nginx/nginx.conf**: API routing and load balancing
- **.env**: Environment variables (copy from .env.tmpl)

### Frequently Modified Files
- **Agent logic**: `services/agent/app/selene_agent.py`
- **Tool definitions**: `services/agent/app/utils/*_tools_defs.py`
- **Home Assistant integration**: `services/agent/app/utils/haos/`
- **Service configuration**: `shared/configs/shared_config.py`

## Critical Timing and Timeout Requirements

### Build Commands (NEVER CANCEL)
```bash
# Set timeout to 120+ minutes for builds
docker compose build --timeout 7200

# Set timeout to 30+ minutes for startup
docker compose up -d --timeout 1800
```

### Expected Timing
- **First-time build**: 90-120 minutes
- **Subsequent builds**: 15-30 minutes
- **Service startup**: 10-15 minutes
- **Model downloads**: 10-30 minutes
- **Health check response**: 30-60 seconds

### Validation Commands
```bash
# Test all API endpoints (run after startup completes)
curl http://localhost/health                    # Gateway health
curl http://localhost:6002/                     # Agent health  
curl http://localhost:6004/                     # TTS UI health
curl http://localhost:8000/v1/models            # LLM models
```

## Integration and API Usage

### OpenAI-Compatible APIs
- **Chat Completions**: `POST /v1/chat/completions`
- **Text-to-Speech**: `POST /v1/audio/speech`
- **Speech-to-Text**: `POST /v1/audio/transcriptions`

### Home Assistant Integration
- Configure `HAOS_URL` and `HAOS_TOKEN` in .env
- Test connection: `curl -H "Authorization: Bearer YOUR_TOKEN" YOUR_HAOS_URL/states`
- Agent can control devices, read states, and execute services

### Web Interfaces
- **Agent Chat**: http://localhost:6002 - Interactive conversation interface
- **TTS Testing**: http://localhost:6004 - Text-to-speech testing and controls
- **API Gateway**: http://localhost - Service status and routing information

**Remember: ALWAYS follow these instructions first. Only search or explore further when information here is incomplete or appears incorrect.**