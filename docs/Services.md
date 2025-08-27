# Service Documentation

This document provides detailed information about each service in the HavenCore architecture, including their purpose, configuration, APIs, and troubleshooting.

## Service Overview

| Service | Ports | Purpose | Technology Stack |
|---------|-------|---------|------------------|
| [Nginx Gateway](#nginx-gateway) | 80 | API Gateway & Load Balancer | Nginx Alpine |
| [Agent Service](#agent-service) | 6002, 6006 | AI Logic & Tool Calling | Python, FastAPI, Gradio |
| [Speech-to-Text](#speech-to-text-service) | 6000, 6001, 5999 | Audio Transcription | Python, Whisper, CUDA |
| [Text-to-Speech](#text-to-speech-service) | 6003, 6004, 6005 | Audio Generation | Python, Kokoro TTS, CUDA |
| [PostgreSQL](#postgresql-database) | 5432 | Data Storage | PostgreSQL 15 Alpine |
| [vLLM Backend](#vllm-backend) | 8000 | LLM Inference | vLLM, CUDA |
| [LlamaCPP Backend](#llamacpp-backend) | 8000 | Alternative LLM | LlamaCPP, CPU/GPU |
| [Qdrant](#qdrant-vector-database) | 6333, 6334 | Vector Database | Qdrant |
| [Embeddings](#embeddings-service) | 3000 | Text Embeddings | Transformer Models |

---

## Nginx Gateway

### Purpose
- API gateway and reverse proxy
- Load balancing across service instances
- SSL termination and CORS handling
- Request routing and transformation

### Configuration
**Location**: `services/nginx/nginx.conf`

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

server {
    listen 80;
    
    # Chat completions routing
    location /v1/chat/completions {
        proxy_pass http://agent_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Audio API routing
    location /v1/audio/speech {
        proxy_pass http://tts_backend;
    }
    
    location /v1/audio/transcriptions {
        proxy_pass http://stt_backend;
    }
}
```

### Features
- **Load Balancing**: Round-robin across service instances
- **Health Checks**: Automatic failover for unhealthy services
- **Rate Limiting**: Configurable request throttling
- **CORS Support**: Cross-origin request handling
- **SSL/TLS**: Encryption termination (when configured)

### Monitoring
```bash
# Check Nginx status
docker compose exec nginx nginx -t

# View access logs
docker compose logs nginx

# Test configuration
curl -I http://localhost/health
```

### Customization
Common customizations in `nginx.conf`:

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req zone=api burst=20 nodelay;

# SSL configuration
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
}

# Custom headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
```

---

## Agent Service

### Purpose
- Core AI conversation engine
- Tool calling and function execution
- Session and conversation management
- Integration orchestration

### Architecture
```
Request → FastAPI → Selene Agent → Tool Registry → External APIs
    ↓         ↓           ↓             ↓              ↓
Response ← JSON ← Agent Logic ← Tool Results ← API Responses
```

### Key Components

#### Selene Agent (`selene_agent.py`)
Main AI logic and conversation handling:
```python
class SeleneAgent:
    def __init__(self):
        self.tool_registry = UnifiedToolRegistry()
        self.conversation_db = conversation_db
        self.haos = HAOSInterface()
        self.tool_executor = AsyncToolExecutor()
```

#### Tool Registry (`unified_tool_registry.py`)
Manages both legacy and MCP tools:
```python
class UnifiedToolRegistry:
    def register_legacy_tool(self, tool_def, function)
    async def get_all_tools(self) -> List[UnifiedTool]
    def get_registry_status(self) -> Dict[str, Any]
```

#### Conversation Database (`conversation_db.py`)
Handles conversation persistence and history.

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Main chat API |
| `/v1/models` | GET | List available models |
| `/health` | GET | Service health check |
| `/tools/status` | GET | Tool registry status |
| `/mcp/status` | GET | MCP connection status |
| `/` | GET | Gradio web interface |

### Configuration
Environment variables in `.env`:
```bash
AGENT_NAME="Selene"
DEV_CUSTOM_API_KEY="your_api_key"
CURRENT_LOCATION="San Francisco, CA"
CURRENT_TIMEZONE="America/Los_Angeles"
```

### Available Tools

#### Home Assistant Tools
- `home_assistant.get_domain_entity_states`
- `home_assistant.get_domain_services`
- `home_assistant.execute_service`

#### External Service Tools
- `get_weather_forecast`: Weather information
- `brave_search`: Web search capabilities
- `wolfram_alpha`: Computational queries
- `query_wikipedia`: Knowledge queries

#### Media Control Tools
- `control_media_player`: Media device control
- `get_media_player_statuses`: Device status
- `play_media`: Content playback
- `find_media_items`: Content discovery

### Conversation Management
- **Session Tracking**: Automatic session management
- **History Storage**: PostgreSQL persistence after timeouts
- **Context Management**: Conversation context preservation
- **Tool Integration**: Seamless external service calls

### Development and Debugging
```bash
# View agent logs
docker compose logs -f agent

# Access Python console
docker compose exec agent python

# Test tool registry
curl http://localhost:6002/tools/status

# Check MCP status
curl http://localhost:6002/mcp/status
```

### Performance Tuning
- **Memory Management**: Conversation history cleanup
- **Tool Optimization**: Parallel tool execution
- **Response Caching**: Repeated query optimization
- **GPU Utilization**: Efficient model inference

---

## Speech-to-Text Service

### Purpose
- Audio transcription using OpenAI Whisper
- Multiple audio format support
- Real-time and batch processing
- OpenAI-compatible API

### Architecture
```
Audio Input → Preprocessing → Whisper Model → Text Output
     ↓              ↓             ↓            ↓
Format Detection  Normalization  GPU Inference  Post-processing
```

### Service Endpoints

| Port | Purpose | API Type |
|------|---------|----------|
| 6001 | OpenAI-compatible API | REST API |
| 6000 | Gradio web interface | Web UI |
| 5999 | Internal service API | Internal |

### Supported Audio Formats
- **WAV**: Uncompressed audio
- **MP3**: MPEG audio compression
- **MP4**: Video container with audio
- **M4A**: Apple audio format
- **FLAC**: Lossless compression
- **OGG**: Open-source audio format
- **WEBM**: Web-optimized audio

### API Usage
```bash
# Transcribe audio file
curl -X POST http://localhost/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "language=en"
```

### Configuration
Environment variables:
```bash
STT_DEVICE="0"           # GPU device index
WHISPER_MODEL="base"     # Model size: tiny, base, small, medium, large
SRC_LAN="en"            # Default language
```

### Model Options
| Model | Size | VRAM | Speed | Accuracy |
|-------|------|------|-------|----------|
| tiny | 39 MB | ~1GB | Fastest | Lowest |
| base | 74 MB | ~1GB | Fast | Good |
| small | 244 MB | ~2GB | Medium | Better |
| medium | 769 MB | ~5GB | Slow | High |
| large | 1550 MB | ~10GB | Slowest | Best |

### Performance Optimization
```python
# Batch processing for multiple files
batch_size = 4
device = "cuda:0"

# Memory optimization
torch.cuda.empty_cache()
```

### Troubleshooting
```bash
# Check GPU usage
nvidia-smi

# Test service directly
curl http://localhost:6001/health

# View processing logs
docker compose logs -f speech-to-text

# Check model loading
docker compose exec speech-to-text ls -la /models/
```

---

## Text-to-Speech Service

### Purpose
- High-quality speech synthesis using Kokoro TTS
- Multiple voice and language options
- OpenAI-compatible API
- Web interface for testing

### Architecture
```
Text Input → Kokoro TTS → Audio Generation → File Storage
     ↓            ↓             ↓              ↓
  Preprocessing  Neural TTS   WAV Output    Static Serving
```

### Service Endpoints

| Port | Purpose | Features |
|------|---------|----------|
| 6005 | OpenAI-compatible API | `/v1/audio/speech` |
| 6004 | Gradio web interface | Testing and controls |
| 6003 | Static file server | Generated audio files |

### Voice Options
Currently all OpenAI voice names map to "af_heart":
- alloy → af_heart
- echo → af_heart  
- fable → af_heart
- onyx → af_heart
- nova → af_heart
- shimmer → af_heart

### API Usage
```bash
# Generate speech
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of text to speech.",
    "model": "tts-1",
    "voice": "alloy",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Configuration
Environment variables:
```bash
TTS_DEVICE="cuda:0"      # GPU device
TTS_LANGUAGE="a"         # Language option
TTS_VOICE="af_heart"     # Voice model
```

### Audio Formats
**Request formats** (accepted but all output as WAV):
- mp3, opus, aac, flac, wav, pcm

**Actual output**: Always WAV format regardless of request

### Performance Features
- **GPU Acceleration**: CUDA-optimized processing
- **Batch Processing**: Multiple text inputs
- **Caching**: Generated audio file storage
- **Streaming**: Real-time audio generation

### Web Interface Features
Access at `http://localhost:6004`:
- Text input with voice selection
- Real-time audio generation
- Audio playback controls
- Download generated files
- Voice parameter tuning

### File Management
```bash
# Generated audio storage
/app/audio_files/timestamp-sessionid.wav

# Static file serving
curl http://localhost:6003/1234567890-abcd.wav
```

### Troubleshooting
```bash
# Check TTS service
curl http://localhost:6005/health

# Test audio generation
curl -X POST http://localhost:6005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "voice": "alloy"}'

# Check GPU usage
nvidia-smi

# View service logs
docker compose logs -f text-to-speech
```

---

## PostgreSQL Database

### Purpose
- Conversation history storage
- User session persistence  
- Configuration data storage
- Analytics and monitoring data

### Database Schema

#### Conversation Histories
```sql
CREATE TABLE conversation_histories (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    conversation_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

### Configuration
Environment variables:
```bash
POSTGRES_HOST="postgres"
POSTGRES_PORT=5432
POSTGRES_DB="havencore"
POSTGRES_USER="havencore" 
POSTGRES_PASSWORD="havencore_password"
```

### Data Storage Patterns

#### Conversation Storage
Triggered automatically when:
- New query received after 3+ minutes of inactivity
- Existing conversation has multiple messages
- Session timeout occurs

#### Metadata Structure
```json
{
  "reset_reason": "timeout_3_minutes",
  "message_count": 5,
  "last_query_timestamp": "2024-01-15T10:30:00Z",
  "agent_name": "Selene",
  "trace_id": "abc123"
}
```

### Database Operations
```bash
# Connect to database
docker compose exec postgres psql -U havencore -d havencore

# Query recent conversations
SELECT session_id, created_at, metadata->>'message_count' 
FROM conversation_histories 
ORDER BY created_at DESC LIMIT 10;

# View conversation content
SELECT jsonb_pretty(conversation_data) 
FROM conversation_histories 
WHERE session_id = 'your-session-id';
```

### Backup and Recovery
```bash
# Create backup
docker compose exec postgres pg_dump -U havencore havencore > backup.sql

# Restore backup
docker compose exec -T postgres psql -U havencore -d havencore < backup.sql

# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker compose exec postgres pg_dump -U havencore havencore | gzip > "backup_${TIMESTAMP}.sql.gz"
```

### Performance Monitoring
```bash
# Check database size
docker compose exec postgres psql -U havencore -d havencore -c "
SELECT pg_size_pretty(pg_database_size('havencore'));
"

# Monitor connections
docker compose exec postgres psql -U havencore -d havencore -c "
SELECT count(*) as connections FROM pg_stat_activity;
"

# Check table sizes
docker compose exec postgres psql -U havencore -d havencore -c "
SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size 
FROM pg_tables WHERE schemaname='public';
"
```

---

## vLLM Backend

### Purpose
- High-performance LLM inference
- OpenAI-compatible API
- GPU-optimized processing
- Model serving and management

### Configuration
Located in `compose.yaml`:
```yaml
vllm:
  image: vllm/vllm-openai:latest
  command: [
    "--model", "TechxGenus/Mistral-Large-Instruct-2411-AWQ",
    "--gpu-memory-utilization", "0.9",
    "--max-model-len", "32768",
    "--dtype", "auto",
    "--api-key", "${DEV_CUSTOM_API_KEY}"
  ]
```

### Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model path | Required |
| `--gpu-memory-utilization` | GPU memory usage (0.0-1.0) | 0.9 |
| `--max-model-len` | Maximum sequence length | 4096 |
| `--dtype` | Data type (auto, float16, bfloat16) | auto |
| `--tensor-parallel-size` | Number of GPUs | 1 |
| `--api-key` | API authentication key | None |

### Supported Models
- **AWQ Quantized**: Optimized inference models
- **GPTQ**: Alternative quantization format
- **Full Precision**: Unquantized models (high VRAM)

Popular model options:
```yaml
# High performance, lower memory
"TechxGenus/Mistral-Large-Instruct-2411-AWQ"

# Alternative options
"hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
"microsoft/Phi-3-mini-4k-instruct"
```

### Performance Tuning
```yaml
# Multi-GPU setup
command: [
  "--model", "your-model",
  "--tensor-parallel-size", "2",  # Use 2 GPUs
  "--gpu-memory-utilization", "0.8"
]

# Memory optimization
command: [
  "--model", "your-model", 
  "--max-model-len", "16384",    # Reduce context length
  "--gpu-memory-utilization", "0.7"
]
```

### API Endpoints
- `GET /v1/models`: List available models
- `POST /v1/chat/completions`: Chat API
- `POST /v1/completions`: Text completion
- `GET /health`: Service health

### Monitoring
```bash
# Check model loading
docker compose logs -f vllm

# Test API directly
curl http://localhost:8000/v1/models

# Monitor GPU usage
nvidia-smi -l 1

# Check memory usage
docker compose exec vllm nvidia-smi
```

---

## LlamaCPP Backend

### Purpose
- CPU-focused LLM inference  
- Lower memory requirements
- GGUF model support
- Alternative to vLLM

### Configuration
```yaml
llamacpp:
  build:
    context: ./services/llamacpp
  command: [
    "python", "-m", "llama_cpp.server",
    "--model", "/models/model.gguf",
    "--n_gpu_layers", "33",
    "--host", "0.0.0.0", 
    "--port", "8000"
  ]
```

### Model Format
Uses GGUF format models:
```bash
# Download GGUF model
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models/
```

### Performance Options
```yaml
# CPU-only inference
command: [
  "python", "-m", "llama_cpp.server",
  "--model", "/models/model.gguf",
  "--n_gpu_layers", "0",        # CPU only
  "--n_threads", "8"            # CPU threads
]

# GPU acceleration
command: [
  "python", "-m", "llama_cpp.server", 
  "--model", "/models/model.gguf",
  "--n_gpu_layers", "33",       # GPU layers
  "--n_batch", "512"            # Batch size
]
```

### When to Use LlamaCPP
- **Limited GPU Memory**: Less than 8GB VRAM
- **CPU Inference**: No GPU available
- **GGUF Models**: Specific model format requirements
- **Resource Constraints**: Lower memory usage needed

---

## Qdrant Vector Database

### Purpose
- Vector storage for embeddings
- Semantic search capabilities
- RAG (Retrieval Augmented Generation)
- Document similarity matching

### Configuration
```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"  # HTTP API
    - "6334:6334"  # gRPC API
  volumes:
    - qdrant_data:/qdrant/storage
```

### API Usage
```bash
# Create collection
curl -X PUT http://localhost:6333/collections/documents \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'

# Add vectors
curl -X PUT http://localhost:6333/collections/documents/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": 1,
        "vector": [0.1, 0.2, 0.3, ...],
        "payload": {"text": "Document content"}
      }
    ]
  }'

# Search similar vectors
curl -X POST http://localhost:6333/collections/documents/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "limit": 5
  }'
```

---

## Embeddings Service

### Purpose
- Convert text to vector embeddings
- Multiple embedding model support
- Integration with vector database
- Semantic similarity computation

### Configuration
```yaml
embeddings:
  image: your-embeddings-image
  ports:
    - "3000:3000"
  environment:
    - MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

### API Usage
```bash
# Generate embeddings
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Another sentence"]
  }'
```

### Supported Models
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- `text-embedding-ada-002` (OpenAI compatible)

---

## Service Communication

### Internal Network
Services communicate via Docker internal networking:
```
agent → postgres:5432
agent → vllm:8000  
agent → qdrant:6333
nginx → agent:6006
nginx → text-to-speech:6005
nginx → speech-to-text:6001
```

### Health Check Chain
```
nginx → agent/health
nginx → tts/health
nginx → stt/health
agent → postgres (connection test)
agent → vllm/v1/models
```

### Service Dependencies
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

---

**Next Steps**:
- [Troubleshooting Guide](Troubleshooting.md) - Service-specific debugging
- [Development Guide](Development.md) - Service modification and testing
- [Performance Tuning](Performance.md) - Service optimization