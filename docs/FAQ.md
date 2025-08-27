# Frequently Asked Questions (FAQ)

This document answers common questions about HavenCore setup, usage, and troubleshooting.

## General Questions

### What is HavenCore?

HavenCore is a self-hosted AI smart home assistant that provides:
- Voice-activated control with natural language processing
- Integration with Home Assistant and other smart home platforms
- OpenAI-compatible APIs for chat, speech-to-text, and text-to-speech
- Fully containerized architecture that runs on your own hardware

**Key Benefits**:
- **Privacy**: All processing happens on your hardware
- **Customization**: Fully open-source and extensible
- **Integration**: Works with existing smart home setups
- **Performance**: Optimized for local inference and low latency

### How is HavenCore different from Alexa, Google Assistant, or Siri?

| Feature | HavenCore | Commercial Assistants |
|---------|-----------|----------------------|
| **Privacy** | Completely local/private | Cloud-based, data collected |
| **Customization** | Fully customizable and extensible | Limited customization |
| **Ownership** | You own and control everything | Controlled by corporation |
| **Integration** | Direct API access, unlimited integrations | Limited to approved partnerships |
| **Voice Data** | Never leaves your network | Sent to cloud for processing |
| **Wake Words** | Customizable (future feature) | Fixed wake words |
| **Responses** | Fully customizable AI responses | Predetermined responses |

### What hardware do I need to run HavenCore?

**Minimum Requirements**:
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 16GB (32GB recommended for optimal performance)
- **Storage**: 50GB free space for models and containers
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for best performance)
- **Network**: Reliable internet for initial setup and external services

**Recommended Setups**:

**Budget Setup** (~$800-1200):
- NVIDIA GTX 1660 Super (6GB) or RTX 3060 (12GB)
- 16GB RAM, modern CPU, SSD storage

**Optimal Setup** (~$1500-2500):
- NVIDIA RTX 4070 (12GB) or RTX 3080 (10GB)
- 32GB RAM, high-end CPU, NVMe SSD

**High-End Setup** (~$3000+):
- NVIDIA RTX 4080/4090 (16GB/24GB)
- 64GB RAM, server-grade CPU, enterprise SSD

### Can I run HavenCore without a GPU?

Yes, but with limitations:
- **CPU-only mode**: Use LlamaCPP backend instead of vLLM
- **Slower inference**: Responses will take longer (10-30 seconds vs 1-3 seconds)
- **Limited models**: Restricted to smaller, quantized models
- **Reduced features**: Some audio processing may be slower

**CPU-only configuration**:
```yaml
# Use LlamaCPP instead of vLLM
llamacpp:
  command: [
    "python", "-m", "llama_cpp.server",
    "--model", "/models/phi-3-mini-4k-instruct-q4.gguf",
    "--n_gpu_layers", "0",  # CPU only
    "--n_threads", "8"
  ]
```

## Installation and Setup

### The build process seems stuck - is this normal?

**Yes, this is completely normal!** The first build takes 60-90 minutes because:

1. **Model Downloads**: AI models are large (2-10GB each)
2. **Docker Images**: Base images with CUDA support are large (3-5GB)
3. **Dependencies**: Python packages and system libraries
4. **Compilation**: Some components need to be compiled for your hardware

**What's happening during the build**:
- Minutes 0-15: Downloading base Docker images
- Minutes 15-45: Downloading AI models from HuggingFace
- Minutes 45-75: Installing Python dependencies and compiling
- Minutes 75-90: Final configuration and startup

**Never cancel the build process!** Let it complete fully.

### Why do I need so many environment variables?

The environment variables configure:
- **Core System**: API keys, host IP, debug settings
- **AI Models**: GPU allocation, model selection, performance tuning
- **External Services**: Home Assistant, weather, search APIs
- **Database**: PostgreSQL connection and credentials
- **Security**: Authentication tokens and access control

**Only these are absolutely required**:
```bash
HOST_IP_ADDRESS="your.ip.address"    # Your Docker host IP
DEV_CUSTOM_API_KEY="your_api_key"    # Any value for API access
```

**Everything else has sensible defaults** and can be configured later.

### Can I use my own AI models?

**Yes!** HavenCore supports:

**LLM Models**:
- Any HuggingFace model compatible with vLLM
- GGUF models with LlamaCPP backend
- Local fine-tuned models
- Custom quantized models

**Audio Models**:
- OpenAI Whisper variants for speech-to-text
- Kokoro TTS for text-to-speech
- Custom trained audio models

**Model Configuration Examples**:
```yaml
# Different LLM models
vllm:
  command: ["--model", "microsoft/Phi-3-medium-4k-instruct"]
  # or
  command: ["--model", "meta-llama/Llama-2-7b-chat-hf"]
  # or
  command: ["--model", "/local/path/to/your-model"]

# Different audio models
speech-to-text:
  environment:
    - WHISPER_MODEL=large-v3  # or medium, small, base, tiny
```

### How do I update HavenCore?

**For stable updates**:
```bash
# Stop services
docker compose down

# Pull latest code
git pull origin main

# Rebuild containers
docker compose build

# Start updated services
docker compose up -d
```

**For development/testing**:
```bash
# Create backup first
docker compose exec postgres pg_dump -U havencore havencore > backup.sql

# Follow update process above
# If issues occur, restore backup
```

## Usage and Features

### What voice commands does HavenCore understand?

HavenCore understands natural language commands for:

**Home Automation**:
- "Turn on the living room lights"
- "Set the thermostat to 72 degrees"
- "Close the bedroom blinds"
- "Start the robot vacuum"

**Media Control**:
- "Play music in the kitchen"
- "Pause the TV"
- "Next song please"
- "Set volume to 50%"

**Information Queries**:
- "What's the weather like today?"
- "How many calories are in an apple?"
- "What time is it in Tokyo?"
- "Search for news about technology"

**System Commands**:
- "Show me all the lights"
- "What sensors are available?"
- "Check the system status"

**The AI understands context and variations** - you don't need exact phrases.

### Can I control HavenCore with text instead of voice?

**Absolutely!** HavenCore provides multiple interfaces:

**Web Interface** (http://localhost:6002):
- Chat-based interface for text conversations
- Real-time responses and tool execution
- Session history and management

**API Access**:
```bash
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Turn on the lights"}]}'
```

**Mobile Apps** (using the API):
- Any app that supports OpenAI-compatible APIs
- Custom mobile apps using the REST API

### How accurate is the speech recognition?

**Speech-to-text accuracy depends on**:
- **Model size**: larger = more accurate but slower
- **Audio quality**: clear audio with minimal background noise
- **Language/accent**: optimized for English by default
- **Context**: smart home commands vs complex conversations

**Typical accuracy rates**:
- **Simple commands**: 95-99% accuracy ("turn on lights")
- **Complex queries**: 85-95% accuracy (longer sentences)
- **Technical terms**: 80-90% accuracy (specific device names)

**Improving accuracy**:
- Use clear, consistent device names
- Speak clearly and at normal pace
- Minimize background noise
- Use larger Whisper models (medium, large)

### Can HavenCore learn my preferences?

**Current capabilities**:
- **Conversation history**: Stores conversations for context
- **Entity recognition**: Learns your device names and preferences
- **Context awareness**: Remembers recent commands and settings

**Future capabilities** (planned):
- **User profiles**: Personal preferences and settings
- **Habit learning**: Automatic routines based on usage patterns
- **Custom responses**: Personalized assistant personality
- **Preference memory**: Long-term memory of user choices

### How does HavenCore handle privacy?

**Complete Local Processing**:
- All voice data processed locally (never sent to cloud)
- Conversations stored locally in your PostgreSQL database
- AI inference happens on your hardware

**External API Usage** (optional):
- Weather data (if weather API key provided)
- Web search (if search API key provided)
- Computational queries (if WolframAlpha key provided)

**Data Control**:
- You control all data storage and retention
- Database backups under your control
- No telemetry or usage data sent anywhere
- Full conversation history encryption possible

## Troubleshooting Common Issues

### Services won't start after installation

**Check these common issues**:

1. **GPU driver problems**:
```bash
nvidia-smi  # Should show GPU information
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

2. **Port conflicts**:
```bash
sudo netstat -tlnp | grep :80  # Check what's using port 80
sudo netstat -tlnp | grep :6002  # Check agent port
```

3. **Environment configuration**:
```bash
grep -E "HOST_IP_ADDRESS|DEV_CUSTOM_API_KEY" .env  # Verify required vars
docker compose config --quiet  # Validate compose file
```

4. **Disk space**:
```bash
df -h  # Check available disk space
docker system df  # Check Docker space usage
```

### API calls return 401 Unauthorized

**Check API key configuration**:
```bash
# Verify API key in environment
grep DEV_CUSTOM_API_KEY .env

# Test with correct authorization header
curl -H "Authorization: Bearer your_actual_api_key" http://localhost/health

# Restart services to pick up changes
docker compose restart agent
```

### Home Assistant integration not working

**Verify connection**:
```bash
# Test HA connection from container
docker compose exec agent curl -H "Authorization: Bearer YOUR_HA_TOKEN" \
  "https://homeassistant.local:8123/api/"

# Check environment variables
docker compose exec agent env | grep HAOS
```

**Common fixes**:
- Ensure Home Assistant URL includes `/api` at the end
- Regenerate long-lived access token in Home Assistant
- Check network connectivity between HavenCore and HA
- Verify Home Assistant is accessible from HavenCore's network

### Voice commands work but responses are slow

**Performance optimization**:

1. **Check GPU utilization**:
```bash
nvidia-smi  # Should show GPU usage during inference
```

2. **Reduce model size**:
```yaml
# Use smaller/faster model
vllm:
  command: ["--model", "microsoft/Phi-3-mini-4k-instruct"]
```

3. **Optimize memory**:
```yaml
# Reduce context length for faster responses
vllm:
  command: [
    "--model", "your-model",
    "--max-model-len", "8192"  # Reduce from 32768
  ]
```

4. **Check system resources**:
```bash
docker stats  # Monitor container resource usage
free -h       # Check system memory
```

### Audio quality is poor

**Text-to-speech improvements**:
- Check TTS service logs: `docker compose logs text-to-speech`
- Test with simple text first
- Verify audio output format compatibility
- Check volume and audio device settings

**Speech-to-text improvements**:
- Use higher quality audio input (16kHz+, mono)
- Minimize background noise
- Use larger Whisper model if needed
- Test with simple commands first

## Advanced Configuration

### Can I run multiple instances of HavenCore?

**Yes, but consider**:
- **Port conflicts**: Each instance needs different ports
- **Resource usage**: Multiple GPU-intensive services
- **Database isolation**: Separate PostgreSQL instances
- **Configuration management**: Different .env files

**Multi-instance setup**:
```bash
# Instance 1 (default ports)
cp .env .env.instance1
docker compose -f compose.yaml --env-file .env.instance1 up -d

# Instance 2 (different ports)
cp .env .env.instance2
# Edit .env.instance2 to change HOST_IP_ADDRESS and ports
docker compose -f compose.instance2.yaml --env-file .env.instance2 up -d
```

### How do I backup my HavenCore configuration?

**Essential backup items**:
```bash
# Configuration files
tar -czf havencore_config_backup.tar.gz .env compose.yaml services/nginx/nginx.conf

# Database backup
docker compose exec postgres pg_dump -U havencore havencore > conversations_backup.sql

# Custom models (if any)
cp -r models/ models_backup/

# Service customizations
tar -czf services_backup.tar.gz services/
```

**Automated backup script**:
```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/${TIMESTAMP}"

mkdir -p "$BACKUP_DIR"
cp .env "$BACKUP_DIR/"
cp compose.yaml "$BACKUP_DIR/"
docker compose exec postgres pg_dump -U havencore havencore > "$BACKUP_DIR/database.sql"
tar -czf "$BACKUP_DIR/services.tar.gz" services/

echo "Backup created in $BACKUP_DIR"
```

### Can I integrate HavenCore with other systems?

**Yes! HavenCore provides OpenAI-compatible APIs** that work with:

**Home Automation**:
- Node-RED flows
- OpenHAB integrations
- Custom automation scripts

**Development Platforms**:
- Any application that supports OpenAI API
- Custom mobile applications
- Web dashboards and interfaces

**Voice Assistants**:
- Integration with existing voice platforms
- Custom wake-word detection systems
- Multi-modal interface development

**Example integrations**:
```python
# Python integration
import openai

client = openai.OpenAI(
    api_key="your_havencore_api_key",
    base_url="http://localhost/v1"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Turn on the lights"}]
)
```

## Performance and Scaling

### How many concurrent users can HavenCore handle?

**Performance depends on**:
- **Hardware specifications**: GPU memory, CPU cores, RAM
- **Model size**: Larger models = slower inference but better quality
- **Request complexity**: Simple commands vs complex conversations
- **Resource allocation**: Container memory and CPU limits

**Typical performance**:
- **Single user**: Near real-time responses (1-3 seconds)
- **2-3 concurrent users**: Good performance with adequate hardware
- **5+ users**: May need horizontal scaling or resource optimization

**Scaling strategies**:
- **Vertical scaling**: More powerful hardware
- **Horizontal scaling**: Multiple HavenCore instances
- **Model optimization**: Smaller/quantized models
- **Load balancing**: Nginx load balancing across instances

### What's the maximum conversation history?

**Default limits**:
- **Active conversation**: No hard limit (limited by memory)
- **Database storage**: Unlimited (limited by disk space)
- **Auto-cleanup**: Conversations stored after 3 minutes of inactivity

**Configuration options**:
```python
# Adjust conversation timeout in agent service
CONVERSATION_TIMEOUT = 180  # seconds (3 minutes)

# Database retention policy
DELETE FROM conversation_histories 
WHERE created_at < NOW() - INTERVAL '30 days';
```

---

**Need more help?**
- Check the [Troubleshooting Guide](Troubleshooting.md) for detailed solutions
- Review the [Configuration Guide](Configuration.md) for advanced settings
- Visit the [Development Guide](Development.md) for customization options
- Join the community discussions on GitHub