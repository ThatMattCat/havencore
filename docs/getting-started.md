# Getting Started with HavenCore

This guide will help you get HavenCore up and running quickly. For detailed configuration options, see the [Configuration Guide](configuration.md).

## Prerequisites

Before starting, ensure you have:

### Hardware Requirements
- **NVIDIA GPU(s)**: Required for AI model inference. The default vLLM
  model (GLM-4.5-Air-AWQ-FP16Mix, MoE ~106B total / ~12B active) needs
  roughly 72 GB of VRAM sharded across 4× 24 GB cards via
  `-tp 4 --enable-expert-parallel`. STT, TTS, vision, and image-gen
  contend for leftover VRAM on whichever card you pin them to — a
  4-GPU host is the target. Fewer-GPU configurations work if you swap
  in a smaller/non-MoE model (e.g. Qwen2.5-72B-AWQ on 2× 24 GB).
- **RAM**: Minimum 32GB, recommended 64GB+
- **Storage**: At least 150GB free space for model weights, container
  images, and Docker volumes.
- **CPU**: Modern multi-core processor (Intel/AMD)

### Software Requirements
- **Docker**: Version 20.10+ with Docker Compose V2
- **NVIDIA Container Toolkit**: For GPU access in containers
- **Git**: For cloning the repository

### Optional Dependencies
- **Home Assistant**: For smart home integration
- **API Keys**: For enhanced functionality (weather, search, etc.)

## Quick Start (5 Minutes)

### 1. Verify GPU Access

First, ensure your NVIDIA GPU is accessible:

```bash
# Test GPU visibility
nvidia-smi

# Test Docker GPU access
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

If either command fails, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 2. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/ThatMattCat/havencore.git
cd havencore

# Copy environment template
cp .env.tmpl .env
```

### 3. Essential Configuration

Edit the `.env` file with these **required** settings:

```bash
# Your Docker host IP address (find with: ip route get 1.1.1.1 | awk '{print $7}')
HOST_IP_ADDRESS="192.168.1.100"

# API access key (set to anything)
LLM_API_KEY="your_secret_key_here"

# GPU configuration
TTS_DEVICE="cuda:0"  # GPU for text-to-speech
STT_DEVICE="0"       # GPU for speech-to-text
```

### 4. Start the System

```bash
# Validate configuration
docker compose config --quiet

# Build and start all services (this takes 60-90 minutes on first run)
docker compose up -d

# Monitor startup progress
docker compose logs -f
```

⚠️ **Important**: The first build takes 60-90 minutes as it downloads AI models and builds containers. Do not cancel the process.

### 5. Verify Installation

Once startup completes (10-15 minutes), verify everything is working:

```bash
# Check service health
curl http://localhost/health

# Test text-to-speech
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, HavenCore is working!", "model": "tts-1", "voice": "alloy"}'

# Test chat completion
curl -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secret_key_here" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Web Interface Access

After successful startup, access these interfaces:

- **Agent Dashboard**: http://localhost (SvelteKit SPA — chat, devices, history, metrics, service playgrounds)
- **System Gateway Health**: http://localhost/health

## Next Steps

### Basic Configuration
1. **Home Assistant Integration**: Configure `HAOS_URL` and `HAOS_TOKEN` in `.env`
2. **API Keys**: Add weather, search, and computational service keys
3. **Location Settings**: Set your timezone, location, and zip code

### Advanced Setup
1. **Custom Models**: Configure different LLM backends
2. **Voice Activation**: Set up wake-word detection
3. **Security**: Configure proper authentication and access controls

### Troubleshooting First Startup

#### Services Won't Start
```bash
# Check Docker configuration
docker compose config --quiet

# Check specific service logs
docker compose logs [service_name]

# Check all service status
docker compose ps
```

#### Out of Memory Errors
- Reduce model size in `compose.yaml`
- Adjust `--gpu-memory-utilization` parameter
- Check available system resources: `nvidia-smi`

#### Build Failures
```bash
# Clean Docker cache
docker system prune -f

# Check disk space
df -h

# Retry build with verbose output
docker compose build --no-cache --progress=plain
```

#### Model Download Issues
```bash
# Pre-download models manually
huggingface-cli download QuantTrio/GLM-4.5-Air-AWQ-FP16Mix

# Check network connectivity
curl -I https://huggingface.co
```

## Configuration Deep Dive

For detailed configuration options, see:
- [Configuration Guide](configuration.md) - Complete environment variable reference
- [Home Assistant Integration](integrations/home-assistant.md) - Smart home setup
- [Media Control](integrations/media-control.md) - Plex + HA TV/media playback

## Architecture Overview

HavenCore consists of these core services:
- **nginx**: API gateway and load balancer
- **agent**: Main AI logic and tool calling
- **speech-to-text**: Audio transcription service
- **text-to-speech**: Audio generation service
- **postgres**: Database for conversation storage
- **vLLM/LlamaCPP**: LLM inference backend

For detailed architecture information, see [Architecture Overview](architecture.md).

## Development Setup

If you plan to contribute or modify HavenCore:

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up development environment**: See [Development Guide](development.md)
4. **Make changes** to Python files (live reload with mounted volumes)
5. **Test your changes** using the provided validation commands

## Getting Help

- **Configuration Issues**: [Configuration Guide](configuration.md)
- **Service Problems**: [Troubleshooting Guide](troubleshooting.md)
- **API Questions**: [API Reference](api-reference.md)
- **Development Help**: [Development Guide](development.md)

---

**Next**: Once you have HavenCore running, explore the [Configuration Guide](configuration.md) to customize your setup, or check out [Home Assistant Integration](integrations/home-assistant.md) to connect your smart home devices.