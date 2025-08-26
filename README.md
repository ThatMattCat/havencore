# HavenCore

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL%20v2.1-blue.svg)](https://github.com/ThatMattCat/havencore/blob/main/LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Required-blue.svg)](https://www.docker.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU%20Required-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Work in Progress](https://img.shields.io/badge/Status-Work%20in%20Progress-orange.svg)](https://github.com/ThatMattCat/havencore)

> **Self-Hosted AI Smart Home Assistant with Voice Control**

A comprehensive AI-powered smart home system designed to run entirely on your own hardware. HavenCore provides voice-activated control, natural language processing, and integrates with popular smart home platforms like Home Assistant.

> ⚠️ **Work in Progress**: This project currently works well for the creator but may require configuration adjustments for other environments. Documentation and templates are being improved for broader compatibility.

## 🚀 Key Features

- 🎤 **Voice Activation**: Wake-word detection and real-time speech processing
- 🗣️ **Natural Conversations**: Advanced LLM-powered responses with tool calling
- 🏠 **Smart Home Integration**: Direct Home Assistant control and monitoring  
- 🔊 **High-Quality TTS**: Kokoro TTS for natural-sounding voice responses
- 📡 **OpenAI-Compatible APIs**: Drop-in replacement for OpenAI services
- 🔍 **Web Search & Knowledge**: Brave Search and WolframAlpha integration
- 🐳 **Self-Contained**: Fully containerized with Docker Compose
- 💻 **Hardware Flexible**: Support for multiple GPU configurations and LLM backends

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

For experienced users who want to get started immediately:

```bash
# Clone the repository
git clone https://github.com/ThatMattCat/havencore.git
cd havencore

# Copy and configure environment
cp .env.tmpl .env
# Edit .env with your specific settings

# Start the services
docker compose up -d

# Access the web interface
open http://localhost:6002  # Agent UI
```

## 🏗️ Architecture

HavenCore is built as a microservices architecture using Docker containers. Each service handles a specific aspect of the AI assistant functionality:

### Core Services

| Service | Port | Purpose | API Endpoints |
|---------|------|---------|---------------|
| **Nginx** | 80 | API Gateway & Load Balancer | Routes to other services |
| **Agent** | 6002, 6006 | LLM Logic & Tool Calling | `/v1/chat/completions` |
| **Speech-to-Text** | 6000, 6001, 5999 | Audio Transcription | `/v1/audio/transcriptions` |
| **Text-to-Speech** | 6003, 6004, 6005 | Audio Generation | `/v1/audio/speech` |
| **PostgreSQL** | 5432 | Database & Conversation Storage | N/A (internal) |
| **vLLM** | 8000 | LLM Inference Backend | OpenAI-compatible API |
| **LlamaCPP** | 8000* | Alternative LLM Backend | OpenAI-compatible API |

*Only one LLM backend runs at a time

### Data Flow

1. **Voice Input**: Edge device captures wake word and audio
2. **Transcription**: Speech-to-text converts audio to text
3. **Processing**: Agent processes text and determines actions
4. **Tool Execution**: Agent calls appropriate tools (Home Assistant, web search, etc.)
5. **Response Generation**: LLM generates natural language response
6. **Audio Output**: Text-to-speech converts response to audio
7. **Playback**: Edge device plays audio response

### Supported Integrations

- **🏠 Home Assistant** - Device control and state monitoring
- **🔍 Brave Search** - Web search capabilities  
- **🧮 WolframAlpha** - Computational queries and factual data
- **🌤️ Weather API** - Weather forecasts and conditions
- **📊 Grafana Loki** - Centralized logging (optional)

## 📋 Prerequisites

### Hardware Requirements

- **NVIDIA GPU**: At least one CUDA-compatible GPU (RTX 3090 or better recommended)
- **RAM**: 16GB+ system RAM (32GB+ recommended for larger models)
- **Storage**: 50GB+ free space for models and data
- **Network**: Stable internet connection for initial model downloads

### Software Requirements

- **Operating System**: Ubuntu 22.04 LTS (tested) or compatible Linux distribution
- **Docker**: Version 20.10+ with Docker Compose V2
- **NVIDIA Container Toolkit**: For GPU support in containers
- **Git**: For cloning the repository

### Optional Hardware

- **Edge Device**: For voice activation (see [HavenCore Edge](https://github.com/ThatMattCat/havencore-edge))
- **ESP32-Box-3**: Alternative edge device with built-in OpenAI integration

### Installation of Prerequisites

#### Install Docker and Docker Compose
```bash
# Remove old Docker versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

#### Install NVIDIA Container Toolkit
```bash
# Configure production repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ThatMattCat/havencore.git
cd havencore
```

### 2. Configure Environment Variables
```bash
# Copy the template
cp .env.tmpl .env

# Edit the configuration file
nano .env  # or your preferred editor
```

### 3. Key Configuration Items

Edit the `.env` file with your specific settings:

| Variable | Description | Example |
|----------|-------------|---------|
| `HOST_IP_ADDRESS` | IP address of your Docker host | `192.168.1.100` |
| `AGENT_NAME` | Name for your AI assistant | `Selene` |
| `HAOS_URL` | Home Assistant URL | `https://homeassistant.local:8123/api` |
| `HAOS_TOKEN` | Home Assistant long-lived token | `eyJ0eXAiOiJKV1QiLCJhbGc...` |
| `WEATHER_API_KEY` | WeatherAPI.com API key | `abc123def456...` |
| `BRAVE_SEARCH_API_KEY` | Brave Search API key (optional) | `BSA...` |
| `WOLFRAM_ALPHA_API_KEY` | WolframAlpha API key (optional) | `ABC123...` |

### 4. Start the Services
```bash
# Start all services in detached mode
docker compose up -d

# View logs to monitor startup
docker compose logs -f

# Check service health
docker compose ps
```

### 5. Verify Installation

Once all services are running, you can verify the installation:

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
  -H "Authorization: Bearer 1234" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## ⚙️ Configuration

### LLM Backend Selection

HavenCore supports multiple LLM backends. Choose one by modifying `compose.yaml`:

#### Option 1: vLLM (Default)
- Best for high-throughput inference
- Supports tensor parallelism across multiple GPUs
- Current model: `TechxGenus/Mistral-Large-Instruct-2411-AWQ`

#### Option 2: LlamaCPP
- More memory efficient
- Better for smaller deployments
- Supports draft model speculative decoding

To switch backends, comment out the unwanted service in `compose.yaml`.

### GPU Configuration

Edit these environment variables in `.env` to configure GPU usage:

```env
TTS_DEVICE="cuda:0"  # GPU for text-to-speech
STT_DEVICE="0"       # GPU for speech-to-text
```

For multi-GPU setups, modify the LLM service configuration in `compose.yaml`:
- vLLM: Adjust the `-tp` parameter for tensor parallelism
- LlamaCPP: Modify the `-dev` parameter for device selection

### Service-Specific Configuration

#### Home Assistant Integration
1. Generate a long-lived access token in Home Assistant:
   - Profile → Security → Long-lived access tokens
2. Add the token to your `.env` file:
   ```env
   HAOS_URL="https://your-homeassistant.local:8123/api"
   HAOS_TOKEN="your_long_lived_token_here"
   ```

#### External API Keys
Configure optional services by adding API keys to `.env`:

```env
# Weather forecasts
WEATHER_API_KEY="your_weatherapi_key"

# Web search capabilities
BRAVE_SEARCH_API_KEY="your_brave_search_key"

# Computational queries
WOLFRAM_ALPHA_API_KEY="your_wolfram_key"
```

## 📱 Usage

### Web Interface Access

- **Agent Interface**: http://localhost:6002 - Interactive chat interface
- **Text-to-Speech**: http://localhost:6004 - TTS testing interface
- **System Health**: http://localhost - Nginx status page

### Edge Device Integration

#### ESP32-Box-3 Setup
1. Flash with Espressif's ChatGPT example code
2. Configure WiFi and endpoint:
   ```
   API Endpoint: http://YOUR_HOST_IP/v1/
   API Key: 1234 (or your configured key)
   ```

#### Custom Edge Device
Use the [HavenCore Edge](https://github.com/ThatMattCat/havencore-edge) project for Raspberry Pi integration.

### Voice Interaction Workflow

1. **Wake Word**: Edge device detects activation phrase
2. **Recording**: Captures audio until silence detection
3. **Transcription**: Converts speech to text via Whisper
4. **Processing**: Agent analyzes request and executes tools
5. **Response**: Generates natural language response
6. **Synthesis**: Converts text to speech via Kokoro TTS
7. **Playback**: Edge device plays audio response

### Example Voice Commands

- *"Turn on the living room lights"*
- *"What's the weather like tomorrow?"*
- *"Set the thermostat to 72 degrees"*
- *"What time is it in Tokyo?"*
- *"Search for local restaurants"*

## 📚 API Reference

HavenCore provides OpenAI-compatible APIs accessible through the Nginx gateway.

### Chat Completions

**Endpoint**: `POST /v1/chat/completions`

```bash
curl -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Turn on the kitchen lights"}
    ]
  }'
```

### Speech Synthesis

**Endpoint**: `POST /v1/audio/speech`

```bash
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test message",
    "model": "tts-1",
    "voice": "alloy",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Audio Transcription

**Endpoint**: `POST /v1/audio/transcriptions`

```bash
curl -X POST http://localhost/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

### Service Health Checks

Individual service health endpoints:

- Agent: `GET http://localhost:6002/`
- Text-to-Speech: `GET http://localhost:6005/health`
- vLLM: `GET http://localhost:8000/v1/models`

## 🔧 Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check Docker and NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

# Verify GPU access
nvidia-smi

# Check service logs
docker compose logs [service_name]
```

#### Out of Memory Errors
- Reduce model size in vLLM configuration
- Adjust `--gpu-memory-utilization` parameter
- Switch to LlamaCPP backend for lower memory usage

#### Model Download Issues
```bash
# Pre-download models manually
huggingface-cli download TechxGenus/Mistral-Large-Instruct-2411-AWQ

# Check Hugging Face token
echo $HF_HUB_TOKEN
```

#### Audio Device Problems
- Ensure audio devices are properly configured on edge device
- Check firewall settings for port access
- Verify network connectivity between edge device and host

#### Home Assistant Connection
```bash
# Test Home Assistant API access
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     YOUR_HAOS_URL/states
```

### Logging and Monitoring

#### View Service Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f agent

# With timestamps
docker compose logs -f -t
```

#### Resource Monitoring
```bash
# GPU usage
nvidia-smi -l 1

# Container resources
docker stats

# Disk usage
docker system df
```

### Performance Optimization

#### GPU Memory Management
- Monitor GPU memory usage with `nvidia-smi`
- Adjust `--gpu-memory-utilization` in vLLM config
- Consider model quantization for lower memory usage

#### Response Time Optimization
- Use faster models for real-time interaction
- Enable model caching
- Optimize network latency to edge devices

## 🤝 Contributing

We welcome contributions to HavenCore! Here's how you can help:

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/havencore.git
   cd havencore
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Areas for Contribution

- 📝 **Documentation**: Improve setup guides and API documentation
- 🐛 **Bug Fixes**: Fix issues and improve stability
- ✨ **Features**: Add new integrations and capabilities
- 🧪 **Testing**: Add automated tests and validation
- 🎨 **UI/UX**: Improve web interfaces and user experience
- 🔧 **DevOps**: Improve deployment and configuration management

### Coding Standards

- Follow Python PEP 8 style guidelines
- Add docstrings to new functions and classes
- Include tests for new functionality
- Update documentation for changes

### Submitting Changes

1. Test your changes thoroughly
2. Update documentation if needed
3. Commit with clear, descriptive messages
4. Push to your fork and create a pull request

### Reporting Issues

When reporting bugs, please include:
- System specifications (OS, GPU, RAM)
- Docker and NVIDIA runtime versions
- Complete error logs
- Steps to reproduce the issue

## 📄 License

This project is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1). See the [LICENSE](LICENSE) file for details.

### Key Points

- ✅ **Use**: Free to use for personal and commercial projects
- ✅ **Modify**: Modify the source code for your needs
- ✅ **Distribute**: Share your modifications under the same license
- ⚠️ **Copyleft**: Derivative works must use LGPL-compatible licenses

## 🙏 Acknowledgments

- **[Kokoro TTS](https://github.com/hexgrad/kokoro)** - High-quality text-to-speech synthesis
- **[Faster Whisper](https://github.com/SYSTRAN/faster-whisper)** - Efficient speech recognition
- **[vLLM](https://github.com/vllm-project/vllm)** - High-performance LLM inference
- **[Home Assistant](https://www.home-assistant.io/)** - Open-source smart home platform
- **[Mistral AI](https://mistral.ai/)** - Advanced language models

---

## 📞 Support

- 📚 **Documentation**: Check this README and service-specific docs
- 🐛 **Issues**: [GitHub Issues](https://github.com/ThatMattCat/havencore/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ThatMattCat/havencore/discussions)
- 🔗 **Edge Device**: [HavenCore Edge Repository](https://github.com/ThatMattCat/havencore-edge)

**Made with ❤️ for the self-hosting and smart home communities**
