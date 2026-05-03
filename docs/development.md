# Development Guide

This guide covers development setup, contributing to HavenCore, and extending the system with custom functionality.

## Development Environment Setup

### Prerequisites

#### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Docker**: Version 20.10+ with Docker Compose V2
- **Git**: For version control
- **Python**: 3.8+ for local development and testing
- **Node.js**: 16+ (optional, for MCP server development)

#### Development Tools (Recommended)
- **IDE**: VS Code with Docker and Python extensions
- **Terminal**: Modern terminal with Docker command completion
- **GPU Tools**: NVIDIA Container Toolkit for GPU development
- **API Testing**: Postman, HTTPie, or curl

### Project Structure Overview

```
havencore/
├── .env.example              # Environment template
├── compose.yaml              # Docker orchestration
├── docs/                     # Documentation (this wiki)
├── services/                 # All microservices
│   ├── agent/               # Main AI agent (Python package: selene-agent)
│   │   ├── selene_agent/    # Python package source
│   │   │   ├── selene_agent.py          # FastAPI app entry point
│   │   │   ├── orchestrator.py          # Event-based agent loop
│   │   │   ├── api/                     # REST/WS routers
│   │   │   ├── modules/                 # Bundled MCP server modules (11 servers, 68 tools)
│   │   │   │   ├── mcp_general_tools/
│   │   │   │   ├── mcp_homeassistant_tools/
│   │   │   │   ├── mcp_face_tools/
│   │   │   │   ├── mcp_vision_tools/
│   │   │   │   ├── mcp_device_action_tools/
│   │   │   │   ├── mcp_github_tools/
│   │   │   │   ├── mcp_plex_tools/
│   │   │   │   ├── mcp_music_assistant_tools/
│   │   │   │   ├── mcp_qdrant_tools/
│   │   │   │   ├── mcp_reminder_tools/
│   │   │   │   └── mcp_mqtt_tools/
│   │   │   ├── autonomy/                # Background engine (engine, turn, schedule, gating, notifiers)
│   │   │   └── utils/                   # Config, MCP client, DB
│   │   ├── frontend/        # SvelteKit SPA (built into the image)
│   │   ├── pyproject.toml
│   │   └── Dockerfile
│   ├── nginx/               # API gateway
│   ├── postgres/            # Database initialization
│   ├── speech-to-text/      # STT service (Faster-Whisper)
│   ├── text-to-speech/      # TTS service (Kokoro)
│   ├── text-to-image/       # ComfyUI
│   ├── vllm/                # Chat LLM backend
│   ├── vllm-vision/         # Vision LLM backend
│   ├── face-recognition/    # InsightFace identity service
│   ├── qdrant/              # Vector DB
│   └── embeddings/          # text-embeddings-inference
└── shared/                  # Shared utilities
    ├── configs/             # Common configuration
    └── libs/                # Logger, trace IDs
```

### Initial Setup

#### 1. Fork and Clone
```bash
# Fork the repository on GitHub first
git clone https://github.com/YOUR_USERNAME/havencore.git
cd havencore

# Add upstream remote
git remote add upstream https://github.com/ThatMattCat/havencore.git
```

#### 2. Development Environment
```bash
# Copy environment template
cp .env.example .env

# Configure for development
# Edit .env with development settings:
DEBUG_LOGGING=1
HOST_IP_ADDRESS="127.0.0.1"
LLM_API_KEY="dev123"
```

#### 3. Development Build
```bash
# Validate configuration
docker compose config --quiet

# Build development environment
docker compose build

# Start core services for development
docker compose up -d postgres qdrant nginx
```

### Development Workflow

#### Live Development with Volume Mounts

The development setup mounts the `selene_agent` package into the agent
container so Python edits land without a rebuild. See the `agent` service
entry in `compose.yaml` for the authoritative list of mounts.

After editing Python files, restart the container to pick up the change:

```bash
docker compose restart agent
```

The SvelteKit dashboard is a compiled SPA baked into the image; UI changes
require a rebuild (`docker compose build agent`).

#### Making Changes

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** to the relevant service code

3. **Test your changes**:
```bash
# Restart the affected service
docker compose restart agent

# Test functionality
curl http://localhost/health
```

4. **Commit your changes**:
```bash
git add .
git commit -m "Add: descriptive commit message"
```

#### Testing Changes

##### Individual Service Testing
```bash
# Test agent service
docker compose exec agent python -c "
from selene_agent import selene_agent
print('Agent module loaded successfully')
"

# Test MCP client manager import
docker compose exec agent python -c "
from selene_agent.utils.mcp_client_manager import MCPClientManager
print('MCP client manager imported successfully')
"

# Run service health checks
curl http://localhost:6002/health
curl http://localhost:6005/health
```

##### Integration Testing
```bash
# Test full API workflow
curl -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev123" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test audio APIs
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Test message", "voice": "alloy"}' \
  --output test.wav
```

---

## Contributing Guidelines

### Code Standards

#### Python Code Style
Follow PEP 8 guidelines:

```python
# Good example
class ToolRegistry:
    """Registry for managing AI tools and functions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.tools = {}
        self.config = config
    
    def register_tool(self, name: str, function: Callable) -> None:
        """Register a new tool with the registry."""
        if not name or not callable(function):
            raise ValueError("Invalid tool name or function")
        
        self.tools[name] = function
        logger.info(f"Registered tool: {name}")
```

#### Documentation Standards
- **Docstrings**: Required for all public functions and classes
- **Type Hints**: Use Python type hints for function parameters and returns
- **Comments**: Explain complex logic and decisions
- **README Updates**: Update service READMEs when adding features

#### Error Handling
```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def safe_api_call(url: str, timeout: int = 30) -> Optional[dict]:
    """Make a safe API call with proper error handling."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection failed for {url}: {e}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling {url}: {e}")
        return None
```

### Commit Guidelines

#### Commit Message Format
```
Type: Brief description of changes

Optional longer description explaining the changes in more detail.
Include motivation for the changes and contrast with previous behavior.

Fixes #123
```

#### Commit Types
- **Add**: New features or capabilities
- **Fix**: Bug fixes
- **Update**: Improvements to existing features
- **Refactor**: Code restructuring without behavior changes
- **Docs**: Documentation updates
- **Test**: Adding or updating tests
- **Style**: Code formatting and style changes

#### Examples
```bash
git commit -m "Add: Home Assistant media player control tool

Implements media player control functionality including play, pause, 
stop, and volume control. Integrates with unified tool registry.

Fixes #45"

git commit -m "Fix: Memory leak in conversation history storage

Resolved issue where conversation objects weren't being properly 
cleaned up after storage to PostgreSQL.

Fixes #67"
```

### Pull Request Process

#### 1. Pre-submission Checklist
- [ ] Code follows project style guidelines
- [ ] All existing tests still pass
- [ ] New functionality includes appropriate documentation
- [ ] Changes are backward compatible (or breaking changes documented)
- [ ] Self-review completed

#### 2. PR Description Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Local testing completed
- [ ] Integration tests pass
- [ ] Manual verification of changes

## Screenshots (if applicable)
Include screenshots of UI changes or new features.

## Additional Notes
Any additional information for reviewers.
```

#### 3. Review Process
- **Automated checks**: Ensure all CI checks pass
- **Code review**: Address reviewer feedback
- **Testing**: Verify changes work as expected
- **Documentation**: Update relevant documentation

---

## Extending HavenCore

### Adding New Tools

All tool surface is delivered by MCP servers bundled inside the agent
package at `services/agent/selene_agent/modules/mcp_*`. New tools
either belong in an existing server (e.g., adding an HA tool to
`mcp_homeassistant_tools`) or in a new server module alongside them.

For the authoring workflow — module layout, tool-definition pattern,
configuration, logging, and testing — see
[Tool Development](services/agent/tools/development.md).

### Adding New Services

#### 1. Service Structure

Create a new service directory:
```bash
mkdir -p services/my-service/app
cd services/my-service
```

Create the service files:
```bash
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ .
CMD ["python", "main.py"]

# requirements.txt
fastapi==0.68.0
uvicorn==0.15.0

# app/main.py
from fastapi import FastAPI

app = FastAPI(title="My Service")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "My Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

#### 2. Add to Docker Compose

Edit `compose.yaml`:
```yaml
services:
  # Existing services...
  
  my-service:
    build:
      context: ./services/my-service
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    env_file:
      - .env
    volumes:
      - ./services/my-service/app:/app
      - ./shared:/app/shared:ro
    depends_on:
      - postgres
    restart: unless-stopped
```

#### 3. Update Nginx Configuration

Add routing in `services/nginx/nginx.conf`:
```nginx
upstream my_service_backend {
    server my-service:8001;
}

server {
    # Existing configuration...
    
    location /my-service/ {
        proxy_pass http://my_service_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### MCP Server Development

HavenCore's in-tree MCP servers are Python modules under
`services/agent/selene_agent/modules/`. Each exposes a stdio server and is
spawned by the agent's `MCPClientManager` per the `MCP_SERVERS` JSON in
`.env`. See [Tool Development](services/agent/tools/development.md) for
the full authoring guide and
[Agent tools overview](services/agent/tools/README.md) for examples of
existing modules.

---

## Testing Framework

### Unit Testing

Create test files in the service directories:

```python
# services/agent/tests/test_example.py
import pytest
from unittest.mock import patch

from selene_agent.modules.mcp_general_tools import mcp_server  # example import

def test_example():
    assert True
```

Run the agent's pytest suite inside the agent container so imports and env resolve correctly (the suite is `pytest-asyncio` with `asyncio_mode=auto`):

```bash
docker compose exec -T agent pytest
```

### Integration Testing

```python
# tests/integration/test_api.py
import requests
import pytest

class TestAPIIntegration:
    def test_health_endpoint(self):
        """Test system health endpoint."""
        response = requests.get("http://localhost/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_chat_completion(self):
        """Test chat completion API."""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }
        headers = {"Authorization": "Bearer dev123"}
        
        response = requests.post(
            "http://localhost/v1/chat/completions",
            json=payload,
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
```

### Performance Testing

```python
# tests/performance/test_load.py
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

def test_concurrent_requests():
    """Test system under concurrent load."""
    def make_request():
        response = requests.get("http://localhost/health")
        return response.status_code == 200
    
    # Test 10 concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [future.result() for future in futures]
    
    # All requests should succeed
    assert all(results)

def test_response_time():
    """Test API response time."""
    start_time = time.time()
    response = requests.get("http://localhost/health")
    end_time = time.time()
    
    assert response.status_code == 200
    assert (end_time - start_time) < 1.0  # Response within 1 second
```

---

## Debugging and Development Tools

### Local Development Setup

#### Python Virtual Environment
```bash
# Create virtual environment for local development
cd services/agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the agent package in editable mode (pulls deps from pyproject.toml)
pip install -e .

# Test imports
python -c "from selene_agent.utils.mcp_client_manager import MCPClientManager; print('OK')"
```

### Debugging Techniques

#### Service Debugging
```bash
# Debug container with shell access
docker compose exec agent bash

# Python debugging inside container
docker compose exec agent python -c "
import pdb; pdb.set_trace()
# Your debugging code here
"

# Check environment variables
docker compose exec agent env | grep -E "(API_KEY|HOST_IP)"

# Test specific modules
docker compose exec agent python -c "
from selene_agent import selene_agent
print('Agent module imported successfully')
"
```

#### API Debugging
```bash
# Verbose API testing
curl -v -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev123" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}'

# Check specific service endpoints
curl -I http://localhost:6002/health
curl -I http://localhost:6005/health
curl -I http://localhost:8000/v1/models
```

#### Log Analysis
```bash
# Real-time log monitoring
docker compose logs -f agent | grep ERROR

# Search for specific patterns
docker compose logs agent | grep -i "tool.*error"

# Export logs for analysis
docker compose logs agent > agent_logs.txt
```

### Development Utilities

#### Helper Scripts

Create `scripts/dev-helpers.sh`:
```bash
#!/bin/bash

# Quick development commands
dev_restart_agent() {
    docker compose restart agent
    echo "Agent restarted"
}

dev_test_api() {
    curl -s http://localhost/health | jq .
}

dev_logs() {
    docker compose logs -f ${1:-agent}
}

dev_shell() {
    docker compose exec ${1:-agent} bash
}

# Usage: source scripts/dev-helpers.sh
```

#### Code Quality Tools

```bash
# Install development tools
pip install black flake8 mypy pytest

# Format code
black services/agent/selene_agent/

# Lint code
flake8 services/agent/selene_agent/

# Type checking
mypy services/agent/selene_agent/

# Run tests (inside the agent container so imports and env resolve)
docker compose exec -T agent pytest
```

---

## Release and Deployment

### Version Management

#### Semantic Versioning
- **Major** (X.0.0): Breaking changes
- **Minor** (1.X.0): New features, backward compatible
- **Patch** (1.1.X): Bug fixes, backward compatible

#### Release Process
1. **Update version** in relevant files
2. **Create release branch**: `git checkout -b release/v1.2.0`
3. **Update CHANGELOG**: Document changes
4. **Test thoroughly**: Run full test suite
5. **Create pull request**: For review
6. **Tag release**: `git tag v1.2.0`
7. **Merge to main**: After approval

### Deployment Considerations

#### Production Configuration
```bash
# .env.prod
DEBUG_LOGGING=0
HOST_IP_ADDRESS="your.production.ip"
LLM_API_KEY="secure_production_key"
```

#### Docker Image Management
```bash
# Build production images
docker compose -f compose.prod.yaml build

# Tag for registry
docker tag havencore_agent:latest registry.example.com/havencore_agent:v1.2.0

# Push to registry
docker push registry.example.com/havencore_agent:v1.2.0
```

---

**Next Steps**:
- [Tool Development](services/agent/tools/development.md) - Deep dive into tool creation
- [Agent tools overview](services/agent/tools/README.md) - Tool inventory and per-module docs