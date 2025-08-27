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
├── .env.tmpl                 # Environment template
├── compose.yaml              # Docker orchestration
├── docs/                     # Documentation (this wiki)
├── services/                 # All microservices
│   ├── agent/               # Main AI agent
│   │   ├── app/             # Python application code
│   │   │   ├── selene_agent.py          # Core agent logic
│   │   │   ├── conversation_db.py       # Database interface
│   │   │   └── utils/                   # Utilities and tools
│   │   │       ├── haos/               # Home Assistant integration
│   │   │       ├── mcp_client_manager.py
│   │   │       ├── unified_tool_registry.py
│   │   │       └── *_tools_defs.py     # Tool definitions
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── nginx/               # API gateway
│   ├── postgres/            # Database initialization
│   ├── speech-to-text/      # STT service
│   ├── text-to-speech/      # TTS service
│   └── vllm/                # LLM backend config
└── shared/                  # Shared utilities
    ├── configs/             # Common configuration
    └── scripts/             # Shared utilities
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
cp .env.tmpl .env

# Configure for development
# Edit .env with development settings:
DEBUG_LOGGING=1
HOST_IP_ADDRESS="127.0.0.1"
DEV_CUSTOM_API_KEY="dev123"
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

The development setup uses volume mounts for live code reloading:

```yaml
# In compose.yaml
volumes:
  - ./services/agent/app:/app              # Live code reload
  - ./shared:/app/shared:ro                # Shared configuration
```

This means changes to Python files are immediately reflected in the running containers.

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
import selene_agent
print('Agent module loaded successfully')
"

# Test imports
docker compose exec agent python -c "
from utils.unified_tool_registry import UnifiedToolRegistry
print('Tool registry imported successfully')
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

#### 1. Legacy Tool Development

Create a new tool in `services/agent/app/utils/`:

```python
# services/agent/app/utils/custom_tools.py

import requests
from typing import Dict, Any, Optional

def get_stock_price(symbol: str, api_key: str) -> Dict[str, Any]:
    """Get current stock price for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        api_key: API key for stock service
        
    Returns:
        Dict containing stock price information
    """
    try:
        url = f"https://api.example.com/quote/{symbol}"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return {
            "symbol": symbol,
            "price": data.get("price"),
            "change": data.get("change"),
            "timestamp": data.get("timestamp")
        }
    except Exception as e:
        return {"error": f"Failed to get stock price: {str(e)}"}
```

#### 2. Tool Definition

Add tool definition in `services/agent/app/utils/general_tools_defs.py`:

```python
# Tool definition for the stock price function
stock_price_tool = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get current stock price for a given stock symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., AAPL, GOOGL, TSLA)"
                }
            },
            "required": ["symbol"]
        }
    }
}
```

#### 3. Register the Tool

In `services/agent/app/selene_agent.py`:

```python
from utils.custom_tools import get_stock_price

class SeleneAgent:
    def _setup_tool_functions(self) -> Dict[str, callable]:
        return {
            # Existing tools...
            'get_stock_price': lambda symbol: get_stock_price(
                symbol, 
                os.getenv('STOCK_API_KEY', '')
            ),
        }
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        return [
            # Existing tools...
            stock_price_tool,
        ]
```

#### 4. Environment Configuration

Add to `.env.tmpl`:
```bash
# Stock API Configuration
STOCK_API_KEY=""  # API key for stock price service
```

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

#### 1. Create MCP Server

Create a new MCP server using Node.js:
```bash
mkdir -p mcp-servers/my-server
cd mcp-servers/my-server

# package.json
{
  "name": "my-mcp-server",
  "version": "1.0.0",
  "type": "module",
  "dependencies": {
    "@modelcontextprotocol/sdk": "latest"
  }
}

npm install
```

#### 2. Implement MCP Server

```javascript
// server.js
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server(
  {
    name: "my-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Define tools
server.setRequestHandler("tools/list", async () => {
  return {
    tools: [
      {
        name: "my_tool",
        description: "A custom tool that does something useful",
        inputSchema: {
          type: "object",
          properties: {
            input: {
              type: "string",
              description: "Input for the tool"
            }
          },
          required: ["input"]
        }
      }
    ]
  };
});

// Handle tool calls
server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;
  
  if (name === "my_tool") {
    // Implement your tool logic here
    return {
      content: [
        {
          type: "text",
          text: `Processed: ${args.input}`
        }
      ]
    };
  }
  
  throw new Error(`Unknown tool: ${name}`);
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("My MCP Server running on stdio");
```

#### 3. Configure MCP Integration

Add to `.env`:
```bash
MCP_ENABLED=true
MCP_SERVERS='[
  {
    "name": "my-server",
    "command": "node",
    "args": ["mcp-servers/my-server/server.js"],
    "enabled": true
  }
]'
```

---

## Testing Framework

### Unit Testing

Create test files in the service directories:

```python
# services/agent/app/tests/test_tools.py
import pytest
from unittest.mock import Mock, patch
from utils.custom_tools import get_stock_price

class TestCustomTools:
    def test_get_stock_price_success(self):
        """Test successful stock price retrieval."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "price": 150.00,
                "change": 2.50,
                "timestamp": "2024-01-15T10:30:00Z"
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = get_stock_price("AAPL", "test_key")
            
            assert result["symbol"] == "AAPL"
            assert result["price"] == 150.00
            assert result["change"] == 2.50
    
    def test_get_stock_price_error(self):
        """Test error handling in stock price retrieval."""
        with patch('requests.get', side_effect=Exception("API Error")):
            result = get_stock_price("INVALID", "test_key")
            
            assert "error" in result
            assert "Failed to get stock price" in result["error"]

# Run tests
# docker compose exec agent python -m pytest tests/
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
cd services/agent/app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH="/path/to/havencore:$PYTHONPATH"

# Test imports
python -c "from utils.unified_tool_registry import UnifiedToolRegistry; print('OK')"
```

#### IDE Configuration

**VS Code Settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./services/agent/app/venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "docker.defaultRegistryPath": "",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
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
from selene_agent import SeleneAgent
agent = SeleneAgent()
print('Agent initialized successfully')
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
black services/agent/app/

# Lint code
flake8 services/agent/app/

# Type checking
mypy services/agent/app/

# Run tests
pytest services/agent/app/tests/
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
DEV_CUSTOM_API_KEY="secure_production_key"
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
- [Tool Development](Tool-Development.md) - Deep dive into tool creation
- [MCP Integration](MCP-Integration.md) - Model Context Protocol details
- [Performance Tuning](Performance.md) - Optimization techniques