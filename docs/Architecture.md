# Architecture Overview

HavenCore is built as a distributed microservices architecture using Docker containers. This design provides scalability, maintainability, and flexibility for different deployment scenarios.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Applications                     │
├─────────────────────────────────────────────────────────────────┤
│  Web UI (6002) │ Voice Clients │ API Clients │ Home Assistant  │
└─────────────────┬───────────────┬─────────────┬─────────────────┘
                  │               │             │
┌─────────────────┴───────────────┴─────────────┴─────────────────┐
│                      Nginx Gateway (80)                        │
├─────────────────────────────────────────────────────────────────┤
│                    Load Balancer & Router                      │
└─────┬─────────┬─────────────┬─────────────┬───────────────────┘
      │         │             │             │
┌─────▼─────┐ ┌─▼───────┐ ┌───▼─────┐ ┌─────▼─────┐
│   Agent   │ │   STT   │ │   TTS   │ │   Data    │
│  (6002)   │ │ (6000)  │ │ (6004)  │ │ Services  │
└─────┬─────┘ └─────────┘ └─────────┘ └─────┬─────┘
      │                                     │
┌─────▼─────────────────────────────────────▼─────┐
│              Backend Services                   │
├─────────────────────────────────────────────────┤
│ vLLM/LlamaCPP (8000) │ PostgreSQL │ Qdrant     │
└─────────────────────────────────────────────────┘
```

## Core Services

### 1. Nginx Gateway (Port 80)
**Purpose**: API Gateway and Load Balancer
- Routes external requests to appropriate services
- Provides SSL termination and rate limiting
- Handles CORS and request preprocessing
- Serves as single entry point for all client interactions

**Key Features**:
- OpenAI-compatible API routing
- Health check aggregation
- Request/response transformation
- Static file serving

### 2. Agent Service (Ports 6002, 6006)
**Purpose**: Main AI Logic and Tool Calling Engine
- Orchestrates conversation flow and context management
- Executes function/tool calling for external integrations
- Manages conversation history and user sessions
- Provides Gradio web interface for direct interaction

**Key Components**:
- **Selene Agent**: Core conversation engine
- **Tool Registry**: Unified tool management (legacy + MCP)
- **Conversation Database**: Session and history management
- **Integration Layer**: Home Assistant, web search, computational tools

**Architecture Pattern**: Event-driven with async tool execution

### 3. Speech-to-Text Service (Ports 6000, 6001, 5999)
**Purpose**: Audio Transcription and Processing
- Converts spoken audio to text using Whisper models
- Handles multiple audio formats and preprocessing
- Provides OpenAI-compatible transcription API
- Supports real-time and batch processing

**Model Pipeline**:
```
Audio Input → Preprocessing → Whisper Model → Text Output
     ↓              ↓             ↓            ↓
  Format Detection  Normalization  GPU Inference  Post-processing
```

### 4. Text-to-Speech Service (Ports 6003, 6004, 6005)
**Purpose**: Audio Generation and Voice Synthesis
- Converts text to natural speech using Kokoro TTS
- Provides multiple voice options and languages
- Serves generated audio files statically
- Offers both API and web interface access

**Service Endpoints**:
- **Port 6005**: OpenAI-compatible API
- **Port 6004**: Gradio web interface
- **Port 6003**: Static audio file server

### 5. PostgreSQL Database (Port 5432)
**Purpose**: Persistent Data Storage
- Conversation history and session management
- User preferences and configuration
- Tool execution logs and analytics
- System metrics and monitoring data

**Schema Design**:
```sql
conversation_histories (
    id, session_id, conversation_data,
    created_at, metadata
)
```

### 6. LLM Backend Services (Port 8000)
**Purpose**: Large Language Model Inference

#### vLLM Backend (Default)
- High-performance inference server
- Optimized for throughput and latency
- Supports AWQ quantized models
- GPU memory optimization

#### LlamaCPP Backend (Alternative)
- CPU-focused inference option
- Lower memory requirements
- GGUF model format support
- Suitable for resource-constrained environments

### 7. Vector Database (Qdrant - Port 6333)
**Purpose**: Embeddings and Semantic Search
- Document embedding storage
- Semantic search capabilities
- RAG (Retrieval Augmented Generation) support
- Context enhancement for conversations

### 8. Embeddings Service (Port 3000)
**Purpose**: Text Embedding Generation
- Converts text to vector representations
- Supports multiple embedding models
- Integration with vector database
- Semantic similarity computation

## Data Flow Architecture

### 1. Voice Interaction Flow
```
Voice Input → STT Service → Agent Service → LLM Backend
     ↓              ↓           ↓            ↓
Audio Processing  Transcription  Tool Calls  Text Generation
     ↓              ↓           ↓            ↓
TTS Service ← Response Text ← Tool Results ← LLM Response
     ↓
Audio Output
```

### 2. API Request Flow
```
Client Request → Nginx Gateway → Target Service
       ↓              ↓             ↓
   Authentication   Routing      Processing
       ↓              ↓             ↓
   Rate Limiting   Load Balance   Response
       ↓              ↓             ↓
Client Response ← Response ← Service Result
```

### 3. Tool Execution Flow
```
User Query → Agent → Tool Registry → Tool Implementation
     ↓         ↓          ↓               ↓
  Intent     Tool       Legacy/MCP      External API
Recognition  Selection   Routing        (HA/Search/etc)
     ↓         ↓          ↓               ↓
  Response ← Result ← Tool Response ← API Response
```

## Integration Architecture

### Home Assistant Integration
- **Direct API Connection**: RESTful API calls to Home Assistant
- **Entity State Management**: Real-time device status monitoring
- **Service Execution**: Command and control capabilities
- **Event Streaming**: Real-time updates and notifications

### External Service Integration
- **Web Search**: Brave Search API integration
- **Computational**: WolframAlpha API for complex queries
- **Weather**: WeatherAPI for location-based forecasts
- **Knowledge**: Wikipedia and other knowledge sources

### MCP (Model Context Protocol) Support
- **Unified Tool Registry**: Manages both legacy and MCP tools
- **Dynamic Tool Loading**: Runtime tool registration and discovery
- **Conflict Resolution**: Preference-based tool selection
- **Extensibility**: Plugin-like architecture for new capabilities

## Deployment Architecture

### Container Orchestration
```yaml
services:
  - nginx (reverse proxy)
  - agent (core logic)
  - speech-to-text (STT)
  - text-to-speech (TTS)
  - postgres (database)
  - vllm (LLM inference)
  - qdrant (vector DB)
  - embeddings (text vectors)
```

### Network Architecture
- **Internal Network**: Services communicate via Docker internal networking
- **External Access**: Only Nginx exposed publicly
- **Service Discovery**: Docker Compose DNS resolution
- **Health Checks**: Automated service health monitoring

### Storage Architecture
- **Persistent Volumes**: Database and model storage
- **Shared Volumes**: Configuration and temporary files
- **Model Cache**: Efficient model loading and caching
- **Audio Storage**: Temporary audio file management

## Scalability Considerations

### Horizontal Scaling
- **Load Balancing**: Nginx distributes requests across service instances
- **Service Replication**: Multiple instances of compute-heavy services
- **Database Clustering**: PostgreSQL read replicas for scaling reads
- **Cache Layers**: Redis for session and response caching

### Vertical Scaling
- **GPU Optimization**: Efficient GPU memory utilization
- **Model Optimization**: Quantization and optimization techniques
- **Resource Allocation**: Dynamic resource allocation per service
- **Memory Management**: Efficient model loading and unloading

## Security Architecture

### Authentication & Authorization
- **API Key Management**: Secure API key validation
- **Service-to-Service**: Internal authentication between services
- **Rate Limiting**: Request throttling and abuse prevention
- **SSL/TLS**: Encrypted communication (when configured)

### Network Security
- **Internal Networks**: Isolated service communication
- **Firewall Rules**: Restricted external access
- **Secret Management**: Secure credential storage
- **Audit Logging**: Comprehensive access and action logging

## Monitoring & Observability

### Logging Architecture
- **Centralized Logging**: Loki aggregation
- **Structured Logs**: JSON format with trace IDs
- **Log Levels**: Configurable verbosity per service
- **Correlation**: Request tracing across services

### Metrics Collection
- **Health Checks**: Service availability monitoring
- **Performance Metrics**: Response times and throughput
- **Resource Usage**: CPU, memory, and GPU utilization
- **Business Metrics**: Conversation counts and user interactions

### Debugging & Troubleshooting
- **Trace IDs**: Request correlation across services
- **Debug Endpoints**: Service status and configuration
- **Container Logs**: Centralized log aggregation
- **Health Dashboards**: Real-time system status

## Technology Stack

### Core Technologies
- **Containerization**: Docker & Docker Compose
- **Languages**: Python (services), JavaScript (some components)
- **Databases**: PostgreSQL (relational), Qdrant (vector)
- **Web Framework**: FastAPI (APIs), Gradio (interfaces)
- **Proxy**: Nginx (reverse proxy, load balancer)

### AI/ML Stack
- **LLM Inference**: vLLM, LlamaCPP
- **Speech-to-Text**: OpenAI Whisper
- **Text-to-Speech**: Kokoro TTS
- **Embeddings**: Various transformer models
- **GPU Compute**: CUDA, NVIDIA Container Toolkit

---

**Next Steps**: 
- [Service Documentation](Services.md) - Detailed service specifications
- [API Reference](API-Reference.md) - Complete API documentation
- [Deployment Guide](Deployment.md) - Production deployment considerations