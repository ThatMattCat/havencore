# Troubleshooting Guide

This guide covers common issues, solutions, and debugging techniques for HavenCore. Issues are organized by category and severity.

## Quick Diagnostics

### System Health Check
Run these commands first to assess overall system status:

```bash
# Overall health check
curl http://localhost/health

# Service status
docker compose ps

# Recent logs
docker compose logs --tail=50

# Resource usage
docker stats --no-stream

# GPU status (if using GPU services)
nvidia-smi
```

### Emergency Recovery
If the system is completely unresponsive:

```bash
# Stop all services
docker compose down

# Clean up resources
docker system prune -f

# Restart with fresh state
docker compose up -d

# Monitor startup
docker compose logs -f
```

---

## Installation and Startup Issues

### Build Failures

#### Symptom: Docker build fails with disk space errors
```
ERROR: failed to solve: write /var/lib/docker/...: no space left on device
```

**Solution**:
```bash
# Check disk space
df -h

# Clean Docker cache
docker system prune -a -f

# Remove unused volumes
docker volume prune -f

# Check Docker root directory size
du -sh /var/lib/docker/
```

#### Symptom: Build times out or gets stuck
```
Building vllm... (hanging)
```

**Solution**:
```bash
# Increase build timeout
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Build with verbose output
docker compose build --no-cache --progress=plain

# Build single service
docker compose build --no-cache text-to-speech
```

#### Symptom: GPU not available during build
```
RuntimeError: No CUDA devices found
```

**Solution**:
```bash
# Check NVIDIA drivers
nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

### Model Download Issues

#### Symptom: HuggingFace model download fails
```
HTTPError: 401 Client Error: Unauthorized for url
```

**Solution**:
```bash
# Set HuggingFace token
export HF_HUB_TOKEN="your_token_here"

# Pre-download models manually
huggingface-cli login
huggingface-cli download TechxGenus/Mistral-Large-Instruct-2411-AWQ

# Check token in container
docker compose exec agent env | grep HF_HUB_TOKEN
```

#### Symptom: Model download slow or times out
```
Downloading... (very slow or hanging)
```

**Solution**:
```bash
# Use different mirror
export HF_ENDPOINT="https://hf-mirror.com"

# Download with resume capability
huggingface-cli download --resume-download TechxGenus/Mistral-Large-Instruct-2411-AWQ

# Check network connectivity
curl -I https://huggingface.co
```

### Service Startup Issues

#### Symptom: Services fail to start or crash immediately
```bash
# Check service status
docker compose ps
```

**Common causes and solutions**:

1. **Port conflicts**:
```bash
# Check what's using ports
sudo netstat -tlnp | grep :80
sudo netstat -tlnp | grep :6002

# Kill conflicting processes
sudo kill -9 <PID>
```

2. **Missing environment variables**:
```bash
# Validate .env file
cat .env | grep -E "HOST_IP_ADDRESS|DEV_CUSTOM_API_KEY"

# Check environment in container
docker compose exec agent env | grep HOST_IP_ADDRESS
```

3. **Database connection issues**:
```bash
# Test database connectivity
docker compose exec postgres psql -U havencore -d havencore -c "SELECT 1;"

# Check database logs
docker compose logs postgres
```

---

## Runtime Issues

### API Errors

#### Symptom: 401 Unauthorized errors
```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "authentication_error"
  }
}
```

**Solution**:
```bash
# Check API key configuration
grep DEV_CUSTOM_API_KEY .env

# Test with correct key
curl -H "Authorization: Bearer your_api_key" http://localhost/health

# Restart services to pick up new API key
docker compose restart agent
```

#### Symptom: 500 Internal Server Error
```json
{
  "error": {
    "message": "Internal server error",
    "type": "internal_error"
  }
}
```

**Solution**:
```bash
# Check service logs for errors
docker compose logs agent --tail=100

# Look for specific error patterns
docker compose logs agent | grep -i error

# Check backend service health
curl http://localhost:8000/v1/models
```

#### Symptom: Requests timing out
```
Request timeout after 30s
```

**Solution**:
```bash
# Check service response times
time curl http://localhost/health

# Monitor resource usage
docker stats

# Check for memory/GPU issues
nvidia-smi
free -h

# Increase timeout in nginx.conf
proxy_read_timeout 300s;
proxy_connect_timeout 300s;
```

### Audio Processing Issues

#### Symptom: Speech-to-text fails with audio format errors
```
Error: Unsupported audio format
```

**Solution**:
```bash
# Check audio file format
file audio.wav

# Convert to supported format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Test with simple WAV file
curl -X POST http://localhost/v1/audio/transcriptions \
  -F "file=@test.wav" \
  -F "model=whisper-1"
```

#### Symptom: Text-to-speech produces no audio or garbled output
```
Response: binary data but no sound
```

**Solution**:
```bash
# Check TTS service logs
docker compose logs text-to-speech

# Test with simple input
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "hello", "voice": "alloy"}' \
  --output test.wav

# Verify audio file
file test.wav
```

#### Symptom: GPU out of memory for audio processing
```
CUDA out of memory
```

**Solution**:
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size in service configuration
# Edit compose.yaml:
environment:
  - BATCH_SIZE=1
  - MAX_CONCURRENT=2

# Restart affected services
docker compose restart speech-to-text text-to-speech
```

### LLM Backend Issues

#### Symptom: vLLM fails to load model
```
RuntimeError: Failed to load model
```

**Solution**:
```bash
# Check available GPU memory
nvidia-smi

# Reduce GPU memory utilization
# Edit compose.yaml:
command: [
  "--model", "TechxGenus/Mistral-Large-Instruct-2411-AWQ",
  "--gpu-memory-utilization", "0.7",  # Reduced from 0.9
  "--max-model-len", "16384"          # Reduced context length
]

# Check model exists
docker compose exec vllm ls -la /root/.cache/huggingface/

# Try smaller model
command: ["--model", "microsoft/Phi-3-mini-4k-instruct"]
```

#### Symptom: LLM responses are slow or timing out
```
Request timeout waiting for LLM response
```

**Solution**:
```bash
# Check LLM service directly
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}]}'

# Monitor GPU utilization
nvidia-smi -l 1

# Adjust model parameters for speed
command: [
  "--model", "your-model",
  "--max-model-len", "8192",     # Shorter context
  "--gpu-memory-utilization", "0.9",
  "--disable-log-stats"          # Reduce logging overhead
]
```

### Database Issues

#### Symptom: Database connection refused
```
psql: error: connection to server on socket failed: Connection refused
```

**Solution**:
```bash
# Check PostgreSQL service status
docker compose ps postgres

# Check database logs
docker compose logs postgres

# Test connection from within network
docker compose exec agent nc -zv postgres 5432

# Restart database service
docker compose restart postgres
```

#### Symptom: Database disk full
```
ERROR: could not extend file: No space left on device
```

**Solution**:
```bash
# Check disk usage
df -h

# Check database size
docker compose exec postgres psql -U havencore -d havencore -c "
SELECT pg_size_pretty(pg_database_size('havencore'));
"

# Clean old conversation history
docker compose exec postgres psql -U havencore -d havencore -c "
DELETE FROM conversation_histories 
WHERE created_at < NOW() - INTERVAL '30 days';
"

# Vacuum database
docker compose exec postgres psql -U havencore -d havencore -c "VACUUM FULL;"
```

---

## Performance Issues

### High Memory Usage

#### Symptom: System running out of memory
```bash
# Check memory usage
free -h
docker stats --no-stream
```

**Solution**:
```bash
# Add memory limits to services
# Edit compose.yaml:
services:
  agent:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

# Optimize model memory usage
command: [
  "--model", "your-model",
  "--gpu-memory-utilization", "0.6",  # Reduce GPU memory
  "--max-model-len", "8192"           # Reduce context length
]

# Enable swap if needed (temporary solution)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### High CPU Usage

#### Symptom: High CPU utilization causing slowdowns
```bash
# Check CPU usage
top
htop
docker stats
```

**Solution**:
```bash
# Limit CPU usage for services
# Edit compose.yaml:
services:
  speech-to-text:
    deploy:
      resources:
        limits:
          cpus: '2.0'

# Use CPU-optimized settings
# For LlamaCPP:
command: [
  "python", "-m", "llama_cpp.server",
  "--model", "/models/model.gguf",
  "--n_threads", "4",        # Limit threads
  "--n_gpu_layers", "0"      # Use CPU only
]
```

### Slow Response Times

#### Symptom: API responses taking too long
```bash
# Measure response times
time curl http://localhost/v1/chat/completions
```

**Solution**:
```bash
# Enable request caching
# Add Redis for caching (optional)
redis:
  image: redis:alpine
  ports:
    - "6379:6379"

# Optimize model inference
command: [
  "--model", "your-model",
  "--tensor-parallel-size", "2",  # Use multiple GPUs
  "--max-num-seqs", "256"         # Increase batch size
]

# Use faster model
command: ["--model", "microsoft/Phi-3-mini-4k-instruct"]
```

---

## Integration Issues

### Home Assistant Integration

#### Symptom: Cannot connect to Home Assistant
```
Error: Failed to connect to Home Assistant API
```

**Solution**:
```bash
# Test Home Assistant connectivity
curl -H "Authorization: Bearer YOUR_TOKEN" https://your-ha-url/api/

# Check environment variables
grep HAOS .env

# Verify SSL certificate (if using HTTPS)
curl -k -H "Authorization: Bearer YOUR_TOKEN" https://your-ha-url/api/

# Update .env with correct values
HAOS_URL="https://homeassistant.local:8123/api"
HAOS_TOKEN="your_long_lived_token"
```

#### Symptom: Home Assistant commands not working
```
Tool execution failed: home_assistant.execute_service
```

**Solution**:
```bash
# Check Home Assistant logs
docker compose logs agent | grep -i "home_assistant"

# Test specific Home Assistant service
curl -X POST "https://your-ha-url/api/services/light/turn_on" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "light.living_room"}'

# Verify entity names in Home Assistant
curl -H "Authorization: Bearer YOUR_TOKEN" https://your-ha-url/api/states
```

### External API Issues

#### Symptom: Weather API not working
```
Error: Weather API key invalid
```

**Solution**:
```bash
# Test weather API directly
curl "http://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q=London"

# Check API key in environment
grep WEATHER_API_KEY .env

# Verify API key is active
# Login to weatherapi.com and check dashboard
```

#### Symptom: Brave Search API failing
```
Error: Brave Search request failed
```

**Solution**:
```bash
# Test Brave Search API
curl -H "X-Subscription-Token: YOUR_TOKEN" \
  "https://api.search.brave.com/res/v1/web/search?q=test"

# Check API key configuration
grep BRAVE_SEARCH_API_KEY .env

# Verify API quota
# Check Brave Search API dashboard
```

---

## Network and Connectivity Issues

### Port Conflicts

#### Symptom: Port already in use
```
Error: bind: address already in use
```

**Solution**:
```bash
# Find what's using the port
sudo netstat -tlnp | grep :80
sudo lsof -i :80

# Kill the conflicting process
sudo kill -9 <PID>

# Use different ports
# Edit compose.yaml:
ports:
  - "8080:80"  # Use port 8080 instead of 80
```

### DNS Resolution Issues

#### Symptom: Services can't reach each other
```
Error: Name resolution failed for service 'postgres'
```

**Solution**:
```bash
# Check Docker network
docker network ls
docker network inspect havencore_default

# Test service connectivity
docker compose exec agent nslookup postgres
docker compose exec agent nc -zv postgres 5432

# Recreate network
docker compose down
docker compose up -d
```

### Firewall Issues

#### Symptom: External access blocked
```
curl: (7) Failed to connect to localhost port 80: Connection refused
```

**Solution**:
```bash
# Check firewall status
sudo ufw status

# Allow necessary ports
sudo ufw allow 80
sudo ufw allow 6002

# Check iptables rules
sudo iptables -L

# Temporarily disable firewall for testing
sudo ufw disable
```

---

## Docker and Container Issues

### Container Crashes

#### Symptom: Services keep restarting
```bash
# Check container status
docker compose ps

# View restart count
docker inspect $(docker compose ps -q agent) | grep RestartCount
```

**Solution**:
```bash
# Check container logs for crash reason
docker compose logs agent --tail=100

# Common crash causes:
# 1. Out of memory - add memory limits
# 2. GPU issues - check nvidia-docker setup
# 3. Configuration errors - validate .env file
# 4. Missing dependencies - rebuild containers

# Force rebuild if needed
docker compose build --no-cache agent
docker compose up -d agent
```

### Volume Mount Issues

#### Symptom: Code changes not reflected
```
Changes to Python files not taking effect
```

**Solution**:
```bash
# Verify volume mounts
docker compose config | grep -A5 volumes

# Check file permissions
ls -la services/agent/app/

# Restart service to pick up changes
docker compose restart agent

# If using Windows/WSL, check line endings
dos2unix services/agent/app/*.py
```

### Image Pull Issues

#### Symptom: Cannot pull Docker images
```
Error: pull access denied for image
```

**Solution**:
```bash
# Check Docker Hub connectivity
curl -I https://registry-1.docker.io

# Login to Docker Hub if needed
docker login

# Use alternative registry
# Edit compose.yaml to use different base images

# Build locally instead of pulling
docker compose build --no-cache
```

---

## Debugging Tools and Techniques

### Logging and Monitoring

#### Enable Debug Logging
```bash
# Add to .env file
DEBUG_LOGGING=1

# Restart services
docker compose restart

# View debug logs
docker compose logs agent | grep DEBUG
```

#### Centralized Logging with Loki
```bash
# Check Loki connectivity
curl http://localhost:3100/ready

# Configure Loki URL in .env
LOKI_URL="http://localhost:3100/loki/api/v1/push"

# View logs in Grafana (if configured)
# http://localhost:3000
```

### Performance Profiling

#### GPU Monitoring
```bash
# Continuous GPU monitoring
nvidia-smi -l 1

# GPU memory timeline
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv -l 1

# Per-process GPU usage
nvidia-smi pmon -i 0
```

#### Memory Profiling
```bash
# Container memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# System memory
watch -n 1 free -h

# Memory breakdown by process
top -o %MEM
```

### Network Debugging

#### Internal Service Communication
```bash
# Test service connectivity
docker compose exec agent nc -zv postgres 5432
docker compose exec agent nc -zv text-to-speech 6005

# Network inspection
docker network inspect $(docker compose config --services | head -1)_default

# DNS resolution testing
docker compose exec agent nslookup postgres
```

#### External API Testing
```bash
# Test external connectivity from containers
docker compose exec agent curl -I https://api.weatherapi.com
docker compose exec agent curl -I https://huggingface.co

# Test with verbose output
docker compose exec agent curl -v https://api.openai.com
```

### Configuration Validation

#### Docker Compose Validation
```bash
# Validate compose file syntax
docker compose config --quiet

# Show resolved configuration
docker compose config

# Check for warnings
docker compose config 2>&1 | grep -i warning
```

#### Environment Variable Validation
```bash
# Check all environment variables
docker compose exec agent env | sort

# Validate specific configurations
docker compose exec agent python -c "
import os
print('API Key:', os.getenv('DEV_CUSTOM_API_KEY'))
print('Host IP:', os.getenv('HOST_IP_ADDRESS'))
print('Debug:', os.getenv('DEBUG_LOGGING'))
"
```

---

## Recovery Procedures

### Complete System Recovery

If the system is severely damaged or corrupted:

```bash
# 1. Stop all services
docker compose down

# 2. Remove all containers and volumes
docker compose down -v
docker system prune -a -f

# 3. Clean up everything
docker volume prune -f
docker network prune -f

# 4. Rebuild from scratch
docker compose build --no-cache
docker compose up -d

# 5. Monitor startup
docker compose logs -f
```

### Data Recovery

#### Database Recovery
```bash
# If you have backups
docker compose exec -T postgres psql -U havencore -d havencore < backup.sql

# If no backups, recreate schema
docker compose exec postgres psql -U havencore -d havencore -f /docker-entrypoint-initdb.d/init.sql
```

#### Model Recovery
```bash
# Re-download models
docker compose exec agent huggingface-cli download TechxGenus/Mistral-Large-Instruct-2411-AWQ

# Clear model cache and restart
docker compose exec vllm rm -rf /root/.cache/huggingface/
docker compose restart vllm
```

### Partial Service Recovery

#### Single Service Issues
```bash
# Restart individual service
docker compose restart agent

# Rebuild and restart
docker compose build --no-cache agent
docker compose up -d agent

# View service-specific logs
docker compose logs -f agent
```

---

## Getting Help

### Before Seeking Help

1. **Check logs**: `docker compose logs [service_name]`
2. **Verify configuration**: `docker compose config --quiet`
3. **Test basics**: `curl http://localhost/health`
4. **Check resources**: `nvidia-smi` and `free -h`
5. **Review this guide**: Many issues are covered above

### Information to Provide

When reporting issues, include:

- **System information**: OS, Docker version, GPU model
- **Configuration**: Relevant parts of `.env` and `compose.yaml`
- **Error messages**: Complete error logs
- **Steps to reproduce**: What you did before the error
- **Resource usage**: Output from `nvidia-smi` and `docker stats`

### Community Resources

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community help
- **Documentation**: Always check the latest wiki pages

---

**Next Steps**:
- [Performance Tuning](Performance.md) - Optimize your setup
- [Development Guide](Development.md) - Modify and extend HavenCore
- [Configuration Guide](Configuration.md) - Advanced configuration options