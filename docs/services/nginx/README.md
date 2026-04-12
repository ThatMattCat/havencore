# Nginx Gateway

API gateway / reverse proxy in front of the agent, TTS, and STT services.

## Purpose

- API gateway and reverse proxy
- Load balancing across service instances
- SSL termination and CORS handling
- Request routing and transformation

## Configuration

**Location**: `services/nginx/nginx.conf`

```nginx
upstream agent_backend {
    server agent:6002;
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

## Features

- **Load Balancing**: Round-robin across service instances
- **Health Checks**: Automatic failover for unhealthy services
- **Rate Limiting**: Configurable request throttling
- **CORS Support**: Cross-origin request handling
- **SSL/TLS**: Encryption termination (when configured)

## Monitoring

```bash
# Check Nginx status
docker compose exec nginx nginx -t

# View access logs
docker compose logs nginx

# Test configuration
curl -I http://localhost/health
```

## Customization

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
