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
upstream tts_backend {
    server text-to-speech:6005;
}

upstream stt_backend {
    server speech-to-text:6001;
}

server {
    listen 80;

    # Re-resolve `agent` against Docker's embedded DNS on a 10s TTL so
    # nginx picks up the new container IP after `docker compose restart agent`
    # without needing nginx itself to be restarted.
    resolver 127.0.0.11 valid=10s ipv6=off;

    # Chat completions routing — variable indirection forces re-resolution.
    location /v1/chat/completions {
        set $agent_upstream "agent:6002";
        proxy_pass http://$agent_upstream;
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

The agent upstream is intentionally **not** declared as a static
`upstream agent_backend { server agent:6002; }` block. nginx resolves
hostnames in static upstream blocks once at startup and caches the IP
forever, so an `agent` restart leaves nginx proxying to a dead IP until
nginx itself restarts. The `resolver` directive plus a `set $var ...;
proxy_pass http://$var;` pattern forces nginx to re-resolve the
hostname on each request (cached for the resolver's `valid=` window).
This is applied to every agent location: `/v1/chat/completions`,
`/api/`, `/ws/`, and the SPA catch-all `/`.

## Satellite OTA firmware (`/firmware/`)

Nginx also static-serves satellite firmware blobs at `http://<host>/firmware/`,
on the same gateway port the satellite already uses for `/api/chat`,
`/v1/audio/transcriptions`, `/v1/audio/speech`, and `/api/status`. Plain HTTP,
no auth, no signing — LAN-only, matching the rest of the API surface.

| URL | Content-Type | Purpose |
|-----|--------------|---------|
| `GET /firmware/satellite.bin`  | `application/octet-stream` | Whole-file firmware blob streamed into `esp_https_ota` on the device |
| `GET /firmware/satellite.json` | `application/json`         | Optional version sidecar (`{"version", "size", "sha256"}`); satellite-side version-skip logic is future work |

Range requests aren't required (the device does whole-file GETs); `gzip` is
explicitly off (the firmware client doesn't decompress). Bare `/firmware/`
returns 403 — `autoindex` is disabled so the directory contents aren't
enumerable.

### Host-side layout

Blobs live on the host at:

```
./volumes/firmware/satellite.bin
./volumes/firmware/satellite.json   (optional)
```

…bind-mounted read-only into the nginx container at
`/usr/share/havencore/firmware`. The directory is owned by the host user
(uid 1000 in the default deployment); the `volumes/**/*` gitignore rule
keeps the blobs out of git.

### Uploading from a build host (the perms gotcha)

The nginx worker inside the container runs as `nginx` (uid 101). On the
bind mount that uid is "other" relative to the host owner, so files
arriving with mode `0600` will return **403 Forbidden** even though the
route is wired correctly. Files need to be at least `0644` (or
group-readable with the right group).

`scp` ships the source file's mode bits literally, so a build host that
writes its manifest under a restrictive umask will trip this on every
upload. Use `rsync --chmod` to normalize perms in flight regardless of
source mode:

```bash
rsync -av --chmod=F644 satellite.bin satellite.json \
    matt@<host>:~/code/havencore/volumes/firmware/
```

Plain `scp` works too if the publish script `chmod 644`s before transfer
or runs a remote `chmod 644` after. Pick whichever fits the build flow.

> **Note**: this is a single-tenant dev setup. In a multi-user
> deployment the route would need at least an allowlist or a token, and
> the upload path/owner would need to be parameterized rather than
> hard-coded to the user's home.

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
