# Text-to-Image Service

ComfyUI-based image generation. Accessed by the agent's MCP `generate_image` tool and by the dashboard's ComfyUI playground.

## Status

Documentation stub — the service is shipping but doesn't yet have a dedicated doc. Add details here (default workflows, model setup, VRAM requirements, prompt conventions) as the service stabilizes.

## Ports

- `8188` — ComfyUI HTTP/WS API

## Where to look in the meantime

- `services/text-to-image/` — service source and Dockerfile
- `compose.yaml` — runtime configuration
- [MCP General Tools → `generate_image`](../agent/tools/general.md) — how the agent calls this service
- [Agent Service](../agent/README.md) — the `/api/comfy/*` proxy endpoints for the dashboard playground
