# Text-to-Image Service

ComfyUI-based image generation. Accessed by the agent's MCP `generate_image` tool and by the dashboard's ComfyUI playground.

## Status

Documentation stub — the service is shipping but doesn't yet have a dedicated doc. Add details here (default workflows, model setup, VRAM requirements, prompt conventions) as the service stabilizes.

## Ports

- `8188` — ComfyUI HTTP/WS API

## GPU placement

The host GPU comes from `TEXT_TO_IMAGE_GPU` in `.env` (default `3`). Unlike the other helpers it is applied through Docker `device_ids` rather than `CUDA_VISIBLE_DEVICES` — ComfyUI still opened a CUDA context on host GPU 0 when pinned by the env var alone — so the container sees exactly one device, and the image's hardcoded `--default-device 3` is overridden to `0` via a compose `command`. With the default chat model, GPUs 0-3 belong to the `vllm` service, so the reference host sets this to `4` — see [configuration.md → GPU Settings](../../configuration.md#gpu-settings).

## Where to look in the meantime

- `services/text-to-image/` — service source and Dockerfile
- `compose.yaml` — runtime configuration
- [MCP General Tools → `generate_image`](../agent/tools/general.md) — how the agent calls this service
- [Agent Service](../agent/README.md) — the `/api/comfy/*` proxy endpoints for the dashboard playground
