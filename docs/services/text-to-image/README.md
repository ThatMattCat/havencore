# Text-to-Image Service

ComfyUI-based image generation. Accessed by the agent's MCP `generate_image` tool and by the dashboard's ComfyUI playground.

## Version

The Dockerfile pins ComfyUI to a release commit (currently **v0.37.4**) and torch to `2.11.0` (the last verified against the host's 580.x driver). Qwen-Image 2.1 needs ComfyUI ≥ 0.37.0; older pins lack the `TextEncodeQwenImage21` / `QwenImage21Cache` nodes and the built-in template. Bump the commit and rebuild (`docker compose build text-to-image && docker compose up -d text-to-image`) to pick up new model families.

`comfyui_controlnet_aux` in `custom_nodes/` has never imported (its own `requirements.txt` — `opencv-python` etc. — is not installed in the image); nothing in the agent's workflows uses it.

## Default model: Qwen-Image 2.1

The agent's `generate_image` tool runs the `qwen_image_2.1` workflow
(`services/agent/selene_agent/modules/mcp_general_tools/comfyui_workflows/qwen_image_2.1.json`):
a 7B single-stream DiT (Apache 2.0) that covers photoreal, anime, illustration, product and
text-in-image prompts from one set of weights. Files live under `services/text-to-image/models/`
(bind-mounted; ComfyUI rescans on demand, no restart needed):

| File | Folder | Size |
|---|---|---|
| `qwen_image_2.1_bf16.safetensors` | `diffusion_models/` | 14.2 GB |
| `qwen3vl_8b_int8_convrot.safetensors` | `text_encoders/` | 9.4 GB |
| `qwen_image_2.1_vae_bf16.safetensors` | `vae/` | 0.7 GB |

Source: [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). An `int8_convrot`
diffusion model (7.3 GB) and a `w4a8` encoder exist for tighter cards; the optional
`qwen3.5_9b_..._pe_t2i` prompt-enhancer encoder is **not** used — the chat LLM writes the prompt.

Graph: `UNETLoader` → `QwenImage21Cache` → `KSampler` (25 steps, cfg 1, euler/simple) with
`CLIPLoader(type=qwen_image)` → `TextEncodeQwenImage21` for conditioning. cfg 1 means the negative
prompt is inert. Sizes are multiples of 32: 1024² (`square`), 1216×832 (`landscape`/`portrait`),
1344×768 (`wide`/`tall`).

Measured on the reference host (RTX 3090, GPU 4 shared with STT/TTS/face/embeddings): first job after
a cold container ≈ 35 s including weight load; ComfyUI 0.37's dynamic VRAM loader stages both encoder
and DiT and fills spare VRAM, so GPU 4 peaks near 24 GB with every helper resident. That is the expected
steady state; `--reserve-vram <GiB>` on the compose `command` is the knob if a helper ever needs headroom.

Prompt convention: natural-language sentences, style named explicitly, any in-image text in double quotes
(tag lists were the SD 1.5 convention and work poorly here).

## Ports

- `8188` — ComfyUI HTTP/WS API

## GPU placement

The host GPU comes from `TEXT_TO_IMAGE_GPU` in `.env` (default `3`). Unlike the other helpers it is applied through Docker `device_ids` rather than `CUDA_VISIBLE_DEVICES` — ComfyUI still opened a CUDA context on host GPU 0 when pinned by the env var alone — so the container sees exactly one device, and the image's hardcoded `--default-device 3` is overridden to `0` via a compose `command`. With the default chat model, GPUs 0-3 belong to the `vllm` service, so the reference host sets this to `4` — see [configuration.md → GPU Settings](../../configuration.md#gpu-settings).

## Where to look in the meantime

- `services/text-to-image/` — service source, Dockerfile, `models/` mounts
- `compose.yaml` — runtime configuration
- [MCP General Tools → `generate_image`](../agent/tools/general.md) — how the agent calls this service
- [Agent Service](../agent/README.md) — the `/api/comfy/*` proxy endpoints for the dashboard playground
