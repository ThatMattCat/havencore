# text-to-image (ComfyUI)

ComfyUI-backed image generation. Exposed as an MCP tool
(`generate_image`) via `mcp_general_tools`, and also available as a
dashboard playground at `/playgrounds/comfy`.

| | |
|---|---|
| **Port** | `8188` (ComfyUI web UI + API) |
| **Health** | `curl -I http://localhost:8188/` |
| **Build** | Local Dockerfile in this directory |

## Key volumes

Mounted from this directory so models and workflows survive rebuilds:

- `models/` — Stable Diffusion / Flux checkpoints, loras, VAEs
- `custom_nodes/` — ComfyUI-Manager is auto-installed by the
  `comfyui-manager-installer` service on first boot
- `input/` — source images for img2img workflows
- `output/` — generated images (also what MCP tool returns URLs into)

## Env

GPU selection is via standard NVIDIA env vars
(`NVIDIA_VISIBLE_DEVICES`). The MCP tool uses workflow JSON files under
`services/agent/selene_agent/modules/mcp_general_tools/comfyui_workflows/`.

## More

- Deep dive: [../../docs/services/text-to-image/README.md](../../docs/services/text-to-image/README.md)
- MCP tool integration:
  [../../docs/services/agent/tools/general.md](../../docs/services/agent/tools/general.md)
