# iav-to-text (vision LLM)

Image / audio / video → text service. A second vLLM instance with a
multimodal model, plus a Gradio UI for manual testing. Surfaced to the
agent via the `query_multimodal_ai` MCP tool — used, for example, to
describe camera snapshots returned by Home Assistant.

| | |
|---|---|
| **Ports** | `8100` (OpenAI-compat API), `8110` (Gradio UI) |
| **Health** | `curl -f http://localhost:8100/v1/models` |
| **Build** | Local Dockerfile in this directory |

## Key env

- `CUDA_VISIBLE_DEVICES` — pinned to `2` in `compose.yaml` by default so
  it doesn't fight vLLM for GPU memory. Change if your topology differs.

## More

- Deep dive: [../../docs/services/iav-to-text/README.md](../../docs/services/iav-to-text/README.md)
- Tool usage (`query_multimodal_ai`, camera snapshot flow):
  [../../docs/services/agent/tools/general.md](../../docs/services/agent/tools/general.md)
- GPU contention troubleshooting:
  [../../docs/troubleshooting.md](../../docs/troubleshooting.md#gpu-and-model-loading-issues)
