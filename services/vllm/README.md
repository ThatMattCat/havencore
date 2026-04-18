# vLLM

Primary LLM inference backend. OpenAI-compatible HTTP server the agent
talks to as `gpt-3.5-turbo`.

| | |
|---|---|
| **Port** | `8000` (HTTP, `/v1/*`) |
| **Health** | `curl http://localhost:8000/v1/models` |
| **Image** | `vllm/vllm-openai` pinned to v0.19.0 digest in `compose.yaml` |
| **Model** | `Qwen/Qwen2.5-72B-Instruct-AWQ` by default (~35 GB, needs ≥48 GB VRAM across 2 cards) |

## Key env / config

Command-line flags live in `compose.yaml` under `services.vllm.command`:
`--model`, `--served-model-name`, `--quantization`, `-tp` (tensor
parallel), `--max-model-len`, `--gpu-memory-utilization`.

The agent reads the endpoint from `LLM_API_BASE` in `.env`.

## More

- Deep dive: [../../docs/services/vllm/README.md](../../docs/services/vllm/README.md)
- Troubleshooting CUDA OOM / slow first load:
  [../../docs/troubleshooting.md](../../docs/troubleshooting.md#gpu-and-model-loading-issues)
- Swapping to a smaller model or to llama.cpp: see compose.yaml
  (the `llamacpp` service is present as a commented-out alternative).

This directory contains runtime bits (`app/` templates), not a build
context — the image is pulled from upstream.
