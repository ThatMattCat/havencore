# vLLM

Primary LLM inference backend. OpenAI-compatible HTTP server the agent
talks to as `gpt-3.5-turbo`.

| | |
|---|---|
| **Port** | `8000` (HTTP, `/v1/*`) |
| **Health** | `curl http://localhost:8000/health` |
| **Image** | `vllm/vllm-openai` pinned to the `qwen38-cu129` special-release digest in `compose.yaml` |
| **Model** | `Qwen/Qwen3.8-27B` by default — dense hybrid-attention, unquantized BF16, multimodal (also the vision backend), 262k context, sharded across 4× 24 GB GPUs with `-tp 4`. `--reasoning-parser qwen3` splits CoT into a separate `reasoning` field so `message.content` stays clean for voice satellites. |

## Key env / config

Command-line flags live in `compose.yaml` under `services.vllm.command`:
`--model`, `--served-model-name`, `-tp` (tensor parallel),
`--max-model-len`, `--gpu-memory-utilization`.

The agent reads the endpoint from `LLM_API_BASE` in `.env`.

## More

- Deep dive: [../../docs/services/vllm/README.md](../../docs/services/vllm/README.md)
- Troubleshooting CUDA OOM / slow first load:
  [../../docs/troubleshooting.md](../../docs/troubleshooting.md#gpu-and-model-loading-issues)
- Swapping to a smaller model or to llama.cpp: see compose.yaml
  (the `llamacpp` service is present as a commented-out alternative).

This directory contains runtime bits (`app/` templates), not a build
context — the image is pulled from upstream.
