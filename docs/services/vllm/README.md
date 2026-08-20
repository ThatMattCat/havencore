# vLLM Backend

Primary LLM inference backend. Exposes an OpenAI-compatible API that the agent treats as `gpt-3.5-turbo`.

## Purpose

- High-performance LLM inference
- OpenAI-compatible API
- GPU-optimized processing
- Model serving and management

## Configuration

Located in `compose.yaml`. The image is pinned to the `qwen38-cu129`
special-release digest — the dedicated vLLM build with Qwen3.8
hybrid-attention support. The CUDA 12.9 (`cu129`) variant is deliberate:
the host driver is 580.x, and the default/`cu130` tags have shipped
builds requiring a newer driver. The model is served under the
OpenAI-compat name `gpt-3.5-turbo` so stock OpenAI SDKs work without
reconfiguration.

```yaml
vllm:
  image: vllm/vllm-openai@sha256:3879c346b0866fe8e1894243bb4fcdfc38f8dfcdb8a710a3b1d456ce4e799999
  command: >
    --model Qwen/Qwen3.8-27B
    --served-model-name gpt-3.5-turbo
    --tensor-parallel-size 4
    --language-model-only
    --max-model-len 262144
    --max-num-seqs 2
    --gpu-memory-utilization 0.80
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --host 0.0.0.0
    --port 8000
```

Qwen3.8-27B is a **dense hybrid-attention** model (48 of 64 layers are
Gated DeltaNet linear attention, 16 are full attention with 4 KV heads)
run unquantized in BF16 — ~53 GB of weights, ~13.3 GB/GPU at `-tp 4`.
It is *not* MoE, so `--enable-expert-parallel` does not apply (that flag
belonged to the previous GLM-4.5-Air default). The hybrid attention
keeps KV cost to ~64 KB/token, which is why the full native `262144`
context fits: ~16.8 GB of KV for one max-length sequence on top of the
weights, inside the `0.80` memory budget. The two `--max-num-seqs` slots
share that pool — both at full context won't fit simultaneously.
`--language-model-only` skips the model's vision tower (the separate
`vllm-vision` service covers images).

`--reasoning-parser qwen3` splits the model's `<think>…</think>`
chain-of-thought into a separate reasoning field on the response so
`message.content` stays clean for voice satellites; the agent surfaces
that reasoning as a `REASONING` event on `/ws/chat` (filtered out of
`/api/chat`'s `events[]` and naturally absent from
`/v1/chat/completions`) and also normalizes it onto the assistant
message as `reasoning_content`, which the chat template reads to render
`<think>…</think>` for the in-progress agentic turn. Note the reasoning
block spends the same completion budget as the visible answer — see
`LLM_MAX_TOKENS` in [configuration.md](../../configuration.md).
`--tool-call-parser qwen3_coder` wires native function-calling on the
same model.

Two quirks of Qwen3.8's chat template / vLLM's request schema the agent
compensates for at send time (`_messages_for_llm`): a system message
anywhere past index 0 raises a template error (mid-history retrieval /
recap blocks are re-tagged as user messages), and explicit
`"tool_calls": null` on assistant messages is rejected (None-valued
keys are stripped).

The healthcheck probes `GET /health` rather than `/v1/models`:
`/health` works with or without `--api-key` auth, so enabling the flag
later can't silently break the container's health status.

### Rollback

The previous known-good pairing is kept in `compose.yaml` comments:
vLLM v0.19.0
(`sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504`)
serving `QuantTrio/GLM-4.5-Air-AWQ-FP16Mix` (MoE, ~106B total / ~12B
active) with `--enable-expert-parallel --tool-call-parser glm45
--reasoning-parser glm45 --trust-remote-code`.

## Authentication

The serving endpoint currently enforces **no API key** — `--api-key` is
deliberately absent from the command (a client app in use mishandles
authenticated endpoints). The agent still sends `LLM_API_KEY` from
`.env` as a bearer token; vLLM ignores it. To turn enforcement on, add
`--api-key <value of LLM_API_KEY>` to the command block and recreate the
container — clients then must send `Authorization: Bearer <key>` on
every `/v1/*` call (`/health` stays open).

## Command-line options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model path | Required |
| `--gpu-memory-utilization` | GPU memory usage (0.0-1.0) | 0.9 |
| `--max-model-len` | Maximum sequence length | 4096 |
| `--dtype` | Data type (auto, float16, bfloat16) | auto |
| `--tensor-parallel-size` | Number of GPUs | 1 |
| `--api-key` | API authentication key | None |

## Supported models

- **Full precision** — unquantized models (current default; high VRAM)
- **AWQ quantized** — optimized inference models
- **GPTQ** — alternative quantization format

Popular model options:

```yaml
# Default — dense hybrid-attention model, BF16, 4× 24 GB GPUs via -tp 4
"Qwen/Qwen3.8-27B"

# Prior default — MoE reasoning model (~106B total / ~12B active); needs
# --enable-expert-parallel, --trust-remote-code and the glm45 parsers
"QuantTrio/GLM-4.5-Air-AWQ-FP16Mix"

# Smaller non-reasoning alternatives (drop --reasoning-parser / --tool-call-parser)
"Qwen/Qwen2.5-72B-Instruct-AWQ"        # 2× 24 GB via -tp 2
"Qwen/Qwen2.5-14B-Instruct-AWQ"
"microsoft/Phi-3-medium-4k-instruct"
```

## Performance tuning

```yaml
# Multi-GPU setup
command: [
  "--model", "your-model",
  "--tensor-parallel-size", "2",  # Use 2 GPUs
  "--gpu-memory-utilization", "0.8"
]

# Memory optimization
command: [
  "--model", "your-model",
  "--max-model-len", "16384",    # Reduce context length
  "--gpu-memory-utilization", "0.7"
]
```

## API endpoints

- `GET /v1/models` — list available models
- `POST /v1/chat/completions` — chat API
- `POST /v1/completions` — text completion
- `GET /health` — service health

## Monitoring

```bash
# Check model loading
docker compose logs -f vllm

# Test API directly
curl http://localhost:8000/v1/models

# Monitor GPU usage
nvidia-smi -l 1

# Check memory usage
docker compose exec vllm nvidia-smi
```
