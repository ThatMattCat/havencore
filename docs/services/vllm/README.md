# vLLM Backend

Primary LLM inference backend. Exposes an OpenAI-compatible API that the agent treats as `gpt-3.5-turbo`.

## Purpose

- High-performance LLM inference
- OpenAI-compatible API
- GPU-optimized processing
- Model serving and management

## Configuration

Located in `compose.yaml`. The image is pinned to a specific digest
(vLLM v0.19.0 — the last release verified against NVIDIA driver 580.x;
`:latest` has shipped builds requiring a newer host driver). The model is
served under the OpenAI-compat name `gpt-3.5-turbo` so stock OpenAI SDKs
work without reconfiguration.

```yaml
vllm:
  image: vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
  command: >
    --model QuantTrio/GLM-4.5-Air-AWQ-FP16Mix
    --served-model-name gpt-3.5-turbo
    --tensor-parallel-size 4
    --enable-expert-parallel
    --max-model-len 32768
    --max-num-seqs 2
    --gpu-memory-utilization 0.77
    --tool-call-parser glm45
    --reasoning-parser glm45
    --enable-auto-tool-choice
    --trust-remote-code
    --host 0.0.0.0
    --port 8000
```

GLM-4.5-Air is a MoE model (~106B total / ~12B active parameters) that
needs `--enable-expert-parallel` when sharded, and `--trust-remote-code`
for the HuggingFace modeling files. `--reasoning-parser glm45` splits the
model's `<think>…</think>` chain-of-thought into a separate `reasoning`
field on the response so `message.content` stays clean for voice
satellites; the agent surfaces that reasoning as a `REASONING` event on
`/ws/chat` (filtered out of `/api/chat`'s `events[]` and naturally absent
from `/v1/chat/completions`) and also normalizes it onto the assistant
message as `reasoning_content`. GLM-4.5-Air's
[`chat_template.jinja`](https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/chat_template.jinja)
reads that field and renders `<think>…</think>` only for assistant
messages newer than the most recent user message — i.e. between tool calls
within the current in-progress turn — so the model can see its own prior
reasoning before the next iteration. Older completed turns get an empty
`<think></think>` from the template regardless of what's stored.
`--tool-call-parser glm45` wires native function-calling on the same
model.

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

- **AWQ quantized** — optimized inference models
- **GPTQ** — alternative quantization format
- **Full precision** — unquantized models (high VRAM)

Popular model options:

```yaml
# Default — MoE reasoning model (~106B total / ~12B active), served across
# 4× 24 GB GPUs via tensor parallelism + expert parallelism
"QuantTrio/GLM-4.5-Air-AWQ-FP16Mix"

# Smaller non-reasoning alternatives (drop --reasoning-parser / --tool-call-parser glm45)
"Qwen/Qwen2.5-72B-Instruct-AWQ"        # prior default, 2× 24 GB via -tp 2
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
