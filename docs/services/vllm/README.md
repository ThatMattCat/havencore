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
    --model Qwen/Qwen2.5-72B-Instruct-AWQ
    --served-model-name gpt-3.5-turbo
    --quantization awq_marlin
    --max-num-seqs 1
    --enforce-eager
    -tp 2
    --max-model-len 16384
```

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
# Default — high quality, fits on a pair of 24GB GPUs with AWQ-Marlin
"Qwen/Qwen2.5-72B-Instruct-AWQ"

# Smaller alternatives
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
