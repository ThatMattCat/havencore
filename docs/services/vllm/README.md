# vLLM Backend

Primary LLM inference backend. Exposes an OpenAI-compatible API that the agent treats as `gpt-3.5-turbo`.

## Purpose

- High-performance LLM inference
- OpenAI-compatible API
- GPU-optimized processing
- Model serving and management

## Configuration

Located in `compose.yaml`:

```yaml
vllm:
  image: vllm/vllm-openai:latest
  command: [
    "--model", "TechxGenus/Mistral-Large-Instruct-2411-AWQ",
    "--gpu-memory-utilization", "0.9",
    "--max-model-len", "32768",
    "--dtype", "auto",
    "--api-key", "${DEV_CUSTOM_API_KEY}"
  ]
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
# High performance, lower memory
"TechxGenus/Mistral-Large-Instruct-2411-AWQ"

# Alternative options
"hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
"microsoft/Phi-3-mini-4k-instruct"
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
