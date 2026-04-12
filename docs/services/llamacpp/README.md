# LlamaCPP Backend

Alternative LLM backend using llama.cpp with GGUF models. Commented out in `compose.yaml` by default — enable if you don't have the GPU capacity to run vLLM.

## Purpose

- CPU-focused LLM inference
- Lower memory requirements
- GGUF model support
- Alternative to [vLLM](../vllm/README.md)

## Configuration

```yaml
llamacpp:
  build:
    context: ./services/llamacpp
  command: [
    "python", "-m", "llama_cpp.server",
    "--model", "/models/model.gguf",
    "--n_gpu_layers", "33",
    "--host", "0.0.0.0",
    "--port", "8000"
  ]
```

## Model format

Uses GGUF format models:

```bash
# Download GGUF model
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
  Phi-3-mini-4k-instruct-q4.gguf --local-dir ./models/
```

## Performance options

```yaml
# CPU-only inference
command: [
  "python", "-m", "llama_cpp.server",
  "--model", "/models/model.gguf",
  "--n_gpu_layers", "0",        # CPU only
  "--n_threads", "8"            # CPU threads
]

# GPU acceleration
command: [
  "python", "-m", "llama_cpp.server",
  "--model", "/models/model.gguf",
  "--n_gpu_layers", "33",       # GPU layers
  "--n_batch", "512"            # Batch size
]
```

## When to use LlamaCPP

- **Limited GPU memory** — less than 8 GB VRAM
- **CPU inference** — no GPU available
- **GGUF models** — specific model-format requirements
- **Resource constraints** — lower memory usage needed
