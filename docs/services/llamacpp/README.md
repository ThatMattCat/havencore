# LlamaCPP Backend

Alternative LLM backend using llama.cpp with GGUF models. Commented out in `compose.yaml` by default — enable if you don't have the GPU capacity to run vLLM.

## Purpose

- CPU-focused LLM inference
- Lower memory requirements
- GGUF model support
- Alternative to [vLLM](../vllm/README.md)

## Configuration

The stanza in `compose.yaml` is commented out — uncomment it (and comment
out `vllm`) to switch backends. It uses the official llama.cpp server
image and reads GGUF weights from `./services/llamacpp/models`:

```yaml
llamacpp:
  image: ghcr.io/ggml-org/llama.cpp:server-cuda-backup-20250816
  volumes:
    - ./services/llamacpp/models:/models
  ports:
    - "8000:8000"
  command: >
    -m /models/Qwen2.5-72B-Instruct-Q6_K/Qwen2.5-72B-Instruct-Q6_K-00001-of-00002.gguf
    -dev CUDA0,CUDA1,CUDA2
    --alias gpt-3.5-turbo
    -md /models/Qwen2.5-7B-Instruct-Q6_K_L.gguf
    -devd CUDA3
    --draft-max 64
    --batch-size 2048
    --ubatch-size 512
    -fa
    -sm layer
```

`--alias gpt-3.5-turbo` is the llama.cpp-server equivalent of vLLM's
`--served-model-name` — it's what lets OpenAI-SDK clients talk to this
backend unchanged.

## Model format

Uses GGUF files under `./services/llamacpp/models/`. Example download:

```bash
huggingface-cli download Qwen/Qwen2.5-72B-Instruct-GGUF \
  Qwen2.5-72B-Instruct-Q6_K-00001-of-00002.gguf \
  --local-dir ./services/llamacpp/models/Qwen2.5-72B-Instruct-Q6_K/
```

## When to use LlamaCPP

- **Limited GPU memory** — less than 8 GB VRAM
- **CPU inference** — no GPU available
- **GGUF models** — specific model-format requirements
- **Resource constraints** — lower memory usage needed
