# vLLM Backend

Primary LLM inference backend. Exposes an OpenAI-compatible API that the agent treats as `gpt-3.5-turbo`.

## Purpose

- High-performance LLM inference
- OpenAI-compatible API
- GPU-optimized processing
- Model serving and management

## Configuration

Located in `compose.yaml`. The image is pinned to the
`qwen38-flash-next-x86_64-cu130` vendor digest — the vLLM dev build with
`qwen4_exp` (Qwen3.8-Flash-Next) support, including the PLE
n-gram-table CPU offload. It is a CUDA 13.0 build; the host driver
(580.x) supports CUDA 13.0. The model is served under the OpenAI-compat
name `gpt-3.5-turbo` so stock OpenAI SDKs work without reconfiguration.

```yaml
vllm:
  image: vllm/vllm-openai@sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3
  environment:
    - VLLM_PLE_CPU_OFFLOAD=1
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    - TORCH_COMPILE_DISABLE=1
    - CUDA_VISIBLE_DEVICES=0,1,2,3
    - VLLM_USE_V2_MODEL_RUNNER=0
  cap_add: [SYS_PTRACE]
  security_opt: ["seccomp=unconfined"]
  init: true
  command: >
    --model VnimanieAI/Qwen3.8-Flash-Next-W4A16
    --served-model-name gpt-3.5-turbo
    --tensor-parallel-size 4
    --enable-expert-parallel
    --max-model-len 98304
    --max-num-seqs 64
    --gpu-memory-utilization 0.94
    --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --trust-remote-code
    --host 0.0.0.0
    --port 8000
```

### Model: Qwen3.8-Flash-Next (W4A16)

`VnimanieAI/Qwen3.8-Flash-Next-W4A16` is an INT4 quant
(compressed-tensors, W4A16, group size 128) of Qwen3.8-Flash-Next — a
**MoE** model with 512 experts / 10 active, a ~125B backbone plus 51B
of "PLE" n-gram lookup tables. It is 168 GB on disk and ~18.2 GiB of
GPU-resident weights per card at `-tp 4`. The official FP8 checkpoint
cannot run on RTX 3090s (SM86 has no FP8), which is why the INT4 quant
is the one we serve.

Flag-by-flag:

- `--enable-expert-parallel` is **mandatory**: `moe_intermediate_size`
  is 640, which does not divide into 128-wide quant groups under plain
  tensor parallelism.
- `--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'`
  — inductor compile of this architecture hangs on Ampere, so
  compilation is off, but decode CUDA graphs are kept (they do not need
  inductor).
- `--max-model-len 98304` / `--max-num-seqs 64` — 4× 24 GB fits weights
  plus ~98k of context *or* weights plus the MTP draft head, not both.
  MTP speculative decoding is therefore off; re-enable it only if the
  context window is dropped. fp8 KV cache is rejected by the model
  ("QSA requires a BF16 main KV cache").
- `--gpu-memory-utilization 0.94` — see the GPU layout section below
  for why the budget has to be this high.
- `--trust-remote-code` — required for the `qwen4_exp` architecture.

Environment and container settings:

- `VLLM_PLE_CPU_OFFLOAD=1` keeps the ~102 GB BF16 PLE tables in host RAM
  instead of VRAM. Needs at least 110 GB of free host RAM (this host has
  251 GB).
- `VLLM_USE_V2_MODEL_RUNNER=0` — **required on this host.** With the
  default v2 model runner every boot deadlocked right after CUDA-graph
  capture (all TP workers at 100% CPU, GPUs idle; with `torch.compile`
  disabled it surfaced as `queue.Full` in `PleOffloadConnector._launch`).
  The v2 runner stages PLE inputs from CUDA tensors through a
  device-to-host copy that waits on an event recorded on the model
  stream, and that handoff races here. The v1 runner feeds the offload
  worker from its CPU input mirrors and boots cleanly.
  `CUDA_LAUNCH_BLOCKING=1` also cures it, but at ~40% of the decode
  speed.
- `TORCH_COMPILE_DISABLE=1` — the TP>1 vocab-masking helper is
  `torch.compile`-decorated regardless of `--compilation-config`, and its
  inductor kernel load hung. Disabling dynamo globally is what actually
  keeps `torch.compile` out of the way.
- `CUDA_VISIBLE_DEVICES=0,1,2,3` — GPUs 0-3 belong to this service;
  GPU 4 is hidden so neither the TP ranks nor the PLE offload worker
  touch it.
- `cap_add: SYS_PTRACE` + `security_opt: seccomp=unconfined` — the PLE
  offload worker hands GPU buffers between processes with the
  `pidfd_getfd` syscall, which Docker's default seccomp profile blocks.
- `init: true` — a hung `vllm serve` otherwise becomes an unkillable
  zombie PID 1 that `docker compose stop` refuses to reap.

### GPU layout and sizing

GPUs 0-3 are **dedicated** to this service. Every other GPU consumer —
`vllm-vision`, `speech-to-text`, `text-to-speech`, `embeddings`,
`face-recognition`, `text-to-image` — is pinned to GPU 4 via the
`*_GPU` / `STT_DEVICE` vars in `.env` (see
[configuration.md → GPU Settings](../../configuration.md#gpu-settings)).

The reason is KV cache. KV cost is ~14.7 KB/token/GPU and does **not**
shrink with tensor parallelism (the model has 2 KV heads, which
replicate once there is less than one head per rank), so a single 98k
request needs 1.24 GiB of KV per GPU. When the helper services shared
GPUs 0-3 the budget capped at 0.88 and only ~0.5 GiB was left for KV
(~32k of context); `--language-model-only` saved nothing (0.47 GiB),
and 0.90 was refused outright on a card shared with Whisper (20.9 GiB
free < 21.2 GiB requested).

Measured at the current settings:

| Metric | Value |
|--------|-------|
| GPU memory per card | ~22.6 GB |
| Host RAM (PLE tables) | ~102 GB |
| KV capacity at 0.94 | 121,048 tokens (1.23x concurrency at 98k) |
| Cold start | ~5 min total (~90 s model load + ~60 s PLE table load) |
| Decode | ~84 tok/s on a short prompt, ~92 tok/s at 31k context |
| Prefill | ~1.7k tok/s at 31k (6.4k-token prompt: TTFT 6.5 s) |
| Warm TTFT | ~0.3 s |

### Parsers and chat-template quirks

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
same model. Both parsers are unchanged from the previous Qwen3.8-27B
pairing, so the switch needed no agent-side code change.

Two quirks of Qwen3.8's chat template / vLLM's request schema the agent
compensates for at send time (`_messages_for_llm`): a system message
anywhere past index 0 raises a template error (mid-history retrieval /
recap blocks are re-tagged as user messages), and explicit
`"tool_calls": null` on assistant messages is rejected (None-valued
keys are stripped).

The healthcheck probes `GET /health` rather than `/v1/models`:
`/health` works with or without `--api-key` auth, so enabling the flag
later can't silently break the container's health status.

### Downloading the weights

```bash
hf download VnimanieAI/Qwen3.8-Flash-Next-W4A16   # ~180 GB
```

The compose services run with `HF_HUB_OFFLINE=1`, and in offline mode
vLLM resolves the model through `refs/main` in the HF cache. Downloading
with `--revision <sha>` does not write that file, and vLLM then
crash-loops with `LocalEntryNotFoundError`. Fix by writing the snapshot
sha into
`~/.cache/huggingface/hub/models--VnimanieAI--Qwen3.8-Flash-Next-W4A16/refs/main`.

A harmless log line on every boot:
`[ERROR] min_frames/max_frames is part of Qwen3VLVideoProcessorInitKwargs, but not documented`.

### Rollback

Both previous known-good pairings are kept in `compose.yaml` comments,
each with its digest and full command:

- **Qwen3.8-27B** — the `qwen38-cu129` special-release digest
  (`sha256:3879c346b0866fe8e1894243bb4fcdfc38f8dfcdb8a710a3b1d456ce4e799999`)
  serving `Qwen/Qwen3.8-27B` (dense hybrid-attention, unquantized BF16,
  ~13.3 GB/GPU) with `--language-model-only --max-model-len 262144
  --max-num-seqs 2 --gpu-memory-utilization 0.80` and the same
  `qwen3` / `qwen3_coder` parsers.
- **GLM-4.5-Air** — vLLM v0.19.0
  (`sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504`)
  serving `QuantTrio/GLM-4.5-Air-AWQ-FP16Mix` (MoE, ~106B total / ~12B
  active) with `--enable-expert-parallel --tool-call-parser glm45
  --reasoning-parser glm45 --trust-remote-code`.

When rolling back, drop `VLLM_PLE_CPU_OFFLOAD`, `cap_add` and
`security_opt` from the service — they are Flash-Next-specific.

## Authentication

The endpoint currently runs **open** — `--api-key` is deliberately
absent from the command while a client app that mishandles auth is in
use. To enforce the key, add `--api-key ${LLM_API_KEY:-changeme}` to the
command block (interpolated from `.env`, so it stays in lockstep with
the bearer token the agent sends) and recreate the container; clients
then must send `Authorization: Bearer <key>` on every `/v1/*` call.
`/health` stays open in both states, which is why the container
healthcheck and the agent's probes are auth-proof either way.

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

- **compressed-tensors / W4A16** — INT4 weight-only quants (current default)
- **Full precision** — unquantized models (high VRAM)
- **AWQ quantized** — optimized inference models
- **GPTQ** — alternative quantization format

Popular model options:

```yaml
# Default — MoE (512 experts / 10 active), INT4 W4A16, 4× 24 GB GPUs via
# -tp 4 + --enable-expert-parallel; PLE tables offloaded to host RAM
"VnimanieAI/Qwen3.8-Flash-Next-W4A16"

# Prior default — dense hybrid-attention model, BF16, 4× 24 GB via -tp 4
# (same qwen3 / qwen3_coder parsers; full 262k context fits)
"Qwen/Qwen3.8-27B"

# Earlier default — MoE reasoning model (~106B total / ~12B active); needs
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

# Test API directly (source .env first, or paste the key)
curl -H "Authorization: Bearer ${LLM_API_KEY}" http://localhost:8000/v1/models

# Monitor GPU usage
nvidia-smi -l 1

# Check memory usage
docker compose exec vllm nvidia-smi
```
