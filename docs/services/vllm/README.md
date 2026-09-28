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
hybrid-attention support (it reports vLLM `0.1.dev19754+g3a0914114`).
The CUDA 12.9 (`cu129`) variant is deliberate: the host driver is 580.x.
The model is served under the OpenAI-compat name `gpt-3.5-turbo` so
stock OpenAI SDKs work without reconfiguration.

```yaml
vllm:
  image: vllm/vllm-openai@sha256:3879c346b0866fe8e1894243bb4fcdfc38f8dfcdb8a710a3b1d456ce4e799999
  environment:
    - HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    - CUDA_VISIBLE_DEVICES=0,1,2,3
  init: true
  command: >
    --model Qwen/Qwen3.8-27B
    --served-model-name gpt-3.5-turbo
    --tensor-parallel-size 4
    --max-model-len 262144
    --max-num-seqs 4
    --limit-mm-per-prompt '{"image": 2, "video": 1}'
    --mm-processor-kwargs '{"size": {"longest_edge": 2097152, "shortest_edge": 65536}}'
    --gpu-memory-utilization 0.90
    --default-chat-template-kwargs '{"reasoning_effort": "medium"}'
    --tool-call-parser qwen3_coder
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --host 0.0.0.0
    --port 8000
```

### Model: Qwen3.8-27B

`Qwen/Qwen3.8-27B` is a **dense hybrid-attention** model (27B
parameters; 48 of its 64 layers are Gated DeltaNet linear attention, 16
are full attention with 4 KV heads), served **unquantized in BF16**
under Apache 2.0. It is 55.6 GB on disk and loads as 13.11 GiB of
weights per card at `-tp 4`. An FP8 checkpoint exists, but RTX 3090s
(SM86) have no native FP8, so BF16 is the highest-quality variant that
fits. The checkpoint is **multimodal** — a 27-layer ViT (~461M vision
parameters) handling image and video input — and it is a reasoning
model.

Flag-by-flag:

- `--tensor-parallel-size 4` with no `--enable-expert-parallel` — the
  model is not MoE, so plain tensor parallelism across GPUs 0-3 is all
  it needs.
- `--max-model-len 262144` — the model's full native context. The
  hybrid attention keeps KV cost to ~64 KB/token, so one max-length
  sequence needs ~16.8 GB of KV (~4.2 GB per GPU) on top of the weights,
  which fits.
- `--max-num-seqs 4` — chat, autonomy turns and vision calls all land on
  this instance. The slots share one KV pool: several sequences at full
  context will not fit at once (see the measured concurrency below),
  which is fine for a household.
- `--limit-mm-per-prompt '{"image": 2, "video": 1}'` /
  `--mm-processor-kwargs '{"size": {"longest_edge": 2097152, "shortest_edge": 65536}}'`
  — there is deliberately no `--language-model-only`: this instance is
  the agent's vision backend (`VISION_API_BASE` / `VISION_SERVED_NAME`
  in `.env` point here). vLLM's default is one image per prompt;
  `compare_snapshots` needs two. The size cap bounds each image at
  ~2 MP (~2k tokens, a full 1080p snapshot), so memory profiling and
  per-request KV stay bounded. Image tokens still come out of the chat
  KV budget below — occasional snapshots are fine, a stream of frames is
  not; the shelved [vllm-vision](../vllm-vision/README.md) service is
  the escape hatch for isolating vision load. Vision calls also pass
  `chat_template_kwargs: {"enable_thinking": false}`
  (`VISION_CHAT_TEMPLATE_KWARGS`) so the reasoning block doesn't eat the
  small vision `max_tokens` budget.
- `--gpu-memory-utilization 0.90` — GPUs 0-3 are dedicated to this
  service, and the vision tower plus image profiling need headroom a
  text-only configuration would not. See the GPU layout section below.
- `--default-chat-template-kwargs '{"reasoning_effort": "medium"}'` —
  the server-side default thinking level for every request that does
  not pass its own `chat_template_kwargs`. The chat template's own
  default is `xhigh`. See
  [Thinking level](#thinking-level) below.
- No `--trust-remote-code` and no `--compilation-config` — the pinned
  build supports the architecture natively and `torch.compile` runs
  normally (it accounts for ~93 s of the cold start).
- The model's built-in MTP draft head is not enabled; it is available
  through `--speculative-config` if speculative decoding is wanted
  later.

Environment and container settings:

- `HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}` — the model is resolved from the
  local HF cache only, so the service does not depend on internet or DNS
  at boot. Set `HF_HUB_OFFLINE=0` in `.env` once to pull a new model,
  then flip it back.
- `CUDA_VISIBLE_DEVICES=0,1,2,3` — GPUs 0-3 belong to this service;
  GPU 4 is hidden so no tensor-parallel rank lands on it.
- `init: true` — a hung `vllm serve` otherwise becomes an unkillable
  zombie PID 1 that `docker compose stop` refuses to reap.

Nothing else is set: the service needs no extra capabilities, no
seccomp override and no special host RAM.

### GPU layout and sizing

GPUs 0-3 are **dedicated** to this service. Every other GPU consumer —
`speech-to-text`, `text-to-speech`, `embeddings`, `face-recognition`,
`text-to-image` — is pinned to GPU 4 via the `*_GPU` / `STT_DEVICE` vars
in `.env` (see
[configuration.md → GPU Settings](../../configuration.md#gpu-settings)).
The optional `vllm-vision` service, when its compose profile is enabled,
is pinned to GPU 4 as well; by default it is off because this instance
serves vision.

The reason is the memory budget. At `--gpu-memory-utilization 0.90`
vLLM claims 90% of each card: 13.11 GiB of weights, the vision tower and
profiling headroom, and whatever is left becomes KV cache. Anything else
holding VRAM on GPUs 0-3 comes straight out of that KV pool — or, if the
card no longer has the requested budget free, vLLM refuses to start.

Measured at the current settings:

| Metric | Value |
|--------|-------|
| Model load (weights) | 13.11 GiB per GPU |
| KV cache memory | 7.52 GiB per GPU |
| KV capacity at 0.90 | 482,775 tokens (1.84x concurrency at 262,144 tokens per request) |
| GPU memory in use per card | ~21.9-22.0 GB |
| Host RAM | no special requirement |
| Cold start | ~5.5 min to healthy (model load ~20 s; engine init 185 s, of which ~93 s is `torch.compile`) |
| Decode | ~44 tok/s on a short prompt, single stream |
| Vision (`/api/vision/ask`, one camera snapshot) | 2,061 prompt tokens, ~5.8 s |

Decode speed is the cost of running the unquantized dense model: ~44
tok/s is roughly half of the ~84 tok/s the Flash-Next quant reached on
the same cards (see [Rollback](#rollback)).

The agent reads `max_model_len` from `/v1/models` **once per process**
and caches it; the conversation context-limit threshold
(`CONVERSATION_CONTEXT_LIMIT_FRACTION`) is derived from that value.
After changing `--max-model-len`, run `docker compose restart agent` or
the threshold keeps the old value.

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
same model. The Qwen3.8-Flash-Next rollback uses the same two parsers,
so switching between them needs no agent-side code change.

### Thinking level

Qwen3.8-27B is a reasoning model, and its chat template takes three
`chat_template_kwargs`:

| Key | Values | Effect |
|-----|--------|--------|
| `enable_thinking` | `true` (template default), `false` | `false` skips the think block entirely |
| `reasoning_effort` | `xhigh` (template default), `medium`, `low` | `xhigh` and `low` each add an instruction to the system prompt; `medium` adds none. Any other value raises a template error |
| `preserve_thinking` | `true` (template default), `false` | `false` drops the reasoning of earlier turns from the rendered history |

The agent's chat paths send no `chat_template_kwargs`, so they run at
the server default set by `--default-chat-template-kwargs`: thinking on,
effort `medium`. Values passed on a request override the server
default, which is how vision calls stay at `enable_thinking: false`
(`VISION_CHAT_TEMPLATE_KWARGS`). To change the chat level, edit the flag
in `compose.yaml` and recreate the `vllm` container.

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
hf download Qwen/Qwen3.8-27B   # ~56 GB
```

The compose services run with `HF_HUB_OFFLINE=1`, and in offline mode
vLLM resolves the model through `refs/main` in the HF cache. Downloading
with `--revision <sha>` does not write that file, and vLLM then
crash-loops with `LocalEntryNotFoundError`. Fix by writing the snapshot
sha into
`~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/refs/main`.

A harmless log line on every boot:
`[ERROR] min_frames/max_frames is part of Qwen3VLVideoProcessorInitKwargs, but not documented`.

### Rollback

Both previous known-good pairings are kept in `compose.yaml` comments,
each with its digest and full command.

#### Qwen3.8-Flash-Next (W4A16)

`VnimanieAI/Qwen3.8-Flash-Next-W4A16` on the
`qwen38-flash-next-x86_64-cu130` vendor digest
(`sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`)
— the vLLM dev build with `qwen4_exp` support, including the PLE
n-gram-table CPU offload. It is an INT4 quant (compressed-tensors,
W4A16, group size 128) of a **MoE** model: 512 experts / 10 active, a
~125B backbone plus 51B of "PLE" n-gram lookup tables, 168 GB on disk
(`hf download VnimanieAI/Qwen3.8-Flash-Next-W4A16`, ~180 GB) and
~18.2 GiB of GPU-resident weights per card at `-tp 4`. It is multimodal
too, with the same `qwen3` / `qwen3_coder` parsers and the same
`--limit-mm-per-prompt` / `--mm-processor-kwargs` flags, so the vision
configuration does not change on rollback.

It is faster (~84 tok/s decode on a short prompt, ~92 tok/s at 31k
context, prefill ~1.7k tok/s, warm TTFT ~0.3 s) but was dropped as the
default for answer quality: the quant is uncalibrated RTN with INT4
attention, and it made false "this word is misspelled" claims and was
unreliable in longer agent loops.

Command flags that differ from the current default:

```
--model VnimanieAI/Qwen3.8-Flash-Next-W4A16
--enable-expert-parallel
--max-model-len 98304
--max-num-seqs 64
--gpu-memory-utilization 0.94
--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
--trust-remote-code
```

- `--enable-expert-parallel` is **mandatory**: `moe_intermediate_size`
  is 640, which does not divide into 128-wide quant groups under plain
  tensor parallelism.
- `--compilation-config` mode 0 — inductor compile of this architecture
  hangs on Ampere, so compilation is off, but decode CUDA graphs are
  kept (they do not need inductor).
- `--max-model-len 98304` / `--max-num-seqs 64` — 4× 24 GB fits weights
  plus ~98k of context *or* weights plus the MTP draft head, not both,
  so MTP stays off. fp8 KV cache is rejected by the model ("QSA requires
  a BF16 main KV cache").
- `--gpu-memory-utilization 0.94` — KV cost is ~14.7 KB/token/GPU and
  does **not** shrink with tensor parallelism (the model has 2 KV heads,
  which replicate once there is less than one head per rank), so a
  single 98k request needs 1.24 GiB of KV per GPU. 0.92 left only
  1.06 GiB; 0.94 measured 121,048 tokens of KV (1.23x concurrency at
  98k) at ~22.6 GB per card. Every step below 0.94 comes straight out of
  KV.
- `--trust-remote-code` — required for the `qwen4_exp` architecture.

Service settings that must be added back, all load-bearing on this
host:

- `VLLM_PLE_CPU_OFFLOAD=1` keeps the ~102 GB BF16 PLE tables in host RAM
  instead of VRAM. Needs at least 110 GB of free host RAM (this host has
  251 GB).
- `VLLM_USE_V2_MODEL_RUNNER=0` — with the default v2 model runner every
  boot deadlocked right after CUDA-graph capture (all TP workers at 100%
  CPU, GPUs idle; with `torch.compile` disabled it surfaced as
  `queue.Full` in `PleOffloadConnector._launch`). The v2 runner stages
  PLE inputs from CUDA tensors through a device-to-host copy that waits
  on an event recorded on the model stream, and that handoff races here.
  The v1 runner feeds the offload worker from its CPU input mirrors and
  boots cleanly. `CUDA_LAUNCH_BLOCKING=1` also cures it, but at ~40% of
  the decode speed.
- `TORCH_COMPILE_DISABLE=1` — the TP>1 vocab-masking helper is
  `torch.compile`-decorated regardless of `--compilation-config`, and
  its inductor kernel load hung. Disabling dynamo globally is what
  actually keeps `torch.compile` out of the way.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- `cap_add: SYS_PTRACE` + `security_opt: seccomp=unconfined` — the PLE
  offload worker hands GPU buffers between processes with the
  `pidfd_getfd` syscall, which Docker's default seccomp profile blocks.

Cold start is ~5 min (~90 s model load + ~60 s PLE table load). The
agent caches `max_model_len`, so restart it after the swap.

#### GLM-4.5-Air

vLLM v0.19.0
(`sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504`)
serving `QuantTrio/GLM-4.5-Air-AWQ-FP16Mix` (MoE, ~106B total / ~12B
active) with `--enable-expert-parallel --max-model-len 46080
--max-num-seqs 2 --gpu-memory-utilization 0.79 --tool-call-parser glm45
--reasoning-parser glm45 --trust-remote-code`. It is text-only as
configured, so vision has to move back to the shelved
[vllm-vision](../vllm-vision/README.md) service.

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

- **Full precision** — unquantized models (current default; high VRAM)
- **compressed-tensors / W4A16** — INT4 weight-only quants
- **AWQ quantized** — optimized inference models
- **GPTQ** — alternative quantization format

Popular model options:

```yaml
# Default — dense hybrid-attention model, unquantized BF16, multimodal,
# 4× 24 GB GPUs via -tp 4; full 262k context fits
"Qwen/Qwen3.8-27B"

# Prior default — MoE (512 experts / 10 active), INT4 W4A16, -tp 4 +
# --enable-expert-parallel; PLE tables offloaded to host RAM (same
# qwen3 / qwen3_coder parsers; faster, lower answer quality)
"VnimanieAI/Qwen3.8-Flash-Next-W4A16"

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
