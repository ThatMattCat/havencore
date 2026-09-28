# vLLM Vision Backend

> **Status: shelved, not deleted.** As of 2026-09-26 the agent's vision pipeline (`/api/vision/*`, `query_multimodal_api`, `mcp_vision_tools`, the autonomy scene-description gather) runs on the main [`vllm`](../vllm/README.md) service — the `Qwen/Qwen3.8-27B` chat model is multimodal (27-layer ViT, image and video input), verified live with an `image_url` chat completion. This service is therefore gated behind the compose profile `vllm-vision` and does not start by default, which returns ~10.5 GiB on GPU 4 to STT/TTS/embeddings/face-recognition/ComfyUI. See [Configuration → Vision Backend](../../configuration.md#vision-backend-configuration) for the default (chat-model) wiring, including `VISION_CHAT_TEMPLATE_KWARGS`.

Second vLLM instance, dedicated to a vision-language model. When enabled it is pinned to the 5th RTX 3090 (`CUDA_VISIBLE_DEVICES=4`), which it shares with every non-LLM GPU service (STT, TTS, embeddings, face-recognition, ComfyUI) — GPUs 0-3 are dedicated to the main `vllm` service. Exposes an OpenAI-compatible API on host port 8001 under its own served-model-name (`gpt-4-vision`).

## When to re-enable it

Image tokens on the chat model come out of the same KV cache the agent's conversations use on GPUs 0-3. Occasional camera snapshots and playground uploads are fine; a steady stream of frames, or a vision workload you want isolated from chat latency, is the case for bringing this instance back. To do so:

```bash
# .env
COMPOSE_PROFILES="kokoro,vllm-vision"       # add the profile beside the active TTS engine's
VISION_API_BASE="http://10.0.0.1:8001/v1"   # this service's host port (LLM_API_BASE stays on :8000)
VISION_SERVED_NAME="gpt-4-vision"           # this service's --served-model-name
```

then `docker compose down && docker compose up -d`. The `VISION_MODEL` / `VISION_MAX_MODEL_LEN` / `VISION_MAX_NUM_SEQS` / `VISION_GPU_MEM_UTIL` lines in `.env` only take effect while the profile is active. `VISION_CHAT_TEMPLATE_KWARGS` is still forwarded on every call; set it to an empty string if the chosen Qwen3-VL build rejects `chat_template_kwargs`. To go back, remove the profile and point `VISION_API_BASE` / `VISION_SERVED_NAME` at `:8000` / `gpt-3.5-turbo` again.

## Purpose

- Image and short-video understanding for the agent
- Automatic scene description on face-recognition autonomy triggers, so "person in backyard" becomes "Matt in a flannel walking the cat" before the triage LLM ever sees it
- General-purpose vision tools surfaced via `mcp_vision_tools` (`describe_image`, `describe_camera_snapshot`, `compare_snapshots`, `identify_object`, `read_text_in_image`)

## Single-card sizing — three-tier ladder

A single 24GB card is right on the edge for the larger Qwen3-VL variants, and since the chat model took GPUs 0-3 for itself, GPU 4 also has to host STT, TTS, embeddings, face-recognition and ComfyUI. The last active config (before the service was shelved) was therefore the **8B** model at the bottom of the ladder, run at half the card (`VISION_GPU_MEM_UTIL=0.50`) so the helpers fit beside it; the `.env.example` values still reflect it. The larger tiers are documented so a swap is a one-line change to `VISION_MODEL` in `.env`, then `docker compose down && up -d vllm-vision` — but they only fit if the helpers move off GPU 4 (or a sixth card appears).

| Tier | Model | Notes |
|------|-------|-------|
| 1 | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | Dense 32B, AWQ ~17–18GB. Highest quality. Tight on 24GB — official Qwen recipe wants tensor-parallel-size 2. Needs the whole card. |
| 2 | `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ` | MoE: 30B total / 3B active. Same weight footprint as Tier 1 but lighter activations / KV-cache pressure. Previous active config, at `VISION_GPU_MEM_UTIL=0.93` / `VISION_MAX_MODEL_LEN=55296`. Needs the whole card. |
| 3 (last active) | `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit` | 7.5 GB on disk, ~7.3 GiB of weights in VRAM. At `VISION_GPU_MEM_UTIL=0.50` / `VISION_MAX_MODEL_LEN=16384` / `VISION_MAX_NUM_SEQS=1` the measured KV capacity is 30,448 tokens. Quality is lower but plenty for camera scene description. Note: `Qwen/Qwen3-VL-8B-Instruct-AWQ` does **not** exist on Hugging Face; the `cyankiwi` repo is the real one. |

## Configuration

```yaml
vllm-vision:
  image: vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
  profiles: ["vllm-vision"]   # off unless COMPOSE_PROFILES includes it
  environment:
    - CUDA_VISIBLE_DEVICES=4
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ports:
    - "8001:8000"
  command: >
    --model ${VISION_MODEL:-QuantTrio/Qwen3-VL-32B-Instruct-AWQ}
    --served-model-name ${VISION_SERVED_NAME:-gpt-4-vision}
    --max-model-len ${VISION_MAX_MODEL_LEN:-16384}
    --max-num-seqs ${VISION_MAX_NUM_SEQS:-2}
    --gpu-memory-utilization ${VISION_GPU_MEM_UTIL:-0.92}
    --kv-cache-dtype fp8_e4m3
    --limit-mm-per-prompt '{"image": 4, "video": 1}'
    --trust-remote-code
```

The image digest is vLLM 0.19.0 — well above the 0.11.0 minimum Qwen3-VL requires, and known-good on NVIDIA driver 580.x. (The main `vllm` service runs a dedicated Qwen3.8 build of vLLM; this service did not need to follow.)

The `compose.yaml` defaults above are the Tier 1 values. The `.env.example` overrides them for the Tier 3 config that was last active:

```bash
VISION_MODEL="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
VISION_GPU_MEM_UTIL=0.50
VISION_MAX_MODEL_LEN=16384
VISION_MAX_NUM_SEQS=1
```

### Env vars (set in `.env`)

| Var | Default | What it does |
|-----|---------|--------------|
| `VISION_MODEL` | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | HF model id (this service only). Last active: `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit` |
| `VISION_API_BASE` | `http://10.0.0.1:8000/v1` (chat vLLM) | Where the agent sends vision calls (host-scoped). Set to `http://10.0.0.1:8001/v1` to use this service. |
| `VISION_API_KEY` | `1234` | Bearer token (neither vLLM enforces one, but the SDK requires a value) |
| `VISION_SERVED_NAME` | `gpt-3.5-turbo` (chat vLLM) | OpenAI-compat alias sent in every request body. Set to `gpt-4-vision` to use this service. |
| `VISION_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking": false}` | Raw JSON forwarded as `chat_template_kwargs` on every vision call (keeps the reasoning chat model from spending the vision `max_tokens` on its think block). Empty string omits the field. |
| `VISION_MAX_MODEL_LEN` | `16384` | This service only. Context window. Drop to 8192 if Tier 1 OOMs. |
| `VISION_MAX_NUM_SEQS` | `2` | This service only. Concurrent sequences. Drop to 1 if Tier 1 OOMs. Last active: `1` |
| `VISION_GPU_MEM_UTIL` | `0.92` | This service only. vLLM reservation fraction. Push to 0.96 if borderline. Last active: `0.50`, leaving the other half of GPU 4 for the helper services. |

## Bring-up

1. Confirm GPU 4 is the 5th RTX 3090 (`nvidia-smi --query-gpu=index,name --format=csv`) and that the helper services' `*_GPU` / `STT_DEVICE` vars in `.env` point at it too — GPUs 0-3 must stay free for the main `vllm` service (see [vLLM → GPU layout](../vllm/README.md#gpu-layout-and-sizing)).
2. Add the `VISION_*` block from `.env.example` to `.env`, add `vllm-vision` to `COMPOSE_PROFILES`, and point `VISION_API_BASE` at `:8001` / `VISION_SERVED_NAME` at `gpt-4-vision` (see [When to re-enable it](#when-to-re-enable-it)).
3. `docker compose up -d vllm-vision`.
4. **Cold start is long** on first download — Qwen3-VL-32B weights are ~17GB (the 8B quant is 7.5 GB). The healthcheck has `start_period: 1200s` (20 min). Track it with `docker compose logs -f vllm-vision`.
5. Once healthy, run `VISION_BASE=http://localhost:8001 VISION_MODEL=gpt-4-vision scripts/vision-smoke-test.sh` from the repo root (the script's defaults now target the chat vLLM on `:8000`). It enforces six acceptance criteria (cold-start within 1200s, steady-state VRAM ≤22.5GB, p50 ≤8s / p95 ≤15s on a ~1MP image with `max_tokens=400`, 50 sequential image queries with no errors, at least one 5-second video clip processed without OOM, and a quality sanity check on a known image). Treat its exit code as the decision gate.

### If sizing fails

Walk down the fallback ladder in order:

1. **Retry Tier 1 with tighter flags** before swapping models. Set `VISION_MAX_MODEL_LEN=8192`, `VISION_MAX_NUM_SEQS=1`, `VISION_GPU_MEM_UTIL=0.96`, and add `--enforce-eager` to the command in `compose.yaml`. `down && up -d vllm-vision`. Rerun smoke test.
2. **Tier 2:** `VISION_MODEL="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"`, restore `VISION_MAX_MODEL_LEN=16384`. The MoE variant trims activation pressure even though weight size is similar.
3. **Tier 3:** `VISION_MODEL="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"` with `VISION_GPU_MEM_UTIL=0.50` / `VISION_MAX_NUM_SEQS=1` (the last active config). If even this fails, suspect a config bug (driver, vLLM image pin, GPU pinning) or a helper service holding more of GPU 4 than expected (`nvidia-smi`) rather than a sizing problem.

## Agent integration

Everything below is backend-agnostic: the agent only knows `VISION_API_BASE` / `VISION_SERVED_NAME`, so it applies equally to the default chat-vLLM wiring and to this service when it is enabled.

### `/api/vision/*` endpoints

`services/agent/selene_agent/api/vision.py` reads `VISION_API_BASE` / `VISION_API_KEY` / `VISION_SERVED_NAME` / `VISION_CHAT_TEMPLATE_KWARGS` from `shared/configs/shared_config.py` (and `selene_agent/utils/config.py`). Every request body includes `model: $VISION_SERVED_NAME` because the vision backend and the chat path may use different served-model aliases, plus `chat_template_kwargs` from `VISION_CHAT_TEMPLATE_KWARGS` unless it is empty. An empty `content` from the model (a reasoning block that exhausted `max_tokens`) is surfaced as a 502 with an explanatory message instead of a blank answer.

- `POST /api/vision/ask` — multipart upload. Branches on the upload's MIME type: `image/*` builds an `image_url` content part, `video/*` builds a `video_url` content part. Unknown MIME → `415 Unsupported Media Type`. The legacy `image` form field still works; new callers should use `file`. A short clip (5-10 s 1080p H.264) fits well within the upload budget.
- `POST /api/vision/ask_url` — JSON sibling. Takes `{text?, image_url, max_tokens?, temperature?}`; the upstream fetches the image URL itself. Used by the `query_multimodal_api` MCP tool so all vision calls go through one chokepoint for logging/auth.
- `GET /api/vision/health` — proxies `/v1/models`.

Both response shapes include the served-model name (`model`) so the dashboard can label which tier handled the call. `/api/vision/ask` additionally returns `media_type` (`image_url` / `video_url`).

The `/api/` location in nginx is sized for these uploads — see [Nginx Gateway → Upload size caps](../nginx/README.md#upload-size-caps).

### Autonomy face-trigger scene description

The autonomy engine's face-trigger triage drives the vision model on every camera event:

- **`watch_llm` gather step.** When an agenda item sets `gather.scene_description: true`, the handler extracts a snapshot URL from the trigger event (preferring `event.sensor_event.snapshot_url`, falling back to `event.payload.snapshot_url`), calls `query_multimodal_api` against it, and threads the description into the triage LLM's user prompt as a `## scene_description` block. Per-seed `gather.scene_description_prompt` overrides the default.
- **Default seeds.** All three `face_*_triage` seeds in `selene_agent/autonomy/seeds/camera_events.py` ship with `scene_description: true`. `face_no_face_triage` ships with a tailored prompt biased toward the disambiguation that seed exists for (resident vs. pet vs. delivery driver vs. wildlife).
- **Failure mode is bounded.** The vision call is wrapped in `_safe_tool`, so a vision-LLM outage or a 404 snapshot path drops a `<tool ... failed>` string into the gather instead of blocking the triage. The triage LLM still gets to judge on the rest of the gather.

See [autonomy/cameras.md](../agent/autonomy/cameras.md#vision-ai-scene-description) for the watch_llm gather schema and worked examples.

### MCP vision tools

`mcp_vision_tools` exposes five purpose-built tools on top of the vision backend so the LLM gets focused tools instead of having to assemble image URLs and prompts manually:

- `describe_image(image_url, prompt?)` — generic "describe this".
- `describe_camera_snapshot(camera_name, prompt?)` — one-shot fresh HA snapshot + vision description. Reuses the `mcp_mqtt_tools` `HACamSnapper` for the capture; matches `camera_name` against returned URLs server-side (substring → token split fallback).
- `compare_snapshots(image_url_a, image_url_b, focus?)` — the only tool that bypasses the `/api/vision/ask_url` chokepoint, posting directly to the vision vLLM's `/v1/chat/completions` because the chokepoint schema is single-image (this is why the chat `vllm` service runs with `--limit-mm-per-prompt image:2`).
- `identify_object(image_url, hint?)` — "what is this thing?" with optional category hint.
- `read_text_in_image(image_url)` — OCR-flavored prompt with `temperature=0.1`.

All five tool names are in the `observe`-tier autonomy allow-list, so autonomy turns can call them. Full reference at [Vision Tools](../agent/tools/vision.md).

### Dashboard playground

`/playgrounds/vision` accepts image+video uploads, renders the appropriate preview element (`<img>` vs `<video controls>`), exposes preset prompts (Describe scene / What's unusual? / Read all text / Identify objects), and displays a served-model badge plus a prompt/completion token breakdown alongside latency.
