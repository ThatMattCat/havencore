# vLLM Vision Backend

Second vLLM instance, dedicated to a vision-language model. Pinned to the 5th RTX 3090 (`CUDA_VISIBLE_DEVICES=4`). Exposes an OpenAI-compatible API on host port 8001.

The main `vllm` service (text-only LLM, served as `gpt-3.5-turbo`) is unaffected; this is a separate process with its own GPU and its own served-model-name.

## Purpose

- Image and short-video understanding for the agent
- Automatic scene description on face-recognition autonomy triggers, so "person in backyard" becomes "Matt in a flannel walking the cat" before the triage LLM ever sees it
- General-purpose vision tools surfaced via `mcp_vision_tools` (`describe_image`, `describe_camera_snapshot`, `compare_snapshots`, `identify_object`, `read_text_in_image`)

## Single-card sizing — three-tier fallback

A single 24GB card is right on the edge for the larger Qwen3-VL variants. The active config is the dense 32B at the top of the ladder; lower tiers are documented so a swap is a one-line change to `VISION_MODEL` in `.env`, then `docker compose down && up -d vllm-vision`. Nothing else changes.

| Tier | Model | Notes |
|------|-------|-------|
| 1 (active) | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | Dense 32B, AWQ ~17–18GB. Highest quality. Tight on 24GB — official Qwen recipe wants tensor-parallel-size 2. |
| 2 (fallback) | `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ` | MoE: 30B total / 3B active. Same weight footprint as Tier 1 but lighter activations / KV-cache pressure. Plausibly fits where Tier 1 doesn't. |
| 3 (fallback) | `Qwen/Qwen3-VL-8B-Instruct-AWQ` | ~5–6GB weights. ~18GB headroom. Fit is essentially guaranteed; quality is lower but plenty for camera scene description. |

## Configuration

```yaml
vllm-vision:
  image: vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504
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

The image digest is the same one the main `vllm` service uses — vLLM 0.19.0, well above the 0.11.0 minimum Qwen3-VL requires, and known-good on NVIDIA driver 580.x.

### Env vars (set in `.env`)

| Var | Default | What it does |
|-----|---------|--------------|
| `VISION_MODEL` | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | HF model id |
| `VISION_API_BASE` | `http://10.0.0.1:8001/v1` | Where the agent reaches the service (host-scoped) |
| `VISION_API_KEY` | `1234` | Bearer token (vllm-vision doesn't enforce, but the SDK requires a value) |
| `VISION_SERVED_NAME` | `gpt-4-vision` | OpenAI-compat alias |
| `VISION_MAX_MODEL_LEN` | `16384` | Context window. Drop to 8192 if Tier 1 OOMs. |
| `VISION_MAX_NUM_SEQS` | `2` | Concurrent sequences. Drop to 1 if Tier 1 OOMs. |
| `VISION_GPU_MEM_UTIL` | `0.92` | vLLM reservation fraction. Push to 0.96 if borderline. |

## Bring-up

1. Confirm GPU 4 is the dedicated RTX 3090 (`nvidia-smi --query-gpu=index,name --format=csv`).
2. Add the `VISION_*` block from `.env.example` to `.env`.
3. `docker compose up -d vllm-vision`.
4. **Cold start is long** — Qwen3-VL-32B weights are ~17GB and the first download takes a while. The healthcheck has `start_period: 1200s` (20 min). Track it with `docker compose logs -f vllm-vision`.
5. Once healthy, run `scripts/vision-smoke-test.sh` from the repo root. It enforces six acceptance criteria (cold-start within 1200s, steady-state VRAM ≤22.5GB, p50 ≤8s / p95 ≤15s on a ~1MP image with `max_tokens=400`, 50 sequential image queries with no errors, at least one 5-second video clip processed without OOM, and a quality sanity check on a known image). Treat its exit code as the decision gate.

### If sizing fails

Walk down the fallback ladder in order:

1. **Retry Tier 1 with tighter flags** before swapping models. Set `VISION_MAX_MODEL_LEN=8192`, `VISION_MAX_NUM_SEQS=1`, `VISION_GPU_MEM_UTIL=0.96`, and add `--enforce-eager` to the command in `compose.yaml`. `down && up -d vllm-vision`. Rerun smoke test.
2. **Tier 2:** `VISION_MODEL="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"`, restore `VISION_MAX_MODEL_LEN=16384`. The MoE variant trims activation pressure even though weight size is similar.
3. **Tier 3:** `VISION_MODEL="Qwen/Qwen3-VL-8B-Instruct-AWQ"`. If even this fails, suspect a config bug (driver, vLLM image pin, GPU pinning) rather than a sizing problem.

## Agent integration

### `/api/vision/*` endpoints

`services/agent/selene_agent/api/vision.py` reads `VISION_API_BASE` / `VISION_API_KEY` / `VISION_SERVED_NAME` from `shared/configs/shared_config.py` (and `selene_agent/utils/config.py`). Every request body includes `model: $VISION_SERVED_NAME` because the vision instance and the main agent vLLM use different served-model aliases.

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

`mcp_vision_tools` exposes five purpose-built tools on top of `vllm-vision` so the LLM gets focused tools instead of having to assemble image URLs and prompts manually:

- `describe_image(image_url, prompt?)` — generic "describe this".
- `describe_camera_snapshot(camera_name, prompt?)` — one-shot fresh HA snapshot + vision description. Reuses the `mcp_mqtt_tools` `HACamSnapper` for the capture; matches `camera_name` against returned URLs server-side (substring → token split fallback).
- `compare_snapshots(image_url_a, image_url_b, focus?)` — the only tool that bypasses the `/api/vision/ask_url` chokepoint, posting directly to `vllm-vision`'s `/v1/chat/completions` because the chokepoint schema is single-image.
- `identify_object(image_url, hint?)` — "what is this thing?" with optional category hint.
- `read_text_in_image(image_url)` — OCR-flavored prompt with `temperature=0.1`.

All five tool names are in the `observe`-tier autonomy allow-list, so autonomy turns can call them. Full reference at [Vision Tools](../agent/tools/vision.md).

### Dashboard playground

`/playgrounds/vision` accepts image+video uploads, renders the appropriate preview element (`<img>` vs `<video controls>`), exposes preset prompts (Describe scene / What's unusual? / Read all text / Identify objects), and displays a served-model badge plus a prompt/completion token breakdown alongside latency.
