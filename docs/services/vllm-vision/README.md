# vLLM Vision Backend

Second vLLM instance, dedicated to a vision-language model. Pinned to the 5th RTX 3090 (`CUDA_VISIBLE_DEVICES=4`). Exposes an OpenAI-compatible API on host port 8001.

The main `vllm` service (text-only LLM, served as `gpt-3.5-turbo`) is unaffected; this is a separate process with its own GPU and its own served-model-name.

## Purpose

- Image and short-video understanding for the agent
- Phase 3 (after this service is validated): automatic scene description on face-recognition autonomy triggers, so "person in backyard" becomes "Matt in a flannel walking the cat" before the triage LLM ever sees it
- General-purpose vision tools (`describe_camera_snapshot`, `read_text_in_image`, `compare_snapshots`, etc.) added in Phase 4

## Single-card sizing — three-tier fallback

A single 24GB card is right on the edge for the larger Qwen3-VL variants. The plan tries the most capable model first and steps down if it OOMs or thrashes. **Swapping models is a one-line change to `VISION_MODEL` in `.env`, then `docker compose down && up -d vllm-vision`** — nothing else changes.

| Tier | Model | Notes |
|------|-------|-------|
| 1 (target) | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | Dense 32B, AWQ ~17–18GB. Highest quality. Tight on 24GB — official Qwen recipe wants tensor-parallel-size 2. |
| 2 (first fallback) | `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ` | MoE: 30B total / 3B active. Same weight footprint as Tier 1 but lighter activations / KV-cache pressure. Plausibly fits where Tier 1 doesn't. |
| 3 (final fallback) | `Qwen/Qwen3-VL-8B-Instruct-AWQ` | ~5–6GB weights. ~18GB headroom. Fit is essentially guaranteed; quality is lower but plenty for camera scene description. |

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

The image digest is the same one the main `vllm` service uses — vLLM 0.19.0, which is well above the 0.11.0 minimum Qwen3-VL requires, and known-good on NVIDIA driver 580.x.

### Env vars (set in `.env`)

| Var | Default | What it does |
|-----|---------|--------------|
| `VISION_MODEL` | `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | HF model id |
| `VISION_API_BASE` | `http://10.0.0.1:8001/v1` | Where the agent reaches the service (host-scoped) |
| `VISION_API_KEY` | `1234` | Bearer token (vllm-vision currently doesn't enforce, but the SDK requires a value) |
| `VISION_SERVED_NAME` | `gpt-4-vision` | OpenAI-compat alias |
| `VISION_MAX_MODEL_LEN` | `16384` | Context window. Drop to 8192 if Tier 1 OOMs. |
| `VISION_MAX_NUM_SEQS` | `2` | Concurrent sequences. Drop to 1 if Tier 1 OOMs. |
| `VISION_GPU_MEM_UTIL` | `0.92` | vLLM reservation fraction. Push to 0.96 if borderline. |

## Bring-up procedure

1. Make sure GPU 4 is the 5th RTX 3090 (`nvidia-smi --query-gpu=index,name --format=csv`).
2. Add the `VISION_*` block from `.env.example` to `.env`.
3. `docker compose up -d vllm-vision`.
4. **Cold start is long** — Qwen3-VL-32B weights are ~17GB and the first download will take a while. The healthcheck has `start_period: 1200s` (20 min). Track it with `docker compose logs -f vllm-vision`.
5. Once healthy, run `scripts/vision-smoke-test.sh` from the repo root. The script enforces the Phase 1 acceptance criteria; treat its exit code as the decision gate.

## Phase 1 decision gate

The smoke test must pass all six criteria before Phase 2 wiring (`/api/vision/ask` repoint, autonomy gather step, MCP tools) starts:

1. Cold-start healthcheck within 1200s
2. Steady-state VRAM ≤22.5GB (≥1.5GB headroom on the 24GB card)
3. p50 ≤8s, p95 ≤15s on a ~1MP image with `max_tokens=400`
4. 50 sequential image queries with no OOM / no errors
5. At least one 5-second video clip processed without OOM (proves video path is alive)
6. Quality sanity: known test image produces a sensible description

### If the gate fails

Walk down the fallback ladder in order:

1. **Retry Tier 1 with tighter flags** before swapping models. Set `VISION_MAX_MODEL_LEN=8192`, `VISION_MAX_NUM_SEQS=1`, `VISION_GPU_MEM_UTIL=0.96`, and add `--enforce-eager` to the command in `compose.yaml`. `down && up -d vllm-vision`. Rerun smoke test.
2. **Tier 2:** `VISION_MODEL="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"`, restore `VISION_MAX_MODEL_LEN=16384`. The MoE variant trims activation pressure even though weight size is similar.
3. **Tier 3:** `VISION_MODEL="Qwen/Qwen3-VL-8B-Instruct-AWQ"`. If even this fails, suspect a config bug (driver, vLLM image pin, GPU pinning) rather than a sizing problem.

## Phase 1 result note (validated 2026-05-01)

```
Tier:               2 (MoE)
Model:              QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ
Flag overrides:     none beyond defaults — max-model-len=16384, max-num-seqs=2,
                    gpu-memory-utilization=0.92, kv-cache-dtype=fp8_e4m3,
                    limit-mm-per-prompt={"image":4,"video":1}, trust-remote-code
Cold start:         (warm cache) — first cold pull is the standard ~17 GB HF download
Steady-state VRAM:  20.45 GB / 24 GB  (3.55 GB headroom)
p50 / p95 latency:  0.47s / 0.61s  on a 1280px scene-description prompt, max_tokens=400
Smoke-test load:    50 sequential image queries, 0 errors
Notes:              Tier 1 (Qwen3-VL-32B-Instruct-AWQ dense) was tested first and worked
                    only with max-model-len=6144, max-num-seqs=1, mem-util=0.97, and
                    --enforce-eager. p50 was 3.25s. Tier 2 was strictly better across
                    every axis (latency, headroom, context, concurrency) at indistinguishable
                    output quality, so Tier 2 is the active config. Tier 3 (8B) untested.
```

This is what the rest of the integration work assumes is true.

## Phase 2 status (landed 2026-05-01)

Agent-side wiring is complete:

- **Endpoints repointed.** `services/agent/selene_agent/api/vision.py` now reads `VISION_API_BASE` / `VISION_API_KEY` / `VISION_SERVED_NAME` from `shared/configs/shared_config.py` (and `selene_agent/utils/config.py`) instead of the hardcoded `iav-to-text:8100`. Every request body includes `model: $VISION_SERVED_NAME` because the vision instance and the main agent vLLM use different served-model aliases.
- **`POST /api/vision/ask_url` (new).** JSON sibling of the multipart `/api/vision/ask`. Takes `{text?, image_url, max_tokens?, temperature?}`; the upstream fetches the image URL itself. Used by the `query_multimodal_api` MCP tool so all vision calls go through one chokepoint for logging/auth.
- **`query_multimodal_api` repointed.** `services/agent/selene_agent/modules/mcp_general_tools/mcp_server.py` now POSTs JSON to `http://agent:6002/api/vision/ask_url`. Schema narrowed to `text` + `image_url` (audio / video parameters were dropped — Qwen3-VL is image+video and the agent-side proxy is image-only; a future `mcp_vision_tools` module will add focused tools per the Phase 4 plan).
- **`iav-to-text` retired.** `services/iav-to-text/`, `docs/services/iav-to-text/`, the commented compose block, and the commented nginx upstreams + `/iav/ui/` / `/iav/api` location blocks have all been removed.

After landing, recreate the agent so `.env` env reads pick up the new VISION vars: `docker compose up -d --force-recreate --no-deps agent` (or the full `down agent && build agent && up -d agent` cycle if you also touched code).

## Phase 3 status (landed 2026-05-01)

The autonomy engine's face-trigger triage now drives the vision model on every camera event:

- **`watch_llm` gather step.** When an agenda item sets `gather.scene_description: true`, the handler extracts a snapshot URL from the trigger event (preferring `event.sensor_event.snapshot_url`, falling back to `event.payload.snapshot_url`), calls `query_multimodal_api` against it, and threads the description into the triage LLM's user prompt as a `## scene_description` block. Per-seed `gather.scene_description_prompt` overrides the default.
- **Default seeds updated.** All three `face_*_triage` seeds in `selene_agent/autonomy/seeds/camera_events.py` ship with `scene_description: true`. `face_no_face_triage` ships with a tailored prompt biased toward the disambiguation that seed exists for (resident vs. pet vs. delivery driver vs. wildlife). A one-time migration patches existing system-seeded rows in place — protected by `created_by='system_camera'` and a missing-key check, so dashboard-edited rows are never clobbered.
- **Verified live (2026-05-01).** A synthetic `haven/face/identified` event against a real backyard detection produced the description *"a person in a maroon shirt holding a small, dark-colored dog on a grassy lawn"*; the triage LLM saw it alongside `subject=Matt confidence=0.95 zone=backyard` and correctly judged `nominal`. End-to-end gather → vllm-vision → triage prompt → judgment loop is working.
- **Failure mode is bounded.** The vision call is wrapped in `_safe_tool`, so a vision-LLM outage or a 404 snapshot path drops a `<tool ... failed>` string into the gather instead of blocking the triage. The triage LLM still gets to judge on the rest of the gather.

See [autonomy/cameras.md](../agent/autonomy/cameras.md#vision-ai-scene-description) for the watch_llm gather schema and the worked examples.

## Phase 4 status (landed 2026-05-02)

A dedicated MCP server, `mcp_vision_tools`, layers five purpose-built
tools on top of `vllm-vision` so the LLM gets focused tools instead of
having to assemble image URLs and prompts manually:

- `describe_image(image_url, prompt?)` — generic "describe this".
- `describe_camera_snapshot(camera_name, prompt?)` — one-shot fresh
  HA snapshot + vision description. Reuses the
  `mcp_mqtt_tools` `HACamSnapper` for the capture; matches
  `camera_name` against returned URLs server-side (substring → token
  split fallback).
- `compare_snapshots(image_url_a, image_url_b, focus?)` — the only
  tool that bypasses the `/api/vision/ask_url` chokepoint, posting
  directly to `vllm-vision`'s `/v1/chat/completions` because the
  chokepoint schema is single-image.
- `identify_object(image_url, hint?)` — "what is this thing?" with
  optional category hint.
- `read_text_in_image(image_url)` — OCR-flavored prompt with
  `temperature=0.1`.

All five tool names are in the `observe`-tier autonomy allow-list, so
autonomy turns can call them. Full reference at
[Vision Tools](../agent/tools/vision.md).

## Phase 5 status (landed 2026-05-02)

Endpoints + dashboard polish:

- **`/api/vision/ask` accepts video.** The endpoint now branches on
  the upload's MIME type — `image/*` builds an `image_url` content
  part (existing behavior), `video/*` builds a `video_url` content
  part. Unknown MIME → `415 Unsupported Media Type`. The legacy
  `image` form field still works; new callers should use `file`. A
  short clip (5-10 s 1080p H.264) is well within the new size budget.
- **Response shape now includes `model`.** Both `/api/vision/ask`
  and `/api/vision/ask_url` return the served-model name so the
  dashboard can label which tier (Tier 1/2/3 from the fallback
  ladder) handled the call. `/api/vision/ask` additionally returns
  `media_type` (`image_url` / `video_url`).
- **Playground page upgraded.** The dashboard playground at
  `/playgrounds/vision` accepts image+video, renders the appropriate
  preview element (`<img>` vs `<video controls>`), exposes four
  preset prompts (Describe scene / What's unusual? / Read all
  text / Identify objects), and displays a served-model badge plus a
  prompt/completion token breakdown alongside latency.
- **nginx upload cap raised.** `client_max_body_size` on the `/api/`
  location went from 25 MB to 100 MB so a typical short video clip
  doesn't 413. The previous cap was sized for phone-camera photos
  only.

The model-selection ladder above is unchanged — Qwen3-VL natively
supports both image and video, and `vllm-vision`'s
`--limit-mm-per-prompt '{"image":4,"video":1}'` flag was already in
place from Phase 1.
