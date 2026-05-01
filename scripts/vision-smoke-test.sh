#!/usr/bin/env bash
# Phase 1 decision gate for the vllm-vision service.
#
# Runs the six acceptance criteria from docs/services/vllm-vision/README.md
# and exits non-zero if any of (steady-state VRAM, latency budget, sustained
# load) fail. The other criteria (cold-start, video parity, quality) are
# reported but don't block — the operator's eyes are the judge.
#
# Usage:
#   scripts/vision-smoke-test.sh                          # uses default fixtures + http://localhost:8001
#   VISION_BASE=http://10.0.0.1:8001 scripts/vision-smoke-test.sh
#   IMAGE=/path/to/test.jpg VIDEO=/path/to/clip.mp4 scripts/vision-smoke-test.sh
#
# Env vars:
#   VISION_BASE   default http://localhost:8001
#   VISION_MODEL  default gpt-4-vision (must match VISION_SERVED_NAME)
#   GPU_INDEX     default 4
#   IMAGE         path to a ~1MP test image (default: scripts/vision-smoke-test-fixtures/test.jpg)
#   VIDEO         optional path to a 5-second clip
#   N_QUERIES     sustained-load query count, default 50

set -euo pipefail

VISION_BASE="${VISION_BASE:-http://localhost:8001}"
VISION_MODEL="${VISION_MODEL:-gpt-4-vision}"
GPU_INDEX="${GPU_INDEX:-4}"
IMAGE="${IMAGE:-$(dirname "$0")/vision-smoke-test-fixtures/test.jpg}"
VIDEO="${VIDEO:-}"
N_QUERIES="${N_QUERIES:-50}"

# Acceptance thresholds (see docs/services/vllm-vision/README.md).
VRAM_HEADROOM_GB_MIN="1.5"   # criterion 2: VRAM used must leave >=1.5GB headroom on a 24GB card -> used <=22.5GB
P50_BUDGET_S="8"             # criterion 3a
P95_BUDGET_S="15"            # criterion 3b

red()    { printf '\033[31m%s\033[0m\n' "$*"; }
green()  { printf '\033[32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { red "missing dependency: $1"; exit 2; }
}

require curl
require jq
require nvidia-smi
require base64
require python3   # used for percentile math

if [[ ! -f "$IMAGE" ]]; then
  red "test image not found: $IMAGE"
  echo "   place a ~1MP jpg/png there, or set IMAGE=/path/to/your.jpg"
  exit 2
fi

bold "vllm-vision smoke test"
echo "  base:    $VISION_BASE"
echo "  model:   $VISION_MODEL"
echo "  GPU:     $GPU_INDEX"
echo "  image:   $IMAGE"
echo "  video:   ${VIDEO:-<not provided, video parity check will be skipped>}"
echo "  queries: $N_QUERIES"
echo

vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$GPU_INDEX" | tr -d ' '
}

vram_used_gb() {
  awk "BEGIN { printf \"%.2f\", $(vram_used_mib) / 1024 }"
}

# -------- 1. Cold-start / health ------------------------------------------
bold "[1/6] Health check"
t0=$(date +%s)
deadline=$(( t0 + 1300 ))
while :; do
  if curl -fsS --max-time 3 "$VISION_BASE/v1/models" >/dev/null 2>&1; then
    elapsed=$(( $(date +%s) - t0 ))
    green "  /v1/models is live after ${elapsed}s"
    break
  fi
  if (( $(date +%s) > deadline )); then
    red "  /v1/models did not come up within 1300s. Check 'docker compose logs vllm-vision'."
    exit 3
  fi
  sleep 5
done

# -------- 2. Build request payload ----------------------------------------
# Body is written to a tempfile and passed via curl --data-binary @file
# because a base64-encoded image overflows the shell's argv limit when inlined.
# The image is downscaled to MAX_IMAGE_DIM on its long edge so the request
# fits inside max-model-len; users drop in whatever they have without caring
# about resolution. The downscale is a copy in /tmp, not a mutation.
MAX_IMAGE_DIM="${MAX_IMAGE_DIM:-1280}"  # long-edge px; tweak via env if you want a bigger or smaller test
BODY_FILE=$(mktemp -t vision-smoke-body.XXXXXX.json)
VBODY_FILE=""
SHRUNK_IMAGE=""
trap 'rm -f "$BODY_FILE" "$VBODY_FILE" "$SHRUNK_IMAGE" 2>/dev/null || true' EXIT

bold "[1.5/6] Downscaling test image to ${MAX_IMAGE_DIM}px on long edge"
SHRUNK_IMAGE=$(mktemp -t vision-smoke-img.XXXXXX.jpg)
python3 - "$IMAGE" "$SHRUNK_IMAGE" "$MAX_IMAGE_DIM" <<'PY'
import sys
from PIL import Image
src, dst, max_dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
img = Image.open(src)
w, h = img.size
print(f"  source: {w}x{h}", file=sys.stderr)
if max(w, h) > max_dim:
    scale = max_dim / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    print(f"  resized: {img.size[0]}x{img.size[1]}", file=sys.stderr)
else:
    print(f"  already within {max_dim}px, no resize", file=sys.stderr)
img.convert("RGB").save(dst, "JPEG", quality=88)
PY

python3 - "$SHRUNK_IMAGE" "image/jpeg" "$VISION_MODEL" > "$BODY_FILE" <<'PY'
import base64, json, sys
img_path, mime, model = sys.argv[1], sys.argv[2], sys.argv[3]
with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
data_url = f"data:{mime};base64,{b64}"
print(json.dumps({
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the scene in this image: who/what is visible, where they are, and anything unusual. 2-3 sentences."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }],
    "temperature": 0.2,
    "max_tokens": 400,
    "stream": False,
}))
PY

# Helper: run a chat-completions request, surface the response body on HTTP
# errors instead of swallowing them. Returns 0 on success, prints body to
# stdout. On failure, prints body+code to stderr and returns the curl exit code.
post_json_body() {
  local body_file="$1" tmp_resp tmp_code rc
  tmp_resp=$(mktemp); tmp_code=$(mktemp)
  curl -sS --max-time "${2:-120}" \
    -o "$tmp_resp" \
    -w "%{http_code}" \
    -H 'Content-Type: application/json' \
    --data-binary "@$body_file" \
    "$VISION_BASE/v1/chat/completions" > "$tmp_code"
  rc=$?
  local code; code=$(cat "$tmp_code")
  if (( rc != 0 )) || [[ "$code" != "200" ]]; then
    {
      red "  HTTP $code (curl rc=$rc)"
      echo "  response body:"
      head -c 1000 "$tmp_resp"
      echo
    } >&2
    rm -f "$tmp_resp" "$tmp_code"
    return $(( rc != 0 ? rc : 1 ))
  fi
  cat "$tmp_resp"
  rm -f "$tmp_resp" "$tmp_code"
}

# -------- 3. Quality sanity check (single call) ---------------------------
bold "[2/6] Quality sanity check (single image)"
sample_resp=$(post_json_body "$BODY_FILE" 120) || { red "  request failed"; exit 3; }
sample_text=$(echo "$sample_resp" | jq -r '.choices[0].message.content')
echo "  description: $(echo "$sample_text" | head -c 400)..."
echo

# -------- 4. Sustained load + latency distribution -------------------------
bold "[3/6] Sustained load: $N_QUERIES sequential single-image queries"
LATENCIES=()
ERRORS=0
for i in $(seq 1 "$N_QUERIES"); do
  start_ns=$(date +%s%N)
  if ! post_json_body "$BODY_FILE" 120 >/dev/null 2>/dev/null; then
    ERRORS=$((ERRORS+1))
    printf '!'
    continue
  fi
  end_ns=$(date +%s%N)
  ms=$(( (end_ns - start_ns) / 1000000 ))
  LATENCIES+=("$ms")
  if (( i % 10 == 0 )); then printf ' %d ' "$i"; else printf '.'; fi
done
echo

# Stats via python (more reliable than awk for percentiles).
LAT_CSV=$(IFS=,; echo "${LATENCIES[*]}")
read -r p50 p95 mean <<EOF
$(python3 -c "
import statistics, sys
xs = sorted(int(x) for x in '$LAT_CSV'.split(',') if x)
if not xs:
    print('0 0 0'); sys.exit(0)
def pct(p):
    k = max(0, min(len(xs)-1, int(round((p/100.0) * (len(xs)-1)))))
    return xs[k]
print(f'{pct(50)} {pct(95)} {int(statistics.mean(xs))}')
")
EOF

p50_s=$(awk "BEGIN { printf \"%.2f\", $p50 / 1000 }")
p95_s=$(awk "BEGIN { printf \"%.2f\", $p95 / 1000 }")
mean_s=$(awk "BEGIN { printf \"%.2f\", $mean / 1000 }")

echo "  errors: $ERRORS / $N_QUERIES"
echo "  p50:  ${p50_s}s    p95: ${p95_s}s    mean: ${mean_s}s"

# -------- 5. Steady-state VRAM --------------------------------------------
bold "[4/6] Steady-state VRAM"
sleep 3
vram_gb=$(vram_used_gb)
echo "  GPU $GPU_INDEX VRAM in use: ${vram_gb} GB / 24.00 GB"
vram_headroom=$(awk "BEGIN { printf \"%.2f\", 24.00 - $vram_gb }")
echo "  headroom: ${vram_headroom} GB"

# -------- 6. Video parity (optional) --------------------------------------
bold "[5/6] Video parity"
if [[ -n "$VIDEO" && -f "$VIDEO" ]]; then
  vmime="video/mp4"; case "$VIDEO" in *.webm) vmime="video/webm" ;; *.mkv) vmime="video/x-matroska" ;; esac
  VBODY_FILE=$(mktemp -t vision-smoke-vbody.XXXXXX.json)
  python3 - "$VIDEO" "$vmime" "$VISION_MODEL" > "$VBODY_FILE" <<'PY'
import base64, json, sys
v_path, mime, model = sys.argv[1], sys.argv[2], sys.argv[3]
with open(v_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
vurl = f"data:{mime};base64,{b64}"
print(json.dumps({
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this short video clip in 2-3 sentences."},
            {"type": "video_url", "video_url": {"url": vurl}},
        ],
    }],
    "temperature": 0.2,
    "max_tokens": 300,
    "stream": False,
}))
PY
  v_resp=$(curl -fsS --max-time 180 -H 'Content-Type: application/json' --data-binary "@$VBODY_FILE" "$VISION_BASE/v1/chat/completions") \
    || { red "  video request failed"; echo "$v_resp"; }
  v_text=$(echo "$v_resp" | jq -r '.choices[0].message.content // "<no content>"')
  echo "  description: $(echo "$v_text" | head -c 400)..."
else
  yellow "  skipped (set VIDEO=/path/to/clip.mp4 to enable)"
fi

# -------- 7. Verdict -------------------------------------------------------
bold "[6/6] Verdict"
fail=0
ok() { green "  PASS  $*"; }
no() { red "  FAIL  $*"; fail=1; }

# Criterion 2: steady-state VRAM headroom
if awk "BEGIN { exit !($vram_headroom >= $VRAM_HEADROOM_GB_MIN) }"; then
  ok "VRAM headroom (${vram_headroom} GB >= ${VRAM_HEADROOM_GB_MIN} GB)"
else
  no "VRAM headroom (${vram_headroom} GB < ${VRAM_HEADROOM_GB_MIN} GB)"
fi

# Criterion 3: latency budget
if awk "BEGIN { exit !($p50_s <= $P50_BUDGET_S) }"; then ok "p50 (${p50_s}s <= ${P50_BUDGET_S}s)"; else no "p50 (${p50_s}s > ${P50_BUDGET_S}s)"; fi
if awk "BEGIN { exit !($p95_s <= $P95_BUDGET_S) }"; then ok "p95 (${p95_s}s <= ${P95_BUDGET_S}s)"; else no "p95 (${p95_s}s > ${P95_BUDGET_S}s)"; fi

# Criterion 4: no OOM under sustained load
if (( ERRORS == 0 )); then ok "sustained load (0 errors / $N_QUERIES)"; else no "sustained load ($ERRORS errors / $N_QUERIES)"; fi

echo
if (( fail == 0 )); then
  green "All gating criteria met. Phase 1 done — proceed to Phase 2."
  echo "  Record the result in docs/services/vllm-vision/README.md (Phase 1 result note)."
else
  red "Decision gate FAILED. Walk the fallback ladder in docs/services/vllm-vision/README.md."
  exit 1
fi
