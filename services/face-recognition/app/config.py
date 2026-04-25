"""Environment-driven configuration for the face-recognition service.

All knobs are read once at import time. `.env` changes require a container
restart (compose down/up), per HavenCore's cross-service convention.
"""

import os


def _bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --- Service ---
PORT = _int(os.getenv("FACE_REC_PORT"), 6006)
ENABLED = _bool(os.getenv("FACE_REC_ENABLED"), True)

# --- Model / GPU ---
# InsightFace's get_model() takes ctx_id which maps to the local CUDA index
# *after* CUDA_VISIBLE_DEVICES is applied. We pin the host GPU via compose
# (CUDA_VISIBLE_DEVICES=3) so ctx_id=0 is correct here.
MODEL_PACK = os.getenv("FACE_REC_MODEL_PACK", "buffalo_l")
CTX_ID = _int(os.getenv("FACE_REC_CTX_ID"), 0)
DET_SIZE = _int(os.getenv("FACE_REC_DET_SIZE"), 640)  # square; 640 is buffalo_l's default
GPU_DEVICE_LABEL = os.getenv("FACE_REC_GPU_DEVICE", "3")  # informational only; for /health

# --- Pipeline thresholds (used in step 4+, defined here so they're discoverable) ---
MATCH_THRESHOLD = _float(os.getenv("FACE_REC_MATCH_THRESHOLD"), 0.50)
IMPROVEMENT_THRESHOLD = _float(os.getenv("FACE_REC_IMPROVEMENT_THRESHOLD"), 0.65)
QUALITY_FLOOR = _float(os.getenv("FACE_REC_QUALITY_FLOOR"), 0.40)
IMPROVEMENT_QUALITY_FLOOR = _float(os.getenv("FACE_REC_IMPROVEMENT_QUALITY_FLOOR"), 0.65)
MAX_EMBEDDINGS_PER_PERSON = _int(os.getenv("FACE_REC_MAX_EMBEDDINGS_PER_PERSON"), 50)

# --- Capture (used in step 4+) ---
TRIGGER_MODE = os.getenv("FACE_REC_TRIGGER_MODE", "ha_person_detected")
BURST_FRAMES = _int(os.getenv("FACE_REC_BURST_FRAMES"), 6)
BURST_INTERVAL_MS = _int(os.getenv("FACE_REC_BURST_INTERVAL_MS"), 500)

# --- Snapshots / retention (used in step 8) ---
SNAPSHOT_DIR = os.getenv("FACE_REC_SNAPSHOT_DIR", "/data/snapshots")
RETENTION_UNKNOWN_DAYS = _int(os.getenv("FACE_SNAPSHOT_RETENTION_UNKNOWN_DAYS"), 30)
RETENTION_KNOWN_DAYS = _int(os.getenv("FACE_SNAPSHOT_RETENTION_KNOWN_DAYS"), 7)

# --- MQTT bridge (used in step 5+) ---
# Default broker host matches the docker compose service name; the agent's
# autonomy listener and mcp_mqtt_tools both use the same env vars.
MQTT_ENABLED = _bool(os.getenv("FACE_REC_MQTT_ENABLED"), True)
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = _int(os.getenv("MQTT_PORT"), 1883)
MQTT_CLIENT_ID = os.getenv("FACE_REC_MQTT_CLIENT_ID", "havencore-face-recognition")
MQTT_RECONNECT_MAX_SEC = _int(os.getenv("FACE_REC_MQTT_RECONNECT_MAX_SEC"), 60)
