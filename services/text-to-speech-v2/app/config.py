"""Config for text-to-speech-v2 (Chatterbox-Turbo).

Reads from shared_config so .env stays the single source of truth for the
whole stack. Falls back to env vars / sane defaults in SOLO mode for
out-of-compose development.
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# /app/voices is mounted as a volume so the operator can add custom reference
# clips (including a Selene clone later) without rebuilding the image.
# /opt/chatterbox-voices is baked in at build time (outside /app so the live
# source mount at /app doesn't shadow it).
VOICES_USER_DIR = os.getenv("CHATTERBOX_VOICES_DIR", "/app/voices")
VOICES_BUNDLED_DIR = os.getenv("CHATTERBOX_VOICES_BUNDLED_DIR", "/opt/chatterbox-voices")

################ Solo mode (single container outside of docker compose project) #######
SOLO = False

if SOLO:
    AGENT_NAME = "Selene"
    MODEL_DEVICE = "cuda:0"
    VOICE = "Olivia"
    TTS_PRONUNCIATIONS = {"Selene": "Suh-leen"}

############################ Do not make changes below this line ######################
else:
    import shared.configs.shared_config as shared_config
    AGENT_NAME = shared_config.AGENT_NAME or "Selene"
    MODEL_DEVICE = os.getenv("CHATTERBOX_DEVICE", "cuda:0")
    VOICE = os.getenv("CHATTERBOX_VOICE", "Olivia")
    # Text-substitution map, applied to the input before synthesis.
    # Distinct from v1's TTS_PRONUNCIATIONS — that one is IPA injected into
    # Kokoro's misaki lexicon. Chatterbox has no lexicon hook
    # (resemble-ai/chatterbox#115), so we work around it by rewriting the
    # spelling at the input level.
    #
    # Empty by default — Chatterbox-Turbo's text encoder handles most proper
    # nouns reasonably without rewriting. Set TTS_V2_PRONUNCIATIONS in .env
    # only when the model mispronounces a word badly enough to fix.
    _raw = os.getenv("TTS_V2_PRONUNCIATIONS", "")
    try:
        TTS_PRONUNCIATIONS = json.loads(_raw) if _raw else {}
    except json.JSONDecodeError:
        TTS_PRONUNCIATIONS = {}

API_HOST = os.getenv("CHATTERBOX_HOST", "0.0.0.0")
API_PORT = int(os.getenv("CHATTERBOX_PORT", "6015"))

# Rhubarb knobs match v1's env names so the same .env block applies to both
# services (operators can keep one configuration for both).
RHUBARB_BIN = os.getenv("RHUBARB_BIN", "rhubarb")
RHUBARB_TIMEOUT_SEC = float(os.getenv("RHUBARB_TIMEOUT_SEC", "10"))
RHUBARB_RECOGNIZER = os.getenv("RHUBARB_RECOGNIZER", "phonetic")
