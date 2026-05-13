import json
import os

import shared.configs.shared_config as _shared_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE_ID = "speech-to-text"
DEVICE_TYPE = "AI-STT"
WARMUP_FILE = "assets/jfk-warmup.wav"
SAMPLE_RATE = 16000
WHISPER_MODEL = "distil-large-v3"

# Bias Whisper's decoder toward the assistant's proper noun. Whisper's
# initial_prompt is a short context string the decoder is conditioned on;
# proper nouns that appear in it become much more likely in the output.
# Default repeats AGENT_NAME in several varied contexts — one mention isn't
# enough to overcome the "Celine" homophone prior on distil-large-v3.
_AN = _shared_config.AGENT_NAME
STT_INITIAL_PROMPT = os.getenv(
    "STT_INITIAL_PROMPT",
    f"My name is {_AN}. Hey {_AN}. Hi {_AN}, how are you? "
    f"{_AN} is your AI assistant. Thanks, {_AN}.",
)

# Hotwords get tokenized and prepended to the decoder prompt (faster-whisper
# transcribe.py:1500-1506) — same mechanism as initial_prompt, just an extra
# nudge per segment. Useful but not a logit-level override.
STT_HOTWORDS = os.getenv("STT_HOTWORDS", _AN)

# Post-transcription substitutions for homophone-spelling cases that prompt
# biasing can't fully fix (e.g., Whisper hears /səˈlin/ correctly but spells
# it "Celine" because that's the more common surname). JSON object mapping
# wrong-spelling -> correct-spelling. Word-boundary, case-preserving regex
# sub is applied after the transcript is assembled.
_DEFAULT_TRANSCRIPT_SUBS = json.dumps({"Celine": _AN})
try:
    STT_TRANSCRIPT_SUBSTITUTIONS = json.loads(
        os.getenv("STT_TRANSCRIPT_SUBSTITUTIONS", _DEFAULT_TRANSCRIPT_SUBS)
    )
    if not isinstance(STT_TRANSCRIPT_SUBSTITUTIONS, dict):
        STT_TRANSCRIPT_SUBSTITUTIONS = {}
except (json.JSONDecodeError, ValueError):
    STT_TRANSCRIPT_SUBSTITUTIONS = {}
