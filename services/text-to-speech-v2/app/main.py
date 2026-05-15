"""text-to-speech-v2 — OpenAI-compatible TTS using Chatterbox-Turbo.

Drop-in compatible with services/text-to-speech (Kokoro/v1) on the API
surface that matters to callers:

    POST /v1/audio/speech   — same request schema (input/voice/response_format/speed)
                              same `X-Visemes` response header (base64 Rhubarb JSON)
                              same `Access-Control-Expose-Headers` CORS hint
    GET  /v1/voices         — same shape (language/default/native/aliases)
    GET  /health            — same shape ({"status":"healthy"})

Architectural deltas from v1:
  - Engine is Chatterbox-Turbo (Resemble AI, MIT). Zero-shot — voice
    selection works by routing the request's `voice` name to a reference
    WAV in /app/voices/ or /app/voices.bundled/.
  - No `exaggeration` / `voice_intensity` slider — Turbo doesn't expose
    one. Expressiveness comes from the model's paralinguistic tags:
    [laugh] [sigh] [chuckle] [cough] [gasp] [clear throat] [shush]
    [groan] [sniff] inserted directly into the input text.
  - Pronunciation overrides are text-substitution (no lexicon hook
    upstream; see app/pronunciation.py).
"""
import base64
import io
import json
import logging
import os

import re

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
import pronunciation
import rhubarb
import voices

# Logging — match v1's pattern: prefer the shared Loki logger when running
# inside the compose stack, fall back to stdlib in SOLO mode.
if not config.SOLO:
    import shared.scripts.logger as logger_module
    logger = logger_module.get_logger("loki")
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("text-to-speech-v2")


# --- Model load --------------------------------------------------------------

# Chatterbox-Turbo lives under chatterbox.tts_turbo; the import path is
# intentionally specific so a future swap to the standard `ChatterboxTTS`
# (which exposes exaggeration/cfg_weight) is a one-line change.
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: E402

_device = config.MODEL_DEVICE if torch.cuda.is_available() else "cpu"
logger.info("Loading Chatterbox-Turbo on device=%s ...", _device)
model = ChatterboxTurboTTS.from_pretrained(device=_device)
SAMPLE_RATE = int(getattr(model, "sr", 24000))
logger.info("Chatterbox-Turbo ready: sample_rate=%d, device=%s", SAMPLE_RATE, _device)
_initial = voices.registry()
logger.info(
    "Registered voices: %d (default=%r)",
    len(_initial), voices.default_voice(_initial),
)


# --- Audio encoding (ported from v1 main.py:104-148) -------------------------

# Formats libsndfile can encode directly from raw samples. mp3/aac fall back
# to WAV since libsndfile can't produce them without extra codec libs — we
# match v1's behavior (honest content-type, no mislabeled bytes).
_SF_ENCODERS = {
    "wav":  ("WAV",  "PCM_16", "audio/wav"),
    "flac": ("FLAC", None,     "audio/flac"),
    "ogg":  ("OGG",  "VORBIS", "audio/ogg"),
    "opus": ("OGG",  "OPUS",   "audio/ogg; codecs=opus"),
}


def encode_audio(samples: np.ndarray, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "wav").lower()

    if fmt == "pcm":
        pcm16 = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
        return pcm16, f"audio/L16; rate={SAMPLE_RATE}"

    spec = _SF_ENCODERS.get(fmt)
    if spec is None:
        logger.warning(
            "Requested TTS format %r not supported by libsndfile; "
            "returning WAV with Content-Type: audio/wav",
            fmt,
        )
        spec = _SF_ENCODERS["wav"]

    sf_format, subtype, content_type = spec
    buf = io.BytesIO()
    try:
        if subtype:
            sf.write(buf, samples, SAMPLE_RATE, format=sf_format, subtype=subtype)
        else:
            sf.write(buf, samples, SAMPLE_RATE, format=sf_format)
    except Exception as e:
        logger.warning(
            "soundfile failed to encode as %s/%s (%s); falling back to WAV",
            sf_format, subtype, e,
        )
        buf = io.BytesIO()
        sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        content_type = "audio/wav"
    return buf.getvalue(), content_type


# --- Synthesis ---------------------------------------------------------------

def synthesize(text: str, audio_prompt_path: str | None) -> np.ndarray:
    """Run Chatterbox-Turbo and return float32 PCM samples at SAMPLE_RATE."""
    text = pronunciation.apply(text, config.TTS_PRONUNCIATIONS)
    kwargs = {}
    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path
    wav = model.generate(text, **kwargs)

    # ChatterboxTurboTTS.generate returns a torch.Tensor shaped [1, N] or [N].
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    samples = np.asarray(wav, dtype=np.float32)
    if samples.ndim == 2:
        # Mono — flatten in case the model emits a leading channel dim.
        samples = samples.squeeze(0) if samples.shape[0] == 1 else samples.mean(axis=0)
    return samples


# --- API ---------------------------------------------------------------------

app = FastAPI(title="HavenCore TTS v2 (Chatterbox-Turbo)")


class SpeechRequest(BaseModel):
    # Field names mirror v1 / OpenAI's audio.speech schema exactly.
    input: str
    voice: str | None = Field(default=None)
    response_format: str | None = Field(default="mp3")
    # Accepted for compat; Chatterbox-Turbo doesn't expose a speed knob, so
    # this is currently a no-op. Logged when != 1.0 so we notice if callers
    # depend on it.
    speed: float | None = Field(default=1.0)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "healthy"})


@app.get("/v1/voices")
def list_voices() -> JSONResponse:
    reg = voices.registry()
    native = sorted(reg.keys())
    user = voices.user_voices()
    return JSONResponse({
        # v1 reports a Kokoro language code; for v2 we report the engine
        # name in the same slot so clients that show this in a UI still get
        # something meaningful. Field name kept the same for drop-in compat.
        "language": "chatterbox-turbo",
        "default": voices.default_voice(reg),
        "native": native,
        # New v2-only fields the agent uses to drive a delete/manage UI:
        "user": user,
        "bundled": [n for n in native if n not in user],
        "aliases": sorted(voices.OPENAI_VOICE_ALIASES),
    })


# Validate uploaded voice names — keeps the filesystem path stable and
# prevents directory traversal via crafted names like "../etc/passwd".
_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


@app.post("/v1/voices/upload")
async def upload_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    """Save an uploaded reference clip under /app/voices/<name>.wav.

    Accepts WAV/FLAC/OGG (anything libsndfile can read). The file is loaded,
    converted to mono, resampled to 24 kHz, and written as 16-bit PCM WAV.
    Custom uploads override bundled voices with the same name.
    """
    if not _VOICE_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Voice name must be 1-40 chars of [A-Za-z0-9_-]",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        # soundfile reads from a BytesIO directly; we get back float32 PCM
        # at whatever sample rate the upload was recorded in.
        import io as _io
        data, sr = sf.read(_io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Could not decode upload as audio ({e}). "
                "Supported formats: WAV, FLAC, OGG. "
                "Convert MP3 to WAV before uploading."
            ),
        )

    # Mix to mono.
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    # Reject obviously-too-short clips so the user gets a clear error rather
    # than a mysterious model output. Chatterbox recommends ~10s of speech.
    duration_sec = arr.shape[0] / sr if sr else 0
    if duration_sec < 3.0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Recording is only {duration_sec:.1f}s long. "
                "Provide at least 3 seconds — 10-30 seconds of clean, "
                "natural speech (single speaker, minimal background "
                "noise) gives best results."
            ),
        )
    if duration_sec > 120.0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Recording is {duration_sec:.1f}s long. "
                "Keep reference clips under 2 minutes — "
                "10-30 seconds is plenty."
            ),
        )

    # Resample to 24 kHz to match the model's native rate. Chatterbox would
    # resample internally anyway, but doing it once here means subsequent
    # generations skip the resample on every request.
    if sr != SAMPLE_RATE:
        try:
            import librosa  # bundled via chatterbox-tts deps
            arr = librosa.resample(arr, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            logger.warning(
                "librosa not available; saving at original sample rate %d "
                "(Chatterbox will resample per request)", sr,
            )

    out_path = os.path.join(config.VOICES_USER_DIR, f"{name}.wav")
    os.makedirs(config.VOICES_USER_DIR, exist_ok=True)
    sf.write(out_path, arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    logger.info(
        "Saved uploaded voice %r → %s (%.1fs, %d Hz original)",
        name, out_path, duration_sec, sr,
    )
    return JSONResponse({
        "name": name,
        "path": out_path,
        "duration_sec": round(duration_sec, 2),
        "original_sample_rate": sr,
        "stored_sample_rate": SAMPLE_RATE,
    })


@app.delete("/v1/voices/{name}")
def delete_voice(name: str) -> JSONResponse:
    """Delete an uploaded voice. Bundled voices cannot be deleted."""
    if not _VOICE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid voice name")
    user_path = os.path.join(config.VOICES_USER_DIR, f"{name}.wav")
    if not os.path.isfile(user_path):
        # Either doesn't exist or is a bundled-only voice (read-only).
        bundled_path = os.path.join(config.VOICES_BUNDLED_DIR, f"{name}.wav")
        if os.path.isfile(bundled_path):
            raise HTTPException(
                status_code=403,
                detail=f"{name!r} is a bundled voice and cannot be deleted",
            )
        raise HTTPException(status_code=404, detail=f"Voice {name!r} not found")
    os.unlink(user_path)
    logger.info("Deleted uploaded voice %r", name)
    return JSONResponse({"deleted": name})


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if not req.input:
        raise HTTPException(status_code=400, detail="Missing required parameter 'input'")

    voice_name, audio_prompt_path = voices.resolve(req.voice)
    if req.speed and abs(req.speed - 1.0) > 1e-3:
        logger.info(
            "Ignoring speed=%s (Chatterbox-Turbo has no speed param)", req.speed,
        )

    logger.info(
        "Generating speech: voice=%r prompt=%r len(input)=%d preview=%r",
        voice_name,
        os.path.basename(audio_prompt_path) if audio_prompt_path else None,
        len(req.input),
        req.input[:100],
    )

    try:
        samples = synthesize(req.input, audio_prompt_path)
    except Exception as e:
        logger.error("synthesis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {e}")

    audio_bytes, content_type = encode_audio(samples, req.response_format or "mp3")

    visemes = rhubarb.compute(samples, SAMPLE_RATE)
    visemes_b64: str | None = None
    if visemes is not None:
        try:
            visemes_b64 = base64.b64encode(
                json.dumps(visemes, separators=(",", ":")).encode()
            ).decode("ascii")
        except Exception as e:
            logger.warning("failed to base64-encode visemes: %s", e)

    headers = {"Content-Length": str(len(audio_bytes))}
    if visemes_b64:
        headers["X-Visemes"] = visemes_b64
        # Browser clients (dashboard) can only read X-* response headers when
        # explicitly exposed via CORS — matches v1's behavior so the UI
        # playgrounds don't need changes.
        headers["Access-Control-Expose-Headers"] = "X-Visemes"

    cue_count = len(visemes.get("mouthCues", [])) if visemes else 0
    logger.info(
        "Served %d bytes as %s (voice=%s, response_format=%r, visemes=%s)",
        len(audio_bytes), content_type, voice_name, req.response_format,
        "omitted" if visemes is None else f"{cue_count} cues",
    )

    return Response(content=audio_bytes, media_type=content_type, headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="info")
