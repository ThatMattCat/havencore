"""Rhubarb Lip Sync post-process.

Ported verbatim from services/text-to-speech/app/main.py:151–202 so the
X-Visemes header the companion app consumes (see havencore-companion-app
app/src/main/kotlin/.../voice/avatar/VisemeTimeline.kt) stays byte-for-byte
identical between v1 (Kokoro) and v2 (Chatterbox).
"""
import json
import logging
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

import config

logger = logging.getLogger("text-to-speech-v2.rhubarb")


def compute(samples: np.ndarray, sample_rate: int) -> dict | None:
    """Run Rhubarb on float32 PCM ``samples`` and return the parsed viseme timeline.

    Returns ``None`` if Rhubarb is unavailable, times out, or errors —
    callers should treat the absence of an X-Visemes header as the
    soft-fallback signal (client renders silent mouth).
    """
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="tts_visemes_")
    os.close(wav_fd)
    json_fd, json_path = tempfile.mkstemp(suffix=".json", prefix="tts_visemes_")
    os.close(json_fd)
    try:
        # Rhubarb requires 16-bit PCM WAV; libsndfile's float default is rejected.
        sf.write(wav_path, samples, sample_rate, format="WAV", subtype="PCM_16")
        result = subprocess.run(
            [config.RHUBARB_BIN, "-f", "json", "-r", config.RHUBARB_RECOGNIZER,
             "-o", json_path, wav_path],
            capture_output=True,
            timeout=config.RHUBARB_TIMEOUT_SEC,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "rhubarb exited %d: %s",
                result.returncode,
                (result.stderr or "").strip()[:500],
            )
            return None
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(
            "rhubarb binary not found at %r; X-Visemes header disabled",
            config.RHUBARB_BIN,
        )
        return None
    except subprocess.TimeoutExpired:
        logger.warning(
            "rhubarb timed out after %.1fs; skipping visemes",
            config.RHUBARB_TIMEOUT_SEC,
        )
        return None
    except Exception as e:
        logger.warning("rhubarb failed: %s", e)
        return None
    finally:
        for p in (wav_path, json_path):
            try:
                os.unlink(p)
            except OSError:
                pass
