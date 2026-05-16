"""Streaming TTS synthesis — yields NDJSON-shaped event dicts per sentence.

The buffered ``/v1/audio/speech`` endpoint synthesizes the whole utterance,
runs Rhubarb on the full WAV, and returns audio + a single base64 viseme
JSON in the ``X-Visemes`` response header. That shape broke at ~30 s of
speech (aiohttp's 8190-byte default ``max_field_size`` rejected the long
header on the agent side) and forced clients to wait for the full reply
before any audio played.

This module synthesizes sentence-by-sentence and emits a stream of NDJSON
events: ``start``, then alternating ``audio`` + ``visemes`` per sentence,
then ``done`` (or ``error`` if a sentence fails). The client reassembles
playback in order using the ``seq`` and ``offset_ms`` fields — it never
has to compute cumulative offsets itself.

The reference-voice prompt is encoded once at stream start (heavy: librosa
load + loudness norm + s3gen.embed_ref + voice-encoder forward pass —
see chatterbox/tts_turbo.py:251 ``prepare_conditionals``). Per-sentence
``model.generate(text)`` calls then reuse ``model.conds`` instead of
re-encoding the prompt N times. Without this hoist a 5-sentence reply
would re-run the prompt pipeline 5×.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import uuid
from typing import AsyncIterator

import numpy as np
import soundfile as sf
import torch

import config
import pronunciation
import rhubarb
from sentences import split_sentences

logger = logging.getLogger("text-to-speech-v2.streaming")

# Chatterbox-Turbo stores the reference-voice conditionals in
# ``model.conds`` as mutable instance state. A concurrent request that
# calls ``prepare_conditionals`` (or ``generate(audio_prompt_path=...)``,
# which calls it implicitly) would corrupt this stream's conds mid-
# utterance. We hold this lock from the start of ``prepare_conditionals``
# through the last ``generate`` call. The buffered endpoint does not
# acquire this lock today — on this single-user host concurrent TTS is
# rare enough that the pre-existing race is accepted rather than retrofit.
_model_lock = threading.Lock()


def _samples_from_tensor(wav) -> np.ndarray:
    """Convert the model's torch.Tensor output to mono float32 numpy."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    samples = np.asarray(wav, dtype=np.float32)
    if samples.ndim == 2:
        # ChatterboxTurboTTS returns [1, N]; flatten the leading dim.
        samples = samples.squeeze(0) if samples.shape[0] == 1 else samples.mean(axis=0)
    return samples


def _encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Force WAV (PCM_16) on the streaming path.

    MP3/Opus/Flac would need fragmented framing that Chatterbox doesn't
    emit — they assume a contiguous bitstream. WAV is trivially splittable
    (each chunk is its own self-contained file) which is what the client's
    ``ConcatenatingMediaSource`` / sequential ``Audio`` elements need.
    """
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _duration_ms(samples: np.ndarray, sample_rate: int) -> int:
    if sample_rate <= 0:
        return 0
    return int(round((samples.shape[0] / sample_rate) * 1000))


async def synthesize_stream(
    model,
    *,
    text: str,
    audio_prompt_path: str | None,
    voice_name: str | None,
    sample_rate: int,
) -> AsyncIterator[dict]:
    """Yield NDJSON event dicts for a streaming TTS response.

    Event order: ``start`` → (``audio``, ``visemes``) × N → ``done``.
    On per-sentence failure: ``error`` is emitted and iteration stops.
    On empty input: a single ``error`` event with ``detail="empty input"``.
    """
    stream_id = uuid.uuid4().hex
    sentences = split_sentences(text)
    if not sentences:
        yield {"type": "error", "seq": 0, "detail": "empty input"}
        return

    logger.info(
        "stream %s: %d sentences, voice=%r prompt=%r",
        stream_id, len(sentences), voice_name,
        audio_prompt_path.rsplit("/", 1)[-1] if audio_prompt_path else None,
    )

    yield {
        "type": "start",
        "stream_id": stream_id,
        "sample_rate": sample_rate,
        "audio_format": "wav",
        "audio_mime": "audio/wav",
        "total_sentences": len(sentences),
        "voice": voice_name,
    }

    loop = asyncio.get_event_loop()

    def _prep() -> None:
        with _model_lock:
            if audio_prompt_path:
                model.prepare_conditionals(audio_prompt_path)

    try:
        await loop.run_in_executor(None, _prep)
    except Exception as e:
        logger.error("stream %s: prepare_conditionals failed: %s", stream_id, e, exc_info=True)
        yield {"type": "error", "seq": 0, "detail": f"prepare_conditionals failed: {e}"}
        return

    offset_ms = 0
    for seq, sentence in enumerate(sentences):
        try:
            normalized = pronunciation.apply(sentence, config.TTS_PRONUNCIATIONS)

            def _gen(s: str = normalized) -> np.ndarray:
                with _model_lock:
                    wav = model.generate(s)
                return _samples_from_tensor(wav)

            samples = await loop.run_in_executor(None, _gen)
            audio_bytes = await loop.run_in_executor(None, _encode_wav, samples, sample_rate)
            dur_ms = _duration_ms(samples, sample_rate)

            yield {
                "type": "audio",
                "seq": seq,
                "offset_ms": offset_ms,
                "duration_ms": dur_ms,
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            }

            visemes = await loop.run_in_executor(None, rhubarb.compute, samples, sample_rate)
            cues = (visemes or {}).get("mouthCues") or []
            yield {
                "type": "visemes",
                "seq": seq,
                "offset_ms": offset_ms,
                "duration_ms": dur_ms,
                "cues": cues,
            }

            offset_ms += dur_ms
        except Exception as e:
            logger.error("stream %s: synthesis failed at seq=%d: %s", stream_id, seq, e, exc_info=True)
            yield {"type": "error", "seq": seq, "detail": f"sentence {seq} failed: {e}"}
            return

    logger.info("stream %s: done, total_ms=%d, sentences=%d", stream_id, offset_ms, len(sentences))
    yield {"type": "done", "total_ms": offset_ms, "sentences": len(sentences)}
