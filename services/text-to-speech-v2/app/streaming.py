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
import os
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

# Cache of ``Conditionals`` objects produced by
# ``ChatterboxTurboTTS.prepare_conditionals``. That call is the dominant
# TTFA cost (~3-4 s of librosa load + loudness norm + voice-encoder +
# s3gen.embed_ref + s3 tokenizer) and runs once per stream today. On this
# single-user host the same voice is reused turn after turn, so caching
# the resulting conds object lets subsequent streams skip the prep step
# entirely. Cache key is ``(absolute path, mtime)`` so any rewrite of the
# voice file (re-upload, manual edit) naturally invalidates. No LRU —
# bundled (~20) + user-uploaded (typically 0-5) voices fit comfortably
# in GPU memory at ~tens of MB each. Reads + writes happen under
# ``_model_lock`` to avoid a race where one stream's cache-hit assignment
# of ``model.conds`` clobbers another stream's prep mid-generate.
_conds_cache: dict[tuple[str, float], object] = {}


def invalidate_conds_cache(audio_prompt_path: str) -> None:
    """Purge any cached conds for ``audio_prompt_path`` (any mtime).

    Called from ``main.py:delete_voice`` so a deleted voice doesn't keep
    a Conditionals object pinned in GPU memory and so a later re-upload
    at the same path can't accidentally hit a stale cache entry if the
    new file lands at the same mtime (unlikely but possible).
    """
    abs_path = os.path.abspath(audio_prompt_path)
    with _model_lock:
        stale = [k for k in _conds_cache if k[0] == abs_path]
        for k in stale:
            _conds_cache.pop(k, None)
    if stale:
        logger.info(
            "conds cache: purged %d entr%s for %s",
            len(stale), "y" if len(stale) == 1 else "ies",
            os.path.basename(abs_path),
        )


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
        if not audio_prompt_path:
            return
        abs_path = os.path.abspath(audio_prompt_path)
        try:
            mtime = os.path.getmtime(abs_path)
            key: tuple[str, float] | None = (abs_path, mtime)
        except OSError:
            # File vanished between voices.resolve() and now; let
            # prepare_conditionals raise its own clear error.
            key = None
        with _model_lock:
            cached = _conds_cache.get(key) if key is not None else None
            if cached is not None:
                # Conditionals.to(device) was already applied when the
                # entry was first stored; reassignment is a pointer swap,
                # no GPU work.
                model.conds = cached
                logger.debug("conds cache hit: %s", os.path.basename(abs_path))
                return
            model.prepare_conditionals(audio_prompt_path)
            if key is not None:
                _conds_cache[key] = model.conds
                logger.info(
                    "conds cache miss: prepared and stored %s (size=%d)",
                    os.path.basename(abs_path), len(_conds_cache),
                )

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
