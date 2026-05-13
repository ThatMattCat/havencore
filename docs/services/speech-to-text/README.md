# Speech-to-Text Service

Audio transcription via Faster Whisper, exposed as an OpenAI-compatible endpoint.

## Purpose

- Audio transcription using Whisper models
- Multiple audio-format support
- Real-time and batch processing
- OpenAI-compatible API

## Architecture

```
Audio Input → Preprocessing → Whisper Model → Text Output
     ↓              ↓             ↓            ↓
Format Detection  Normalization  GPU Inference  Post-processing
```

## Service endpoints

| Port | Purpose | API Type |
|------|---------|----------|
| 6001 | OpenAI-compatible API (`/v1/audio/transcriptions`, `/health`) | REST |

The service is HTTP-only — the legacy streaming WebSocket on port 6000 was retired. The dashboard STT playground and the Chat page's push-to-talk mic both use the record-then-upload HTTP endpoint via `/api/stt/transcribe`.

## Supported audio formats

- **WAV** — Uncompressed audio
- **MP3** — MPEG audio compression
- **MP4** — Video container with audio
- **M4A** — Apple audio format
- **FLAC** — Lossless compression
- **OGG** — Open-source audio format
- **WEBM** — Web-optimized audio

## API usage

```bash
# Transcribe audio file
curl -X POST http://localhost/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "language=en"
```

## Configuration

Env-driven:

```bash
STT_DEVICE="0"   # GPU device index (passed to Faster Whisper)
SRC_LAN="en"     # Default source language
```

The Whisper model is *not* an env var — it's a Python constant in
`services/speech-to-text/app/config.py` (currently `distil-large-v3`).
Edit the file and restart the service to change it:

```python
# services/speech-to-text/app/config.py
WHISPER_MODEL = "distil-large-v3"
```

### Proper-noun bias and homophone fix-ups

Whisper transcribes `AGENT_NAME` correctly phonetically but sometimes picks
the wrong spelling for homophones (e.g., "Selene" → "Celine"). Three opt-in
layers, in increasing strength:

```bash
# 1) initial_prompt — short context string fed to the decoder. Biases the
# whole transcription, more diffuse. Default repeats AGENT_NAME in several
# varied contexts so a single mention does not get diluted.
STT_INITIAL_PROMPT="My name is Selene. Hey Selene. Hi Selene, how are you? Selene is your AI assistant. Thanks, Selene."

# 2) hotwords — additional prompt-context tokens prepended per segment.
# Implemented in faster-whisper 1.1+ as an extra token block in the same
# decoder prompt buffer; helps once the name has appeared earlier in the
# same turn (condition_on_previous_text feeds it back), but does not
# logit-bias the first cold-start occurrence. Space-separated word list.
STT_HOTWORDS="Selene"

# 3) Transcript substitutions — the bulletproof layer. JSON object mapping
# wrong-spelling → correct-spelling, applied via word-boundary regex after
# transcription. Case-preserving (Celine→Selene, celine→selene, CELINE→
# SELENE). Use this when prompt-based bias hits the architectural ceiling.
STT_TRANSCRIPT_SUBSTITUTIONS='{"Celine":"Selene"}'
```

All three are applied together; lower layers help reduce the number of
substitutions needed but are not sufficient on their own for homophone
spellings. Extend the substitutions dict with more pairs as you discover
other names with the same problem:

```bash
STT_TRANSCRIPT_SUBSTITUTIONS='{"Celine":"Selene","Marie":"Murray"}'
```

Invalid JSON is silently ignored (substitution falls back to empty). The
mechanism is configured in `services/speech-to-text/app/config.py` and
applied in `app/main.py`'s `_apply_substitutions()`.

## Model options

| Model              | Size    | VRAM  | Speed   | Accuracy |
|--------------------|---------|-------|---------|----------|
| tiny               | 39 MB   | ~1GB  | Fastest | Lowest   |
| base               | 74 MB   | ~1GB  | Fast    | Good     |
| small              | 244 MB  | ~2GB  | Medium  | Better   |
| medium             | 769 MB  | ~5GB  | Slow    | High     |
| large-v3           | 1550 MB | ~10GB | Slow    | Best     |
| distil-large-v3    | ~750 MB | ~4GB  | Fast    | Near-best (default) |

## Performance optimization

```python
# Batch processing for multiple files
batch_size = 4
device = "cuda:0"

# Memory optimization
torch.cuda.empty_cache()
```

## Troubleshooting

```bash
# Check GPU usage
nvidia-smi

# Test service directly
curl http://localhost:6001/health

# View processing logs
docker compose logs -f speech-to-text

# Check model loading
docker compose exec speech-to-text ls -la /models/
```
