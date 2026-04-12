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

The service is HTTP-only — the legacy streaming WebSocket on port 6000 was retired. The dashboard STT playground uses the record-then-upload HTTP endpoint.

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

Environment variables:

```bash
STT_DEVICE="0"           # GPU device index
WHISPER_MODEL="base"     # Model size: tiny, base, small, medium, large
SRC_LAN="en"             # Default language
```

## Model options

| Model  | Size    | VRAM  | Speed   | Accuracy |
|--------|---------|-------|---------|----------|
| tiny   | 39 MB   | ~1GB  | Fastest | Lowest   |
| base   | 74 MB   | ~1GB  | Fast    | Good     |
| small  | 244 MB  | ~2GB  | Medium  | Better   |
| medium | 769 MB  | ~5GB  | Slow    | High     |
| large  | 1550 MB | ~10GB | Slowest | Best     |

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
