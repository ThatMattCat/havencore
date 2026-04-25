# face-recognition

Event-driven facial recognition service for HavenCore. Loads InsightFace
(`buffalo_l`: RetinaFace detect + ArcFace R100 embed) on a dedicated GPU and
exposes an HTTP API that other HavenCore services consume.

Implementation tracked in `~/.claude/plans/parallel-purring-sundae.md`.

## Status

Step 1 of 9: service skeleton + InsightFace smoke test.

- [x] FastAPI lifespan loads `buffalo_l` on the configured GPU
- [x] `/health` reports model/provider/load-time readiness
- [ ] Postgres + Qdrant bootstrap (step 2)
- [ ] People CRUD + enrollment HTTP API (step 3)
- [ ] HA snapshot burst + identification pipeline (step 4)
- [ ] MQTT bridge (step 5)
- [ ] Agent MCP module (step 6)
- [ ] SvelteKit `/people` UI (step 7)
- [ ] Continuous improvement + retention sweeper (step 8)
- [ ] Docs (step 9)

## GPU pinning

Pinned via compose to `CUDA_VISIBLE_DEVICES=3`. Inside the container the GPU
appears as `cuda:0`, so `FACE_REC_CTX_ID=0`. Move to GPU 4 when the 5th GPU
comes online by bumping `CUDA_VISIBLE_DEVICES` in `compose.yaml`.

## Smoke test

```bash
docker compose build face-recognition
docker compose up -d face-recognition
docker compose logs -f face-recognition       # expect "InsightFace buffalo_l loaded on CUDA"
curl http://localhost:6006/health
nvidia-smi                                    # face-recognition process on GPU 3
```

`/health` should return `ready: true` and `providers` containing
`CUDAExecutionProvider`. CPU-only fallback is logged as a warning.

## Configuration

All env vars are optional; defaults match the implementation plan.

| Var | Default | Notes |
|---|---|---|
| `FACE_REC_ENABLED` | `true` | Set false to skip model load (debugging) |
| `FACE_REC_PORT` | `6006` | |
| `FACE_REC_MODEL_PACK` | `buffalo_l` | InsightFace pack name |
| `FACE_REC_CTX_ID` | `0` | Local CUDA index after `CUDA_VISIBLE_DEVICES` |
| `FACE_REC_DET_SIZE` | `640` | RetinaFace input edge length |
| `FACE_REC_GPU_DEVICE` | `3` | Informational; actual pin is in compose |

Pipeline-stage env vars (`FACE_REC_MATCH_THRESHOLD`, `FACE_REC_BURST_FRAMES`,
etc.) are parsed but not yet used in step 1.
