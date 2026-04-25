"""FastAPI entrypoint for the face-recognition service.

Step 1 scope: model loads on GPU at startup, /health reports readiness.
Database, Qdrant, MQTT, and pipeline endpoints arrive in later steps.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from embedder import embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("face-recognition")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.ENABLED:
        logger.warning("FACE_REC_ENABLED=false — skipping model load")
        yield
        return

    logger.info("Loading InsightFace %s ...", config.MODEL_PACK)
    embedder.prepare()
    logger.info("Service ready on port %d", config.PORT)
    yield
    logger.info("Shutting down face-recognition service")


app = FastAPI(title="HavenCore Face Recognition", lifespan=lifespan)


@app.get("/health")
def health():
    info = embedder.info
    return {
        "ready": embedder.ready,
        "enabled": config.ENABLED,
        "model": {
            "pack": config.MODEL_PACK,
            "ctx_id": config.CTX_ID,
            "det_size": config.DET_SIZE,
            "providers": info.providers if info else [],
            "load_seconds": info.load_seconds if info else None,
        },
        "gpu_device_label": config.GPU_DEVICE_LABEL,
    }
