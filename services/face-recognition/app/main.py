"""FastAPI entrypoint for the face-recognition service.

Step 2 scope: model loads on GPU, Postgres pool initializes + migrations
run, Qdrant `faces` collection is bootstrapped. /health reports each.
HA snapshot capture, MQTT bridge, and the detection pipeline arrive in
later steps.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from api import detections as detections_api
from api import people as people_api
from db import db
from embedder import embedder
from face_qdrant import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("face-recognition")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.ENABLED:
        logger.warning("FACE_REC_ENABLED=false — skipping startup")
        yield
        return

    logger.info("Loading InsightFace %s ...", config.MODEL_PACK)
    embedder.prepare()

    await db.initialize()
    await db.migrate()

    vector_store.ensure_collection()

    logger.info("Service ready on port %d", config.PORT)
    try:
        yield
    finally:
        logger.info("Shutting down face-recognition service")
        await db.close()


app = FastAPI(title="HavenCore Face Recognition", lifespan=lifespan)
app.include_router(people_api.router)
app.include_router(detections_api.router)


@app.get("/health")
async def health():
    info = embedder.info
    db_ok = await db.health() if config.ENABLED else False
    qdrant_ok = vector_store.health() if config.ENABLED else False
    return {
        "ready": embedder.ready and db_ok and qdrant_ok,
        "enabled": config.ENABLED,
        "model": {
            "pack": config.MODEL_PACK,
            "ctx_id": config.CTX_ID,
            "det_size": config.DET_SIZE,
            "providers": info.providers if info else [],
            "load_seconds": info.load_seconds if info else None,
        },
        "gpu_device_label": config.GPU_DEVICE_LABEL,
        "db": "ok" if db_ok else ("disabled" if not config.ENABLED else "error"),
        "qdrant": {
            "status": "ok" if qdrant_ok else ("disabled" if not config.ENABLED else "error"),
            "collection": vector_store.collection_name,
            "dim": vector_store.dim,
        },
    }
