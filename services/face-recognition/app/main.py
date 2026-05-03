"""FastAPI entrypoint for the face-recognition service.

Lifespan ordering:
  embedder.prepare()  →  db.initialize() + migrate()  →  qdrant ensure
  →  mqtt bridge.start()
The MQTT bridge starts last because it's the first component that can
actually trigger pipeline work; everything it depends on must be ready.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from api import admin as admin_api
from api import cameras as cameras_api
from api import detections as detections_api
from api import face_images as face_images_api
from api import identify as identify_api
from api import people as people_api
from db import db
from embedder import embedder
from face_qdrant import vector_store
from mqtt_bridge import bridge as mqtt_bridge
from retention import sweeper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("face-recognition")

# Quiet third-party libraries that log every HTTP call at INFO. The qdrant
# client routes through httpx for every query; uvicorn's access logger fires
# on every Docker /health poll (every 30s). Both drown out the lines we
# actually care about.
for noisy in ("httpx", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


class _SuppressHealthAccessLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # uvicorn access format: '127.0.0.1:xxxxx - "GET /health HTTP/1.1" 200 OK'
        return "/health" not in msg


logging.getLogger("uvicorn.access").addFilter(_SuppressHealthAccessLog())


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

    if config.MQTT_ENABLED:
        await mqtt_bridge.start()
    else:
        logger.warning("FACE_REC_MQTT_ENABLED=false — skipping MQTT bridge")

    await sweeper.start()

    logger.info("Service ready on port %d", config.PORT)
    try:
        yield
    finally:
        logger.info("Shutting down face-recognition service")
        await sweeper.stop()
        if config.MQTT_ENABLED:
            await mqtt_bridge.stop()
        await db.close()


app = FastAPI(title="HavenCore Face Recognition", lifespan=lifespan)
app.include_router(people_api.router)
app.include_router(detections_api.router)
app.include_router(face_images_api.router)
app.include_router(identify_api.router)
app.include_router(cameras_api.router)
app.include_router(admin_api.router)


@app.get("/health")
async def health():
    info = embedder.info
    db_ok = await db.health() if config.ENABLED else False
    qdrant_ok = vector_store.health() if config.ENABLED else False
    mqtt_connected = mqtt_bridge.is_connected() if config.ENABLED and config.MQTT_ENABLED else False
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
        "mqtt": {
            "enabled": config.MQTT_ENABLED,
            "connected": mqtt_connected,
            "broker": f"{config.MQTT_BROKER}:{config.MQTT_PORT}",
            "subscribed_topics": mqtt_bridge.subscribed_topics() if config.MQTT_ENABLED else [],
        },
        "retention": sweeper.info(),
    }
