"""People + enrollment endpoints.

Atomicity note: enrollment writes to disk → Qdrant → Postgres in that
order. A failure between Qdrant upsert and DB insert leaves an orphan
Qdrant point + JPEG (the row never lands). The Step 8 admin
`rebuild-embeddings` endpoint will reconcile drift; for v1 we accept
this and surface the failure as a 500 so the client can retry.
"""

import logging
import uuid
from pathlib import Path

import asyncpg
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Path as PathParam, UploadFile

import config
from db import db
from embedder import embedder
from face_qdrant import vector_store
from models import (
    EnrollmentResult,
    FaceImageOut,
    PersonCreate,
    PersonDetailOut,
    PersonOut,
)


logger = logging.getLogger("face-recognition.api.people")

router = APIRouter(prefix="/api/people", tags=["people"])


ENROLLMENT_DIR = Path(config.SNAPSHOT_DIR) / "enrollments"


@router.post("", response_model=PersonOut, status_code=201)
async def create_person(payload: PersonCreate) -> PersonOut:
    try:
        row = await db.create_person(
            name=payload.name,
            access_level=payload.access_level,
            notes=payload.notes,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail=f"person '{payload.name}' already exists")
    return PersonOut(**row)


@router.get("", response_model=list[PersonOut])
async def list_people() -> list[PersonOut]:
    rows = await db.list_people()
    return [PersonOut(**r) for r in rows]


@router.get("/{person_id}", response_model=PersonDetailOut)
async def get_person(person_id: uuid.UUID = PathParam(...)) -> PersonDetailOut:
    person = await db.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")
    images = await db.list_face_images(person_id)
    return PersonDetailOut(
        **person,
        images=[FaceImageOut(**img) for img in images],
    )


@router.post(
    "/{person_id}/images",
    response_model=EnrollmentResult,
    status_code=201,
)
async def enroll_image(
    person_id: uuid.UUID = PathParam(...),
    file: UploadFile = File(...),
    is_primary: bool = Form(False),
    source: str = Form("upload"),
) -> EnrollmentResult:
    person = await db.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    faces = embedder.detect_and_embed(img)
    if not faces:
        raise HTTPException(status_code=422, detail="no face detected in image")

    # Highest detector confidence wins. Real quality scoring (blur, pose,
    # bbox area, brightness) lands in step 4; until then det_score is the
    # best signal we have and gets stored as quality_score.
    faces_sorted = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
    best = faces_sorted[0]
    quality_score = float(best.det_score)

    face_image_id = uuid.uuid4()
    qdrant_point_id = uuid.uuid4()

    person_dir = ENROLLMENT_DIR / str(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    img_path = person_dir / f"{face_image_id}.jpg"
    if not cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise HTTPException(status_code=500, detail="failed to write enrollment image to disk")

    try:
        vector_store.upsert_point(
            point_id=str(qdrant_point_id),
            vector=best.normed_embedding.tolist(),
            payload={
                "person_id": str(person_id),
                "face_image_id": str(face_image_id),
                "source": source,
            },
        )
    except Exception as e:
        img_path.unlink(missing_ok=True)
        logger.exception("Qdrant upsert failed during enrollment")
        raise HTTPException(status_code=500, detail=f"qdrant upsert failed: {e}")

    try:
        row = await db.insert_face_image(
            face_image_id=face_image_id,
            person_id=person_id,
            path=str(img_path),
            qdrant_point_id=qdrant_point_id,
            source=source,
            quality_score=quality_score,
            is_primary=is_primary,
        )
    except Exception as e:
        # Best-effort cleanup of file; the orphan Qdrant point will be
        # caught by the step-8 reconcile pass. Surface a 500 either way.
        img_path.unlink(missing_ok=True)
        logger.exception("DB insert failed after Qdrant upsert (orphan point %s)", qdrant_point_id)
        raise HTTPException(status_code=500, detail=f"db insert failed: {e}")

    return EnrollmentResult(
        id=row["id"],
        qdrant_point_id=qdrant_point_id,
        quality_score=quality_score,
        faces_detected=len(faces),
    )
