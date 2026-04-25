"""Pydantic schemas for the face-recognition HTTP API."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Mirrors the access_level values the plan calls out. Schema doesn't enforce
# a CHECK constraint yet; this is the source of truth at the API boundary.
ACCESS_LEVEL_PATTERN = r"^(unknown|resident|guest|blocked)$"


class PersonCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    access_level: str = Field("unknown", pattern=ACCESS_LEVEL_PATTERN)
    notes: Optional[str] = None


class PersonOut(BaseModel):
    id: UUID
    name: str
    access_level: str
    notes: Optional[str] = None
    image_count: int = 0
    created_at: datetime
    updated_at: datetime


class FaceImageOut(BaseModel):
    id: UUID
    path: str
    is_primary: bool
    source: str
    quality_score: Optional[float] = None
    created_at: datetime


class PersonDetailOut(PersonOut):
    images: list[FaceImageOut]


class EnrollmentResult(BaseModel):
    id: UUID
    qdrant_point_id: UUID
    quality_score: float
    faces_detected: int


class PipelineResult(BaseModel):
    """Result of a single trigger event through the detection pipeline.

    `outcome` enumerates every terminal state so the caller can branch on a
    string instead of inferring from null fields:
      identified  — top-1 cosine similarity > MATCH_THRESHOLD
      unknown     — face(s) found, top-1 below threshold
      no_face     — frames captured, but nothing cleared QUALITY_FLOOR
      no_frames   — burst returned zero decoded frames
    """

    outcome: str
    event_id: UUID
    captured_at: datetime
    camera: str
    detection_id: Optional[UUID] = None
    person_id: Optional[UUID] = None
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    snapshot_path: Optional[str] = None
    frames_processed: int = 0
    faces_kept: int = 0
    embedding_contributed: bool = False
