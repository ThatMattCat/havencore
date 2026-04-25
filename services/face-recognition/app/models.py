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
