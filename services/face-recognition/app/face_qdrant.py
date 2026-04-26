"""Qdrant access for the face-recognition service.

File is named `face_qdrant.py` (not `qdrant_client.py` as the plan
originally specified) so it doesn't shadow the upstream `qdrant_client`
package — `from qdrant_client import QdrantClient` would otherwise resolve
to this module and break the import.
"""

import logging
import os

from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


logger = logging.getLogger("face-recognition.qdrant")


COLLECTION_NAME = os.getenv("FACE_REC_QDRANT_COLLECTION", "faces")
# ArcFace R100 (buffalo_l) emits 512-d embeddings. Pinned in code rather than
# env: this dimension is a property of the model, not a deployment knob.
EMBEDDING_DIM = 512


class FaceVectorStore:
    def __init__(self) -> None:
        self.host = os.getenv("QDRANT_HOST", "qdrant")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = COLLECTION_NAME
        self.dim = EMBEDDING_DIM
        self.client = QdrantClient(host=self.host, port=self.port)

    def ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection_name)
            logger.info("Qdrant collection '%s' already exists", self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection '%s' (size=%d, cosine)",
                        self.collection_name, self.dim)

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed: %s", e)
            return False

    def upsert_point(self, point_id: str, vector: Iterable[float], payload: dict) -> None:
        """Insert (or replace) a single embedding point.

        `vector` should already be L2-normalized — InsightFace's
        `Face.normed_embedding` is, and the collection uses cosine distance,
        so no extra normalization is needed here.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=str(point_id), vector=list(vector), payload=payload)],
        )

    def query(self, vector: Iterable[float], limit: int = 3) -> list:
        """Return up to `limit` nearest points (with payload) by cosine similarity.

        Each hit exposes `.id`, `.score` (cosine similarity in [-1, 1] for
        unit vectors; effectively [0, 1] for ArcFace embeddings), and
        `.payload` (the dict stored at upsert time).
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def delete_point(self, point_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[str(point_id)]),
        )

    def delete_points(self, point_ids: Iterable[str]) -> int:
        """Delete a batch of points by id; returns the count attempted."""
        ids = [str(p) for p in point_ids]
        if not ids:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)

    def scroll_all_payload(self) -> list[tuple[str, dict]]:
        """Walk every point in the collection, returning (id, payload) pairs.

        Used by the rebuild-embeddings admin endpoint to find orphans (points
        whose payload.face_image_id has no matching face_images row).
        """
        out: list[tuple[str, dict]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=512,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for p in points:
                out.append((str(p.id), p.payload or {}))
            if offset is None:
                break
        return out

    def delete_by_person(self, person_id: str) -> None:
        """Cascade-delete every point with payload.person_id == person_id.

        Used when a person is removed via DELETE /api/people/{id}. Caller is
        responsible for removing the person row + face_images files; this is
        the Qdrant side of the cascade.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(
                        key="person_id",
                        match=MatchValue(value=str(person_id)),
                    )]
                )
            ),
        )


vector_store = FaceVectorStore()
