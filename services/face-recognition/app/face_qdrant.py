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
from qdrant_client.models import Distance, PointStruct, VectorParams


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


vector_store = FaceVectorStore()
